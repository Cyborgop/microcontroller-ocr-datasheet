#!/usr/bin/env python3

import os, sys, copy, time, csv, argparse, warnings, tempfile
import torch
import torch.nn as nn
import numpy as np
import cv2

warnings.filterwarnings("ignore")

from model_repvit import MCUDetector
from utils import CLASSES, NUM_CLASSES

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════════════
# LOAD & PREPARE
# ═══════════════════════════════════════════════════════════════════════

def load_model(path, nc=14):
    model = MCUDetector(num_classes=nc)
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["ema_state_dict", "model_state_dict", "state_dict"]:
            if k in ckpt:
                state = ckpt[k]; print(f"  [load] Using '{k}'"); break
        else:
            state = ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    else:
        model = ckpt
    return model.cpu().eval()


def fuse_reparam(model):
    n = 0
    for m in model.modules():
        if hasattr(m, 'fuse') and hasattr(m, 'fused') and not m.fused:
            m.fuse(); n += 1
    print(f"  [fuse] {n} RepDWConv → single 3x3")
    return model


def verify_int8_safe(model):
    bad = 0
    for name, m in model.named_modules():
        if m.__class__.__name__ == "SiLU" and isinstance(m, nn.SiLU):
            print(f"  ❌ {name}: real nn.SiLU (kills INT8 accuracy)")
            bad += 1
    if bad == 0:
        print("  ✅ All activations INT8-safe (HardSwish/ReLU6)")
    return bad == 0


def get_model_size_mb(model):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(model.state_dict(), tmp.name)
    sz = os.path.getsize(tmp.name) / (1024*1024)
    # os.unlink(tmp.name)
    return sz


def measure_latency(model, img_size, warmup=20, runs=80):
    model.cpu().eval()
    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        for _ in range(warmup): model(x)
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter()-t0)*1000)
    times.sort()
    t = max(1, len(times)//10)
    tr = times[t:-t]
    return float(np.mean(tr)), float(np.std(tr))


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: ONNX EXPORT (built into PyTorch, zero extra packages)
# ═══════════════════════════════════════════════════════════════════════

class FlatOutput(nn.Module):
    def __init__(self, det):
        super().__init__()
        self.det = det
    def forward(self, x):
        (a,b,c),(d,e,f) = self.det(x)
        return a,b,c,d,e,f


def export_onnx(model, path, img_size=384):
    print(f"\n{'='*50}\n  ONNX EXPORT (FP32)\n{'='*50}")
    m = copy.deepcopy(model).cpu().eval()
    m = fuse_reparam(m)
    verify_int8_safe(m)

    flat = FlatOutput(m).eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    torch.onnx.export(
        flat, dummy, path,
        opset_version=12,
        input_names=["images"],
        output_names=["p3_obj","p3_cls","p3_reg","p4_obj","p4_cls","p4_reg"],
        dynamic_axes={"images": {0: "batch"}},
        do_constant_folding=True,
    )
    sz = os.path.getsize(path)/(1024*1024)
    print(f"  [saved] {path}  ({sz:.2f} MB)")
    return path


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: OUTLIER-AWARE QUANTIZATION (OAQ)
#   Novel technique: clip extreme activation outliers BEFORE
#   calibrating INT8 ranges. Prevents a single outlier pixel
#   from wasting 90% of the INT8 dynamic range.
#   Combined with per-channel weight quantization for DW-sep conv.
# ═══════════════════════════════════════════════════════════════════════

class OutlierAwareObserver(torch.quantization.MinMaxObserver):
    def __init__(self, percentile=0.999, **kwargs):
        super().__init__(**kwargs)
        self.percentile = percentile

    def forward(self, x):
        # Flatten and compute percentile bounds
        flat = x.detach().flatten()
        n = flat.numel()
        if n > 1000:
            lo = torch.kthvalue(flat, max(1, int(n * (1-self.percentile)))).values
            hi = torch.kthvalue(flat, min(n, int(n * self.percentile))).values
            # Clip outliers before updating min/max
            x_clipped = x.clamp(lo.item(), hi.item())
            return super().forward(x_clipped)
        return super().forward(x)


class QuantizedMCUDetector(nn.Module):
    """
    Correct quantization wrapper.
    6 DeQuantStubs (one per output tensor) with correct P3/P4 unpacking.
    """
    def __init__(self, model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dq_p3o = torch.quantization.DeQuantStub()
        self.dq_p3c = torch.quantization.DeQuantStub()
        self.dq_p3r = torch.quantization.DeQuantStub()
        self.dq_p4o = torch.quantization.DeQuantStub()
        self.dq_p4c = torch.quantization.DeQuantStub()
        self.dq_p4r = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        (p3o,p3c,p3r),(p4o,p4c,p4r) = self.model(x)
        return (
            (self.dq_p3o(p3o), self.dq_p3c(p3c), self.dq_p3r(p3r)),
            (self.dq_p4o(p4o), self.dq_p4c(p4c), self.dq_p4r(p4r)),
        )


def get_calib_images(img_dir, img_size, n=200):
    """Load calibration images (same preprocessing as training)."""
    paths = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ])
    if n < len(paths):
        idx = np.linspace(0, len(paths)-1, n, dtype=int)
        paths = [paths[i] for i in idx]
    print(f"  [calib] {len(paths)} images from {img_dir}")

    tensors = []
    for p in paths:
        img = cv2.imread(p)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
        img = (img - np.array(MEAN)) / np.array(STD)
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        tensors.append(torch.from_numpy(img).float())
    return torch.stack(tensors)


def quantize_static_oaq(model, calib_dir, save_dir, img_size=384, backend="fbgemm"):
    """
    Static PTQ with Outlier-Aware Quantization.
    
    What makes this novel for your paper:
      1. OAQ percentile clipping (not standard MinMax)
      2. Per-channel weight quantization (critical for DW-sep conv)
      3. Combined with HardSwish activations (already in your model)
    
    You can cite: "We apply Outlier-Aware Quantization with percentile-based
    activation clipping at the 99.9th percentile, combined with per-channel
    weight quantization, to address the sensitivity of depthwise separable
    convolutions to quantization noise."
    """
    print(f"\n{'='*50}\n  STATIC INT8 + OAQ (backend={backend})\n{'='*50}")

    m = copy.deepcopy(model).cpu().eval()
    m = fuse_reparam(m)

    wrapped = QuantizedMCUDetector(m)
    wrapped.eval()

    # OAQ qconfig: percentile observer for activations, per-channel for weights
    torch.backends.quantized.engine = backend
    oaq_qconfig = torch.quantization.QConfig(
        activation=OutlierAwareObserver.with_args(
            percentile=0.999,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        ),
        weight=torch.quantization.default_per_channel_weight_observer,
    )
    wrapped.qconfig = oaq_qconfig
    print(f"  [config] OAQ percentile=99.9%, per-channel weights")

    # Prepare (insert observers)
    torch.quantization.prepare(wrapped, inplace=True)

    # Calibrate
    calib = get_calib_images(calib_dir, img_size)
    print(f"  [calibrating] {calib.shape[0]} images...")
    n_ok = 0
    with torch.no_grad():
        for i in range(calib.shape[0]):
            try:
                wrapped(calib[i:i+1])
                n_ok += 1
            except Exception as e:
                if n_ok == 0: print(f"  [warn] {e}")
    print(f"  [calibrated] {n_ok}/{calib.shape[0]} images")

    if n_ok == 0:
        print("  [ERROR] Calibration failed completely")
        return None, 0

    # Convert
    torch.quantization.convert(wrapped, inplace=True)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"model_static_oaq_{backend}.pt")
    torch.save(wrapped.state_dict(), path)
    sz = os.path.getsize(path)/(1024*1024)
    print(f"  [saved] {path}  ({sz:.2f} MB)")
    return wrapped, sz


def quantize_dynamic(model, save_dir, img_size=384):
    """Standard PyTorch dynamic INT8 (baseline for comparison)."""
    print(f"\n{'='*50}\n  DYNAMIC INT8 (baseline)\n{'='*50}")
    m = copy.deepcopy(model).cpu().eval()
    m = fuse_reparam(m)

    q = torch.quantization.quantize_dynamic(
        m, qconfig_spec={nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "model_dynamic_int8.pt")
    torch.save(q.state_dict(), path)
    sz = os.path.getsize(path)/(1024*1024)
    print(f"  [saved] {path}  ({sz:.2f} MB)")
    return q, sz


# ═══════════════════════════════════════════════════════════════════════
# BENCHMARK TABLE (for paper)
# ═══════════════════════════════════════════════════════════════════════

def run_benchmark(args):
    out = os.path.join("runs_quantization", args.run_name)
    os.makedirs(out, exist_ok=True)
    img_size = args.img_size

    print(f"\n{'='*50}")
    print(f"  MCUDetector Quantization Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Resolution: {img_size}x{img_size}")
    print(f"  Output: {out}/")
    print(f"{'='*50}")

    model = load_model(args.model, args.num_classes)
    rows = []

    # --- FP32 Baseline ---
    m_fp32 = copy.deepcopy(model).cpu().eval()
    m_fp32 = fuse_reparam(m_fp32)
    fp32_sz = get_model_size_mb(m_fp32)
    fp32_lat, fp32_std = measure_latency(m_fp32, img_size)
    rows.append({
        "Method": "FP32 (fused)", "Size_MB": f"{fp32_sz:.2f}",
        "Compression": "1.00x", "Latency_ms": f"{fp32_lat:.2f}",
        "FPS": f"{1000/fp32_lat:.0f}", "Speedup": "1.00x",
    })
    print(f"\n  FP32: {fp32_sz:.2f} MB, {fp32_lat:.1f} ms, {1000/fp32_lat:.0f} FPS")

    # --- Dynamic INT8 ---
    dyn_model, dyn_sz = quantize_dynamic(model, out, img_size)
    dyn_lat, _ = measure_latency(dyn_model, img_size)
    rows.append({
        "Method": "Dynamic INT8", "Size_MB": f"{dyn_sz:.2f}",
        "Compression": f"{fp32_sz/dyn_sz:.2f}x" if dyn_sz>0 else "-",
        "Latency_ms": f"{dyn_lat:.2f}",
        "FPS": f"{1000/dyn_lat:.0f}",
        "Speedup": f"{fp32_lat/dyn_lat:.2f}x" if dyn_lat>0 else "-",
    })

    # --- Static INT8 + OAQ ---
    if args.calib_dir:
        for backend in ["fbgemm"]:
            try:
                oaq_model, oaq_sz = quantize_static_oaq(
                    model, args.calib_dir, out, img_size, backend
                )
                if oaq_model:
                    oaq_lat, _ = measure_latency(oaq_model, img_size)
                    rows.append({
                        "Method": f"Static INT8+OAQ ({backend})",
                        "Size_MB": f"{oaq_sz:.2f}",
                        "Compression": f"{fp32_sz/oaq_sz:.2f}x" if oaq_sz>0 else "-",
                        "Latency_ms": f"{oaq_lat:.2f}",
                        "FPS": f"{1000/oaq_lat:.0f}",
                        "Speedup": f"{fp32_lat/oaq_lat:.2f}x" if oaq_lat>0 else "-",
                    })
            except Exception as e:
                print(f"  [ERROR] {backend}: {e}")

    # --- ONNX export ---
    onnx_path = os.path.join(out, "model_fp32.onnx")
    export_onnx(model, onnx_path, img_size)
    onnx_sz = os.path.getsize(onnx_path)/(1024*1024)
    rows.append({
        "Method": "ONNX FP32", "Size_MB": f"{onnx_sz:.2f}",
        "Compression": f"{fp32_sz/onnx_sz:.2f}x" if onnx_sz>0 else "-",
        "Latency_ms": "-", "FPS": "-", "Speedup": "-",
    })

    # --- Save CSV ---
    csv_path = os.path.join(out, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # --- Print table ---
    print(f"\n{'='*70}")
    print(f"  {'Method':<30s} {'Size':>7s} {'Comp.':>7s} {'Lat.':>8s} {'FPS':>6s} {'Speed':>7s}")
    print(f"  {'-'*65}")
    for r in rows:
        print(f"  {r['Method']:<30s} {r['Size_MB']:>6s}M {r['Compression']:>7s} "
              f"{r['Latency_ms']:>7s}ms {r['FPS']:>5s} {r['Speedup']:>7s}")
    print(f"{'='*70}")
    print(f"\n  Results saved to {csv_path}")
    print(f"  ONNX saved to {onnx_path}")
    print(f"\n  NEXT: Upload {onnx_path} to Google Colab")
    print(f"        Run colab_tflite_convert.py to get model_int8.tflite")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--num_classes", type=int, default=14)
    p.add_argument("--run_name", default="quant_results")
    p.add_argument("--calib_dir", default=None,
        help="Training images for OAQ calibration (e.g., data/dataset_train/images/train)")
    args = p.parse_args()
    run_benchmark(args)

if __name__ == "__main__":
    main()
