#!/usr/bin/env python3
"""
Standalone latency benchmark — tests RepViT and/or StarNet model back-to-back
with proper reparameterization fusion. Run BEFORE generating paper numbers.

Usage:
  # Test RepViT only
  python3 benchmark_latency.py \
      --model_repvit runs_repvit_models/detect/train/weights/best_mcu.pt \
      --img_size 384

  # Test StarNet only
  python3 benchmark_latency.py \
      --model_star runs_starnet_models/detect/train/weights/best_mcu.pt \
      --img_size 384

  # Test BOTH (recommended — back-to-back comparison)
  python3 benchmark_latency.py \
      --model_repvit runs_repvit_models/detect/train/weights/best_mcu.pt \
      --model_star runs_starnet_models/detect/train/weights/best_mcu.pt \
      --img_size 384

  # Multi-resolution sweep
  python3 benchmark_latency.py \
      --model_repvit runs_repvit_models/detect/train/weights/best_mcu.pt \
      --model_star runs_starnet_models/detect/train/weights/best_mcu.pt \
      --img_sizes 320 384 512
"""

import os, sys, time, copy, argparse, gc
import torch
import torch.nn as nn
import numpy as np

# ──────────────────────────────────────────────────────────────
# Fuse RepDWConv multi-branch blocks → single 3x3 DW conv
# ──────────────────────────────────────────────────────────────
def fuse_reparam(model):
    n = 0
    for m in model.modules():
        if hasattr(m, 'fuse') and hasattr(m, 'fused') and not m.fused:
            m.fuse()
            n += 1
    return n


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model, img_size, device="cpu"):
    """Simple FLOPs estimation via hooks."""
    flops_list = []

    def conv_hook(module, inp, out):
        batch = out.shape[0]
        out_ch = out.shape[1]
        oh, ow = out.shape[2], out.shape[3]
        in_ch = module.in_channels
        kh, kw = module.kernel_size
        groups = module.groups
        f = 2 * out_ch * oh * ow * (in_ch // groups) * kh * kw
        if module.bias is not None:
            f += out_ch * oh * ow
        flops_list.append(f)

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))

    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, img_size, img_size, device=device))

    for h in hooks:
        h.remove()

    return sum(flops_list)


# ──────────────────────────────────────────────────────────────
# GPU latency (with proper cuda.synchronize)
# ──────────────────────────────────────────────────────────────
def measure_gpu(model, img_size, warmup=100, runs=300):
    device = next(model.parameters()).device
    if device.type != "cuda":
        return None, None
    
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times = sorted(times)
    # Trim 10% outliers from each end
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim]
    return float(np.mean(trimmed)), float(np.std(trimmed))


# ──────────────────────────────────────────────────────────────
# CPU latency (isolated — deepcopy to CPU, gc, no interference)
# ──────────────────────────────────────────────────────────────
def measure_cpu(model, img_size, warmup=20, runs=100):
    # Force garbage collection before measurement
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model_cpu = copy.deepcopy(model).cpu().eval()
    
    # Verify all parameters are on CPU
    for name, p in model_cpu.named_parameters():
        assert p.device.type == "cpu", f"{name} still on {p.device}"
    for name, b in model_cpu.named_buffers():
        assert b.device.type == "cpu", f"Buffer {name} still on {b.device}"

    x = torch.randn(1, 3, img_size, img_size)

    with torch.no_grad():
        for _ in range(warmup):
            model_cpu(x)

    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model_cpu(x)
            times.append((time.perf_counter() - t0) * 1000)

    del model_cpu
    gc.collect()

    times = sorted(times)
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim]
    return float(np.mean(trimmed)), float(np.std(trimmed))


# ──────────────────────────────────────────────────────────────
# Load & prepare model
# ──────────────────────────────────────────────────────────────
def load_model(model_cls, path, nc=14, device="cpu"):
    model = model_cls(num_classes=nc)
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["ema_state_dict", "model_state_dict", "state_dict"]:
            if k in ckpt:
                state = ckpt[k]
                break
        else:
            state = ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    else:
        model = ckpt
    model = model.to(device).eval()
    return model


def benchmark_model(name, model, img_sizes, device):
    """Run full benchmark for one model across multiple resolutions."""
    rows = []
    params = count_params(model)

    # Fuse RepDWConv blocks
    n_fused = fuse_reparam(model)
    print(f"  [fuse] {n_fused} RepDWConv blocks fused")

    # Verify INT8 safety
    bad = 0
    for mname, m in model.named_modules():
        if m.__class__.__name__ == "SiLU" and isinstance(m, nn.SiLU):
            bad += 1
    if bad > 0:
        print(f"  ⚠️  {bad} real nn.SiLU found (will hurt INT8 quantization)")
    else:
        print(f"  ✅ All activations INT8-safe")

    for sz in img_sizes:
        print(f"\n  --- {name} @ {sz}×{sz} ---")

        flops = estimate_flops(copy.deepcopy(model).cpu(), sz)
        gflops = flops / 1e9

        gpu_ms, gpu_std = measure_gpu(model, sz)
        cpu_ms, cpu_std = measure_cpu(model, sz)

        row = {
            "Model": name,
            "Resolution": f"{sz}×{sz}",
            "Params_M": f"{params/1e6:.3f}",
            "GFLOPs": f"{gflops:.2f}",
            "GPU_ms": f"{gpu_ms:.2f}" if gpu_ms else "-",
            "GPU_FPS": f"{1000/gpu_ms:.1f}" if gpu_ms else "-",
            "CPU_ms": f"{cpu_ms:.2f}",
            "CPU_FPS": f"{1000/cpu_ms:.1f}",
        }
        rows.append(row)

        if gpu_ms:
            print(f"    GPU: {gpu_ms:.2f} ± {gpu_std:.2f} ms  ({1000/gpu_ms:.1f} FPS)")
        print(f"    CPU: {cpu_ms:.2f} ± {cpu_std:.2f} ms  ({1000/cpu_ms:.1f} FPS)")
        print(f"    FLOPs: {gflops:.2f} G,  Params: {params/1e6:.3f}M")

    return rows


def main():
    p = argparse.ArgumentParser(description="Latency benchmark for MCUDetector variants")
    p.add_argument("--model_repvit", type=str, default=None, help="RepViT model checkpoint")
    p.add_argument("--model_star", type=str, default=None, help="StarNet model checkpoint")
    p.add_argument("--img_size", type=int, default=None, help="Single resolution")
    p.add_argument("--img_sizes", type=int, nargs="+", default=None, help="Multiple resolutions")
    p.add_argument("--num_classes", type=int, default=14)
    args = p.parse_args()

    if not args.model_repvit and not args.model_star:
        print("Error: specify at least one of --model_repvit or --model_star")
        sys.exit(1)

    # Determine resolutions
    if args.img_sizes:
        img_sizes = args.img_sizes
    elif args.img_size:
        img_sizes = [args.img_size]
    else:
        img_sizes = [320, 384]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  MCUDetector Latency Benchmark")
    print(f"  Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Resolutions: {img_sizes}")
    print(f"{'='*65}")

    all_rows = []

    if args.model_repvit:
        print(f"\n\n{'─'*65}")
        print(f"  REPVIT MODEL: {args.model_repvit}")
        print(f"{'─'*65}")
        from model_repvit import MCUDetector as RepViTDetector
        model = load_model(RepViTDetector, args.model_repvit, args.num_classes, device)
        all_rows += benchmark_model("RepViT (V2-Lite)", model, img_sizes, device)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.model_star:
        print(f"\n\n{'─'*65}")
        print(f"  STARNET MODEL: {args.model_star}")
        print(f"{'─'*65}")
        from model_star import MCUDetector as StarDetector
        model = load_model(StarDetector, args.model_star, args.num_classes, device)
        all_rows += benchmark_model("StarNet (V3-Star)", model, img_sizes, device)
        del model
        gc.collect()

    # ── Print comparison table ──
    print(f"\n\n{'='*85}")
    print(f"  {'Model':<22} {'Res':<8} {'Params':>7} {'GFLOPs':>7} "
          f"{'GPU ms':>8} {'GPU FPS':>8} {'CPU ms':>8} {'CPU FPS':>8}")
    print(f"  {'─'*80}")
    for r in all_rows:
        print(f"  {r['Model']:<22} {r['Resolution']:<8} {r['Params_M']:>6}M {r['GFLOPs']:>7} "
              f"{r['GPU_ms']:>7}ms {r['GPU_FPS']:>7} {r['CPU_ms']:>7}ms {r['CPU_FPS']:>7}")
    print(f"{'='*85}")

    # Save CSV
    import csv
    csv_path = "latency_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n  📄 Results saved to {csv_path}")


if __name__ == "__main__":
    main()
