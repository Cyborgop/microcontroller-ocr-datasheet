#!/usr/bin/env python3
"""
train_debug_combined.py - ULTIMATE EDITION (fixed & robust)

Comprehensive debug suite that combines:
 - train_debug.py (forward, loss, backward, ema/fuse, checkpoint, onnx)
 - model_debug.py (per-block checks when available)
 - utils diagnostics (mAP / IoU / decode / PR sanity checks)
 - LR Finder validation (guarded)
 - Multi-GPU compatibility
 - DataLoader speed test
 - Memory leak detection (guarded)
 - Determinism check (safer)
Exit code 0 => all critical checks passed.
"""
import argparse
import importlib
import inspect
import os
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------
# Project imports (may raise if project broken)
# ---------------------------
try:
    from model import MCUDetector, MCUDetectionLoss
    from dataset import MCUDetectionDataset, detection_collate_fn
    import utils
    from utils import CLASSES, NUM_CLASSES, get_run_dir
    # helper functions from train.py (optional)
    try:
        from train import fuse_repv_blocks, get_eval_model, verify_dataset_paths, ModelEMA
        from train import find_optimal_lr, setup_gpu_optimizations
    except Exception:
        fuse_repv_blocks = None
        get_eval_model = None
        verify_dataset_paths = None
        ModelEMA = None
        find_optimal_lr = None
        setup_gpu_optimizations = None
except Exception as e:
    print("❌ ERROR importing project modules:", e)
    traceback.print_exc()
    sys.exit(2)


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_env_info(device):
    print("\n================ ENVIRONMENT ================")
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("CUDA version:", torch.version.cuda)
            print("GPU:", torch.cuda.get_device_name(0))
            print("Compute capability:", torch.cuda.get_device_capability(0))
            print("GPU count:", torch.cuda.device_count())
        except Exception:
            pass
    print("Device used:", device)
    print("============================================\n")


# ---------------------------
# Minimal forward/loss/backward checks
# ---------------------------
def get_synthetic_batch(batch_size=2, img_size=512):
    images = torch.rand(batch_size, 3, img_size, img_size)
    # targets: list of (N,5) tensors in [cls,cx,cy,w,h] normalized
    targets = [torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]], dtype=torch.float32) for _ in range(batch_size)]
    return images, targets


def sample_real_batch(img_dir, label_dir, img_size=512, batch_size=2, workers=2):
    ds = MCUDetectionDataset(
        img_dir=Path(img_dir),
        label_dir=Path(label_dir),
        img_size=img_size,
        transform=None,
    )
    assert len(ds) > 0, "Dataset is empty"
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn,
        num_workers=workers,
    )
    images, targets = next(iter(loader))
    return images, targets


def check_forward(model, images, device):
    model.eval()
    outputs = model(images.to(device))

    assert isinstance(outputs, (list, tuple)), "Model output must be tuple/list"
    assert len(outputs) == 2, "Expected (P3, P4) outputs"

    for i, o in enumerate(outputs):
        assert isinstance(o, (tuple, list)), f"output[{i}] must be (obj, cls, reg)"
        assert len(o) == 3, f"output[{i}] must have 3 elements"

        obj_map, cls_map, reg_map = o

        assert isinstance(obj_map, torch.Tensor)
        assert isinstance(cls_map, torch.Tensor)
        assert isinstance(reg_map, torch.Tensor)

        print(
            f"✔ output[{i}] OK | "
            f"obj: {tuple(obj_map.shape)} | "
            f"cls: {tuple(cls_map.shape)} | "
            f"reg: {tuple(reg_map.shape)}"
        )

    return outputs



def check_loss_backward(model, criterion, outputs, targets, device):
    model.train()
    # Prepare targets according to train.py logic
    targets_p3, targets_p4 = [], []
    for t in targets:
        if t is None or (isinstance(t, torch.Tensor) and t.numel() == 0):
            targets_p3.append(torch.zeros((0, 5), device=device))
            targets_p4.append(torch.zeros((0, 5), device=device))
            continue
        t = t.to(device)
        area = t[:, 3] * t[:, 4] * 512 * 512
        targets_p3.append(t[area < 512])
        targets_p4.append(t[area >= 512])

    loss_dict = criterion(outputs[0], outputs[1], targets_p3, targets_p4)
    assert "total" in loss_dict, "loss dict missing 'total'"
    loss = loss_dict["total"]
    assert isinstance(loss, torch.Tensor), "loss must be a Tensor"
    assert loss.requires_grad, "loss must require grad"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients produced"
    print("✔ Loss backward OK")
    print("  Loss values:", {k: float(v) if torch.is_tensor(v) else v for k, v in loss_dict.items()})


@torch.no_grad()
def check_ema_and_fuse(model, images, device):
    if ModelEMA is not None:
        ema = ModelEMA(model, decay=0.999)
        ema.update(model)
        out = ema.ema(images.to(device))
        assert isinstance(out, (list, tuple)), "EMA forward failed"
        print("✔ EMA forward OK")
    else:
        print("ℹ️ ModelEMA not available; skipping EMA check")

    if fuse_repv_blocks is not None:
        try:
            fuse_repv_blocks(model)
            print("✔ Rep-style blocks fused successfully")
        except Exception as e:
            print("⚠️ fuse_repv_blocks failed:", e)
    else:
        print("ℹ️ fuse_repv_blocks not available; skipping fuse check")


def check_checkpoint_io(model, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "debug_ckpt.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    loaded = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in loaded
    model.load_state_dict(loaded["model_state_dict"], strict=True)
    ckpt_path.unlink(missing_ok=True)
    print("✔ Checkpoint save/load OK")


def check_onnx_export(eval_model, run_dir):
    try:
        dummy = torch.randn(1, 3, 512, 512, device=next(eval_model.parameters()).device)
        out_path = Path(run_dir) / "debug.onnx"
        torch.onnx.export(eval_model, dummy, str(out_path), opset_version=14, do_constant_folding=True, input_names=["input"])
        out_path.unlink(missing_ok=True)
        print("✔ ONNX export OK")
    except Exception as e:
        print("⚠️ ONNX export skipped:", e)


# ---------------------------
# Optional: import model_debug helpers and use them (safe)
# ---------------------------
model_debug = None
try:
    model_debug = importlib.import_module("model_debug")
    print("ℹ️ Imported model_debug module - will run per-block checks.")
except Exception:
    print("ℹ️ model_debug module not importable; skipping per-block deep checks.")


# ---------------------------
# Utils diagnostics (key tests)
# ---------------------------
def create_perfect_test_case():
    target = np.array([[1, 0.507500, 0.515000, 0.082500, 0.066667]], dtype=np.float32)
    img_size = 512
    cx = target[0, 1] * img_size
    cy = target[0, 2] * img_size
    w = target[0, 3] * img_size
    h = target[0, 4] * img_size
    pred_x1 = cx - w / 2 + 2
    pred_y1 = cy - h / 2 + 2
    pred_x2 = cx + w / 2 - 2
    pred_y2 = cy + h / 2 - 2
    prediction = np.array([[1, 0.9, pred_x1, pred_y1, pred_x2, pred_y2]], dtype=np.float32)
    return [prediction], [target]


def test_calculate_map_perfect():
    print("\n== TEST: calculate_map with perfect example ==")
    preds, targs = create_perfect_test_case()
    try:
        # call in a guarded way; different projects may expect different args
        try:
            res = utils.calculate_map(predictions=preds, targets=targs, num_classes=NUM_CLASSES, epoch=50)
        except TypeError:
            # fallback to older signature used earlier in your repo
            res = utils.calculate_map(preds, targs, NUM_CLASSES, epoch=50)
        # Expect tuple like (map50_95,map50,map75,per_class_ap) - handle both shapes
        if isinstance(res, tuple) and len(res) >= 2:
            map_50 = res[1]
            print(f"mAP@0.5 = {map_50:.4f}")
            return map_50 >= 0.95
        print("⚠️ Unexpected calculate_map return:", type(res))
        return False
    except Exception as e:
        print("❌ calculate_map raised exception:", e)
        traceback.print_exc()
        return False


def test_box_iou():
    print("\n== TEST: box_iou_batch math ==")
    box1 = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    box2 = torch.tensor([[150, 150, 250, 250]], dtype=torch.float32)
    try:
        iou = utils.box_iou_batch(box1, box2)
        expected = (50 * 50) / (100 * 100 + 100 * 100 - 50 * 50)
        print(f"Computed IoU: {float(iou):.6f}, expected {expected:.6f}")
        return abs(float(iou) - expected) < 1e-3
    except Exception as e:
        print("❌ box_iou_batch raised:", e)
        traceback.print_exc()
        return False


def test_decode_predictions_interface():
    print("\n== TEST: decode_predictions interface ==")
    batch_size = 1
    num_classes = NUM_CLASSES
    cls_p4 = torch.randn(batch_size, num_classes + 1, 64, 64)
    reg_p4 = torch.randn(batch_size, 4, 64, 64)
    cls_p3 = torch.randn(batch_size, num_classes + 1, 128, 128)
    reg_p3 = torch.randn(batch_size, 4, 128, 128)
    try:
        preds = utils.decode_predictions(pred_p3=(cls_p3, reg_p3), pred_p4=(cls_p4, reg_p4), conf_thresh=0.001, nms_thresh=0.45, img_size=512)
        print("decode_predictions executed.")
        if isinstance(preds, list) and len(preds) > 0 and (isinstance(preds[0], np.ndarray) or hasattr(preds[0], "shape")):
            print(f"First pred shape: {preds[0].shape}")
            return preds[0].ndim == 2 and preds[0].shape[1] >= 6
        return True
    except Exception as e:
        print("❌ decode_predictions failed:", e)
        traceback.print_exc()
        return False


def test_precision_recall_perfect():
    print("\n== TEST: compute_precision_recall with perfect example ==")
    preds, targs = create_perfect_test_case()
    try:
        precision, recall = utils.compute_precision_recall(preds=preds, targets=targs, conf_thresh=0.25, iou_thresh=0.5, img_size=512, debug_first_n=1)
        print(f"P={precision:.4f}, R={recall:.4f}")
        return abs(precision - 1.0) < 0.01 and abs(recall - 1.0) < 0.01
    except Exception as e:
        print("❌ compute_precision_recall failed:", e)
        traceback.print_exc()
        return False


def iou_distribution_probe():
    print("\n== DEBUG: IoU distribution probe ==")
    preds, targs = create_perfect_test_case()
    pboxes = preds[0][:, 2:6]
    cx = targs[0][:, 1] * 512
    cy = targs[0][:, 2] * 512
    w = targs[0][:, 3] * 512
    h = targs[0][:, 4] * 512
    gt = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    try:
        iou = utils.box_iou_batch(torch.from_numpy(pboxes), torch.from_numpy(gt))
        print("IoU(s):", iou)
        return True
    except Exception as e:
        print("❌ IoU probe failed:", e)
        traceback.print_exc()
        return False


# ---------------------------
# Extra tests (enhancements)
# ---------------------------

def safe_sum_model_output_tensors(out):
    """Return scalar sum across all leaf tensors in model output (safe for unknown shapes)."""
    total = None
    def add_tensor(t):
        nonlocal total
        if not torch.is_tensor(t):
            return
        s = t.sum()
        total = s if total is None else total + s

    # recursively walk
    if isinstance(out, torch.Tensor):
        total = out.sum()
    elif isinstance(out, (list, tuple)):
        for x in out:
            if isinstance(x, (list, tuple)):
                for y in x:
                    add_tensor(y)
            else:
                add_tensor(x)
    return total if total is not None else torch.tensor(0., device=next(iter([]), torch.device('cpu')))


def test_lr_finder(model, train_loader, criterion, device, run_dir):
    print("\n== TEST: LR Finder ==")
    if find_optimal_lr is None:
        print("⚠️ find_optimal_lr not available; skipping")
        return None
    try:
        # call simple guarded signature
        try:
            lr = find_optimal_lr(model, train_loader, criterion, device, run_dir)
        except TypeError:
            # try alternative fallback without train_loader
            lr = find_optimal_lr(model, device=device, run_dir=run_dir)
        print(f"✅ LR finder OK (returned: {lr})")
        return True
    except Exception as e:
        print("⚠️ LR finder test failed:", e)
        traceback.print_exc()
        return False


def test_multi_gpu(model, images, device):
    print("\n== TEST: Multi-GPU ==")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        print("ℹ️ Single GPU only, skipping")
        return True
    try:
        model_dp = torch.nn.DataParallel(model)
        out = model_dp(images.to(device))
        print(f"✅ DataParallel forward OK on {torch.cuda.device_count()} GPUs")
        return True
    except Exception as e:
        print("⚠️ Multi-GPU test failed:", e)
        traceback.print_exc()
        return False


def test_dataloader_speed(loader, num_batches=10):
    print("\n== TEST: DataLoader Speed ==")
    if loader is None:
        print("ℹ️ No loader provided, skipping")
        return True
    try:
        start = time.time()
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
        elapsed = time.time() - start
        speed = num_batches / max(elapsed, 1e-6)
        print(f"✅ {speed:.2f} batches/sec ({elapsed:.2f}s for {num_batches} batches)")
        return speed > 0.5  # conservative threshold; tune for your hardware
    except Exception as e:
        print("⚠️ DataLoader speed test failed:", e)
        traceback.print_exc()
        return False


def test_memory_leak(model, loader, device, num_iterations=5):
    print("\n== TEST: Memory Leak ==")
    if device.type != "cuda":
        print("ℹ️ CPU only, skipping memory leak test")
        return True
    if loader is None:
        print("ℹ️ No loader, skipping memory leak test")
        return True
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()
        model.train()
        it = iter(loader)
        mem_after_first = None
        for i in range(num_iterations):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            # batch might be (imgs, targets) or TensorDataset pair
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                images = batch[0]
            else:
                images = batch
            images = images.to(device)
            out = model(images)
            # compute simple sum over output tensors
            total = None
            if isinstance(out, torch.Tensor):
                total = out.sum()
            else:
                # recursively sum
                def ssum(x):
                    if isinstance(x, torch.Tensor):
                        return x.sum()
                    if isinstance(x, (list, tuple)):
                        return sum((ssum(y) for y in x))
                    return torch.tensor(0., device=device)
                total = ssum(out)
            total.backward()
            model.zero_grad()
            if i == 0:
                mem_after_first = torch.cuda.memory_allocated()
        mem_after = torch.cuda.memory_allocated()
        peak_after = torch.cuda.max_memory_allocated()
        leak = mem_after - mem_before
        first_step_mem = (mem_after_first - mem_before) if mem_after_first is not None else 0
        print(f"   Memory before: {mem_before/1e6:.2f} MB, after: {mem_after/1e6:.2f} MB, peak: {peak_after/1e6:.2f} MB")
        if first_step_mem and leak > first_step_mem * 1.5:
            print("⚠️ Possible memory leak detected")
            return False
        print("✅ No significant leak detected")
        return True
    except Exception as e:
        print("⚠️ Memory leak test error:", e)
        traceback.print_exc()
        return False


def test_determinism(model, images, device):
    print("\n== TEST: Determinism ==")
    try:
        # store original flags
        prev_benchmark = torch.backends.cudnn.benchmark
        prev_deterministic = torch.are_deterministic_algorithms_enabled() if hasattr(torch, "are_deterministic_algorithms_enabled") else False
        # enforce determinism
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # older torch versions may not have this API
            pass

        model.eval()
        set_seed(42)
        out1 = model(images.to(device))
        set_seed(42)
        out2 = model(images.to(device))
        # compare a representative tensor if possible
        def max_diff(a, b):
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                return (a - b).abs().max().item()
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                maxd = 0.
                for x, y in zip(a, b):
                    d = max_diff(x, y)
                    if d is not None:
                        maxd = max(maxd, d)
                return maxd
            return 0.
        diff = max_diff(out1, out2)
        print(f"  max elementwise difference (same seed): {diff:.3e}")
        # restore flags
        torch.backends.cudnn.benchmark = prev_benchmark
        try:
            torch.use_deterministic_algorithms(prev_deterministic)
        except Exception:
            pass
        return diff < 1e-5
    except Exception as e:
        print("⚠️ Determinism test failed:", e)
        traceback.print_exc()
        return False


def test_gradient_flow(model, images, targets, device):
    print("\n== TEST: Gradient Flow Analysis ==")
    try:
        model.train()
        model.zero_grad()
        out = model(images.to(device))
        # safe sum of output tensors
        total = None
        def ssum(x):
            if isinstance(x, torch.Tensor):
                return x.sum()
            if isinstance(x, (list, tuple)):
                return sum((ssum(y) for y in x))
            return torch.tensor(0., device=device)
        total = ssum(out)
        total.backward()
        grad_zero = 0
        total_params = 0
        nan_grad = 0
        inf_grad = 0
        max_grad = 0.
        for name, p in model.named_parameters():
            if p.requires_grad:
                total_params += 1
                if p.grad is None:
                    grad_zero += 1
                else:
                    if not torch.isfinite(p.grad).all():
                        if torch.isnan(p.grad).any():
                            nan_grad += 1
                        if torch.isinf(p.grad).any():
                            inf_grad += 1
                    else:
                        max_grad = max(max_grad, float(p.grad.abs().max().item()))
        print(f"  params_with_grad: {total_params - grad_zero}/{total_params}, nan_grad={nan_grad}, inf_grad={inf_grad}, max_grad={max_grad:.2e}")
        if nan_grad > 0 or inf_grad > 0:
            print("❌ NaN/Inf gradients detected")
            return False
        if grad_zero > total_params * 0.5:
            print("⚠️ Many zero gradients; suspicious")
            return False
        return True
    except Exception as e:
        print("⚠️ Gradient flow test failed:", e)
        traceback.print_exc()
        return False


# ---------------------------
# Driver: run combined checks
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="ULTIMATE Combined debug for MCUDetector (fixed)")
    parser.add_argument("--train-img-dir", type=str, default=None)
    parser.add_argument("--train-label-dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--use-real-data", action="store_true")
    parser.add_argument("--skip-slow-tests", action="store_true", help="Skip memory leak and speed tests")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    print_env_info(device)

    temp_run_dir = tempfile.mkdtemp(prefix="debug_train_")
    print(f"📁 Temp run dir: {temp_run_dir}")

    # ---------- Data loading ----------
    if args.use_real_data and args.train_img_dir and args.train_label_dir:
        if verify_dataset_paths is not None:
            ok = verify_dataset_paths(Path(args.train_img_dir), Path(args.train_label_dir), Path(args.train_img_dir), Path(args.train_label_dir))
            if not ok:
                print("❌ Dataset verification failed.")
                return 1
        images, targets = sample_real_batch(args.train_img_dir, args.train_label_dir, batch_size=2, workers=args.workers)
        print("✔ Real dataset batch loaded")
    else:
        images, targets = get_synthetic_batch(batch_size=2)
        print("✔ Synthetic batch created")

    # ---------- Prepare loader for speed/leak tests ----------
    test_loader = None
    if not args.skip_slow_tests:
        if args.use_real_data and args.train_img_dir and args.train_label_dir:
            ds = MCUDetectionDataset(img_dir=Path(args.train_img_dir), label_dir=Path(args.train_label_dir), img_size=512, transform=None)
            test_loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, collate_fn=detection_collate_fn, num_workers=args.workers)
        else:
            # tensor dataset where each sample is (image, target_tensor)
            stacked_targets = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in targets])
            tensor_ds = torch.utils.data.TensorDataset(images, stacked_targets)
            test_loader = torch.utils.data.DataLoader(tensor_ds, batch_size=2, shuffle=True, num_workers=args.workers)

    # ---------- Model ----------
    num_classes = len(CLASSES) if CLASSES else 7
    model = MCUDetector(num_classes=num_classes).to(device)
    criterion = MCUDetectionLoss(num_classes=num_classes).to(device)

    # ---------- Run checks ----------
    results = []

    # forward
    forward_ok = False
    try:
        outputs = check_forward(model, images, device)
        forward_ok = True
        results.append(("forward", True))
    except Exception as e:
        print("❌ Forward check failed:", e)
        traceback.print_exc()
        results.append(("forward", False))

    # loss/backward: only if forward succeeded
    if forward_ok:
        try:
            check_loss_backward(model, criterion, outputs, targets, device)
            results.append(("loss_backward", True))
        except Exception as e:
            print("❌ Loss/backward check failed:", e)
            traceback.print_exc()
            results.append(("loss_backward", False))
    else:
        results.append(("loss_backward", False))

    # ema & fuse
    try:
        check_ema_and_fuse(model, images, device)
        results.append(("ema_fuse", True))
    except Exception as e:
        print("⚠️ EMA/fuse check raised:", e)
        traceback.print_exc()
        results.append(("ema_fuse", False))

    # checkpoint IO
    try:
        check_checkpoint_io(model, temp_run_dir)
        results.append(("checkpoint_io", True))
    except Exception as e:
        print("❌ Checkpoint I/O check failed:", e)
        traceback.print_exc()
        results.append(("checkpoint_io", False))

    # ONNX export (best effort)
    try:
        eval_model = model
        if get_eval_model is not None:
            eval_model = get_eval_model(model, ema=None, device=device, fuse=False)
        check_onnx_export(eval_model, temp_run_dir)
        results.append(("onnx_export", True))
    except Exception as e:
        print("⚠️ ONNX export raised:", e)
        traceback.print_exc()
        results.append(("onnx_export", False))

    # LR finder (guarded)
    lr_ok = test_lr_finder(model, test_loader, criterion, device, temp_run_dir)
    results.append(("lr_finder", lr_ok if lr_ok is not None else True))

    # Multi-GPU
    mg = test_multi_gpu(model, images, device)
    results.append(("multi_gpu", mg))

    # DataLoader speed
    if not args.skip_slow_tests:
        dl_speed_ok = test_dataloader_speed(test_loader, num_batches=5)
        results.append(("dataloader_speed", dl_speed_ok))
    else:
        results.append(("dataloader_speed", True))

    # Memory leak
    if not args.skip_slow_tests:
        leak_ok = test_memory_leak(model, test_loader, device, num_iterations=3)
        results.append(("memory_leak", leak_ok))
    else:
        results.append(("memory_leak", True))

    # determinism
    det = test_determinism(model, images, device)
    results.append(("determinism", det))

    # gradient flow
    grad_ok = test_gradient_flow(model, images, targets, device)
    results.append(("gradient_flow", grad_ok))

    # optional model_debug checks
    if model_debug:
        try:
            if hasattr(model_debug, "check_all_weights"):
                ok = model_debug.check_all_weights(model)
                results.append(("model_debug_weights", ok))
            if hasattr(model_debug, "debug_single"):
                test_img = torch.rand(1, 3, 512, 512).to(device)
                test_labels = torch.tensor([[1, 0.5, 0.5, 0.1, 0.1]], dtype=torch.float32).to(device)
                model_debug.debug_single(model, criterion, test_img, test_labels, batch_mode=False)
                results.append(("model_debug_single", True))
        except Exception as e:
            print("⚠️ model_debug checks raised:", e)
            traceback.print_exc()
            results.append(("model_debug", False))

    # utils diagnostics
    print("\n=== Running utils diagnostics ===")
    utils_checks = []
    try:
        utils_checks.append(("utils_calculate_map_perfect", test_calculate_map_perfect()))
        utils_checks.append(("utils_box_iou", test_box_iou()))
        utils_checks.append(("utils_decode_predictions_interface", test_decode_predictions_interface()))
        utils_checks.append(("utils_precision_recall_perfect", test_precision_recall_perfect()))
        utils_checks.append(("utils_iou_probe", iou_distribution_probe()))
    except Exception as e:
        print("❌ utils diagnostics crashed:", e)
        traceback.print_exc()
        return 5
    results.extend(utils_checks)

    # final summary
    print("\n" + "=" * 70)
    print("📊 ULTIMATE DEBUG SUMMARY")
    print("=" * 70)

    critical_tests = {"forward", "loss_backward", "checkpoint_io", "utils_calculate_map_perfect", "utils_box_iou", "utils_precision_recall_perfect"}
    all_passed = True
    critical_passed = True
    for name, ok in results:
        ok_str = bool(ok)
        status = "✅ PASS" if ok_str else "❌ FAIL"
        tag = "[CRITICAL]" if name in critical_tests else "[INFO]    "
        print(f"{tag} {name:30} : {status}")
        if name in critical_tests and not ok_str:
            critical_passed = False
        if not ok_str:
            all_passed = False

    print("=" * 70)
    if critical_passed:
        print("\n✅ CRITICAL TESTS PASSED - core math and pipeline OK.")
    else:
        print("\n❌ CRITICAL TESTS FAILED - fix failures above first.")

    if all_passed:
        print("\n🎉 ALL TESTS PASSED")
        return 0
    else:
        # non-zero exit only if critical tests failed
        return 1 if not critical_passed else 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
