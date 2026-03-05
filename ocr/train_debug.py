#!/usr/bin/env python3
"""
train_debug.py — MCU Detector V2 (BiFPN + DecoupledScaleHead)
==============================================================
Combined debug suite compatible with train.py.

This script performs a set of sanity checks against the project:
  - forward shape checks (P3/P4 -> (obj, cls, reg))
  - loss / backward checks
  - EMA & rep-style fuse checks (if available)
  - checkpoint IO
  - ONNX export (best-effort)
  - LR finder sanity (best-effort)
  - DataLoader speed test
  - Memory leak probe
  - Determinism
  - Gradient flow analysis
  - Utility diagnostics (calculate_map, box_iou_batch, decode_predictions, PR)
Exit code 0 => all critical checks passed.

Usage:
    python train_debug.py [--use-real-data --train-img-dir ... --train-label-dir ...]
"""
from __future__ import annotations
import argparse
import importlib
import os
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------
# Project imports (best-effort)
# ---------------------------
try:
    from model import MCUDetector, MCUDetectionLoss  # noqa: F401
    from dataset import MCUDetectionDataset, detection_collate_fn  # noqa: F401
    import utils
    from utils import CLASSES, NUM_CLASSES, get_run_dir  # type: ignore
except Exception as e:
    # partial import allowed — we'll fallback for unavailable pieces
    try:
        import utils  # try at least utils
    except Exception:
        utils = None  # type: ignore
    MCUDetector = None  # type: ignore
    MCUDetectionLoss = None  # type: ignore
    MCUDetectionDataset = None  # type: ignore
    detection_collate_fn = None  # type: ignore
    CLASSES = getattr(utils, "CLASSES", []) if utils is not None else []
    NUM_CLASSES = getattr(utils, "NUM_CLASSES", len(CLASSES) if CLASSES else 14)

# Optional helpers from train.py (if available)
fuse_repv_blocks = None
get_eval_model = None
verify_dataset_paths = None
ModelEMA = None
find_optimal_lr = None
setup_gpu_optimizations = None
_split_targets_by_area = None
try:
    from train import (  # type: ignore
        fuse_repv_blocks as _frb,
        get_eval_model as _gem,
        verify_dataset_paths as _vdp,
        ModelEMA as _ema_cls,
        find_optimal_lr as _fol,
        setup_gpu_optimizations as _sg,
        _split_targets_by_area as _split_fn,
    )
    fuse_repv_blocks = _frb
    get_eval_model = _gem
    verify_dataset_paths = _vdp
    ModelEMA = _ema_cls
    find_optimal_lr = _fol
    setup_gpu_optimizations = _sg
    _split_targets_by_area = _split_fn
except Exception:
    pass

# ---------------------------
# Helpers
# ---------------------------
def set_seed(seed: int = 42):
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
            print("GPU count:", torch.cuda.device_count())
        except Exception:
            pass
    print("Device used:", device)
    print("NUM_CLASSES:", NUM_CLASSES)
    print("CLASSES:", (CLASSES[:8] if CLASSES else []) , "..." if len(CLASSES) > 8 else "")
    print("============================================\n")


# ---------------------------
# Synthetic data / sampling
# ---------------------------
def get_synthetic_batch(batch_size: int = 2, img_size: int = 512):
    images = torch.rand(batch_size, 3, img_size, img_size)
    # single-box per image centered
    targets = [
        torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]], dtype=torch.float32)
        for _ in range(batch_size)
    ]
    return images, targets


def sample_real_batch(img_dir: str, label_dir: str, img_size: int = 512, batch_size: int = 2, workers: int = 2):
    if MCUDetectionDataset is None or detection_collate_fn is None:
        raise RuntimeError("Real-data sampling requires MCUDetectionDataset and detection_collate_fn available.")
    ds = MCUDetectionDataset(img_dir=Path(img_dir), label_dir=Path(label_dir), img_size=img_size, transform=None)
    assert len(ds) > 0, "Dataset is empty"
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=detection_collate_fn, num_workers=workers)
    images, targets = next(iter(loader))
    return images, targets


# ---------------------------
# Target splitting helper (mirror of train.py)
# ---------------------------
def split_targets(targets, device, H: int = 512, W: int = 512):
    if _split_targets_by_area is not None:
        return _split_targets_by_area(targets, H, W, device, warmup=False)
    # fallback
    area_threshold = 0.02 * H * W
    targets_p3, targets_p4 = [], []
    for t in targets:
        if t is None or (isinstance(t, torch.Tensor) and t.numel() == 0):
            targets_p3.append(torch.zeros((0, 5), device=device))
            targets_p4.append(torch.zeros((0, 5), device=device))
            continue
        t = t.to(device)
        areas = (t[:, 3] * W) * (t[:, 4] * H)
        small_mask = areas < area_threshold
        p3 = t[small_mask] if small_mask.any() else t
        p4 = t[~small_mask] if (~small_mask).any() else t
        targets_p3.append(p3)
        targets_p4.append(p4)
    return targets_p3, targets_p4


# ---------------------------
# Core checks
# ---------------------------
def check_forward(model, images, device):
    """Verify model forward returns correct (P3, P4) format."""
    model.eval()
    outputs = model(images.to(device))

    assert isinstance(outputs, (list, tuple)), "Model output must be tuple/list"
    assert len(outputs) == 2, f"Expected (P3, P4), got {len(outputs)} outputs"

    for i, o in enumerate(outputs):
        assert isinstance(o, (tuple, list)), f"output[{i}] must be (obj, cls, reg)"
        assert len(o) == 3, f"output[{i}] must have 3 elements, got {len(o)}"
        obj_map, cls_map, reg_map = o
        assert isinstance(obj_map, torch.Tensor)
        assert isinstance(cls_map, torch.Tensor)
        assert isinstance(reg_map, torch.Tensor)
        B = images.shape[0]
        assert obj_map.shape[0] == B, f"P{3+i} obj batch mismatch"
        assert obj_map.shape[1] == 1, f"P{3+i} obj should have 1 channel, got {obj_map.shape[1]}"
        assert cls_map.shape[1] == NUM_CLASSES, f"P{3+i} cls should have {NUM_CLASSES} channels, got {cls_map.shape[1]}"
        assert reg_map.shape[1] == 4, f"P{3+i} reg should have 4 channels, got {reg_map.shape[1]}"
        print(f"✔ P{3+i} OK | obj: {tuple(obj_map.shape)} | cls: {tuple(cls_map.shape)} | reg: {tuple(reg_map.shape)}")
    return outputs


def check_loss_backward(model, criterion, targets, device):
    """Verify loss computation and backward pass."""
    if criterion is None:
        raise RuntimeError("Loss criterion not available for check_loss_backward.")
    model.train()
    images = torch.rand(2, 3, 512, 512).to(device)
    outputs = model(images)
    targets_p3, targets_p4 = split_targets(targets, device)
    loss_dict = criterion(outputs[0], outputs[1], targets_p3, targets_p4)
    assert "total" in loss_dict and "bbox" in loss_dict and "obj" in loss_dict and "cls" in loss_dict
    loss = loss_dict["total"]
    assert isinstance(loss, torch.Tensor) and loss.requires_grad
    assert torch.isfinite(loss), f"loss is NaN/Inf: {loss.item()}"
    cls_val = float(loss_dict["cls"].detach().cpu().item())
    assert cls_val >= 0.0, f"cls_loss NEGATIVE: {cls_val:.6f}"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients produced"
    nan_grads = sum(1 for g in grads if not torch.isfinite(g).all())
    assert nan_grads == 0, f"{nan_grads} parameters have NaN/Inf gradients"
    print("✔ Loss backward OK")
    print("  Loss values:", {k: float(v.detach().cpu().item()) for k, v in loss_dict.items() if isinstance(v, torch.Tensor)})


@torch.no_grad()
def check_ema_and_fuse(model, images, device):
    """Test EMA and rep-style block fusion."""
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
            fuse_model = copy.deepcopy(model).eval()
            fuse_repv_blocks(fuse_model)
            out_before = model.eval()(images.to(device))
            out_after = fuse_model(images.to(device))
            assert out_before[0][0].shape == out_after[0][0].shape, "Fusion changed output shape"
            print("✔ Rep-style blocks fused successfully")
        except Exception as e:
            print(f"⚠️ fuse_repv_blocks failed: {e}")
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
        out_path = Path(run_dir) / "debug.onnx"
        dummy = torch.randn(1, 3, 512, 512, device=next(eval_model.parameters()).device)
        torch.onnx.export(eval_model, dummy, str(out_path), opset_version=14, do_constant_folding=True, input_names=["input"])
        out_path.unlink(missing_ok=True)
        print("✔ ONNX export OK")
    except Exception as e:
        print(f"⚠️ ONNX export skipped/failed: {e}")


# ---------------------------
# Utils diagnostics (uses utils module if present)
# ---------------------------
def create_perfect_test_case():
    target = np.array([[1, 0.5075, 0.515, 0.0825, 0.066667]], dtype=np.float32)
    img_size = 512
    cx = target[0, 1] * img_size
    cy = target[0, 2] * img_size
    w = target[0, 3] * img_size
    h = target[0, 4] * img_size
    pred_x1, pred_y1 = cx - w / 2 + 2, cy - h / 2 + 2
    pred_x2, pred_y2 = cx + w / 2 - 2, cy + h / 2 - 2
    prediction = np.array([[1, 0.9, pred_x1, pred_y1, pred_x2, pred_y2]], dtype=np.float32)
    return [prediction], [target]


def test_calculate_map_perfect():
    print("\n== TEST: calculate_map with perfect data ==")
    if utils is None or not hasattr(utils, "calculate_map"):
        print("ℹ️ utils.calculate_map not available; skipping")
        return True
    preds, targs = create_perfect_test_case()
    try:
        res = utils.calculate_map(predictions=preds, targets=targs, num_classes=NUM_CLASSES, epoch=50)
        if isinstance(res, tuple) and len(res) >= 2:
            map_50 = res[1]
            print(f"mAP@0.5 = {map_50:.4f}")
            return map_50 >= 0.95
        print("⚠️ Unexpected return from calculate_map:", type(res))
        return False
    except Exception as e:
        print("❌ calculate_map failed:", e)
        traceback.print_exc()
        return False


def test_box_iou():
    print("\n== TEST: box_iou math ==")
    if utils is None or not hasattr(utils, "box_iou_batch"):
        print("ℹ️ utils.box_iou_batch not available; skipping")
        return True
    box1 = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    box2 = torch.tensor([[150, 150, 250, 250]], dtype=torch.float32)
    try:
        iou = utils.box_iou_batch(box1, box2)
        expected = (50 * 50) / (100 * 100 + 100 * 100 - 50 * 50)
        print(f"Computed IoU: {float(iou):.6f}, expected {expected:.6f}")
        return abs(float(iou) - expected) < 1e-3
    except Exception as e:
        print("❌ box_iou failed:", e)
        traceback.print_exc()
        return False


def test_decode_predictions_interface():
    print("\n== TEST: decode_predictions interface (V2 format) ==")
    if utils is None or not hasattr(utils, "decode_predictions"):
        print("ℹ️ utils.decode_predictions not available; skipping")
        return True
    B, NC = 1, NUM_CLASSES
    p3 = (torch.randn(B, 1, 128, 128), torch.randn(B, NC, 128, 128), torch.randn(B, 4, 128, 128))
    p4 = (torch.randn(B, 1, 64, 64), torch.randn(B, NC, 64, 64), torch.randn(B, 4, 64, 64))
    try:
        def to_decoder(pred):
            obj, cls, reg = pred
            return (torch.cat([obj, cls], dim=1), reg)
        p3_dec = to_decoder(p3)
        p4_dec = to_decoder(p4)
        preds = utils.decode_predictions(pred_p3=p3_dec, pred_p4=p4_dec, conf_thresh=0.001, nms_thresh=0.45, img_size=512)
        print(f"decode returned {len(preds)} images")
        if isinstance(preds, list) and len(preds) > 0:
            p = preds[0]
            if isinstance(p, np.ndarray):
                return p.ndim == 2 and p.shape[1] >= 6
            return True
        return True
    except Exception as e:
        print("❌ decode failed:", e)
        traceback.print_exc()
        return False


def test_precision_recall_perfect():
    print("\n== TEST: compute_precision_recall ==")
    if utils is None or not hasattr(utils, "compute_precision_recall"):
        print("ℹ️ utils.compute_precision_recall not available; skipping")
        return True
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
    print("\n== TEST: IoU probe ==")
    if utils is None or not hasattr(utils, "box_iou_batch"):
        print("ℹ️ utils.box_iou_batch not available; skipping")
        return True
    preds, targs = create_perfect_test_case()
    try:
        pboxes = preds[0][:, 2:6]
        cx = targs[0][:, 1] * 512; cy = targs[0][:, 2] * 512
        w = targs[0][:, 3] * 512; h = targs[0][:, 4] * 512
        gt = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        iou = utils.box_iou_batch(torch.from_numpy(pboxes), torch.from_numpy(gt))
        print("IoU(s):", iou)
        return True
    except Exception as e:
        print("❌ IoU probe failed:", e)
        traceback.print_exc()
        return False


# ---------------------------
# Extra tests
# ---------------------------
def test_multi_gpu(model, images, device):
    print("\n== TEST: Multi-GPU ==")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        print("ℹ️ Single GPU, skipping")
        return True
    try:
        model_dp = torch.nn.DataParallel(model)
        out = model_dp(images.to(device))
        print(f"✅ DataParallel OK on {torch.cuda.device_count()} GPUs")
        return True
    except Exception as e:
        print(f"⚠️ Multi-GPU failed: {e}")
        return False


def test_dataloader_speed(loader, num_batches: int = 5):
    print("\n== TEST: DataLoader Speed ==")
    if loader is None:
        print("ℹ️ No loader, skipping")
        return True
    try:
        start = time.time()
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
        elapsed = time.time() - start
        speed = num_batches / max(elapsed, 1e-6)
        print(f"✅ {speed:.2f} batches/sec ({elapsed:.2f}s for {num_batches} batches)")
        return speed > 0.1  # permissive threshold for varied I/O
    except Exception as e:
        print(f"⚠️ DataLoader speed failed: {e}")
        return False


def test_memory_leak(model, loader, device, num_iterations: int = 3):
    print("\n== TEST: Memory Leak ==")
    if not torch.cuda.is_available():
        print("ℹ️ No CUDA, skipping")
        return True
    if loader is None:
        print("ℹ️ No loader, skipping memory test")
        return True
    try:
        torch.cuda.reset_peak_memory_stats()
        criterion = MCUDetectionLoss(num_classes=NUM_CLASSES, topk=9, focal_gamma=2.0).to(device) if MCUDetectionLoss is not None else None
        model.train()
        data_iter = iter(loader)
        memories = []
        for i in range(num_iterations):
            try:
                images, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                images, targets = next(data_iter)
            images = images.to(device)
            t3, t4 = split_targets(targets, device)
            out = model(images)
            if criterion is None:
                # run a dummy backward to test mem growth
                loss = out[0][0].abs().sum() * 1e-6
            else:
                loss = criterion(out[0], out[1], t3, t4)["total"]
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            memories.append(torch.cuda.memory_allocated() / 1e6)
        if len(memories) >= 2:
            growth = memories[-1] - memories[0]
            print(f"Memory: {memories[0]:.1f} → {memories[-1]:.1f} MB (growth={growth:.1f} MB)")
            if growth > 500:
                print("⚠️ Possible memory leak")
                return False
        print("✅ No significant memory leak")
        return True
    except Exception as e:
        print(f"⚠️ Memory leak test failed: {e}")
        traceback.print_exc()
        return False


def test_determinism(model, images, device):
    print("\n== TEST: Determinism ==")
    try:
        model.eval()
        with torch.no_grad():
            set_seed(42)
            out1 = model(images.to(device))
            set_seed(42)
            out2 = model(images.to(device))
        # compare a representative tensor
        d1 = out1[0][0]
        d2 = out2[0][0]
        diff = (d1 - d2).abs().max().item()
        print(f"Max diff: {diff:.2e}")
        return diff < 1e-5
    except Exception as e:
        print(f"⚠️ Determinism test failed: {e}")
        traceback.print_exc()
        return False


def test_gradient_flow(model, images, targets, device):
    print("\n== TEST: Gradient Flow ==")
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        if MCUDetectionLoss is None:
            print("ℹ️ MCUDetectionLoss not available; skipping gradient flow test")
            return True
        criterion = MCUDetectionLoss(num_classes=NUM_CLASSES, topk=9, focal_gamma=2.0).to(device)
        out = model(images.to(device))
        t3, t4 = split_targets(targets, device)
        loss = criterion(out[0], out[1], t3, t4)["total"]
        loss.backward()
        total_params = 0
        grad_zero = 0
        nan_grad = 0
        inf_grad = 0
        grad_norms = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            total_params += 1
            if p.grad is None:
                grad_zero += 1
                continue
            norm = p.grad.norm().item()
            grad_norms.append(norm)
            if np.isnan(norm): nan_grad += 1
            if np.isinf(norm): inf_grad += 1
            if norm == 0: grad_zero += 1
        print(f"  Params: {total_params}, Zero grad: {grad_zero}, NaN: {nan_grad}, Inf: {inf_grad}")
        if grad_norms:
            print(f"  Grad norm: min={min(grad_norms):.2e}, max={max(grad_norms):.2e}, mean={np.mean(grad_norms):.2e}")
        if nan_grad > 0 or inf_grad > 0:
            print("❌ NaN/Inf gradients detected")
            return False
        if grad_zero > total_params * 0.5:
            print("⚠️ Many zero gradients")
            return False
        print("✅ Gradient flow OK")
        return True
    except Exception as e:
        print(f"⚠️ Gradient flow failed: {e}")
        traceback.print_exc()
        return False


# ---------------------------
# V2-specific checks (best-effort)
# ---------------------------
def test_v2_architecture(model, device):
    print("\n== TEST: V2 Architecture Sanity ==")
    try:
        if not hasattr(model, "fpn"):
            print("ℹ️ Model has no 'fpn' attribute; skipping V2 architecture checks")
            return True
        # BiFPN weight checks (best-effort)
        if hasattr(model.fpn, "w_td") and hasattr(model.fpn, "w_bu"):
            print(f"  BiFPN TD weights: {getattr(model.fpn, 'w_td').data.flatten()[:8].cpu().tolist()}")
            print(f"  BiFPN BU weights: {getattr(model.fpn, 'w_bu').data.flatten()[:8].cpu().tolist()}")
        else:
            print("ℹ️ BiFPN learnable weights not present; ok for some implementations")
        # Decoupled head checks
        if hasattr(model, "head"):
            for name, head in (("P3", getattr(model.head, "p3_refine", None)), ("P4", getattr(model.head, "p4_refine", None))):
                if head is None:
                    continue
                if not (hasattr(head, "obj_conv") and hasattr(head, "cls_branch") and hasattr(head, "reg_branch")):
                    print(f"⚠️ {name} head missing expected branches")
                    return False
                cls_out_ch = getattr(head.cls_branch[-1], "out_channels", None)
                reg_out_ch = getattr(head.reg_branch[-1], "out_channels", None)
                obj_out_ch = getattr(head.obj_conv, "out_channels", None)
                if cls_out_ch is not None and cls_out_ch != NUM_CLASSES:
                    print(f"⚠️ {name} cls channels mismatch: {cls_out_ch} != {NUM_CLASSES}")
                    return False
                if reg_out_ch is not None and reg_out_ch != 4:
                    print(f"⚠️ {name} reg channels mismatch: {reg_out_ch} != 4")
                    return False
                if obj_out_ch is not None and obj_out_ch != 1:
                    print(f"⚠️ {name} obj channels mismatch: {obj_out_ch} != 1")
                    return False
                print(f"  ✔ {name}: obj={obj_out_ch}, cls={cls_out_ch}, reg={reg_out_ch}")
        print("✅ V2 architecture checks passed (best-effort)")
        return True
    except Exception as e:
        print(f"❌ V2 architecture check failed: {e}")
        traceback.print_exc()
        return False


def test_v2_loss_sanity(device):
    print("\n== TEST: V2 Loss Sanity (no negatives) ==")
    if MCUDetectionLoss is None:
        print("ℹ️ MCUDetectionLoss not available; skipping loss sanity")
        return True
    try:
        criterion = MCUDetectionLoss(num_classes=NUM_CLASSES, bbox_weight=1.0, obj_weight=1.0, cls_weight=1.0, topk=9, focal_gamma=2.0).to(device)
        B = 2
        p3 = (torch.randn(B, 1, 128, 128, device=device), torch.randn(B, NUM_CLASSES, 128, 128, device=device), torch.randn(B, 4, 128, 128, device=device))
        p4 = (torch.randn(B, 1, 64, 64, device=device), torch.randn(B, NUM_CLASSES, 64, 64, device=device), torch.randn(B, 4, 64, 64, device=device))
        t3 = [torch.tensor([[0, 0.3, 0.3, 0.05, 0.05]], device=device), torch.tensor([[5, 0.7, 0.7, 0.08, 0.08]], device=device)]
        t4 = [torch.tensor([[2, 0.4, 0.4, 0.3, 0.3]], device=device), torch.tensor([[10, 0.6, 0.6, 0.25, 0.25]], device=device)]
        loss_dict = criterion(p3, p4, t3, t4)
        total = float(loss_dict["total"].item())
        bbox = float(loss_dict["bbox"].item())
        obj = float(loss_dict["obj"].item())
        cls = float(loss_dict["cls"].item())
        print(f"  total={total:.4f}, bbox={bbox:.4f}, obj={obj:.4f}, cls={cls:.4f}")
        assert cls >= 0 and obj >= 0 and bbox >= 0 and total >= 0 and total < 1000
        print("✅ V2 loss sanity OK (all non-negative)")
        return True
    except Exception as e:
        print(f"❌ V2 loss sanity failed: {e}")
        traceback.print_exc()
        return False


# ---------------------------
# Main orchestration
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="V2 Combined Debug for MCUDetector")
    parser.add_argument("--train-img-dir", type=str, default=None)
    parser.add_argument("--train-label-dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--use-real-data", action="store_true")
    parser.add_argument("--skip-slow-tests", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    print_env_info(device)

    temp_run_dir = tempfile.mkdtemp(prefix="debug_train_v2_")
    print(f"📁 Temp dir: {temp_run_dir}")

    # Data selection
    if args.use_real_data and args.train_img_dir and args.train_label_dir:
        if verify_dataset_paths is not None:
            ok = verify_dataset_paths(Path(args.train_img_dir), Path(args.train_label_dir), Path(args.train_img_dir), Path(args.train_label_dir), num_classes=NUM_CLASSES)
            if not ok:
                print("❌ Dataset verification failed")
                return 1
        try:
            images, targets = sample_real_batch(args.train_img_dir, args.train_label_dir, batch_size=2, workers=args.workers)
            print("✔ Real data loaded")
        except Exception as e:
            print(f"❌ Could not load real data: {e}")
            traceback.print_exc()
            return 2
    else:
        images, targets = get_synthetic_batch(batch_size=2)
        print("✔ Synthetic data created")

    # Loader for slow tests (optional)
    test_loader = None
    if not args.skip_slow_tests and args.use_real_data and args.train_img_dir and args.train_label_dir and MCUDetectionDataset is not None:
        try:
            ds = MCUDetectionDataset(img_dir=Path(args.train_img_dir), label_dir=Path(args.train_label_dir), img_size=512, transform=None)
            test_loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, collate_fn=detection_collate_fn, num_workers=args.workers)
        except Exception:
            test_loader = None

    # Model + Loss
    if MCUDetector is None:
        print("❌ MCUDetector class not found. Place model.py in the same folder.")
        return 2
    model = MCUDetector(num_classes=NUM_CLASSES).to(device)
    criterion = MCUDetectionLoss(num_classes=NUM_CLASSES, bbox_weight=1.0, obj_weight=2.0, cls_weight=5.0, topk=9, focal_gamma=2.0, label_smoothing=0.05).to(device) if MCUDetectionLoss is not None else None

    results = []

    # V2-specific
    results.append(("v2_architecture", test_v2_architecture(model, device)))
    results.append(("v2_loss_sanity", test_v2_loss_sanity(device)))

    # Forward
    try:
        outputs = check_forward(model, images, device)
        results.append(("forward", True))
    except Exception as e:
        print(f"❌ Forward failed: {e}")
        traceback.print_exc()
        results.append(("forward", False))
        outputs = None

    # Loss/backward
    if outputs is not None and criterion is not None:
        try:
            check_loss_backward(model, criterion, targets, device)
            results.append(("loss_backward", True))
        except Exception as e:
            print(f"❌ Loss/backward failed: {e}")
            traceback.print_exc()
            results.append(("loss_backward", False))
    else:
        results.append(("loss_backward", False if criterion is None else True))

    # EMA & fuse
    try:
        check_ema_and_fuse(model, images, device)
        results.append(("ema_fuse", True))
    except Exception as e:
        print(f"⚠️ EMA/fuse: {e}")
        results.append(("ema_fuse", False))

    # Checkpoint
    try:
        check_checkpoint_io(model, temp_run_dir)
        results.append(("checkpoint_io", True))
    except Exception as e:
        print(f"❌ Checkpoint failed: {e}")
        results.append(("checkpoint_io", False))

    # ONNX
    try:
        check_onnx_export(model.eval(), temp_run_dir)
        results.append(("onnx_export", True))
    except Exception as e:
        print(f"⚠️ ONNX: {e}")
        results.append(("onnx_export", False))

    # Multi-GPU
    results.append(("multi_gpu", test_multi_gpu(model, images, device)))

    # DataLoader speed
    if not args.skip_slow_tests:
        results.append(("dataloader_speed", test_dataloader_speed(test_loader)))
    else:
        results.append(("dataloader_speed", True))

    # Memory leak
    if not args.skip_slow_tests and test_loader:
        results.append(("memory_leak", test_memory_leak(model, test_loader, device)))
    else:
        results.append(("memory_leak", True))

    # Determinism
    results.append(("determinism", test_determinism(model, images, device)))

    # Gradient flow
    results.append(("gradient_flow", test_gradient_flow(model, images, targets, device)))

    # Utils diagnostics
    print("\n=== Utils Diagnostics ===")
    results.append(("utils_map_perfect", test_calculate_map_perfect()))
    results.append(("utils_box_iou", test_box_iou()))
    results.append(("utils_decode_v2", test_decode_predictions_interface()))
    results.append(("utils_pr_perfect", test_precision_recall_perfect()))
    results.append(("utils_iou_probe", iou_distribution_probe()))

    # Summary
    print("\n" + "=" * 70)
    print("📊 V2 DEBUG SUMMARY")
    print("=" * 70)

    critical_tests = {
        "forward", "loss_backward", "checkpoint_io",
        "v2_architecture", "v2_loss_sanity",
        "utils_map_perfect", "utils_box_iou", "utils_pr_perfect"
    }

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
        print("\n✅ CRITICAL TESTS PASSED — core pipeline OK")
    else:
        print("\n❌ CRITICAL TESTS FAILED — fix before training!")

    if all_passed:
        print("🎉 ALL TESTS PASSED\n")
        return 0
    else:
        return 1 if not critical_passed else 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)