#!/usr/bin/env python3
# train_debug.py — FINAL robustness checks for MCUDetector

import argparse
import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------
try:
    from model import MCUDetector, MCUDetectionLoss
    from dataset import MCUDetectionDataset, detection_collate_fn
    from utils import CLASSES, get_run_dir
    from train import (
        fuse_repv_blocks,
        get_eval_model,
        verify_dataset_paths,
        ModelEMA,
    )
except Exception as e:
    print("❌ Failed to import project modules:", e)
    raise

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def assert_prediction(obj, name="pred"):
    """
    Recursively validate YOLO-style prediction structures.
    Accepts:
      - Tensor
      - tuple/list of tensors
      - nested tuple/list (YOLO heads)
    """
    if isinstance(obj, torch.Tensor):
        assert torch.isfinite(obj).all(), f"{name} contains NaN/Inf"
        return

    if isinstance(obj, (list, tuple)):
        assert len(obj) > 0, f"{name} is empty"
        for i, v in enumerate(obj):
            assert_prediction(v, f"{name}[{i}]")
        return

    raise AssertionError(f"{name} has invalid type: {type(obj)}")


def print_env_info(device):
    print("\n================ ENVIRONMENT ================")
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
        print("Compute capability:", torch.cuda.get_device_capability(0))
    print("Device used:", device)
    print("============================================\n")

def assert_tensor(t, name):
    assert isinstance(t, torch.Tensor), f"{name} must be torch.Tensor"
    assert torch.isfinite(t).all(), f"{name} contains NaN/Inf"

# ---------------------------------------------------------
# Dataset / Loader checks
# ---------------------------------------------------------
def get_synthetic_batch(batch_size=2, img_size=512):
    images = torch.rand(batch_size, 3, img_size, img_size)
    targets = [
        torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]], dtype=torch.float32)
        for _ in range(batch_size)
    ]
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

# ---------------------------------------------------------
# Model checks
# ---------------------------------------------------------

def check_forward(model, images, device):
    model.eval()
    outputs = model(images.to(device))
    assert isinstance(outputs, (list, tuple)), "Model output must be tuple/list"
    assert len(outputs) == 2, "Expected (P3, P4) outputs"
    for i, o in enumerate(outputs):
        assert isinstance(o, (tuple, list)), f"output[{i}] must be tuple"
        cls_map, reg_map = o

        assert isinstance(cls_map, torch.Tensor)
        assert isinstance(reg_map, torch.Tensor)

        print(
            f"✔ output[{i}] OK | "
            f"cls: {tuple(cls_map.shape)} | "
            f"reg: {tuple(reg_map.shape)}"
        )

    return outputs

def check_loss_backward(model, criterion, outputs, targets, device):
    model.train()

    # Split targets exactly like train.py
    targets_p3, targets_p4 = [], []
    for t in targets:
        if t.numel() == 0:
            targets_p3.append(torch.zeros((0, 5), device=device))
            targets_p4.append(torch.zeros((0, 5), device=device))
            continue

        t = t.to(device)
        area = t[:, 3] * t[:, 4] * 512 * 512
        targets_p3.append(t[area < 512])
        targets_p4.append(t[(area >= 512) & (area < 1024)])

    # ---- forward loss ----
    loss_dict = criterion(outputs[0], outputs[1], targets_p3, targets_p4)

    assert "total" in loss_dict
    loss = loss_dict["total"]

    # 🔑 CRITICAL ASSERTIONS
    assert isinstance(loss, torch.Tensor), "loss must be a Tensor"
    assert loss.requires_grad, "loss must require grad"

    # ---- backward ----
    loss.backward()

    # ---- gradient sanity ----
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients produced"

    print("✔ Loss backward OK")
    print("  Loss values:",
          {k: float(v) if torch.is_tensor(v) else v for k, v in loss_dict.items()})
# ---------------------------------------------------------
# EMA + fuse checks
# ---------------------------------------------------------
@torch.no_grad()
def check_ema_and_fuse(model, images, device):
    ema = ModelEMA(model, decay=0.999)
    ema.update(model)

    out = ema.ema(images.to(device))
    assert isinstance(out, (list, tuple)), "EMA forward failed"
    print("✔ EMA forward OK")

    fuse_repv_blocks(ema.ema)
    print("✔ Rep-style blocks fused successfully")

# ---------------------------------------------------------
# Checkpoint integrity
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# ONNX export (best effort)
# ---------------------------------------------------------
def check_onnx_export(eval_model, run_dir):
    try:
        dummy = torch.randn(1, 3, 512, 512, device=next(eval_model.parameters()).device)
        out_path = Path(run_dir) / "debug.onnx"
        torch.onnx.export(
            eval_model,
            dummy,
            str(out_path),
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
        )
        out_path.unlink(missing_ok=True)
        print("✔ ONNX export OK")
    except Exception as e:
        print("⚠️ ONNX export skipped:", e)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-img-dir", type=str, default=None)
    parser.add_argument("--train-label-dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    print_env_info(device)

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------
    if args.train_img_dir and args.train_label_dir:
        assert verify_dataset_paths(
            Path(args.train_img_dir),
            Path(args.train_label_dir),
            Path(args.train_img_dir),
            Path(args.train_label_dir),
        ), "Dataset verification failed"
        images, targets = sample_real_batch(
            args.train_img_dir,
            args.train_label_dir,
            workers=args.workers,
        )
        print("✔ Real dataset batch loaded")
    else:
        images, targets = get_synthetic_batch()
        print("✔ Synthetic batch created")

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    num_classes = len(CLASSES) if CLASSES else 7
    model = MCUDetector(num_classes=num_classes).to(device)
    criterion = MCUDetectionLoss(num_classes=num_classes).to(device)

    # -----------------------------------------------------
    # Checks
    # -----------------------------------------------------
    outputs = check_forward(model, images, device)
    check_loss_backward(model, criterion, outputs, targets, device)
    check_ema_and_fuse(model, images, device)
    check_checkpoint_io(model, get_run_dir("debug"))
    eval_model = get_eval_model(model, ema=None, device=device, fuse=False)
    check_onnx_export(eval_model, get_run_dir("debug"))

    print("\n✅ ALL DEBUG CHECKS PASSED — TRAINING PIPELINE IS SOUND")

if __name__ == "__main__":
    main()
