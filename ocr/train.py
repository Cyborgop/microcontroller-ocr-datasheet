#!/usr/bin/env python3
"""
MCUDetector YOLO Training (Detection ONLY) - RTX 2080Ti Optimized
7-class object detection without OCR
With EMA, LR Finder, TQDM, and Full Metrics
"""

import os
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import json
from datetime import datetime
import time
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import local modules
try:
    from model import MCUDetector, MCUDetectionLoss
    from dataset import MCUDetectionDataset, detection_collate_fn
    from utils import (
    get_run_dir, count_parameters, print_gpu_memory,
    CLASSES, plot_confusion_matrix,
    plot_label_heatmap, plot_f1_confidence_curve,
    decode_predictions, calculate_map, save_precision_recall_curves, 
    compute_precision_recall_curves, compute_epoch_precision_recall,  
    plot_yolo_results,  save_test_summary
)
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure model.py, dataset.py, and utils.py are in the same directory")
    sys.exit(1)

def fuse_repv_blocks(model):
    assert not model.training, "Rep-style fusion must be performed only in eval() mode (call model.eval() first)."
    """Call .fuse() on all modules that provide it (RepDWConvSR / RepvitBlock)."""
    
    for m in model.modules():
        if hasattr(m, "fuse") and callable(getattr(m, "fuse")):
            try:
                m.fuse()
            except Exception:
                # fail-safe: ignore modules that can't be fused
                pass

def get_eval_model(model, ema=None, device=None, fuse=False):
    """
    Return the model to use for evaluation:
      - prefer ema.ema if ema is provided
      - move to device and call .eval()
      - optionally fuse reparam blocks (for latency / final eval)
    """
    eval_model = ema.ema if ema is not None else model
    if device is not None:
        eval_model = eval_model.to(device)
    eval_model.eval()
    if fuse:
        fuse_repv_blocks(eval_model)
    return eval_model

# =================== PATHS ===================
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
NUM_CLASSES = len(CLASSES) if CLASSES else 7
print("DEBUG[train] NUM_CLASSES:", NUM_CLASSES)


# =================== EMA (EXPONENTIAL MOVING AVERAGE) ===================
class ModelEMA:
    """
    Exponential Moving Average (EMA) of model weights.
    Based on YOLOv5 / YOLOv8 implementation.
    """
    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = self._clone_model(model).eval()  # FP32 EMA model
        self.updates = updates
        self.decay = lambda x: decay * (1.0 - np.exp(-x / 2000))  # ramp-up decay

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _clone_model(model):
        """
        Create a clean EMA copy using state_dict (SAFE, FAST).
        """
        device = next(model.parameters()).device
        model_copy = type(model)(num_classes=model.num_classes).to(device)
        model_copy.load_state_dict(model.state_dict())
        return model_copy

    @torch.no_grad()
    def update(self, model):
        """
        Update EMA parameters.
        """
        self.updates += 1
        d = self.decay(self.updates)

        msd = model.state_dict()
        esd = self.ema.state_dict()

        for k in esd.keys():
            if torch.is_floating_point(esd[k]):
                esd[k].mul_(d).add_(msd[k].detach(), alpha=1.0 - d)

        self.ema.load_state_dict(esd)

    def __call__(self, x):
        return self.ema(x)


# =================== EARLY STOPPING ===================
class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
# =================== TRAINING METRICS TRACKER ===================
class TrainingMetricsTracker:
    """Track and save training metrics every epoch."""
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_box": [],
            "train_cls": [],
            "train_obj": [],
            "val_loss": [],
            "val_box": [],
            "val_cls": [],
            "val_obj": [],
            "precision": [],
            "recall": [],
            "map50": [],
            "map5095": [],
            "lr": []
        }
    
    def update(self, epoch, train_metrics, val_metrics, lr):
        """Update history with new epoch metrics."""
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_metrics["loss"])
        self.history["train_box"].append(train_metrics["box"])
        self.history["train_cls"].append(train_metrics["cls"])
        self.history["train_obj"].append(train_metrics["obj"])
        self.history["val_loss"].append(val_metrics["loss"])
        self.history["val_box"].append(val_metrics["box"])
        self.history["val_cls"].append(val_metrics["cls"])
        self.history["val_obj"].append(val_metrics["obj"])
        self.history["precision"].append(val_metrics["precision"])
        self.history["recall"].append(val_metrics["recall"])
        self.history["map50"].append(val_metrics["map50"])
        self.history["map5095"].append(val_metrics["map5095"])
        self.history["lr"].append(lr)
    
    def save_csv(self):
        """Save metrics to CSV file."""
        import csv
        csv_path = self.run_dir / "results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.history.keys())
            rows = zip(*self.history.values())
            writer.writerows(rows)
        print(f"📊 Training metrics saved to: {csv_path}")

# =================== DATASET VERIFICATION ===================
# SAME extensions as dataset.py (must match)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def verify_dataset_paths(train_img_dir, train_label_dir, val_img_dir, val_label_dir):
    print("\n🔍 Verifying dataset paths...")

    paths = [
        ("Train Images", train_img_dir),
        ("Train Labels", train_label_dir),
        ("Val Images", val_img_dir),
        ("Val Labels", val_label_dir),
    ]

    for name, path in paths:
        if not path.exists():
            print(f"❌ {name}: {path} NOT FOUND")
            return False
        print(f"✅ {name}: {path}")

    def collect_images(img_dir):
        files = []
        for ext in IMG_EXTS:
            files.extend(img_dir.glob(f"*{ext}"))
        return sorted(files)

    train_images = collect_images(train_img_dir)
    train_labels = list(train_label_dir.glob("*.txt"))

    if not train_images:
        print("❌ ERROR: No training images found!")
        return False

    if not train_labels:
        print("❌ ERROR: No training labels found!")
        return False

    train_img_stems = {f.stem for f in train_images}
    train_lbl_stems = {f.stem for f in train_labels}

    missing_labels = train_img_stems - train_lbl_stems
    if missing_labels:
        print(f"⚠️ Warning: {len(missing_labels)} images without labels")
        if len(missing_labels) <= 10:
            print("Missing labels for:", list(missing_labels))

    print(f"📊 Train images: {len(train_images)}")
    print(f"📊 Train labels: {len(train_labels)}")

    print("✅ Dataset verification complete!")
    return True

# =================== GPU OPTIMIZATION ===================
def setup_gpu_optimizations():
    """
    Safe GPU optimizations for:
      - RTX 2080 Ti (Turing, CC 7.5)
      - RTX 3050 (Ampere, CC 8.6)
      - Any newer NVIDIA GPU

    AMP is handled separately via autocast + GradScaler.
    """
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, running on CPU")
        return False

    # cuDNN autotuner (VERY important for YOLO-style fixed input sizes)
    torch.backends.cudnn.benchmark = True

    # Detect compute capability
    major, minor = torch.cuda.get_device_capability(0)

    # TF32 is SAFE only on Ampere+
    tf32_enabled = False
    if major >= 8:  # Ampere or newer (RTX 30xx, 40xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        tf32_enabled = True
    else:
        # Explicitly disable TF32 on Turing (RTX 20xx)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Reduce memory fragmentation (helps 6GB GPUs like RTX 3050)
    try:
        torch.cuda.set_per_process_memory_fraction(0.95, device=0)
    except Exception:
        pass  # older torch versions may not support this

    torch.cuda.empty_cache()

    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    memory_gb = props.total_memory / 1e9

    print("🖥️  GPU OPTIMIZATION ENABLED")
    print(f"   • GPU: {gpu_name}")
    print(f"   • Compute Capability: {major}.{minor}")
    print(f"   • Memory: {memory_gb:.1f} GB")
    print(f"   • CUDA: {torch.version.cuda}")
    print(f"   • TF32 Enabled: {tf32_enabled}")
    print(f"   • cuDNN Benchmark: {torch.backends.cudnn.benchmark}")

    return True

# =================== LEARNING RATE FINDER ===================
def find_optimal_lr(
    model,
    train_loader,
    criterion,
    device,
    run_dir,
    start_lr=1e-7,
    end_lr=0.1,
    num_iter=100,
):
    """
    Find optimal learning rate using Leslie Smith's LR range test.
    Automatically adapts to SMALL vs LARGE datasets.

    - Small dataset (e.g. MCU images):
        • Fewer iterations
        • Narrower, safer LR range
    - Large dataset:
        • Full LR sweep
    """

    print("\n🔍 Finding optimal learning rate...")

    # ------------------ DATASET-AWARE ADAPTATION ------------------
    dataset_size = len(train_loader.dataset)
    batches = len(train_loader)

    # Cap iterations: max 2 passes over dataset
    max_reasonable_iters = batches * 2
    num_iter = min(num_iter, max_reasonable_iters)

    # Adjust LR range for tiny datasets (MCU / small object datasets)
    if dataset_size < 500:
        start_lr = 1e-6
        end_lr = 5e-3
        print("🧠 Small dataset detected → conservative LR range")
    else:
        print("🧠 Large dataset detected → full LR sweep")

    print(f"🔁 LR finder iterations: {num_iter}")
    print(f"📈 LR range: {start_lr:.1e} → {end_lr:.1e}")

    # --------------------------------------------------------------
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    lrs = []
    losses = []

    avg_loss = 0.0
    best_loss = float("inf")
    beta = 0.98  # smoothing factor

    # Exponential LR increase per step
    lr_mult = (end_lr / start_lr) ** (1 / max(1, num_iter))

    data_iter = iter(train_loader)
    pbar = tqdm(range(num_iter), desc="LR Finder", leave=False)

    for i in pbar:
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)

        images = images.to(device, non_blocking=True)

        # ------------------ YOLO target preparation ------------------
        targets_p3, targets_p4 = [], []
        for t in targets:
            if t.numel() == 0:
                targets_p3.append(torch.zeros((0, 5), device=device))
                targets_p4.append(torch.zeros((0, 5), device=device))
                # targets_p5.append(torch.zeros((0, 5), device=device))
                continue

            t = t.to(device)
            area = t[:, 3] * t[:, 4] * 512 * 512
            targets_p3.append(t[area < 512])
            targets_p4.append(t[(area >= 512) & (area < 1024)])
            # targets_p5.append(t[area >= 1024])

        optimizer.zero_grad(set_to_none=True)

        # ------------------ Forward / Backward ------------------
        with autocast(enabled=(device.type == "cuda")):
            pred = model(images)
            loss_dict = criterion(
                pred[0], pred[1],
                targets_p3, targets_p4
            )
            loss = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ------------------ Smoothed loss ------------------
        loss_val = loss.item()
        avg_loss = beta * avg_loss + (1 - beta) * loss_val
        smoothed_loss = avg_loss / (1 - beta ** (i + 1))

        # Stop early if loss explodes
        if i > 10 and smoothed_loss > 4 * best_loss:
            print("⚠️ LR finder stopped early (loss exploded)")
            break

        best_loss = min(best_loss, smoothed_loss)

        # Record
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        losses.append(smoothed_loss)

        # Increase LR
        for pg in optimizer.param_groups:
            pg["lr"] *= lr_mult

        pbar.set_postfix(
            LR=f"{current_lr:.2e}",
            Loss=f"{smoothed_loss:.4f}",
        )

    # ------------------ Select optimal LR ------------------
    if len(lrs) < 10:
        print("⚠️ Not enough points for LR selection, using default")
        return None

    # Smooth loss curve
    window = max(1, len(losses) // 10)
    if window > 1:
        kernel = np.ones(window) / window
        losses_smooth = np.convolve(losses, kernel, mode="valid")
        lrs_smooth = lrs[window - 1 :]
    else:
        losses_smooth = losses
        lrs_smooth = lrs

    # Steepest negative gradient
    gradients = np.gradient(losses_smooth)
    best_idx = np.argmin(gradients)
    optimal_lr = lrs_smooth[best_idx]

    print(f"✅ Optimal LR found: {optimal_lr:.2e}")

    # ------------------ Plot ------------------
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(lrs_smooth, losses_smooth, linewidth=2)
    plt.axvline(optimal_lr, color="red", linestyle="--",
                label=f"Optimal LR: {optimal_lr:.2e}")
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(run_dir, "plots", "lr_finder.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📈 LR finder plot saved to: {plot_path}")
    return optimal_lr


# =================== TRAIN LOOP WITH TQDM ===================
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    accum_steps,
    scheduler,
    epoch,
    warmup_epochs,
    writer=None,
    ema=None,
):
    """Train for one epoch with AMP, gradient accumulation, EMA, warmup, and TQDM."""
    model.train()

    total_loss = 0.0
    bbox_loss_total = 0.0
    obj_loss_total = 0.0
    cls_loss_total = 0.0

    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    def _to_float(v):
        if isinstance(v, torch.Tensor):
            return v.item()
        try:
            return float(v)
        except Exception:
            return 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1:03d} [Train]", leave=False)

    for i, batch in enumerate(pbar):
        if batch is None:
            continue

        images, targets = batch

        images = images.to(device, non_blocking=True)
        targets = [t.to(device) for t in targets]
        # ------------------------------
        # Prepare scale-aware targets
        # ------------------------------
        targets_p3, targets_p4 = [], []
        for t in targets:
            if t.numel() == 0:
                targets_p3.append(torch.zeros((0, 5), device=device))
                targets_p4.append(torch.zeros((0, 5), device=device))
                continue

            t = t.to(device)
            area = t[:, 3] * t[:, 4] * 512 * 512
            targets_p3.append(t[area < 512])                      # P3 (stride 4)
            targets_p4.append(t[(area >= 512) & (area < 1024)])   # P4 (stride 8)

        # ------------------------------
        # Forward (AMP)
        # ------------------------------
        with autocast(enabled=(device.type == "cuda")):
            pred = model(images)
            loss_dict = criterion(pred[0], pred[1], targets_p3, targets_p4)
            raw_loss = loss_dict["total"]
            loss = raw_loss / accum_steps

        # ------------------------------
        # Backward
        # ------------------------------
        scaler.scale(loss).backward()

        # ------------------------------
        # Optimizer / EMA / Scheduler
        # ------------------------------
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

            # ✅ Scheduler steps ONLY after warmup
            if epoch >= warmup_epochs:
                scheduler.step()

        # ------------------------------
        # Warmup (LR override)
        # ------------------------------
        if epoch < warmup_epochs:
            warmup_factor = (epoch * len(loader) + i + 1) / (warmup_epochs * len(loader))
            for g in optimizer.param_groups:
                g["lr"] = g["initial_lr"] * warmup_factor

        # ------------------------------
        # Loss accounting (UNSCALED)
        # ------------------------------
        total_loss += _to_float(raw_loss)
        bbox_loss_total += _to_float(loss_dict.get("bbox", 0))
        obj_loss_total += _to_float(loss_dict.get("obj", 0))
        cls_loss_total += _to_float(loss_dict.get("cls", 0))

        # ------------------------------
        # Progress bar
        # ------------------------------
        pbar.set_postfix({
            "Loss": f"{raw_loss.item():.3f}",
            "LR": f"{optimizer.param_groups[0]['lr']:.1e}"
        })

        # ------------------------------
        # TensorBoard (batch)
        # ------------------------------
        if writer and i % 10 == 0:
            global_step = epoch * len(loader) + i
            writer.add_scalar("LR/train", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("Loss/train_batch", raw_loss.item(), global_step)

    # ------------------------------
    # Epoch averages
    # ------------------------------
    n_batches = max(len(loader), 1)
    avg_total = total_loss / n_batches
    avg_bbox = bbox_loss_total / n_batches
    avg_obj = obj_loss_total / n_batches
    avg_cls = cls_loss_total / n_batches

    # ------------------------------
    # TensorBoard (epoch)
    # ------------------------------
    if writer:
        writer.add_scalar("Loss/train", avg_total, epoch)
        writer.add_scalar("Loss/bbox", avg_bbox, epoch)
        writer.add_scalar("Loss/obj", avg_obj, epoch)
        writer.add_scalar("Loss/cls", avg_cls, epoch)
        writer.add_scalar("LR/epoch", optimizer.param_groups[0]["lr"], epoch)

    epoch_time = time.time() - start_time
    batches = max(len(loader), 1)
    print(f"  ⏱️  Epoch time: {epoch_time:.1f}s ({epoch_time / batches:.2f}s/batch)")

    # Return order MUST match main()
    return avg_total, avg_bbox, avg_cls, avg_obj


@torch.no_grad()
def validate(model, loader, criterion, device, run_dir, epoch, writer=None, ema=None, calculate_metrics=False, plot=False):
    """
    Validate model on `loader`.

    Returns:
      avg_total, avg_bbox, avg_cls, avg_obj, metrics_dict, plot_data

    Behavior:
      - If `calculate_metrics` is True: decodes predictions (P3+P4), accumulates per-image preds & targets
        and computes mAP via `calculate_map`.
      - If `plot` is True: saves confusion matrix, F1/confidence curve, PR curves, heatmap via
        save_plots_from_validation(...) (which uses plot_confusion_matrix, plot_f1_confidence_curve, etc.)
      - Uses EMA model if `ema` provided (prefers ema.ema).
      - Defensive: handles empty batches, CPU/torch types, and avoids crashes on degenerate inputs.
    """
    # Prepare evaluation model (prefer EMA)
    eval_model = ema.ema if ema is not None else model
    if device is not None:
        eval_model = eval_model.to(device)
    eval_model.eval()

    # accumulators
    total_loss = 0.0
    bbox_loss_total = 0.0
    obj_loss_total = 0.0
    cls_loss_total = 0.0

    all_targets = []      # list (per image) of np.ndarray (M,5) in YOLO format (cls, cx, cy, w, h)
    all_preds = []        # list (per image) of np.ndarray (N,6) in decode format (cls, conf, x1,y1,x2,y2)
    all_box_centers = []  # list of (cx_norm, cy_norm) for heatmap

    start_time = time.time()
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]", leave=False)

    def _to_float(v):
        if isinstance(v, torch.Tensor):
            return v.item()
        try:
            return float(v)
        except Exception:
            return 0.0

    for batch_idx, (images, targets) in enumerate(pbar):
        # images: Tensor (B,C,H,W)
        images = images.to(device)

        # Build scale-aware targets for loss and also image-aligned targets for metrics
        targets_p3 = []
        targets_p4 = []
        batch_targets_per_image = []

        for t in targets:
            # t expected shape (N,5) or empty tensor
            if t is None or (isinstance(t, torch.Tensor) and t.numel() == 0):
                targets_p3.append(torch.zeros((0, 5), device=device))
                targets_p4.append(torch.zeros((0, 5), device=device))
                batch_targets_per_image.append(np.zeros((0, 5), dtype=np.float32))
                continue

            t = t.to(device)
            # area uses normalized w*h scaled to image pixels (512x512)
            area = t[:, 3] * t[:, 4] * 512.0 * 512.0
            targets_p3.append(t[area < 512])
            targets_p4.append(t[(area >= 512) & (area < 1024)])

            # image-aligned GT (keep YOLO format normalized)
            gt_cpu = t.detach().cpu().numpy().astype(np.float32)
            if gt_cpu.shape[0] > 0:
                batch_targets_per_image.append(gt_cpu)
            else:
                batch_targets_per_image.append(np.zeros((0, 5), dtype=np.float32))

            # collect centers for heatmap (normalized cx, cy)
            try:
                centers = t[:, 1:3].detach().cpu().numpy().tolist()
                all_box_centers.extend(centers)
            except Exception:
                pass

        # Forward with eval model (EMA or raw eval_model)
        pred = eval_model(images)

        # Loss computation — adapt to returned tuple format (pred[0], pred[1])
        loss_dict = criterion(pred[0], pred[1], targets_p3, targets_p4)

        # Accumulate safe scalars
        total_val = _to_float(loss_dict.get("total", 0))
        bbox_val  = _to_float(loss_dict.get("bbox",  0))
        obj_val   = _to_float(loss_dict.get("obj",   0))
        cls_val   = _to_float(loss_dict.get("cls",   0))

        total_loss += total_val
        bbox_loss_total += bbox_val
        obj_loss_total += obj_val
        cls_loss_total += cls_val

        pbar.set_postfix({"Loss": f"{total_val:.3f}"})

        # Decode predictions only if requested for metrics
        if calculate_metrics:
            # decode_predictions returns list of numpy arrays per image (on CPU)
            try:
                decoded = decode_predictions(
                    pred[0], pred[1],  # P3, P4
                    conf_thresh=0.01,
                    nms_thresh=0.45
                )
            except Exception as e:
                # Defensive fallback: create empty predictions for this batch
                decoded = [np.zeros((0, 6), dtype=np.float32) for _ in range(images.shape[0])]
                print(f"⚠️ decode_predictions failed at batch {batch_idx}: {e}")

            # Ensure alignment between decoded results and images in this batch
            if len(decoded) != len(batch_targets_per_image):
                # If mismatch, try to coerce by repeating/trim — but warn
                decoded = decoded[:len(batch_targets_per_image)] if len(decoded) >= len(batch_targets_per_image) else decoded + [np.zeros((0, 6), dtype=np.float32)] * (len(batch_targets_per_image) - len(decoded))
                print(f"⚠️ Warning: decoded length ({len(decoded)}) != batch size ({len(batch_targets_per_image)}), aligned forcefully.")

            # Append per-image preds and gts (image-aligned)
            for pb, gt in zip(decoded, batch_targets_per_image):
                # normalize types
                if isinstance(pb, torch.Tensor):
                    if pb.numel() == 0:
                        pb_arr = np.zeros((0, 6), dtype=np.float32)
                    else:
                        pb_arr = pb.detach().cpu().numpy().astype(np.float32)
                elif isinstance(pb, np.ndarray):
                    pb_arr = pb.astype(np.float32) if pb.size else np.zeros((0, 6), dtype=np.float32)
                else:
                    pb_arr = np.zeros((0, 6), dtype=np.float32)

                # ensure shape (N,6)
                if pb_arr.ndim == 1 and pb_arr.size > 0:
                    pb_arr = pb_arr.reshape(1, -1)

                all_preds.append(pb_arr)
                # gt already image-aligned; ensure float32 numpy array
                if isinstance(gt, torch.Tensor):
                    gt_arr = gt.detach().cpu().numpy().astype(np.float32)
                else:
                    gt_arr = np.array(gt, dtype=np.float32).reshape(-1, 5) if gt is not None else np.zeros((0,5), dtype=np.float32)
                all_targets.append(gt_arr)

    # End batches
    n_batches = len(loader) if len(loader) > 0 else 1
    avg_total = total_loss / n_batches
    avg_bbox = bbox_loss_total / n_batches
    avg_obj = obj_loss_total / n_batches
    avg_cls = cls_loss_total / n_batches

    # Compute dataset-level metrics (mAP) when requested and when we have predictions & targets
    metrics = {}
    if calculate_metrics and len(all_targets) > 0 and len(all_preds) > 0:
        try:
            map_50_95, map_50, map_75, per_class_ap = calculate_map(
                predictions=all_preds,
                targets=all_targets,
                num_classes=NUM_CLASSES,
                iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                epoch=epoch
            )
            metrics = {
                "mAP_50": map_50,
                "mAP_75": map_75,
                "mAP_50_95": map_50_95,
                "per_class_ap": per_class_ap
            }
            if writer:
                writer.add_scalar("Metrics/mAP_50", map_50, epoch)
                writer.add_scalar("Metrics/mAP_75", map_75, epoch)
                writer.add_scalar("Metrics/mAP_50_95", map_50_95, epoch)
        except Exception as e:
            print(f"⚠️ calculate_map failed: {e}")
            metrics = {}
    else:
        # Ensure consistent keys (empty) for downstream code
        metrics = {}

    # Write validation losses to TensorBoard
    if writer:
        writer.add_scalar("Loss/val", avg_total, epoch)
        writer.add_scalar("Loss/val_bbox", avg_bbox, epoch)
        writer.add_scalar("Loss/val_obj", avg_obj, epoch)
        writer.add_scalar("Loss/val_cls", avg_cls, epoch)

    # Prepare plot_data to return (image-aligned lists)
    plot_data = {
        "all_preds": all_preds,
        "all_targets": all_targets,
        "all_box_centers": all_box_centers
    }

    # Plot and save diagnostics only when requested (single call)
    if plot:
        try:
            # fallback: if all_preds empty, create empty arrays per image to keep alignment
            if len(plot_data["all_preds"]) == 0:
                plot_data["all_preds"] = [np.zeros((0, 6), dtype=np.float32) for _ in range(len(plot_data["all_targets"]))]

            # Use your helper to save confusion, F1/conf curves, PR curves, heatmap, etc.
            save_plots_from_validation(plot_data, run_dir, epoch)
        except Exception as e:
            print(f"⚠️ Plotting error in validate(): {e}")

    elapsed = time.time() - start_time
    print(f"  ⏱️ Validation time: {elapsed:.1f}s")

    return avg_total, avg_bbox, avg_cls, avg_obj, metrics, plot_data





# =================== SAVE LOSS PLOT ===================
def save_loss_plot(train_losses, val_losses, run_dir, title="Training and Validation Loss"):
    """
    Save loss plot with adaptive smoothing for small datasets.
    Optimized for microcontroller training (few epochs, early stopping).
    """
    # ---- Guard against empty data ----
    if not train_losses or not val_losses:
        print("⚠️ No loss data to plot")
        return
    
    if len(train_losses) != len(val_losses):
        print(f"⚠️ Length mismatch: train={len(train_losses)}, val={len(val_losses)}")
        min_len = min(len(train_losses), len(val_losses))
        train_losses = train_losses[:min_len]
        val_losses = val_losses[:min_len]
    
    # Convert to numpy arrays for safety
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    
    plt.figure(figsize=(12, 8))
    
    # ---- ADAPTIVE SMOOTHING for small datasets ----
    # Small datasets (< 20 epochs): minimal smoothing
    # Large datasets (> 100 epochs): aggressive smoothing
    num_epochs = len(train_losses)
    
    if num_epochs < 10:
        # Very short training - no smoothing (preserve all data)
        window = 1
    elif num_epochs < 50:
        # Small dataset - light smoothing (show trends without losing detail)
        window = max(3, num_epochs // 10)
    else:
        # Large dataset - standard smoothing
        window = max(5, num_epochs // 20)
    
    # Apply smoothing
    if window > 1 and len(train_losses) > window:
        kernel = np.ones(window) / window
        train_smooth = np.convolve(train_losses, kernel, mode='valid')
        val_smooth = np.convolve(val_losses, kernel, mode='valid')
        x_train = np.arange(window - 1, len(train_losses))
        x_val = np.arange(window - 1, len(val_losses))
    else:
        train_smooth = train_losses
        val_smooth = val_losses
        x_train = np.arange(len(train_losses))
        x_val = np.arange(len(val_losses))
    
    # ---- Plot smoothed lines ----
    plt.plot(x_train, train_smooth, label='Train Loss', 
             color='blue', linewidth=2, alpha=0.8)
    plt.plot(x_val, val_smooth, label='Val Loss', 
             color='red', linewidth=2, alpha=0.8)
    
    # ---- Plot original points (only if not too many) ----
    if num_epochs <= 100:
        plt.scatter(range(len(train_losses)), train_losses, 
                    color='blue', alpha=0.3, s=10)
        plt.scatter(range(len(val_losses)), val_losses, 
                    color='red', alpha=0.3, s=10)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # ---- Add best validation loss annotation ----
    best_loss = None
    if len(val_losses) > 0:
        best_epoch = int(np.argmin(val_losses))
        best_loss = float(val_losses[best_epoch])
        
        # Vertical line at best epoch
        plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
        
        # ---- ADAPTIVE TEXT PLACEMENT ----
        # Prevent text from going off-screen for small datasets
        y_range = np.ptp(val_losses) if np.ptp(val_losses) > 0 else 1.0
        y_min = np.min(val_losses)
        
        # Place text in upper portion but not too high
        text_y = y_min + y_range * 0.85
        
        # Horizontal alignment based on epoch position
        if best_epoch < num_epochs * 0.3:
            ha = 'left'
            text_x = best_epoch + 1
        elif best_epoch > num_epochs * 0.7:
            ha = 'right'
            text_x = best_epoch - 1
        else:
            ha = 'center'
            text_x = best_epoch
        
        plt.text(text_x, text_y, 
                f'Best: {best_loss:.4f} @ epoch {best_epoch + 1}',
                fontsize=10, color='green', ha=ha,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='green', alpha=0.8))
    
    plt.tight_layout()
    
    # ---- Save plot ----
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    plot_path = os.path.join(run_dir, "plots", "loss_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if best_loss is not None:
        print(f"📈 Loss plot saved → {plot_path} | Best val: {best_loss:.4f}")
    else:
        print(f"📈 Loss plot saved → {plot_path}")

# =================== WARMUP LR PLOT ===================
def plot_warmup_lr(optimizer, total_steps, warmup_steps, run_dir):
    """Plot learning rate during warmup."""
    lrs = []
    steps = list(range(total_steps))
    
    for step in steps:
        if step < warmup_steps:
            warmup_factor = (step + 1) / warmup_steps
            lr = optimizer.param_groups[0]['initial_lr'] * warmup_factor
        else:
            # Continue with scheduler
            lr = optimizer.param_groups[0]['initial_lr']
        lrs.append(lr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs, 'b-', linewidth=2)
    plt.axvline(x=warmup_steps, color='r', linestyle='--', 
                label=f'Warmup end: step {warmup_steps}')
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Warmup Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(run_dir, "plots", "warmup_lr.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Warmup LR plot saved to: {plot_path}")

# =================== Helper: Save plots from validation data ===================
def save_plots_from_validation(plot_data, run_dir, epoch):
    """Given plot_data from validate(), save confusion, f1/confidence, PR/precision curves, and heatmap."""
    try:
        all_preds = plot_data.get("all_preds", [])
        all_targets = plot_data.get("all_targets", [])
        all_box_centers = plot_data.get("all_box_centers", [])

        # fallback: if no preds, create empty preds per image
        if len(all_preds) == 0:
            all_preds = [np.zeros((0, 6), dtype=np.float32) for _ in range(len(all_targets))]

        # Confusion matrix (top-pred per image)
        y_true_cm = []
        y_pred_cm = []
        for gt, pred in zip(all_targets, all_preds):
            if isinstance(gt, np.ndarray) and gt.shape[0] > 0:
                y_true_cm.append(int(gt[0, 0]))
            else:
                y_true_cm.append(-1)
            if isinstance(pred, np.ndarray) and pred.shape[0] > 0:
                y_pred_cm.append(int(pred[0, 0]))
            else:
                y_pred_cm.append(-1)

        y_true_clean = []
        y_pred_clean = []
        for t, p in zip(y_true_cm, y_pred_cm):
            if t == -1 and p == -1:
                continue
            y_true_clean.append(t if t != -1 else 0)
            y_pred_clean.append(p if p != -1 else 0)

        plots_root = os.path.join(run_dir, "plots")
        os.makedirs(plots_root, exist_ok=True)

        plot_confusion_matrix(
            y_true=y_true_clean,
            y_pred=y_pred_clean,
            labels=CLASSES,
            run_dir=plots_root,
            base_title=f"Confusion Matrix - Epoch {epoch}"
        )

        plot_f1_confidence_curve(
            predictions=all_preds,
            targets=all_targets,
            class_names=CLASSES,
            run_dir=plots_root
        )

        if len(all_box_centers) > 0:
            plot_label_heatmap(
                box_centers=all_box_centers,
                run_dir=plots_root
            )

       
        if len(all_preds) > 0:
            confidences, precisions, recalls = compute_precision_recall_curves(
                all_preds=all_preds,
                all_targets=all_targets,
                num_classes=len(CLASSES),
                img_size=512,
                adaptive_iou=True
            )
            
            save_precision_recall_curves(
                confidences=confidences,
                precisions=precisions,
                recalls=recalls,
                class_names=CLASSES,
                run_dir=plots_root
            )
        else:
            print("⚠️ No predictions - skipping PR curves")

        print(f"📊 Validation plots saved to: {plots_root} (epoch {epoch})")
    except Exception as e:
        print(f"⚠️ Error saving validation plots: {e}")


# =================== MAIN ===================
def main():
    parser = argparse.ArgumentParser(description="Train MCUDetector (7 classes) - YOLO Detection Only")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=12, help="RTX 2080Ti optimized")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--no_tb", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--train_img_dir", type=str, help="Custom train images directory")
    parser.add_argument("--train_label_dir", type=str, help="Custom train labels directory")
    parser.add_argument("--val_img_dir", type=str, help="Custom validation images directory")
    parser.add_argument("--val_label_dir", type=str, help="Custom validation labels directory")
    parser.add_argument("--find_lr", action="store_true", help="Run learning rate finder before training")
    parser.add_argument("--use_ema", action="store_true", help="Use Exponential Moving Average")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--calculate_map", action="store_true", help="Calculate mAP during validation")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    # GPU optimizations
    gpu_enabled = setup_gpu_optimizations() if device.type == "cuda" else False
    
    # Set paths
    if args.train_img_dir:
        TRAIN_IMG_DIR = Path(args.train_img_dir)
    else:
        TRAIN_IMG_DIR = DATA_DIR / "dataset_train" / "images" / "train"
    
    if args.train_label_dir:
        TRAIN_LABEL_DIR = Path(args.train_label_dir)
    else:
        TRAIN_LABEL_DIR = DATA_DIR / "dataset_train" / "labels" / "train"
    
    if args.val_img_dir:
        VAL_IMG_DIR = Path(args.val_img_dir)
    else:
        VAL_IMG_DIR = DATA_DIR / "dataset_test" / "images" / "train"
    
    if args.val_label_dir:
        VAL_LABEL_DIR = Path(args.val_label_dir)
    else:
        VAL_LABEL_DIR = DATA_DIR / "dataset_test" / "labels" / "train"
    
    # Verify dataset paths
    if not verify_dataset_paths(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, VAL_IMG_DIR, VAL_LABEL_DIR):
        sys.exit(1)
    
    print(f"\n📊 Number of classes: {NUM_CLASSES}")
    if CLASSES:
        print(f"🎯 Classes: {', '.join(CLASSES)}")
    
    run_dir = Path(get_run_dir("detect/train"))
    print(f"📂 Run directory: {run_dir}")
    # Train directory structure
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "images").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    # ensure weights folder exists (was missing previously)
    (run_dir / "weights").mkdir(exist_ok=True)

    
    # =================== TRAINING METRICS TRACKER ===================
    metrics_tracker = TrainingMetricsTracker(run_dir)


    # =================== TEST RUN DIRECTORY ===================
    test_run_dir = Path(get_run_dir("detect/test"))
    (test_run_dir / "plots").mkdir(parents=True, exist_ok=True)
    print(f"📂 Test run directory: {test_run_dir}")
    
    # Save config
    config = {
        "num_classes": NUM_CLASSES,
        "classes": CLASSES if CLASSES else [],
        "args": vars(args),
        "paths": {
            "train_images": str(TRAIN_IMG_DIR),
            "train_labels": str(TRAIN_LABEL_DIR),
            "val_images": str(VAL_IMG_DIR),
            "val_labels": str(VAL_LABEL_DIR),
        },
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "gpu_enabled": gpu_enabled
    }
    
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📝 Config saved to: {config_path}")
    
    # Setup TensorBoard
    writer = None
    if not args.no_tb:
        try:
            tb_dir = run_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)
            print(f"📊 TensorBoard logs: tensorboard --logdir={tb_dir}")
        except Exception as e:
            print(f"⚠️ Could not setup TensorBoard: {e}")
    
    # Data transforms
    transform = transforms.Compose([
        
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Data loaders
    print("\n📦 Creating data loaders...")
    try:
        train_dataset = MCUDetectionDataset(
            img_dir=TRAIN_IMG_DIR,
            label_dir=TRAIN_LABEL_DIR,
            img_size=512,
            transform=transform
        )
        
        val_dataset = MCUDetectionDataset(
            img_dir=VAL_IMG_DIR,
            label_dir=VAL_LABEL_DIR,
            img_size=512,
            transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=detection_collate_fn,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True if args.workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=detection_collate_fn,
            pin_memory=False,
            persistent_workers=True if args.workers > 0 else False
        )
        
        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Validation samples: {len(val_dataset)}")
        print(f"✅ Training batches: {len(train_loader)}")
        
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        sys.exit(1)
    
    # Model and criterion
    print("\n🤖 Creating model...")
    try:
        model = MCUDetector(num_classes=NUM_CLASSES).to(device)
        criterion = MCUDetectionLoss(num_classes=NUM_CLASSES).to(device)
        print(f"✅ Model parameters: {count_parameters(model) / 1e6:.2f}M")
        print_gpu_memory()
        print("📏 Using adaptive IoU thresholds for small object detection")
        print("   • Objects <16×16: IoU threshold × 0.7")
        print("   • Objects 16-32: IoU threshold × 0.85")
        print("   • Objects >32: IoU threshold × 1.0")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        sys.exit(1)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Store initial LR for warmup
    for g in optimizer.param_groups:
        g['initial_lr'] = args.lr
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=20, 
        T_mult=2, 
        eta_min=1e-6
    )
    
    # Mixed precision training
    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    # EMA
    ema = None
    if args.use_ema:
        ema = ModelEMA(model, decay=args.ema_decay)
        print(f"✅ Using EMA with decay={args.ema_decay}")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    best_map = 0.0
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint.get('epoch', 0)
            best_loss = checkpoint.get('best_loss', float('inf'))
            best_map = checkpoint.get('best_map', 0.0)
            
             # ✅ RESTORE EMA STATE HERE (CORRECT PLACE)
            if args.use_ema and checkpoint.get("ema_state_dict") is not None:
                try:
                    ema.ema.load_state_dict(checkpoint["ema_state_dict"])
                    print("🔁 Restored EMA state from checkpoint.")
                except Exception as e:
                    print(f"⚠️ Could not restore EMA state: {e}")

            print(f"✅ Loaded checkpoint from epoch {start_epoch}")
            print(f"   Best loss: {best_loss:.4f}, Best mAP: {best_map:.4f}")

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
    
    # Learning rate finder
    if args.find_lr:
        optimal_lr = find_optimal_lr(
            model, train_loader, criterion, device, run_dir
        )
        if optimal_lr:
            for g in optimizer.param_groups:
                g['lr'] = optimal_lr
                g['initial_lr'] = optimal_lr
            print(f"🎯 Using found LR: {optimal_lr:.2e}")
    
    # Plot warmup LR schedule
    if args.warmup_epochs > 0:
        plot_warmup_lr(optimizer, args.warmup_epochs * len(train_loader), 
                      args.warmup_epochs * len(train_loader), run_dir)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"🚀 Starting training for {args.epochs} epochs...")
    print(f"📦 Batch size: {args.batch_size} × {args.accum_steps} = {args.batch_size * args.accum_steps}")
    print(f"📈 Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"🔥 Warmup epochs: {args.warmup_epochs}")
    print(f"🛑 Early stopping patience: {args.patience}")
    print(f"📊 mAP calculation: {'Enabled' if args.calculate_map else 'Disabled'}")
    print(f"📈 EMA: {'Enabled' if args.use_ema else 'Disabled'}")
    print(f"{'='*60}")
    
    train_losses = []
    train_f1_history = []
    val_f1_history = []
    val_losses = []
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1:03d}/{args.epochs}")
        
        # Train
        train_loss, train_bbox, train_cls, train_obj = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, args.accum_steps, scheduler, epoch, 
            args.warmup_epochs, writer, ema
        )
        train_losses.append(train_loss)
        
        
        # Validate with EMA model if available
        
        val_loss, val_bbox, val_cls, val_obj, metrics, plot_data = validate(
            model, val_loader, criterion, device,
            run_dir, epoch+1, writer, ema, args.calculate_map, plot=False
        )
        val_losses.append(val_loss)
        if plot_data["all_preds"] and plot_data["all_targets"]:
            p, r = compute_epoch_precision_recall(
                plot_data["all_preds"],
                plot_data["all_targets"],
                conf_thresh=0.01,
                epoch=epoch
            )
        else:
            p, r = 0.0, 0.0
       
        # ✅ Update metrics tracker
        metrics_tracker.update(
        epoch=epoch+1,
        train_metrics={
            "loss": train_loss,
            "box": train_bbox,
            "cls": train_cls,
            "obj": train_obj
        },
        val_metrics={
            "loss": val_loss,
            "box": val_bbox,
            "cls": val_cls,
            "obj": val_obj,
            "precision": p,
            "recall": r,
            "map50": metrics.get("mAP_50", 0.0),
            "map5095": metrics.get("mAP_50_95", 0.0)
        },
        lr=optimizer.param_groups[0]['lr']
    )

        
        # Print metrics
        metric_str = ""
        if metrics:
            metric_str = f" | mAP@0.5: {metrics.get('mAP_50', 0):.4f}"
        
        print(f"  📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{metric_str}")
        print(f"  📊 Train Box: {train_bbox:.3f} | Cls: {train_cls:.3f} | Obj: {train_obj:.3f}")
        print(f"  📈 P: {p:.3f} | R: {r:.3f} | F1: {2*p*r/(p+r+1e-12):.3f}")
        print(f"  📈 LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        
        # Save best model based on validation loss AND mAP
        current_metric = metrics.get('mAP_50', 0) if args.calculate_map else -val_loss
        
        if args.calculate_map:
            # Use mAP for model selection
            if current_metric > best_map:
                best_map = current_metric
                best_loss = val_loss
                
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.ema.state_dict() if ema else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "best_loss": best_loss,
                    "best_map": best_map,
                    "metrics": metrics,
                    "args": vars(args),
                    "classes": CLASSES,
                    "num_classes": NUM_CLASSES
                }
                checkpoint_path = run_dir / "weights" / "best_mcu.pt"
                torch.save(checkpoint, checkpoint_path)
                # torch.save(checkpoint, run_dir / "model" / "best_mcu.pt")
                print(f"  💾 BEST mAP! Model saved (mAP@0.5: {current_metric:.4f})")

                # --- NEW: save validation plots once when best model updates ---
                save_plots_from_validation(plot_data, run_dir, epoch+1)
        else:
            # Use validation loss for model selection
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.ema.state_dict() if ema else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "best_loss": best_loss,
                    "best_map": best_map,
                    "metrics": metrics,
                    "args": vars(args),
                    "classes": CLASSES,
                    "num_classes": NUM_CLASSES
                }
                checkpoint_path = run_dir / "weights" / "best_mcu.pt"
                torch.save(checkpoint, checkpoint_path)
                # torch.save(checkpoint, run_dir / "model" / "best_mcu.pt")
                print(f"  💾 BEST! Model saved (val_loss: {val_loss:.4f})")

                # --- NEW: save validation plots once when best model updates ---
                save_plots_from_validation(plot_data, run_dir, epoch+1)
        
        # Periodic checkpoint (overwrite single latest file to avoid accumulation)
        if (epoch + 1) % 10 == 0:
            checkpoint_latest = run_dir / "weights" / "checkpoint_latest.pt"
            tmp_path = run_dir / "weights" / "checkpoint_tmp.pt"
            latest_data = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.ema.state_dict() if ema else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "best_loss": best_loss,
                "best_map": best_map,
                "metrics": metrics,
            }
            # atomic write: save to tmp then replace
            torch.save(latest_data, tmp_path)
            try:
                os.replace(str(tmp_path), str(checkpoint_latest))
            except Exception:
                # fallback to simple save if os.replace not available
                torch.save(latest_data, checkpoint_latest)
            # also update model/ copy (overwrites)
            
            # torch.save(latest_data)
            print(f"  💾 Checkpoint (latest) saved at epoch {epoch+1}")
            
        # --- REMOVED: save loss plot every 20 epochs ---
        # We no longer save periodic loss plots to reduce clutter.
        # Instead, we save loss plot only when best model updates and at final save.
        # ✅ Save CSV EVERY epoch
        metrics_tracker.save_csv()
        # Check early stopping
        if early_stopping(val_loss):
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            print(f"   No improvement for {early_stopping.counter} epochs")
            break
    
    # Final save
    final_checkpoint = {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.ema.state_dict() if ema else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss if 'val_loss' in locals() else 0,
        "train_loss": train_loss if 'train_loss' in locals() else 0,
        "best_loss": best_loss,
        "best_map": best_map,
        "metrics": metrics if 'metrics' in locals() else {},
        "args": vars(args),
        "classes": CLASSES,
        "num_classes": NUM_CLASSES
    }
    torch.save(final_checkpoint, run_dir / "weights" / "final_mcu.pt")
    
    # ✅ SAVE FINAL TRAINING METRICS
    metrics_tracker.save_csv()
    
    # Save final loss plot (using old train_losses/val_losses)
    print(f"DEBUG: Saving loss curve to {run_dir / 'plots' / 'loss_curve.png'}")
    save_loss_plot(train_losses, val_losses, run_dir, "Final Training and Validation Loss")
    
    # ✅ Plot YOLOv8-style results using tracker
    plot_yolo_results(
        metrics_tracker.history,
        save_path=os.path.join(run_dir, "plots", "results.png")
    )
    
    # --- NEW: Run validation with the BEST model and save TEST plots to test_run_dir/plots ---
    try:
        best_checkpoint_path = run_dir / "weights" / "best_mcu.pt"
        if best_checkpoint_path.exists():
            best_ckpt = torch.load(best_checkpoint_path, map_location=device)
            # load best weights into model (prefer EMA state if present)
            if best_ckpt.get("ema_state_dict", None) is not None:
                try:
                    model.load_state_dict(best_ckpt["ema_state_dict"])
                    print("🔁 Loaded EMA weights for final evaluation.")
                except Exception:
                    model.load_state_dict(best_ckpt["model_state_dict"])
                    print("🔁 Loaded model_state_dict for final evaluation.")
            else:
                model.load_state_dict(best_ckpt["model_state_dict"])
                print("🔁 Loaded best model for final evaluation.")

            
        else:
            print("⚠️ Best checkpoint not found; using current model for final test plots.")

        # Prepare evaluation model (move to device, set eval, and fuse reparam blocks)
        eval_model = get_eval_model(model, ema=None, device=device, fuse=True)

        # Validate on val set with plotting and save plots to test folder (force calculate_metrics=True for full PR/F1)
        val_loss_f, vb, vc, vd, metrics_f, plot_data_f = validate(eval_model,                      # eval model already moved to device and fused
                val_loader,
                criterion,
                device,
                test_run_dir,
                args.epochs,
                writer=None,
                ema=None,
                calculate_metrics=True,
                plot=True
            )
        
        # ✅ Compute final precision/recall
        p_test, r_test = compute_epoch_precision_recall(
            plot_data_f["all_preds"],
            plot_data_f["all_targets"],
            conf_thresh=0.01,
            epoch=args.epochs
        )
        
        # ✅ Save test summary
        test_metrics = {
            'precision': p_test,
            'recall': r_test,
            'mAP_50': metrics_f.get('mAP_50', 0),
            'mAP_75': metrics_f.get('mAP_75', 0),
            'mAP_50_95': metrics_f.get('mAP_50_95', 0),
            'loss': val_loss_f,
            'box_loss': vb,
            'cls_loss': vc,
            'obj_loss': vd
        }
        
        save_test_summary(test_metrics, test_run_dir / "test_summary.txt")
        
        print(f"\n📊 TEST RESULTS:")
        print(f"   - Precision:     {p_test:.3f}")
        print(f"   - Recall:        {r_test:.3f}")
        print(f"   - mAP@0.5:       {metrics_f.get('mAP_50', 0):.3f}")
        print(f"   - mAP@0.5:0.95:  {metrics_f.get('mAP_50_95', 0):.3f}")
        print(f"   - F1 Score:      {2*p_test*r_test/(p_test+r_test+1e-12):.3f}")
        print(f"\n📊 TEST plots saved to: {test_run_dir / 'plots'}")
    except Exception as e:
        print(f"⚠️ Final evaluation error: {e}")
    
    # Close TensorBoard writer
    if writer:
        writer.close()
    
    print(f"\n{'='*60}")
    print("✅ Training complete!")
    print(f"📊 Best validation loss: {best_loss:.4f}")
    if args.calculate_map:
        print(f"🎯 Best mAP@0.5: {best_map:.4f}")
    print(f"📈 Final validation loss: {val_losses[-1] if val_losses else 0:.4f}")
    print(f"💾 Models saved to: {run_dir}/weights/")
    print(f"📊 Plots saved to: {run_dir}/plots/ and {test_run_dir}/plots/")
    print(f"📝 Logs saved to: {run_dir}/logs/")
    
    # Print summary
    print(f"\n📋 Summary:")
    print(f"   - Total epochs trained: {len(train_losses)}")
    if val_losses:
        print(f"   - Best epoch: {np.argmin(val_losses) + 1}")
    print(f"   - Training time: ~{(len(train_losses) * 10):.0f} minutes (est.)")
    
    if not args.no_tb and writer:
        print(f"   - TensorBoard: tensorboard --logdir={run_dir}/tensorboard")

if __name__ == "__main__":
    main()
