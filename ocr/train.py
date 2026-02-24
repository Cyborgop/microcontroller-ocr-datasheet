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
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
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
    compute_precision_recall_curves, compute_precision_recall,  
    plot_yolo_results,  save_test_summary, decode_single_scale, _unpack_pred
)
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure model.py, dataset.py, and utils.py are in the same directory")
    sys.exit(1)

def fuse_repv_blocks(model):# checked okay
    assert not model.training, "Rep-style fusion must be performed only in eval() mode (call model.eval() first)."
    """Call .fuse() on all modules that provide it (RepDWConvSR / RepvitBlock)."""
    
    for m in model.modules():
        if hasattr(m, "fuse") and callable(getattr(m, "fuse")):
            try:
                m.fuse()
            except Exception:
                # fail-safe: ignore modules that can't be fused
                pass

def get_eval_model(model, ema=None, device=None, fuse=False):#checked okay
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
class ModelEMA:#checked correct
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
class EarlyStopping:# checked okay but change for 17k dataset
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=40, min_delta=1e-4):
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
class TrainingMetricsTracker:#checked okay but changed
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

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

    def _s(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().item()
        return float(x)

    def update(self, epoch, train_metrics, val_metrics, lr):
        h = self.history

        h["epoch"].append(epoch)

        h["train_loss"].append(self._s(train_metrics.get("loss", 0)))
        h["train_box"].append(self._s(train_metrics.get("box", 0)))
        h["train_cls"].append(self._s(train_metrics.get("cls", 0)))
        h["train_obj"].append(self._s(train_metrics.get("obj", 0)))

        h["val_loss"].append(self._s(val_metrics.get("loss", 0)))
        h["val_box"].append(self._s(val_metrics.get("box", 0)))
        h["val_cls"].append(self._s(val_metrics.get("cls", 0)))
        h["val_obj"].append(self._s(val_metrics.get("obj", 0)))

        h["precision"].append(self._s(val_metrics.get("precision", 0)))
        h["recall"].append(self._s(val_metrics.get("recall", 0)))
        h["map50"].append(self._s(val_metrics.get("map50", 0)))
        h["map5095"].append(self._s(val_metrics.get("map5095", 0)))

        h["lr"].append(self._s(lr))

        self.save_csv()

    def save_csv(self):
        import csv
        path = self.run_dir / "results.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.history.keys())
            writer.writerows(zip(*self.history.values()))


# =================== DATASET VERIFICATION ===================
# SAME extensions as dataset.py (must match)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")#changed


def verify_dataset_paths(
    train_img_dir,
    train_label_dir,
    val_img_dir,
    val_label_dir,
    num_classes=None
):
    print("\n🔍 Verifying dataset paths (YOLO-style)...")

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

    # -----------------------------
    # Image collection (case-insensitive)
    # -----------------------------
    def collect_images(img_dir):
        files = []
        for ext in IMG_EXTS:
            files.extend(img_dir.glob(f"*{ext.lower()}"))
            files.extend(img_dir.glob(f"*{ext.upper()}"))
        return sorted(set(files))

    # -----------------------------
    # Label validation
    # -----------------------------
    def validate_label_file(label_path):
        """
        YOLO label format:
          cls cx cy w h
        All floats, normalized to [0,1]
        """
        try:
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue  # empty label file allowed
                    parts = line.split()
                    if len(parts) != 5:
                        return False, "Invalid column count"

                    cls, x, y, w, h = map(float, parts)

                    if num_classes is not None:
                        if not (0 <= int(cls) < num_classes):
                            return False, "Class index out of range"

                    if not (0 <= x <= 1 and 0 <= y <= 1):
                        return False, "Center coords out of range"

                    if not (0 < w <= 1 and 0 < h <= 1):
                        return False, "Width/height out of range"

            return True, None
        except Exception as e:
            return False, str(e)

    # -----------------------------
    # Verify a split (train or val)
    # -----------------------------
    def verify_split(name, img_dir, lbl_dir):
        images = collect_images(img_dir)
        labels = list(lbl_dir.glob("*.txt"))

        if not images:
            print(f"❌ ERROR: No {name} images found!")
            return False

        image_stems = {f.stem for f in images}
        label_stems = {f.stem for f in labels}

        missing_labels = image_stems - label_stems
        if missing_labels:
            print(f"⚠️ {name}: {len(missing_labels)} images without labels (background images allowed)")
            if len(missing_labels) <= 10:
                print("   Missing labels for:", list(missing_labels))

        empty_labels = 0
        invalid_labels = []

        for lbl in labels:
            if lbl.stat().st_size == 0:
                empty_labels += 1
                continue

            ok, reason = validate_label_file(lbl)
            if not ok:
                invalid_labels.append((lbl.name, reason))

        if invalid_labels:
            print(f"❌ {name}: Found invalid label files!")
            for fname, reason in invalid_labels[:10]:
                print(f"   {fname}: {reason}")
            return False

        print(f"📊 {name} images: {len(images)}")
        print(f"📊 {name} labels: {len(labels)}")
        print(f"📊 {name} empty labels (background): {empty_labels}")

        return True

    # -----------------------------
    # Run checks
    # -----------------------------
    if not verify_split("Train", train_img_dir, train_label_dir):
        return False

    if not verify_split("Val", val_img_dir, val_label_dir):
        return False

    print("✅ Dataset verification complete (Ultralytics-style)")
    
    return True


# --- ADD this after verify_dataset_paths(...) definition ---
def debug_label_stats(label_files, class_names, run_dir=None):
    """Print per-class instance counts and images-with-class. Save to runs/detect/debug_label_stats.txt"""
    from collections import Counter
    gt_counts = Counter()
    imgs_with_class = Counter()
    for lf in sorted(label_files):
        classes = set()
        try:
            for l in Path(lf).read_text().splitlines():
                if not l.strip():
                    continue
                cls_token = l.split()[0]
                try:
                    cls = int(float(cls_token))
                except Exception:
                    cls = cls_token.lower().replace(' ', '').replace('_','')
                    # fallback: skip non-int tokens
                    continue
                gt_counts[int(cls)] += 1
                classes.add(int(cls))
        except Exception as e:
            print(f"⚠️ debug_label_stats: could not read {lf}: {e}")
        for c in classes:
            imgs_with_class[c] += 1

    print("\n=== DATASET LABEL STATS ===")
    for i, name in enumerate(class_names):
        print(f"  Class {i:02d} {name:20s}  instances={gt_counts.get(i,0):6d}  images={imgs_with_class.get(i,0):6d}")
    total = sum(gt_counts.values())
    if total > 0:
        most = gt_counts.most_common(1)[0]
        imbalance = most[1] / (total / max(1, len(class_names)))
        print(f"  total instances: {total} | largest class: {most} ratio vs mean: {imbalance:.2f}")
    else:
        print("  WARNING: no instances found in label files!")

    if run_dir:
        out = Path(run_dir) / "debug_label_stats.txt"
        with open(out, "a") as f:
            f.write(f"{datetime.now().isoformat()} STATS: total={total} top={gt_counts.most_common(3)}\n")
    print("=== END STATS ===\n")
# --- end add ---

# =================== GPU OPTIMIZATION ===================
def setup_gpu_optimizations():#checked fine
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
def find_optimal_lr(#checked okay
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
            img_size = getattr(train_loader.dataset, "img_size", 512)
            
            area = t[:, 3] * t[:, 4] * img_size * img_size

            # Use the same scale/threshold as train_one_epoch: 0.02 * img_size*img_size (adjust if you want a different split)
            area_threshold = 0.02 * img_size * img_size

            targets_p3.append(t[area < area_threshold])
            targets_p4.append(t[area >= area_threshold])
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
def train_one_epoch(#checked okay
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
    """Train for one epoch with AMP, gradient accumulation, EMA, warmup, and TQDM.

    Notes:
    - Scale-aware target splitting uses image H, W -> area normalized to pixels.
    - Losses are accumulated as raw (unscaled) values for logging; backward uses loss/accum_steps.
    - Scheduler.step() is executed after an optimizer step (only once per accumulation block)
      and only after warmup epochs are complete.
    """
    model.train()
    # DEBUG: collect GT classes seen in this epoch
    epoch_classes: set[int] = set()

    total_loss = 0.0
    bbox_loss_total = 0.0
    obj_loss_total = 0.0
    cls_loss_total = 0.0

    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    def _to_float(v):
        if isinstance(v, torch.Tensor):
            # prefer scalar extraction but guard
            try:
                return float(v.detach().cpu().item())
            except Exception:
                return 0.0
        try:
            return float(v)
        except Exception:
            return 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1:03d} [Train]", leave=False)

    for i, batch in enumerate(pbar):
        if batch is None:
            # defensive: some collate fns may produce None
            continue

        try:
            images, targets = batch
        except Exception:
            # if batch structure unexpected, skip
            print(f"⚠️ Unexpected batch structure at index {i}, skipping.")
            continue

        # Move images to device; we keep non_blocking to leverage pin_memory if available
        images = images.to(device, non_blocking=True)

        # Targets: list of tensors (N,5) or empty tensors
        # Ensure each element is a tensor on the correct device
        targets = [
            (t.to(device) if isinstance(t, torch.Tensor) and t.numel() > 0 else (torch.zeros((0, 5), device=device)))
            for t in targets
        ]
        # DEBUG: accumulate GT classes for the epoch
        for t in targets:
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                epoch_classes.update(
                    t[:, 0].detach().cpu().numpy().tolist()
                )

        # ------------------------------
        # Prepare scale-aware targets (pixel-area based)
        # ------------------------------
        targets_p3, targets_p4 = [], []
        # guard: images may be (B,C,H,W)
        if images.dim() < 3:
            H = 512
            W = 512
        else:
            H, W = images.shape[-2], images.shape[-1]

        if epoch < warmup_epochs:
            area_threshold = 0.035 * H * W   # more objects to P3 early
        else:
            area_threshold = 0.02 * H * W

        for t in targets:
            if not isinstance(t, torch.Tensor) or t.numel() == 0:
                targets_p3.append(torch.zeros((0, 5), device=device))
                targets_p4.append(torch.zeros((0, 5), device=device))
                continue
            try:
                widths = t[:, 3] * W
                heights = t[:, 4] * H
                areas = (widths * heights)
                small_mask = areas < area_threshold
                targets_p3.append(t[small_mask])
                targets_p4.append(t[~small_mask])
            except Exception:
                # defensive fallback
                targets_p3.append(torch.zeros((0, 5), device=device))
                targets_p4.append(torch.zeros((0, 5), device=device))

        if epoch < warmup_epochs:
            for idx in range(len(targets)):
                if targets_p3[idx].numel() == 0 and targets_p4[idx].numel() > 0:
                    targets_p3[idx] = targets_p4[idx]
                elif targets_p4[idx].numel() == 0 and targets_p3[idx].numel() > 0:
                    targets_p4[idx] = targets_p3[idx]
        if epoch < 5:
            p3_cnt = sum(t.shape[0] for t in targets_p3)
            p4_cnt = sum(t.shape[0] for t in targets_p4)

            if i == 0:  # print once per epoch
                print(f"[DEBUG] Epoch {epoch}: P3 targets={p3_cnt}, P4 targets={p4_cnt}")
                

        # ------------------------------
        # Forward (AMP)
        # ------------------------------
        gt_counts = Counter()
        for t in targets_p3 + targets_p4:
            if t is not None and t.numel() > 0:
                for row in t.detach().cpu().numpy():
                    gt_counts[int(row[0])] += 1
        with autocast(enabled=(device.type == "cuda")):
            pred = model(images)
            # model expected to return ((p3_obj, p3_cls, p3_reg), (p4_obj, p4_cls, p4_reg))
            loss_dict = criterion(pred[0], pred[1], targets_p3, targets_p4)
            

            raw_loss = loss_dict.get("total", torch.tensor(0.0, device=device))
            # scale for accumulation
            loss = raw_loss / max(1, accum_steps)

        # ------------------------------
        # Backward (with GradScaler)
        # ------------------------------
        scaler.scale(loss).backward()

        # ------------------------------
        # Optimizer / EMA / Scheduler (on accumulation boundary)
        # ------------------------------
        step_now = ((i + 1) % accum_steps == 0) or ((i + 1) == len(loader))
        if step_now:
            # gradient clipping (unscaled grads)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            try:
                scaler.step(optimizer)
                scaler.update()
            except Exception as e:
                print(f"⚠️ Optimizer step failed at batch {i}: {e}")
                scaler.update()  # still update scaler to avoid deadlock

            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                try:
                    ema.update(model)
                except Exception as e:
                    print(f"⚠️ EMA update failed: {e}")

            # Scheduler step only after warmup epochs
            if epoch >= warmup_epochs:
                try:
                    # works for step-based schedulers; if you use epoch-based schedulers, handle elsewhere
                    scheduler.step()
                except Exception:
                    # some schedulers expect different calling; ignore if incompatible
                    pass

        # ------------------------------
        # Warmup (LR override) - per-step warmup using initial_lr
        # ------------------------------
        if epoch < warmup_epochs:
            # safe guard: len(loader) might be 0
            denom = max(1, warmup_epochs * max(1, len(loader)))
            warmup_factor = (epoch * max(1, len(loader)) + i + 1) / denom
            warmup_factor = min(1.0, max(0.0, warmup_factor))
            for g in optimizer.param_groups:
                # ensure initial_lr exists
                base_lr = g.get("initial_lr", g.get("lr", 1e-3))
                g["lr"] = base_lr * warmup_factor

        # ------------------------------
        # Loss accounting (UNSCALED values)
        # ------------------------------
        total_loss += _to_float(raw_loss)
        bbox_loss_total += _to_float(loss_dict.get("bbox", 0))
        obj_loss_total += _to_float(loss_dict.get("obj", 0))
        cls_loss_total += _to_float(loss_dict.get("cls", 0))

        # ------------------------------
        # Progress bar
        # ------------------------------
        try:
            cur_lr = optimizer.param_groups[0].get("lr", 0.0)
            pbar.set_postfix({
                "Loss": f"{raw_loss.item():.3f}" if isinstance(raw_loss, torch.Tensor) else f"{raw_loss:.3f}",
                "LR": f"{cur_lr:.1e}"
            })
        except Exception:
            pass

        # ------------------------------
        # TensorBoard (batch-level)
        # ------------------------------
        if writer and (i % 10 == 0):
            global_step = epoch * max(1, len(loader)) + i
            try:
                writer.add_scalar("LR/train", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("Loss/train_batch", _to_float(raw_loss), global_step)
            except Exception:
                pass

    # ------------------------------
    # Epoch averages (safe division)
    # ------------------------------
    n_batches = max(len(loader), 1)
    avg_total = total_loss / n_batches
    avg_bbox = bbox_loss_total / n_batches
    avg_obj = obj_loss_total / n_batches
    avg_cls = cls_loss_total / n_batches

    # ------------------------------
    # TensorBoard (epoch-level)
    # ------------------------------
    if writer:
        try:
            writer.add_scalar("Loss/train", avg_total, epoch)
            writer.add_scalar("Loss/bbox", avg_bbox, epoch)
            writer.add_scalar("Loss/obj", avg_obj, epoch)
            writer.add_scalar("Loss/cls", avg_cls, epoch)
            writer.add_scalar("LR/epoch", optimizer.param_groups[0].get("lr", 0.0), epoch)
        except Exception:
            pass

    epoch_time = time.time() - start_time
    batches = max(len(loader), 1)
    print(f"  ⏱️  Epoch time: {epoch_time:.1f}s ({epoch_time / batches:.2f}s/batch)")
    print(f"[DEBUG] Epoch {epoch} GT classes seen:", sorted(epoch_classes))
    # Return order MUST match main()
    return avg_total, avg_bbox, avg_cls, avg_obj



@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    run_dir,
    epoch,
    writer=None,
    ema=None,
    calculate_metrics=False,
    plot=False,
):
    """
    Validate model on `loader`.

    Returns:
        avg_total, avg_bbox, avg_cls, avg_obj, metrics_dict, plot_data
    """
    # ----------------------------
    # Select eval model (EMA preferred)
    # ----------------------------
    eval_model = ema.ema if ema is not None else model
    if device is not None:
        eval_model = eval_model.to(device)
    eval_model.eval()

    # ----------------------------
    # Accumulators
    # ----------------------------
    total_loss = 0.0
    bbox_loss_total = 0.0
    obj_loss_total = 0.0
    cls_loss_total = 0.0

    all_targets = []      # per-image: (M,5) YOLO norm
    all_preds = []        # per-image: (N,6) [cls,conf,x1,y1,x2,y2] pixel coords
    all_box_centers = []  # (cx_norm, cy_norm)
    global_val_pred_counter = Counter()  # 🔍 NEW: Track predicted classes across all batches
    start_time = time.time()
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]", leave=False)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _to_float(v):
        if isinstance(v, torch.Tensor):
            try:
                return float(v.detach().cpu().item())
            except Exception:
                return 0.0
        try:
            return float(v)
        except Exception:
            return 0.0

    def _to_decoder_format(scale_pred):
        """
        Convert model output to decoder-compatible format.
        Accepts:
            (obj, cls, reg) -> (cls_combined, reg)
            (cls_map, reg_map) -> passthrough
        """
        if isinstance(scale_pred, (tuple, list)) and len(scale_pred) == 3:
            obj, cls, reg = scale_pred
            cls_comb = torch.cat([obj, cls], dim=1)
            return (cls_comb, reg)

        if isinstance(scale_pred, (tuple, list)) and len(scale_pred) == 2:
            return scale_pred

        raise ValueError("Unexpected scale_pred format for decoder")

    # ----------------------------
    # Validation loop
    # ----------------------------
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)

        H, W = int(images.shape[-2]), int(images.shape[-1])
        area_threshold = 0.02 * (H * W)

        targets_p3, targets_p4 = [], []
        batch_targets_per_image = []

        for t in targets:
            if t is None or (isinstance(t, torch.Tensor) and t.numel() == 0):
                targets_p3.append(torch.zeros((0, 5), device=device))
                targets_p4.append(torch.zeros((0, 5), device=device))
                batch_targets_per_image.append(np.zeros((0, 5), dtype=np.float32))
                continue

            t = t.to(device)
            areas = (t[:, 3] * W) * (t[:, 4] * H)
            small_mask = areas < area_threshold

            targets_p3.append(t[small_mask])
            targets_p4.append(t[~small_mask])

            gt_np = t.detach().cpu().numpy().astype(np.float32)
            batch_targets_per_image.append(
                gt_np if gt_np.shape[0] > 0 else np.zeros((0, 5), dtype=np.float32)
            )

            try:
                all_box_centers.extend(t[:, 1:3].detach().cpu().numpy().tolist())
            except Exception:
                pass

        # ----------------------------
        # Forward + loss
        # ----------------------------
        pred = eval_model(images)
        loss_dict = criterion(pred[0], pred[1], targets_p3, targets_p4)

        total_val = _to_float(loss_dict.get("total", 0))
        bbox_val  = _to_float(loss_dict.get("bbox",  0))
        obj_val   = _to_float(loss_dict.get("obj",   0))
        cls_val   = _to_float(loss_dict.get("cls",   0))

        total_loss += total_val
        bbox_loss_total += bbox_val
        obj_loss_total += obj_val
        cls_loss_total += cls_val

        pbar.set_postfix({"Loss": f"{total_val:.3f}"})

        # ----------------------------
        # Decode for metrics
        # ----------------------------
        if calculate_metrics:
            try:
                p3_dec = _to_decoder_format(pred[0])
                p4_dec = _to_decoder_format(pred[1])
                decoded = decode_predictions(
                    p3_dec,
                    p4_dec,
                    conf_thresh = 0.005 if epoch < 10 else 0.02,
                    nms_thresh=0.45
                )
                # 🔍 Per-batch predicted class frequency
                batch_pred_counter = Counter()
                for p in decoded:  # list of np.arrays per image
                    if p is None or len(p) == 0:
                        continue
                    batch_pred_counter.update([int(x[0]) for x in p])  # x[0] = class ID
                print(f"[VAL BATCH {batch_idx}] Predicted class counts:", batch_pred_counter)

                # 🔁 Accumulate over entire val set
                global_val_pred_counter.update(batch_pred_counter)

            except Exception as e:
                print(f"⚠️ decode_predictions failed at batch {batch_idx}: {e}")
                decoded = [np.zeros((0, 6), dtype=np.float32) for _ in range(images.shape[0])]

            # Align decoded length
            if len(decoded) != len(batch_targets_per_image):
                if len(decoded) > len(batch_targets_per_image):
                    decoded = decoded[:len(batch_targets_per_image)]
                else:
                    decoded += [np.zeros((0, 6), dtype=np.float32)] * (
                        len(batch_targets_per_image) - len(decoded)
                    )

            for pb, gt in zip(decoded, batch_targets_per_image):
                if isinstance(pb, torch.Tensor):
                    pb_arr = pb.detach().cpu().numpy().astype(np.float32) if pb.numel() else np.zeros((0, 6), dtype=np.float32)
                elif isinstance(pb, np.ndarray):
                    pb_arr = pb.astype(np.float32) if pb.size else np.zeros((0, 6), dtype=np.float32)
                else:
                    pb_arr = np.zeros((0, 6), dtype=np.float32)

                if pb_arr.ndim == 1 and pb_arr.size > 0:
                    pb_arr = pb_arr.reshape(1, -1)

                all_preds.append(pb_arr)
                all_targets.append(gt)

    # ----------------------------
    # Epoch averages
    # ----------------------------
    n_batches = max(len(loader), 1)
    avg_total = total_loss / n_batches
    avg_bbox  = bbox_loss_total / n_batches
    avg_obj   = obj_loss_total / n_batches
    avg_cls   = cls_loss_total / n_batches

    # ----------------------------
    # Metrics
    # ----------------------------
    metrics = {}
    if calculate_metrics and len(all_targets) > 0 and len(all_preds) > 0:
        try:
            map_50_95, map_50, map_75, per_class_ap = calculate_map(
                predictions=all_preds,
                targets=all_targets,
                num_classes=NUM_CLASSES,
                iou_thresholds=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
                epoch=epoch
            )
            metrics = {
                "mAP_50": map_50,
                "mAP_75": map_75,
                "mAP_50_95": map_50_95,
                "per_class_ap": per_class_ap,
            }
            if writer:
                writer.add_scalar("Metrics/mAP_50", map_50, epoch)
                writer.add_scalar("Metrics/mAP_75", map_75, epoch)
                writer.add_scalar("Metrics/mAP_50_95", map_50_95, epoch)
        except Exception as e:
            print(f"⚠️ calculate_map failed: {e}")

    # ----------------------------
    # TensorBoard losses
    # ----------------------------
    if writer:
        writer.add_scalar("Loss/val", avg_total, epoch)
        writer.add_scalar("Loss/val_bbox", avg_bbox, epoch)
        writer.add_scalar("Loss/val_obj", avg_obj, epoch)
        writer.add_scalar("Loss/val_cls", avg_cls, epoch)

    plot_data = {
        "all_preds": all_preds,
        "all_targets": all_targets,
        "all_box_centers": all_box_centers,
    }

    if plot:
        try:
            if len(plot_data["all_preds"]) == 0:
                plot_data["all_preds"] = [
                    np.zeros((0, 6), dtype=np.float32)
                    for _ in range(len(plot_data["all_targets"]))
                ]
            save_plots_from_validation(plot_data, run_dir, epoch)
        except Exception as e:
            print(f"⚠️ Plotting error in validate(): {e}")

    elapsed = time.time() - start_time
    print(f"  ⏱️ Validation time: {elapsed:.1f}s")
    print("=== VAL EPOCH predicted class totals ===")
    print(global_val_pred_counter)
    # after printing global_val_pred_counter
    total_preds = sum(global_val_pred_counter.values())
    if total_preds > 0:
        most_cls, most_cnt = global_val_pred_counter.most_common(1)[0]
        if most_cnt / total_preds > 0.90:
            print(f"⚠️ WARNING: predictions collapsed — class {most_cls} accounts for {most_cnt}/{total_preds} ({most_cnt/total_preds:.2f}) of predictions.")
    log_debug = Path(run_dir) / "logs" / "debug_stats.txt"
    log_debug.parent.mkdir(parents=True, exist_ok=True)
    with open(log_debug, "a") as f:
        f.write(f"epoch {epoch} predicted_counts: {dict(global_val_pred_counter)}\n")


    return avg_total, avg_bbox, avg_cls, avg_obj, metrics, plot_data







# =================== SAVE LOSS PLOT ===================
def save_loss_plot(train_losses, val_losses, run_dir, title="Training and Validation Loss"):#cheked and changed
    """
    Save loss plot with adaptive smoothing for small datasets.
    Optimized for microcontroller training (few epochs, early stopping).

    Copy-paste ready.
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

    # Convert to numpy arrays (float) for safety
    train_losses = np.array(train_losses, dtype=float)
    val_losses = np.array(val_losses, dtype=float)

    # ---- Replace NaN / inf with large finite numbers to avoid plotting crashes ----
    # Use a large but finite sentinel instead of inf so plotting/np.percentile works.
    sentinel = 1e6
    train_losses = np.nan_to_num(train_losses, nan=sentinel, posinf=sentinel, neginf=-sentinel)
    val_losses = np.nan_to_num(val_losses, nan=sentinel, posinf=sentinel, neginf=-sentinel)

    # ---- Optional: clip extreme outliers to 99th percentile to keep y-axis readable ----
    try:
        finite_mask = np.isfinite(val_losses)
        if finite_mask.any():
            clip_max = np.percentile(val_losses[finite_mask], 99)
            # ensure clip_max is not zero or negative (avoid degenerate clipping)
            clip_max = max(clip_max, 1e-6)
            val_losses_clipped = np.clip(val_losses, a_min=None, a_max=clip_max)
        else:
            val_losses_clipped = val_losses.copy()
    except Exception:
        val_losses_clipped = val_losses.copy()

    plt.figure(figsize=(12, 8))

    # ---- ADAPTIVE SMOOTHING for small datasets ----
    # Small datasets (< 20 epochs): minimal smoothing
    # Large datasets (> 100 epochs): aggressive smoothing
    num_epochs = len(train_losses)

    if num_epochs < 10:
        window = 1
    elif num_epochs < 50:
        window = max(3, num_epochs // 10)
    else:
        window = max(5, num_epochs // 20)

    # Apply smoothing (explicit kernel and indices)
    N = len(train_losses)
    if window > 1 and N > window:
        kernel = np.ones(window) / window
        train_smooth = np.convolve(train_losses, kernel, mode='valid')
        val_smooth = np.convolve(val_losses_clipped, kernel, mode='valid')
        # explicit indices for smoothed arrays (valid mode -> length = N - window + 1)
        x_train = np.arange(window - 1, window - 1 + train_smooth.shape[0])
        x_val = np.arange(window - 1, window - 1 + val_smooth.shape[0])
    else:
        train_smooth = train_losses
        val_smooth = val_losses_clipped
        x_train = np.arange(N)
        x_val = np.arange(N)

    # ---- Plot smoothed lines ----
    plt.plot(x_train, train_smooth, label='Train Loss', linewidth=2, alpha=0.9)
    plt.plot(x_val, val_smooth, label='Val Loss', linewidth=2, alpha=0.9)

    # ---- Plot original points (only if not too many) ----
    if num_epochs <= 100:
        plt.scatter(range(len(train_losses)), train_losses, label='_nolegend_', alpha=0.3, s=10)
        plt.scatter(range(len(val_losses_clipped)), val_losses_clipped, label='_nolegend_', alpha=0.3, s=10)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # ---- Add best validation loss annotation (1-based epoch) ----
    best_loss = None
    try:
        if len(val_losses) > 0:
            # use the unclipped array when reporting exact best (but guard if sentinel present)
            finite_vals_mask = np.isfinite(val_losses) & (val_losses < sentinel)
            if finite_vals_mask.any():
                best_epoch = int(np.argmin(val_losses))
                best_loss = float(val_losses[best_epoch])
                # Vertical line at best epoch (0-based x -> annotate with 1-based)
                plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)

                # Adaptive text placement using clipped values for visual stability
                y_range = np.ptp(val_losses_clipped) if np.ptp(val_losses_clipped) > 0 else 1.0
                y_min = np.min(val_losses_clipped)
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

                plt.text(
                    text_x, text_y,
                    f'Best: {best_loss:.4f} @ epoch {best_epoch + 1}',
                    fontsize=10, color='green', ha=ha,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', alpha=0.8)
                )
            else:
                # if no finite val values, skip annotation
                best_loss = None
    except Exception:
        best_loss = None

    plt.tight_layout()

    # ---- Save plot ----
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    plot_path = os.path.join(run_dir, "plots", "loss_curve.png")
    try:
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"⚠️ Could not save loss plot: {e}")
    finally:
        plt.close()

    if best_loss is not None:
        print(f"📈 Loss plot saved → {plot_path} | Best val: {best_loss:.4f}")
    else:
        print(f"📈 Loss plot saved → {plot_path}")


# =================== WARMUP LR PLOT ===================
def plot_warmup_lr(optimizer, total_steps, warmup_steps, run_dir):
    """Plot learning rate during warmup. Copy-paste ready and robust to warmup_steps==0."""
    # safe guards
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    # read base lr safely
    base_lr = optimizer.param_groups[0].get('initial_lr', optimizer.param_groups[0].get('lr', 1e-3))

    lrs = []
    steps = list(range(total_steps))

    for step in steps:
        if warmup_steps > 0 and step < warmup_steps:
            warmup_factor = (step + 1) / warmup_steps
            lr = base_lr * warmup_factor
        else:
            lr = base_lr
        lrs.append(lr)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs, '-', linewidth=2)
    if warmup_steps > 0:
        plt.axvline(x=warmup_steps, color='r', linestyle='--', label=f'Warmup end: step {warmup_steps}')
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Warmup Schedule')
    if warmup_steps > 0:
        plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    plot_path = os.path.join(run_dir, "plots", "warmup_lr.png")
    try:
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"⚠️ Could not save warmup LR plot: {e}")
    finally:
        plt.close()
    print(f"📈 Warmup LR plot saved to: {plot_path}")

# =================== Helper: Save plots from validation data ===================
def save_plots_from_validation(plot_data, run_dir, epoch):
    """
    Given plot_data from validate(), save confusion, f1/confidence, PR/precision curves, and heatmap.

    Defensive fixes:
      - Do NOT map missing GT+pred -> class 0 (silently corrupts class 0). Instead:
        * Skip pairs where both GT and pred are missing.
        * Map missing->'background' index only for confusion matrix (append an extra label).
      - Ensure all_preds length aligns with all_targets.
      - Skip plotting steps gracefully if inputs are empty.
      - Return boolean indicating whether any plots were created.
    Returns:
      created (bool) -- True if at least one plot was produced & saved, False otherwise.
    """
    created_any = False
    try:
        all_preds = plot_data.get("all_preds", [])
        all_targets = plot_data.get("all_targets", [])
        all_box_centers = plot_data.get("all_box_centers", [])

        # Ensure lists
        if all_preds is None:
            all_preds = []
        if all_targets is None:
            all_targets = []

        # Align lengths: if preds shorter/longer than targets, coerce to same length (defensive)
        if len(all_preds) != len(all_targets):
            if len(all_preds) < len(all_targets):
                # pad preds with empty arrays
                all_preds = list(all_preds) + [np.zeros((0, 6), dtype=np.float32) for _ in range(len(all_targets) - len(all_preds))]
            else:
                # pad targets with empty arrays
                all_targets = list(all_targets) + [np.zeros((0, 5), dtype=np.float32) for _ in range(len(all_preds) - len(all_targets))]

        # fallback: if still zero-length overall, nothing to plot (except maybe heatmap)
        if len(all_preds) == 0 and len(all_box_centers) == 0:
            print("⚠️ No validation data for plotting (no preds, no box centers).")
            return False

        # ----------------------------
        # CONFUSION MATRIX PREPARATION
        # ----------------------------
        # Build y_true / y_pred but DO NOT map missing->class0.
        # Option chosen: create an explicit "background" class index = len(CLASSES)
        background_index = len(CLASSES)
        y_true_clean = []
        y_pred_clean = []

        for gt, pred in zip(all_targets, all_preds):
            gt_present = isinstance(gt, np.ndarray) and gt.shape[0] > 0
            pred_present = isinstance(pred, np.ndarray) and pred.shape[0] > 0

            # skip images where both are absent (background-only images, don't pollute confusion)
            if not gt_present and not pred_present:
                continue

            # top-1 GT / pred mapping (keep it simple & consistent with previous behavior)
            if gt_present:
                try:
                    gt_cls = int(gt[0, 0])
                except Exception:
                    # defensive fallback if gt dtype unexpected
                    gt_cls = background_index
            else:
                gt_cls = background_index  # missing GT -> background

            if pred_present:
                try:
                    pred_cls = int(pred[0, 0])
                except Exception:
                    pred_cls = background_index
            else:
                pred_cls = background_index  # missing pred -> background

            y_true_clean.append(gt_cls)
            y_pred_clean.append(pred_cls)

        plots_root = os.path.join(run_dir, "plots")
        os.makedirs(plots_root, exist_ok=True)

        # Only plot confusion matrix if we have at least one pair
        if len(y_true_clean) > 0:
            # Build labels: real classes + background
            conf_labels = list(CLASSES) + ["background"]
            try:
                plot_confusion_matrix(
                    y_true=y_true_clean,
                    y_pred=y_pred_clean,
                    labels=conf_labels,
                    run_dir=plots_root,
                    base_title=f"Confusion Matrix - Epoch {epoch}"
                )
                created_any = True
            except Exception as e:
                print(f"⚠️ Confusion matrix plotting failed: {e}")
        else:
            print("⚠️ Skipping confusion matrix: no paired gt/pred samples after filtering.")

        # ----------------------------
        # F1 / Confidence curve
        # ----------------------------
        try:
            # plot_f1_confidence_curve is robust to empty predictions/targets per your utils
            plot_f1_confidence_curve(
                predictions=all_preds,
                targets=all_targets,
                class_names=CLASSES,
                run_dir=plots_root
            )
            created_any = True
        except Exception as e:
            print(f"⚠️ plot_f1_confidence_curve failed: {e}")

        # ----------------------------
        # Label heatmap (if centers present)
        # ----------------------------
        if isinstance(all_box_centers, (list, tuple)) and len(all_box_centers) > 0:
            try:
                plot_label_heatmap(
                    box_centers=all_box_centers,
                    run_dir=plots_root
                )
                created_any = True
            except Exception as e:
                print(f"⚠️ plot_label_heatmap failed: {e}")
        else:
            # no centers — skip silently
            pass

        # ----------------------------
        # Precision–Recall curves (class-wise)
        # ----------------------------
        # compute_precision_recall_curves expects predictions in pixel coords (N,6) and targets in YOLO normalized (M,5).
        # Defensive: ensure arrays are numpy and shapes are what we expect. If everything empty, skip.
        has_any_preds = any(isinstance(p, np.ndarray) and p.shape[0] > 0 for p in all_preds)
        has_any_targets = any(isinstance(t, np.ndarray) and t.shape[0] > 0 for t in all_targets)

        if has_any_preds or has_any_targets:
            try:
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
                created_any = True
            except Exception as e:
                print(f"⚠️ PR curve computation or saving failed: {e}")
        else:
            print("⚠️ No predictions/targets for PR curves - skipping PR plotting.")

        if created_any:
            print(f"📊 Validation plots saved to: {plots_root} (epoch {epoch})")
        else:
            print("⚠️ No plots were created from validation data.")

        return created_any

    except Exception as e:
        print(f"⚠️ Error saving validation plots: {e}")
        return False



# =================== MAIN ===================
def main():#checked
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
        # 1) compute class instance counts from train labels (do once)#changed
        label_dir = Path(TRAIN_LABEL_DIR)  # e.g. "data/dataset_train/labels/train"
        label_paths = sorted(list(label_dir.glob("*.txt")))  # keep deterministic order
        debug_label_stats(label_paths, CLASSES if CLASSES else [str(i) for i in range(NUM_CLASSES)], run_dir)
        # count instances per class
        from collections import Counter
        inst_counts = Counter()
        for p in label_paths:
            for line in p.read_text().splitlines():
                if not line.strip(): continue
                c = int(float(line.split()[0]))
                inst_counts[c] += 1

        # turn into list aligned with class ids 0..C-1
        num_classes = NUM_CLASSES
        class_counts = [inst_counts.get(i, 0) for i in range(num_classes)]

        # 2) class weights (inverse frequency)
        total = float(sum(class_counts))
        # avoid division by zero
        class_weights = [ (total / (c if c>0 else 1.0)) for c in class_counts ]

        # 3) per-image weight (average weight of classes present in image)
        # 👇 Build label paths from dataset order to match sample-to-weight
        if hasattr(train_dataset, "image_paths"):
            dataset_label_paths = [Path(p).with_suffix(".txt") for p in train_dataset.image_paths]
        elif hasattr(train_dataset, "samples"):
            dataset_label_paths = [Path(p[0]).with_suffix(".txt") for p in train_dataset.samples]
        else:
            dataset_label_paths = sorted(label_dir.glob("*.txt"))  # fallback

        # 👇 Now compute image weights aligned to dataset sample order
        image_weights = []

        for p in dataset_label_paths:
            cls_set = set()
            if p.exists():
                for line in p.read_text().splitlines():
                    if not line.strip():
                        continue
                    cls_set.add(int(float(line.split()[0])))

            if len(cls_set) == 0:
                image_weights.append(min(class_weights) * 0.05)
            else:
                image_weights.append(
                    float(np.mean([class_weights[c] for c in cls_set]))
                )
            
        image_weights = np.array(image_weights, dtype=np.float32)
        image_weights = image_weights / image_weights.mean()
        image_weights = np.clip(image_weights, 0.1, 10.0)

        # 4) create sampler & DataLoader (replace your old train_loader)
        sampler = WeightedRandomSampler(
                        weights=image_weights.tolist(),
                        num_samples=len(image_weights),
                        replacement=True
                    )

        # === Final DataLoader ===
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,         # <--- sampler instead of shuffle
            num_workers=args.workers,
            collate_fn=detection_collate_fn,
            pin_memory=False,        # True if using CUDA
            drop_last=True,
            persistent_workers=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=detection_collate_fn,
            pin_memory=False,
            persistent_workers=False if args.workers > 0 else False
        )
        print(f"DEBUG class_counts: {class_counts}")
        with open(run_dir / "logs" / "dataset_counts.txt", "w") as f:
            f.write(f"class_counts: {class_counts}\n")
            f.write(f"image_weights min/max: {min(image_weights):.6f}/{max(image_weights):.6f}\n")
                
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
        # Split classifier vs. others
        classifier_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"⚠️ WARNING: {name} does not require grad!")
            if 'cls_conv' in name:  # or more specific: "cls_conv"
                classifier_params.append(param)
            else:
                other_params.append(param)
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
    
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": args.lr, "weight_decay": 1e-4},
        {"params": classifier_params, "lr": args.lr * 1.5, "weight_decay": 1e-5}
    ], betas=(0.9, 0.999))
    
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
    
    # helper: robust epoch-level PR computation with fallback
    def _safe_epoch_pr(preds_list, targets_list, conf_thresh=0.01, iou_thresh=0.5):
        """Return (precision, recall) as floats. Uses compute_epoch_precision_recall if available,
           otherwise falls back to compute_precision_recall on whole lists."""
        # quick non-empty checks
        has_preds = any(isinstance(a, np.ndarray) and a.size > 0 for a in preds_list)
        has_gts  = any(isinstance(t, np.ndarray) and t.size > 0 for t in targets_list)
        if not (has_preds and has_gts):
            return 0.0, 0.0

        try:
            # prefer compute_precision_recall (if defined in your utils)
            return compute_precision_recall(preds_list, targets_list, conf_thresh=conf_thresh)
        except Exception:
            # fallback to compute_precision_recall (your existing utility)
            try:
                return compute_precision_recall(preds_list, targets_list, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
            except Exception as e:
                print(f"⚠️ PR computation fallback failed: {e}")
                return 0.0, 0.0

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

        # Robustly compute epoch precision/recall using helper
        p, r = _safe_epoch_pr(plot_data.get("all_preds", []), plot_data.get("all_targets", []), conf_thresh=0.01)

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
                print(f"  💾 BEST mAP! Model saved (mAP@0.5: {current_metric:.4f})")

                # save validation plots once when best model updates (plots function internally handles empty checks)
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
                print(f"  💾 BEST! Model saved (val_loss: {val_loss:.4f})")

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
            print(f"  💾 Checkpoint (latest) saved at epoch {epoch+1}")
            
        # Save CSV EVERY epoch
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
        # replace model loading / eval selection to correctly prefer EMA if used
        if best_checkpoint_path.exists():
            best_ckpt = torch.load(best_checkpoint_path, map_location=device)
            # Restore model weights (prefer model_state_dict for model, ema stored separately)
            if best_ckpt.get("model_state_dict", None) is not None:
                try:
                    model.load_state_dict(best_ckpt["model_state_dict"])
                    print("🔁 Loaded model_state_dict for final evaluation.")
                except Exception as e:
                    print(f"⚠️ Could not load model_state_dict into model: {e}")

            # If we used EMA during training and ema state exists, restore ema.ema
            if args.use_ema and best_ckpt.get("ema_state_dict", None) is not None:
                try:
                    if ema is None:
                        ema = ModelEMA(model, decay=args.ema_decay)  # create local ema wrapper
                    ema.ema.load_state_dict(best_ckpt["ema_state_dict"])
                    print("🔁 Restored EMA weights for final evaluation.")
                except Exception as e:
                    print(f"⚠️ Could not restore EMA weights: {e}")
        else:
            print("⚠️ Best checkpoint not found; using current model for final evaluation.")
        # Use EMA if exists, otherwise model; also fuse if requested
        eval_model = get_eval_model(model, ema=(ema if args.use_ema else None), device=device, fuse=True)

        # Validate on val set with plotting and save plots to test folder (force calculate_metrics=True for full PR/F1)
        val_loss_f, vb, vc, vd, metrics_f, plot_data_f = validate(
            eval_model,                      # eval model already moved to device and fused
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
        
        # ✅ Compute final precision/recall robustly
        p_test, r_test = _safe_epoch_pr(plot_data_f.get("all_preds", []), plot_data_f.get("all_targets", []), conf_thresh=0.01)
        
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
