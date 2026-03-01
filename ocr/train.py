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


# =================== PATHS ===================
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
NUM_CLASSES = len(CLASSES) if CLASSES else 14
print("DEBUG[train] NUM_CLASSES:", NUM_CLASSES)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

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
    def __init__(self, patience=30, min_delta=1e-4):
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

# =================== LR FINDER ===================
def find_optimal_lr(model, train_loader, criterion, device, run_dir,
                    start_lr=1e-7, end_lr=0.1, num_iter=100):
    print("\n🔍 Finding optimal learning rate...")

    dataset_size = len(train_loader.dataset)
    batches = len(train_loader)
    num_iter = min(num_iter, batches * 2)

    if dataset_size < 500:
        start_lr, end_lr = 1e-6, 5e-3
        print("   Small dataset → conservative LR range")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    lrs, losses = [], []
    avg_loss, best_loss, beta = 0.0, float("inf"), 0.98
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
        targets_p3, targets_p4 = _split_targets_by_area(
            targets, images.shape[-2], images.shape[-1], device, warmup=True
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == "cuda")):
            pred = model(images)
            loss_dict = criterion(pred[0], pred[1], targets_p3, targets_p4)
            loss = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        
        # ✅ SAFETY: Stop if loss is NaN/Inf
        if not np.isfinite(loss_val):
            print("⚠️ LR finder stopped: loss is NaN/Inf")
            break
        
        avg_loss = beta * avg_loss + (1 - beta) * loss_val
        smoothed_loss = avg_loss / (1 - beta ** (i + 1))

        if i > 10 and smoothed_loss > 4 * best_loss:
            print("⚠️ LR finder stopped early (loss exploded)")
            break
        best_loss = min(best_loss, smoothed_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        losses.append(smoothed_loss)

        for pg in optimizer.param_groups:
            pg["lr"] *= lr_mult

        pbar.set_postfix(LR=f"{current_lr:.2e}", Loss=f"{smoothed_loss:.4f}")

    if len(lrs) < 10:
        print("⚠️ Not enough points, using default LR")
        return None

    window = max(1, len(losses) // 10)
    if window > 1:
        kernel = np.ones(window) / window
        losses_smooth = np.convolve(losses, kernel, mode="valid")
        lrs_smooth = lrs[window - 1:]
    else:
        losses_smooth, lrs_smooth = losses, lrs

    gradients = np.gradient(losses_smooth)
    best_idx = np.argmin(gradients)
    optimal_lr = lrs_smooth[best_idx]

    print(f"✅ Optimal LR: {optimal_lr:.2e}")

    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(lrs_smooth, losses_smooth, linewidth=2)
    plt.axvline(optimal_lr, color="red", linestyle="--", label=f"Optimal: {optimal_lr:.2e}")
    plt.xscale("log")
    plt.xlabel("Learning Rate"); plt.ylabel("Loss"); plt.title("LR Finder")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(run_dir, "plots", "lr_finder.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return optimal_lr


# =================== TARGET SPLITTING HELPER ===================
def _split_targets_by_area(targets, H, W, device, warmup=False, area_frac=0.02):
    """
    Split targets into P3 (small) and P4 (large) by pixel area.
    During warmup, duplicate targets to both scales for stronger gradients.
    """
    area_threshold = area_frac * H * W
    if warmup:
        area_threshold = 0.035 * H * W  # push more to P3 during warmup

    targets_p3, targets_p4 = [], []
    for t in targets:
        if t is None or (isinstance(t, torch.Tensor) and t.numel() == 0):
            targets_p3.append(torch.zeros((0, 5), device=device))
            targets_p4.append(torch.zeros((0, 5), device=device))
            continue

        t = t.to(device)
        areas = (t[:, 3] * W) * (t[:, 4] * H)
        small_mask = areas < area_threshold

        p3_targets = t[small_mask]
        p4_targets = t[~small_mask]

        # During warmup: if one scale has no targets, duplicate from the other
        if warmup:
            if p3_targets.numel() == 0 and p4_targets.numel() > 0:
                p3_targets = p4_targets
            elif p4_targets.numel() == 0 and p3_targets.numel() > 0:
                p4_targets = p3_targets

        targets_p3.append(p3_targets)
        targets_p4.append(p4_targets)

    return targets_p3, targets_p4



# =================== TRAIN ONE EPOCH ===================
def train_one_epoch(
    model, loader, criterion, optimizer, scaler,
    device, accum_steps, scheduler, epoch, warmup_epochs,
    writer=None, ema=None,
):
    model.train()
    epoch_classes = set()
    total_loss = bbox_loss_total = obj_loss_total = cls_loss_total = 0.0

    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    def _f(v):
        if isinstance(v, torch.Tensor):
            try: return float(v.detach().cpu().item())
            except: return 0.0
        try: return float(v)
        except: return 0.0

    H, W = 512, 512  # default
    is_warmup = epoch < warmup_epochs
    pbar = tqdm(loader, desc=f"Epoch {epoch+1:03d} [Train]", leave=False)

    for i, batch in enumerate(pbar):
        if batch is None:
            continue
        try:
            images, targets = batch
        except Exception:
            continue

        images = images.to(device, non_blocking=True)
        if images.dim() >= 3:
            H, W = images.shape[-2], images.shape[-1]

        targets = [
            (t.to(device) if isinstance(t, torch.Tensor) and t.numel() > 0
             else torch.zeros((0, 5), device=device))
            for t in targets
        ]

        # Track GT classes (epoch-level only — no per-batch spam)
        for t in targets:
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                epoch_classes.update(t[:, 0].detach().cpu().numpy().tolist())

        # Split targets by area
        targets_p3, targets_p4 = _split_targets_by_area(
            targets, H, W, device, warmup=is_warmup
        )

        # Debug: print once per epoch at batch 0
        if epoch < 5 and i == 0:
            p3_cnt = sum(t.shape[0] for t in targets_p3)
            p4_cnt = sum(t.shape[0] for t in targets_p4)
            print(f"  [DEBUG] Epoch {epoch}: P3={p3_cnt}, P4={p4_cnt} targets")

        # Forward (AMP)
        with autocast(enabled=(device.type == "cuda")):
            pred = model(images)
            loss_dict = criterion(pred[0], pred[1], targets_p3, targets_p4)
            raw_loss = loss_dict.get("total", torch.tensor(0.0, device=device))
            loss = raw_loss / max(1, accum_steps)

        # ✅ NaN GUARD: skip batch if loss is NaN/Inf
        if not torch.isfinite(loss):
            print(f"  ⚠️ NaN/Inf loss at batch {i}, skipping (box={loss_dict.get('bbox','?')}, cls={loss_dict.get('cls','?')}, obj={loss_dict.get('obj','?')})")
            optimizer.zero_grad(set_to_none=True)
            continue

        # Backward
        scaler.scale(loss).backward()

        # Optimizer step on accumulation boundary
        step_now = ((i + 1) % accum_steps == 0) or ((i + 1) == len(loader))
        if step_now:
            scaler.unscale_(optimizer)
            # ✅ CHANGED: clip=10.0 (loss already internally clamped)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
            try:
                scaler.step(optimizer)
                scaler.update()
            except Exception as e:
                print(f"⚠️ Optimizer step failed at batch {i}: {e}")
                scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

            
        # Warmup LR override
        if is_warmup:
            denom = max(1, warmup_epochs * max(1, len(loader)))
            warmup_factor = min(1.0, (epoch * max(1, len(loader)) + i + 1) / denom)
            for g in optimizer.param_groups:
                base_lr = g.get("initial_lr", g.get("lr", 1e-3))
                g["lr"] = base_lr * warmup_factor

        # Accumulate
        total_loss += _f(raw_loss)
        bbox_loss_total += _f(loss_dict.get("bbox", 0))
        obj_loss_total += _f(loss_dict.get("obj", 0))
        cls_loss_total += _f(loss_dict.get("cls", 0))

        try:
            pbar.set_postfix({
                "Loss": f"{raw_loss.item():.3f}" if isinstance(raw_loss, torch.Tensor) else f"{raw_loss:.3f}",
                "LR": f"{optimizer.param_groups[0].get('lr', 0):.1e}"
            })
        except Exception:
            pass

        if writer and (i % 10 == 0):
            global_step = epoch * max(1, len(loader)) + i
            writer.add_scalar("LR/train", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("Loss/train_batch", _f(raw_loss), global_step)

    n = max(len(loader), 1)
    avg_total = total_loss / n
    avg_bbox = bbox_loss_total / n
    avg_obj = obj_loss_total / n
    avg_cls = cls_loss_total / n

    if writer:
        writer.add_scalar("Loss/train", avg_total, epoch)
        writer.add_scalar("Loss/bbox", avg_bbox, epoch)
        writer.add_scalar("Loss/obj", avg_obj, epoch)
        writer.add_scalar("Loss/cls", avg_cls, epoch)

    elapsed = time.time() - start_time
    print(f"  ⏱️ {elapsed:.1f}s | GT classes seen: {sorted(int(c) for c in epoch_classes)}")

    return avg_total, avg_bbox, avg_cls, avg_obj



# =================== VALIDATION ===================
@torch.no_grad()
def validate(
    model, loader, criterion, device, run_dir, epoch,
    writer=None, ema=None, calculate_metrics=False, plot=False,
):
    # 🔴 FIX #4: Verify class count matches model
    assert NUM_CLASSES == model.num_classes, \
        f"❌ Class mismatch: NUM_CLASSES={NUM_CLASSES}, model.num_classes={model.num_classes}"
    
    # ✅ EMA selection with logging
    use_ema = epoch >= 15 and ema is not None
    if epoch == 15 and ema is not None:
        print("🔁 Validation switched to EMA model")
    eval_model = ema.ema if use_ema else model
    if device is not None:
        eval_model = eval_model.to(device)
    eval_model.eval()

    total_loss = bbox_loss_total = obj_loss_total = cls_loss_total = 0.0
    all_targets, all_preds, all_box_centers = [], [], []
    global_val_pred_counter = Counter()
    start_time = time.time()

    # ✅ Adaptive confidence threshold
    if epoch < 10:
        val_conf_thresh = 0.001  # See everything early
    elif epoch < 30:
        val_conf_thresh = 0.01   # Medium confidence mid-training
    else:
        val_conf_thresh = 0.05    # High confidence late

    def _f(v):
        if isinstance(v, torch.Tensor):
            try: return float(v.detach().cpu().item())
            except: return 0.0
        try: return float(v)
        except: return 0.0

    def _to_decoder_format(scale_pred):
        if isinstance(scale_pred, (tuple, list)) and len(scale_pred) == 3:
            obj, cls, reg = scale_pred
            return (torch.cat([obj, cls], dim=1), reg)
        if isinstance(scale_pred, (tuple, list)) and len(scale_pred) == 2:
            return scale_pred
        raise ValueError("Unexpected pred format")

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]", leave=False)

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        H, W = int(images.shape[-2]), int(images.shape[-1])

        # 🔴 FIX #1: Explicitly move targets to device
        targets = [
            t.to(device) if isinstance(t, torch.Tensor) and t.numel() > 0 
            else torch.zeros((0, 5), device=device)
            for t in targets
        ]

        # Split targets by area (now safely on device)
        targets_p3, targets_p4 = _split_targets_by_area(
            targets, H, W, device, warmup=False
        )

        batch_targets_per_image = []
        for t in targets:
            if t.numel() == 0:
                batch_targets_per_image.append(np.zeros((0, 5), dtype=np.float32))
                continue
            
            gt_np = t.detach().cpu().numpy().astype(np.float32)
            batch_targets_per_image.append(gt_np)
            
            # 🔴 FIX #3: Only collect box centers when plotting
            if plot:
                try:
                    all_box_centers.extend(t[:, 1:3].cpu().numpy().tolist())
                except Exception:
                    pass

        # Forward + loss
        pred = eval_model(images)
        loss_dict = criterion(pred[0], pred[1], targets_p3, targets_p4)

        total_loss += _f(loss_dict.get("total", 0))
        bbox_loss_total += _f(loss_dict.get("bbox", 0))
        obj_loss_total += _f(loss_dict.get("obj", 0))
        cls_loss_total += _f(loss_dict.get("cls", 0))

        pbar.set_postfix({"Loss": f"{_f(loss_dict.get('total', 0)):.3f}"})

        # Decode for metrics
        if calculate_metrics:
            try:
                p3_dec = _to_decoder_format(pred[0])
                p4_dec = _to_decoder_format(pred[1])
                decoded = decode_predictions(
                    p3_dec, p4_dec,
                    conf_thresh=val_conf_thresh,
                    nms_thresh=0.40
                )
                for p in decoded:
                    if p is not None and len(p) > 0:
                        global_val_pred_counter.update([int(x[0]) for x in p])
            except Exception as e:
                print(f"⚠️ decode failed batch {batch_idx}: {e}")
                decoded = [np.zeros((0, 6), dtype=np.float32) for _ in range(images.shape[0])]

            # Align lengths
            while len(decoded) < len(batch_targets_per_image):
                decoded.append(np.zeros((0, 6), dtype=np.float32))
            decoded = decoded[:len(batch_targets_per_image)]

            for pb, gt in zip(decoded, batch_targets_per_image):
                if isinstance(pb, torch.Tensor):
                    pb = pb.detach().cpu().numpy().astype(np.float32) if pb.numel() else np.zeros((0, 6), dtype=np.float32)
                elif isinstance(pb, np.ndarray):
                    pb = pb.astype(np.float32) if pb.size else np.zeros((0, 6), dtype=np.float32)
                else:
                    pb = np.zeros((0, 6), dtype=np.float32)
                if pb.ndim == 1 and pb.size > 0:
                    pb = pb.reshape(1, -1)
                all_preds.append(pb)
                all_targets.append(gt)

    n = max(len(loader), 1)
    avg_total = total_loss / n
    avg_bbox = bbox_loss_total / n
    avg_obj = obj_loss_total / n
    avg_cls = cls_loss_total / n

    # Metrics
    metrics = {}
    if calculate_metrics and len(all_targets) > 0 and len(all_preds) > 0:
        try:
            map_50_95, map_50, map_75, per_class_ap = calculate_map(
                predictions=all_preds, targets=all_targets,
                num_classes=NUM_CLASSES,
                iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                epoch=epoch
            )
            metrics = {"mAP_50": map_50, "mAP_75": map_75, "mAP_50_95": map_50_95, "per_class_ap": per_class_ap}
            if writer:
                writer.add_scalar("Metrics/mAP_50", map_50, epoch)
                writer.add_scalar("Metrics/mAP_75", map_75, epoch)
                writer.add_scalar("Metrics/mAP_50_95", map_50_95, epoch)
        except Exception as e:
            print(f"⚠️ calculate_map failed: {e}")

    if writer:
        writer.add_scalar("Loss/val", avg_total, epoch)
        writer.add_scalar("Loss/val_bbox", avg_bbox, epoch)
        writer.add_scalar("Loss/val_obj", avg_obj, epoch)
        writer.add_scalar("Loss/val_cls", avg_cls, epoch)

    plot_data = {"all_preds": all_preds, "all_targets": all_targets, "all_box_centers": all_box_centers}

    if plot:
        try:
            save_plots_from_validation(plot_data, run_dir, epoch)
        except Exception as e:
            print(f"⚠️ Plotting error: {e}")

    elapsed = time.time() - start_time
    print(f"  ⏱️ Val: {elapsed:.1f}s | conf_thresh={val_conf_thresh}")

    # Collapse warning
    total_preds = sum(global_val_pred_counter.values())
    if total_preds > 0:
        print(f"  📊 Val predictions: {dict(global_val_pred_counter)}")
        most_cls, most_cnt = global_val_pred_counter.most_common(1)[0]
        if most_cnt / total_preds > 0.85:
            print(f"  ⚠️ WARNING: class {most_cls} = {most_cnt}/{total_preds} ({most_cnt/total_preds:.0%}) — possible collapse")
    else:
        print(f"  📊 Val predictions: NONE (model not producing detections yet)")

    # Log
    log_path = Path(run_dir) / "logs" / "debug_stats.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"epoch {epoch} pred_counts: {dict(global_val_pred_counter)} conf_thresh={val_conf_thresh}\n")

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


def plot_warmup_lr(optimizer, total_steps, warmup_steps, run_dir):
    """
    Plot learning rate during warmup.
    - robust to warmup_steps == 0
    - supports multiple optimizer param groups (plots each group's base LR)
    - saves plot and returns saved path
    """
    import os
    import matplotlib.pyplot as plt

    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    # collect base LR for each param group
    base_lrs = []
    for i, pg in enumerate(optimizer.param_groups):
        base_lrs.append(pg.get('initial_lr', pg.get('lr', 1e-3)))

    steps = list(range(total_steps))
    lrs_groups = []
    for base_lr in base_lrs:
        lrs = []
        for step in steps:
            if warmup_steps > 0 and step < warmup_steps:
                warmup_factor = (step + 1) / warmup_steps
                lr = base_lr * warmup_factor
            else:
                lr = base_lr
            lrs.append(lr)
        lrs_groups.append(lrs)

    plt.figure(figsize=(10, 6))
    for idx, lrs in enumerate(lrs_groups):
        label = f'group {idx} base_lr={base_lrs[idx]:.2e}'
        plt.plot(steps, lrs, '-', linewidth=2, label=label)

    if warmup_steps > 0:
        plt.axvline(x=warmup_steps, color='r', linestyle='--', label=f'Warmup end: step {warmup_steps}')

    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Warmup Schedule')
    if warmup_steps > 0 or len(base_lrs) > 1:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    plot_path = os.path.join(run_dir, "plots", "warmup_lr.png")
    try:
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"⚠️ Could not save warmup LR plot: {e}")
    finally:
        plt.close()

    print(f"📈 Warmup LR plot saved to: {plot_path}")
    return plot_path

def save_plots_from_validation(plot_data, run_dir, epoch):
    """
    Save confusion matrix, F1/confidence, PR curves, and heatmap from validation data.

    Robust / defensive version that:
      - Aligns preds/targets lengths (pads with empty arrays if needed)
      - Uses an explicit 'background' index instead of mapping missing -> class 0
      - Accepts torch.Tensor, np.ndarray, lists; converts safely to numpy
      - Skips plotting steps gracefully if insufficient data
      - Returns True if at least one plot was created, else False
    """
    created_any = False
    try:
        all_preds = plot_data.get("all_preds", []) or []
        all_targets = plot_data.get("all_targets", []) or []
        all_box_centers = plot_data.get("all_box_centers", []) or []

        # Ensure lists
        all_preds = list(all_preds)
        all_targets = list(all_targets)

        # Convert torch tensors to numpy and ensure each entry is numpy array
        def _to_np(arr, expected_cols):
            # expected_cols: number of columns (6 for preds, 5 for targets) used for empty shape
            if arr is None:
                return np.zeros((0, expected_cols), dtype=np.float32)
            if isinstance(arr, torch.Tensor):
                if arr.numel() == 0:
                    return np.zeros((0, expected_cols), dtype=np.float32)
                return arr.detach().cpu().numpy().astype(np.float32)
            if isinstance(arr, (list, tuple)):
                try:
                    arr = np.asarray(arr, dtype=np.float32)
                    if arr.size == 0:
                        return np.zeros((0, expected_cols), dtype=np.float32)
                    return arr
                except Exception:
                    return np.zeros((0, expected_cols), dtype=np.float32)
            if isinstance(arr, np.ndarray):
                if arr.size == 0:
                    return np.zeros((0, expected_cols), dtype=np.float32)
                return arr.astype(np.float32)
            # unknown type -> empty
            return np.zeros((0, expected_cols), dtype=np.float32)

        all_preds = [_to_np(p, 6) for p in all_preds]
        all_targets = [_to_np(t, 5) for t in all_targets]

        # Align lengths: pad the shorter list with empty arrays
        if len(all_preds) < len(all_targets):
            all_preds += [np.zeros((0, 6), dtype=np.float32) for _ in range(len(all_targets) - len(all_preds))]
        elif len(all_targets) < len(all_preds):
            all_targets += [np.zeros((0, 5), dtype=np.float32) for _ in range(len(all_preds) - len(all_targets))]

        # If everything is empty and no centers, nothing to plot
        has_any_preds = any(isinstance(p, np.ndarray) and p.shape[0] > 0 for p in all_preds)
        has_any_targets = any(isinstance(t, np.ndarray) and t.shape[0] > 0 for t in all_targets)
        has_any_centers = isinstance(all_box_centers, (list, tuple)) and len(all_box_centers) > 0

        if (not has_any_preds) and (not has_any_targets) and (not has_any_centers):
            print("⚠️ No validation data to plot (no preds, no targets, no centers).")
            return False

        plots_root = os.path.join(run_dir, "plots")
        os.makedirs(plots_root, exist_ok=True)

        # ----------------------------
        # CONFUSION MATRIX PREPARATION
        # ----------------------------
        # Use explicit background index instead of mapping to class 0
        background_index = len(CLASSES)
        y_true_clean = []
        y_pred_clean = []

        for gt, pred in zip(all_targets, all_preds):
            gt_present = isinstance(gt, np.ndarray) and gt.shape[0] > 0
            pred_present = isinstance(pred, np.ndarray) and pred.shape[0] > 0

            # skip images where both are absent (background-only images)
            if not gt_present and not pred_present:
                continue

            # simple top-1 mapping
            if gt_present:
                try:
                    gt_cls = int(gt[0, 0])
                except Exception:
                    gt_cls = background_index
            else:
                gt_cls = background_index

            if pred_present:
                try:
                    pred_cls = int(pred[0, 0])
                except Exception:
                    pred_cls = background_index
            else:
                pred_cls = background_index

            y_true_clean.append(gt_cls)
            y_pred_clean.append(pred_cls)

        # Confusion matrix plotting
        if len(y_true_clean) > 0:
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
        if has_any_centers:
            try:
                plot_label_heatmap(
                    box_centers=all_box_centers,
                    run_dir=plots_root
                )
                created_any = True
            except Exception as e:
                print(f"⚠️ plot_label_heatmap failed: {e}")

        # ----------------------------
        # Precision–Recall curves (class-wise)
        # ----------------------------
        if has_any_preds or has_any_targets:
            try:
                # compute_precision_recall_curves signature in your repo variably named;
                # call defensively with named args
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
    parser = argparse.ArgumentParser(description="Train MCUDetector V2 (14 classes) - YOLO Detection Only")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs (increased for 14 classes)")
    parser.add_argument("--batch_size", type=int, default=8, help="RTX 2080Ti optimized")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs (increased for 14 classes)")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (increased for 14 classes)")
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

    # Set paths (using reference's cleaner syntax)
    TRAIN_IMG_DIR = Path(args.train_img_dir) if args.train_img_dir else DATA_DIR / "dataset_train" / "images" / "train"
    TRAIN_LABEL_DIR = Path(args.train_label_dir) if args.train_label_dir else DATA_DIR / "dataset_train" / "labels" / "train"
    VAL_IMG_DIR = Path(args.val_img_dir) if args.val_img_dir else DATA_DIR / "dataset_test" / "images" / "train"
    VAL_LABEL_DIR = Path(args.val_label_dir) if args.val_label_dir else DATA_DIR / "dataset_test" / "labels" / "train"

    # Verify dataset paths (with num_classes for validation)
    if not verify_dataset_paths(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, VAL_IMG_DIR, VAL_LABEL_DIR, num_classes=NUM_CLASSES):
        sys.exit(1)

    # Better class display (from reference)
    print(f"\n📊 Classes: {NUM_CLASSES}")
    if CLASSES:
        for i, c in enumerate(CLASSES):
            print(f"  {i:2d}: {c}")

    run_dir = Path(get_run_dir("detect/train"))
    print(f"📂 Run directory: {run_dir}")

    # Train directory structure (using reference's loop)
    for sub in ["plots", "images", "logs", "weights"]:
        (run_dir / sub).mkdir(exist_ok=True)

    # =================== TRAINING METRICS TRACKER ===================
    metrics_tracker = TrainingMetricsTracker(run_dir)

    # =================== TEST RUN DIRECTORY ===================
    test_run_dir = Path(get_run_dir("detect/test"))
    (test_run_dir / "plots").mkdir(parents=True, exist_ok=True)
    print(f"📂 Test run directory: {test_run_dir}")

    # Save config (keeping your format with gpu_enabled)
    config = {
        "num_classes": NUM_CLASSES,
        "classes": list(CLASSES) if CLASSES else [],
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

    # Setup TensorBoard (keeping your error message)
    writer = None
    if not args.no_tb:
        try:
            tb_dir = run_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)
            print(f"📊 TensorBoard logs: tensorboard --logdir={tb_dir}")
        except Exception as e:
            print(f"⚠️ Could not setup TensorBoard: {e}")

    # Data transforms — TRAINING: heavy augmentation to prevent memorization
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])
    
    # VALIDATION: clean transform only
    val_transform = transforms.Compose([
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
            transform=train_transform,
            augment=True
        )
        
        val_dataset = MCUDetectionDataset(
            img_dir=VAL_IMG_DIR,
            label_dir=VAL_LABEL_DIR,
            img_size=512,
            transform=val_transform
        )
        
        # Label stats
        label_dir = Path(TRAIN_LABEL_DIR)
        label_paths = sorted(list(label_dir.glob("*.txt")))
        debug_label_stats(label_paths, CLASSES if CLASSES else [str(i) for i in range(NUM_CLASSES)], run_dir)
        
        # Count instances per class
        from collections import Counter
        inst_counts = Counter()
        for p in label_paths:
            for line in p.read_text().splitlines():
                if not line.strip(): continue
                c = int(float(line.split()[0]))
                inst_counts[c] += 1

        class_counts = [inst_counts.get(i, 0) for i in range(NUM_CLASSES)]
        
        # Class weights (inverse frequency)
        total = float(sum(class_counts))
        class_weights = [(total / max(c, 1.0)) for c in class_counts]

        # Per-image weights
        if hasattr(train_dataset, "image_paths"):
            dataset_label_paths = [Path(p).with_suffix(".txt") for p in train_dataset.image_paths]
        elif hasattr(train_dataset, "samples"):
            dataset_label_paths = [Path(p[0]).with_suffix(".txt") for p in train_dataset.samples]
        else:
            dataset_label_paths = sorted(label_dir.glob("*.txt"))

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
                image_weights.append(float(np.mean([class_weights[c] for c in cls_set])))

        image_weights = np.array(image_weights, dtype=np.float32)
        image_weights = image_weights / image_weights.mean()
        image_weights = np.clip(image_weights, 0.1, 10.0)

        # Sampler
        sampler = WeightedRandomSampler(
            weights=image_weights.tolist(),
            num_samples=len(image_weights),
            replacement=True
        )

        # ✅ FIXED: Enable pin_memory and persistent_workers for faster data loading
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            collate_fn=detection_collate_fn,
            pin_memory=False,        # Changed from False to True
            drop_last=True,
            persistent_workers=args.workers > 0  # Changed
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=detection_collate_fn,
            pin_memory=False,        # Changed from False to True
            persistent_workers=args.workers > 0  # Changed
        )
        
        print(f"DEBUG class_counts: {class_counts}")
        with open(run_dir / "logs" / "dataset_counts.txt", "w") as f:
            f.write(f"class_counts: {class_counts}\n")
            f.write(f"image_weights min/max: {min(image_weights):.6f}/{max(image_weights):.6f}\n")
                
        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Validation samples: {len(val_dataset)}")
        print(f"✅ Training batches: {len(train_loader)}")
        
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        import traceback; traceback.print_exc()  # Added traceback for debugging
        sys.exit(1)

    # =================== MODEL + LOSS ===================
    print("\n🤖 Creating model...")
    try:
        model = MCUDetector(num_classes=NUM_CLASSES).to(device)
        
        # Split classifier vs. others
        classifier_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"⚠️ WARNING: {name} does not require grad!")
                continue

            if "cls" in name and "bn" not in name:
                classifier_params.append(param)
            else:
                other_params.append(param)

        print(f"DEBUG optimizer params: cls={len(classifier_params)}, other={len(other_params)}")
        
        # ✅ FIXED: Proper loss initialization with all parameters
        criterion = MCUDetectionLoss(
            num_classes=NUM_CLASSES,
            bbox_weight=1.0,
            obj_weight=4.0,
            cls_weight=3.0,
            topk=9,
            focal_gamma=2.0,
            label_smoothing=0.05,
        ).to(device)
        
        print(f"✅ Model parameters: {count_parameters(model) / 1e6:.2f}M")
        print_gpu_memory()
        print("📏 Using adaptive IoU thresholds for small object detection")
        print("   • Objects <16×16: IoU threshold × 0.7")
        print("   • Objects 16-32: IoU threshold × 0.85")
        print("   • Objects >32: IoU threshold × 1.0")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ✅ FIXED: Increased classifier LR to 2.0x for decoupled head
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": args.lr, "weight_decay": 1e-4},
        {"params": classifier_params, "lr": args.lr * 2.0, "weight_decay": 1e-5}  # Changed from 1.5x to 2.0x
    ], betas=(0.9, 0.999))

    # Store initial LR for warmup
    for g in optimizer.param_groups:
        g['initial_lr'] = g['lr']

    # ✅ FIXED: Adjusted T_0 for 200 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs - args.warmup_epochs,  # Full cosine over remaining epochs
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
        # ✅ CRITICAL FIX: Save state BEFORE LR finder (it corrupts weights!)
        saved_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        saved_optimizer_state = optimizer.state_dict()
        saved_scaler_state = scaler.state_dict()
        saved_ema_state = None
        if ema is not None:
            saved_ema_state = {k: v.clone() for k, v in ema.ema.state_dict().items()}

        optimal_lr = find_optimal_lr(
            model, train_loader, criterion, device, run_dir
        )

        # ✅ CRITICAL FIX: Restore clean weights after LR finder
        model.load_state_dict(saved_model_state)
        optimizer.load_state_dict(saved_optimizer_state)
        scaler.load_state_dict(saved_scaler_state)
        if ema is not None and saved_ema_state is not None:
            ema.ema.load_state_dict(saved_ema_state)
        print("🔁 Model weights restored after LR finder")

        if optimal_lr:
            # FIX: Safety cap — LR finder returned 8.71e-2 last time which is way too high
            # For AdamW with this model, >3e-3 causes instability
            max_lr = 3e-3
            if optimal_lr > max_lr:
                print(f"  ⚠️ LR finder suggested {optimal_lr:.2e} — capping at {max_lr:.2e}")
                optimal_lr = max_lr
            # Preserve the relative LR ratios between param groups
            # Group 0: backbone/fpn (1x), Group 1: classifier (2x)
            for i, g in enumerate(optimizer.param_groups):
                multiplier = g.get('initial_lr', g.get('lr', optimal_lr)) / optimizer.param_groups[0].get('initial_lr', args.lr)
                multiplier = max(multiplier, 1.0)  # at least 1x
                g['lr'] = optimal_lr * multiplier
                g['initial_lr'] = optimal_lr * multiplier
            print(f"🎯 Using found LR: {optimal_lr:.2e} (cls group: {optimizer.param_groups[-1]['lr']:.2e})")
        
        del saved_model_state, saved_optimizer_state, saved_scaler_state, saved_ema_state
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # Plot warmup LR schedule
    if args.warmup_epochs > 0:
        plot_warmup_lr(optimizer, args.warmup_epochs * len(train_loader), 
                    args.warmup_epochs * len(train_loader), run_dir)
    
# =================== TRAINING LOOP ===================
    print(f"\n{'='*60}")
    print(f"🚀 Training {args.epochs} epochs | batch={args.batch_size}×{args.accum_steps}")
    print(f"   LR={args.lr:.1e} | warmup={args.warmup_epochs} | patience={args.patience}")
    print(f"   mAP: {'ON' if args.calculate_map else 'OFF'} | EMA: {'ON' if args.use_ema else 'OFF'}")
    print(f"{'='*60}")

    train_losses, val_losses = [], []

    def _safe_pr(preds_list, targets_list, conf_thresh=0.25):
        has_p = any(isinstance(a, np.ndarray) and a.size > 0 for a in preds_list)
        has_g = any(isinstance(t, np.ndarray) and t.size > 0 for t in targets_list)
        if not (has_p and has_g):
            return 0.0, 0.0
        try:
            return compute_precision_recall(preds_list, targets_list, conf_thresh=conf_thresh)
        except Exception:
            return 0.0, 0.0

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1:03d}/{args.epochs}")

        train_loss, train_bbox, train_cls, train_obj = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, args.accum_steps, scheduler, epoch,
            args.warmup_epochs, writer, ema
        )
        train_losses.append(train_loss)
        # FIX: Scheduler step ONCE PER EPOCH (was per-batch = 51 cycles/epoch)

        if epoch >= args.warmup_epochs:
            scheduler.step()

        val_loss, val_bbox, val_cls, val_obj, metrics, plot_data = validate(
            model, val_loader, criterion, device,
            run_dir, epoch + 1, writer, ema, args.calculate_map, plot=False
        )
        val_losses.append(val_loss)

        p, r = _safe_pr(plot_data.get("all_preds", []), plot_data.get("all_targets", []))

        metrics_tracker.update(
            epoch=epoch + 1,
            train_metrics={"loss": train_loss, "box": train_bbox, "cls": train_cls, "obj": train_obj},
            val_metrics={
                "loss": val_loss, "box": val_bbox, "cls": val_cls, "obj": val_obj,
                "precision": p, "recall": r,
                "map50": metrics.get("mAP_50", 0.0), "map5095": metrics.get("mAP_50_95", 0.0)
            },
            lr=optimizer.param_groups[0]['lr']
        )

        metric_str = f" | mAP@0.5: {metrics.get('mAP_50', 0):.4f}" if metrics else ""
        print(f"  📊 Train: {train_loss:.4f} | Val: {val_loss:.4f}{metric_str}")
        print(f"     Box: {train_bbox:.3f}/{val_bbox:.3f} | Cls: {train_cls:.3f}/{val_cls:.3f} | Obj: {train_obj:.3f}/{val_obj:.3f}")
        print(f"     P={p:.3f} R={r:.3f} F1={2*p*r/(p+r+1e-12):.3f} | LR={optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        current_metric = metrics.get('mAP_50', 0) if args.calculate_map else -val_loss
        is_best = False

        if args.calculate_map:
            if current_metric > best_map:
                best_map = current_metric
                best_loss = val_loss
                is_best = True
        else:
            if val_loss < best_loss:
                best_loss = val_loss
                is_best = True

        if is_best:
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.ema.state_dict() if ema else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss, "train_loss": train_loss,
                "best_loss": best_loss, "best_map": best_map,
                "metrics": metrics, "args": vars(args),
                "classes": list(CLASSES), "num_classes": NUM_CLASSES
            }
            torch.save(ckpt, run_dir / "weights" / "best_mcu.pt")
            reason = f"mAP@0.5={current_metric:.4f}" if args.calculate_map else f"val_loss={val_loss:.4f}"
            print(f"  💾 BEST! ({reason})")
            save_plots_from_validation(plot_data, run_dir, epoch + 1)

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            latest = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.ema.state_dict() if ema else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss, "best_loss": best_loss, "best_map": best_map,
            }
            tmp = run_dir / "weights" / "checkpoint_tmp.pt"
            torch.save(latest, tmp)
            try:
                os.replace(str(tmp), str(run_dir / "weights" / "checkpoint_latest.pt"))
            except Exception:
                torch.save(latest, run_dir / "weights" / "checkpoint_latest.pt")
            print(f"  💾 Checkpoint @ epoch {epoch+1}")

        metrics_tracker.save_csv()

        if early_stopping(val_loss):
            print(f"\n🛑 Early stopping at epoch {epoch+1} (no improvement for {early_stopping.counter} epochs)")
            break

    # =================== FINAL ===================
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.ema.state_dict() if ema else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss, "best_map": best_map,
        "classes": list(CLASSES), "num_classes": NUM_CLASSES
    }, run_dir / "weights" / "final_mcu.pt")

    metrics_tracker.save_csv()
    save_loss_plot(train_losses, val_losses, run_dir, "Final Loss Curves")
    plot_yolo_results(metrics_tracker.history, save_path=os.path.join(run_dir, "plots", "results.png"))

    # Final evaluation with best model
    try:
        best_path = run_dir / "weights" / "best_mcu.pt"
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            if args.use_ema and best_ckpt.get("ema_state_dict"):
                if ema is None:
                    ema = ModelEMA(model, decay=args.ema_decay)
                ema.ema.load_state_dict(best_ckpt["ema_state_dict"])

        eval_model = get_eval_model(model, ema=(ema if args.use_ema else None), device=device, fuse=True)

        vl, vb, vc, vd, mf, pdf = validate(
            eval_model, val_loader, criterion, device, test_run_dir,
            args.epochs, writer=None, ema=None, calculate_metrics=True, plot=True
        )

        pt, rt = _safe_pr(pdf.get("all_preds", []), pdf.get("all_targets", []))

        test_metrics = {
            'precision': pt, 'recall': rt,
            'mAP_50': mf.get('mAP_50', 0), 'mAP_75': mf.get('mAP_75', 0),
            'mAP_50_95': mf.get('mAP_50_95', 0),
            'loss': vl, 'box_loss': vb, 'cls_loss': vc, 'obj_loss': vd
        }
        save_test_summary(test_metrics, test_run_dir / "test_summary.txt")

        print(f"\n📊 FINAL TEST:")
        print(f"   P={pt:.3f} R={rt:.3f} F1={2*pt*rt/(pt+rt+1e-12):.3f}")
        print(f"   mAP@0.5={mf.get('mAP_50',0):.3f} mAP@0.5:0.95={mf.get('mAP_50_95',0):.3f}")
    except Exception as e:
        print(f"⚠️ Final eval error: {e}")
        import traceback; traceback.print_exc()

    if writer:
        writer.close()

    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"   Best val loss: {best_loss:.4f}")
    if args.calculate_map:
        print(f"   Best mAP@0.5:  {best_map:.4f}")
    print(f"   Weights: {run_dir / 'weights'}")
    print(f"   Plots:   {run_dir / 'plots'} & {test_run_dir / 'plots'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()