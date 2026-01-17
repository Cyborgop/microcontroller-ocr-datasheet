#!/usr/bin/env python3
"""
MCUDetector YOLO Training (Detection ONLY) - RTX 2080Ti Optimized
7-class object detection without OCR
With EMA, LR Finder, TQDM, and Full Metrics
"""

import os
import tensorboard
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
        save_results_csv, plot_label_heatmap, plot_f1_confidence_curve,
        decode_predictions, calculate_map, save_precision_recall_curves,  plot_yolo_results, box_iou_batch  # Add these to utils.py
    )
    from pathlib import Path
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure model.py, dataset.py, and utils.py are in the same directory")
    sys.exit(1)

# =================== PATHS ===================
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
NUM_CLASSES = len(CLASSES) if CLASSES else 7

# =================== EMA (EXPONENTIAL MOVING AVERAGE) ===================
class ModelEMA:
    """Model Exponential Moving Average from YOLOv5"""
    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = self.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1 - np.exp(-x / 2000))  # decay exponential ramp
        for p in self.ema.parameters():
            p.requires_grad_(False)
    
    @staticmethod
    def deepcopy(model):
        """Create a deep copy of the model"""
        model_copy = type(model)(num_classes=model.num_classes).to(next(model.parameters()).device)
        model_copy.load_state_dict(model.state_dict())
        return model_copy
    
    def update(self, model):
        """Update EMA parameters"""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            
            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()
    
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

# =================== DATASET VERIFICATION ===================
def verify_dataset_paths(train_img_dir, train_label_dir, val_img_dir, val_label_dir):
    """Verify all dataset paths exist."""
    print("\n🔍 Verifying dataset paths...")
    
    paths = [
        ("Train Images", train_img_dir),
        ("Train Labels", train_label_dir),
        ("Val Images", val_img_dir),
        ("Val Labels", val_label_dir),
    ]
    
    all_exist = True
    for name, path in paths:
        if not path.exists():
            print(f"❌ {name}: {path} - NOT FOUND")
            all_exist = False
        else:
            files = list(path.glob("*"))
            file_count = len(files)
            print(f"✅ {name}: {path} ({file_count} files)")
            
            if "images" in str(path).lower():
                img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
                img_count = sum(1 for f in files if f.suffix.lower() in img_exts)
                print(f"   Images: {img_count} files with image extensions")
            
            if "labels" in str(path).lower():
                txt_count = sum(1 for f in files if f.suffix.lower() == '.txt')
                print(f"   Labels: {txt_count} .txt files")
    
    if not all_exist:
        print("\n❌ ERROR: Some dataset paths don't exist!")
        print("Please check your data directory structure.")
        return False
    
    # Check for matching files
    train_images = {f.stem for f in train_img_dir.glob("*") if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
    train_labels = {f.stem for f in train_label_dir.glob("*.txt")}
    
    if not train_images:
        print("❌ ERROR: No training images found!")
        return False
    
    if not train_labels:
        print("❌ ERROR: No training labels found!")
        return False
    
    missing_labels = train_images - train_labels
    if missing_labels:
        print(f"⚠️ Warning: {len(missing_labels)} images without labels")
        if len(missing_labels) < 10:
            print(f"   Missing: {list(missing_labels)[:5]}...")
    
    print("✅ Dataset verification complete!")
    return True

# =================== GPU OPTIMIZATION ===================
def setup_gpu_optimizations():
    """Set up GPU optimizations for RTX 2080Ti."""
    if torch.cuda.is_available():
        # Enable TF32 for faster training (RTX 20 series and newer)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Memory optimization
        torch.cuda.empty_cache()
        
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️  GPU: {gpu_name}")
        print(f"🧠 Memory: {memory_gb:.1f} GB")
        print(f"🎯 CUDA: {torch.version.cuda}")
        
        return True
    return False

# =================== LEARNING RATE FINDER ===================
def find_optimal_lr(model, train_loader, criterion, device, run_dir, 
                    start_lr=1e-7, end_lr=0.1, num_iter=100):
    """Find optimal learning rate using Leslie Smith's method."""
    print("\n🔍 Finding optimal learning rate...")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    scaler = GradScaler()
    
    lrs = []
    losses = []
    avg_loss = 0.0
    best_loss = float('inf')
    beta = 0.98
    
    # Exponential learning rate schedule
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    
    data_iter = iter(train_loader)
    pbar = tqdm(range(num_iter), desc="LR Finder")
    
    for i in pbar:
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)
        
        images = images.to(device)
        
        # Prepare targets
        targets_p4, targets_p5 = [], []
        for t in targets:
            if t.numel() == 0:
                targets_p4.append(torch.zeros((0, 5), device=device))
                targets_p5.append(torch.zeros((0, 5), device=device))
                continue
            
            t = t.to(device)
            area = t[:, 3] * t[:, 4] * 512 * 512
            targets_p4.append(t[area < 1024])
            targets_p5.append(t[area >= 1024])
        
        # Forward/backward
        optimizer.zero_grad()
        with autocast():
            pred = model(images)
            loss_dict = criterion(pred[0], pred[1], targets_p4, targets_p5)
            loss = loss_dict["total"]
        
        # Scale loss for mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Exponential moving average of loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (i + 1))
        
        # Stop if loss explodes
        if i > 10 and smoothed_loss > 4 * best_loss:
            pbar.set_description(f"LR Finder stopped early (loss exploded)")
            break
        
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        
        # Record
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        losses.append(smoothed_loss)
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
        
        pbar.set_postfix({"LR": f"{current_lr:.2e}", "Loss": f"{smoothed_loss:.4f}"})
    
    # Find optimal LR (minimum gradient point)
    if len(lrs) > 10:
        # Smooth losses
        window = min(5, len(losses) // 10)
        if window > 1:
            kernel = np.ones(window) / window
            smoothed = np.convolve(losses, kernel, mode='valid')
            lrs_smooth = lrs[window-1:]
        else:
            smoothed = losses
            lrs_smooth = lrs
        
        # Find point with steepest negative gradient
        gradients = np.gradient(smoothed)
        min_gradient_idx = np.argmin(gradients)
        optimal_lr = lrs_smooth[min_gradient_idx]
        
        print(f"✅ Optimal LR found: {optimal_lr:.2e}")
        
        # Plot LR finder curve
        plt.figure(figsize=(10, 6))
        plt.plot(lrs_smooth, smoothed, 'b-', linewidth=2)
        plt.axvline(x=optimal_lr, color='r', linestyle='--', 
                   label=f'Optimal LR: {optimal_lr:.2e}')
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(run_dir, "plots", "lr_finder.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 LR finder plot saved to: {plot_path}")
        
        return optimal_lr
    
    print("⚠️ Could not determine optimal LR, using default")
    return None

# =================== TRAIN LOOP WITH TQDM ===================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, 
                    accum_steps, scheduler, epoch, warmup_epochs, writer=None, ema=None):
    """Train for one epoch with gradient accumulation and TQDM."""
    model.train()
    total_loss = 0.0
    bbox_loss_total = 0.0
    obj_loss_total = 0.0
    cls_loss_total = 0.0
    
    optimizer.zero_grad(set_to_none=True)
    
    start_time = time.time()
    
    # Use tqdm for progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch+1:03d} [Train]", leave=False)
    
    for i, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)

        # Prepare scale-aware targets for YOLO
        targets_p4, targets_p5 = [], []
        for t in targets:
            if t.numel() == 0:
                targets_p4.append(torch.zeros((0, 5), device=device))
                targets_p5.append(torch.zeros((0, 5), device=device))
                continue
            
            t = t.to(device)
            area = t[:, 3] * t[:, 4] * 512 * 512
            targets_p4.append(t[area < 1024])   # Small objects → P4
            targets_p5.append(t[area >= 1024])  # Large objects → P5

        # Mixed precision forward
        with autocast():
            pred = model(images)
            loss_dict = criterion(pred[0], pred[1], targets_p4, targets_p5)
            loss = loss_dict["total"] / accum_steps

        # Backward with gradient scaling
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Update EMA if available
            if ema is not None:
                ema.update(model)

        # Accumulate losses
        total_loss += loss_dict["total"].item()
        bbox_loss_total += loss_dict.get("bbox", 0)
        obj_loss_total  += loss_dict.get("obj", 0)
        cls_loss_total  += loss_dict.get("cls", 0)



        # Update progress bar
        pbar.set_postfix({
            "Loss": f"{loss_dict['total'].item():.3f}",
            "LR": f"{optimizer.param_groups[0]['lr']:.1e}"
        })

        # Linear warmup
        if epoch < warmup_epochs:
            warmup_factor = (epoch * len(loader) + i + 1) / (warmup_epochs * len(loader))
            for g in optimizer.param_groups:
                g['lr'] = g['initial_lr'] * warmup_factor
        
        # TensorBoard logging
        if writer and i % 10 == 0:
            writer.add_scalar('LR/train', optimizer.param_groups[0]['lr'], 
                            epoch * len(loader) + i)
            writer.add_scalar('Loss/train_batch', loss_dict["total"].item(),
                            epoch * len(loader) + i)
    
    scheduler.step()
    
    # Calculate average losses
    avg_total = total_loss / len(loader)
    avg_bbox = bbox_loss_total / len(loader)
    avg_obj = obj_loss_total / len(loader)
    avg_cls = cls_loss_total / len(loader)
    
    # TensorBoard logging
    if writer:
        writer.add_scalar('Loss/train', avg_total, epoch)
        writer.add_scalar('Loss/bbox', avg_bbox, epoch)
        writer.add_scalar('Loss/obj', avg_obj, epoch)
        writer.add_scalar('Loss/cls', avg_cls, epoch)
        writer.add_scalar('LR/epoch', optimizer.param_groups[0]['lr'], epoch)
    
    epoch_time = time.time() - start_time
    print(f"  ⏱️  Epoch time: {epoch_time:.1f}s ({epoch_time/len(loader):.2f}s/batch)")
    
    return avg_total, avg_bbox, avg_cls, avg_obj


@torch.no_grad()
def validate(model, loader, criterion, device, run_dir, epoch, writer=None, ema=None, calculate_metrics=False):
    """Validate model with full metrics, TQDM, EMA, confusion matrix, PR curve, and label heatmap."""
    model.eval()
    ema_model = ema.ema if ema is not None else model

    total_loss = 0.0
    bbox_loss_total = 0.0
    obj_loss_total = 0.0
    cls_loss_total = 0.0

    all_targets = []    # list of per-image np.ndarray (M,5) => [cls, cx, cy, w, h]
    all_preds = []      # list of per-image np.ndarray (N,6) => [cls, conf, x1, y1, x2, y2]
    all_pred_scores = []  # kept for compatibility (currently unused)
    all_box_centers = []

    start_time = time.time()
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]", leave=False)

    def _to_float(v):
        """Convert tensor or numeric-like to Python float safely."""
        if isinstance(v, torch.Tensor):
            return v.item()
        try:
            return float(v)
        except Exception:
            return 0.0

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)

        # -------------------------------------------------------
        # Target handling (YOLOv8-compatible, per-image GT)
        # -------------------------------------------------------
        targets_p4, targets_p5 = [], []
        batch_targets_per_image = []   # per-image GT storage

        for t in targets:
            if t.numel() == 0:
                targets_p4.append(torch.zeros((0, 5), device=device))
                targets_p5.append(torch.zeros((0, 5), device=device))
                # empty GT for this image (shape (0,5))
                batch_targets_per_image.append(np.zeros((0, 5), dtype=np.float32))
                continue

            t = t.to(device)

            # Scale-aware split (unchanged)
            area = t[:, 3] * t[:, 4] * 512 * 512
            targets_p4.append(t[area < 1024])
            targets_p5.append(t[area >= 1024])

            # store full GT per image (M,5): cls, cx, cy, w, h
            batch_targets_per_image.append(t.detach().cpu().numpy())

            # Heatmap centers (cx, cy) normalized
            centers = t[:, 1:3].cpu().numpy().tolist()
            all_box_centers.extend(centers)

        # -------------------------------------------------------
        # Forward (EMA model if enabled)
        # -------------------------------------------------------
        pred = ema_model(images)
        loss_dict = criterion(pred[0], pred[1], targets_p4, targets_p5)

        # -------------------------------------------------------
        # Decode predictions for metrics
        # -------------------------------------------------------
        if calculate_metrics:
            decoded_preds = decode_predictions(
                pred[0], pred[1],
                conf_thresh=0.25,
                nms_thresh=0.45
            )

            for pb in decoded_preds:
                if isinstance(pb, torch.Tensor) and pb.numel() > 0:
                    all_preds.append(pb.detach().cpu().numpy())
                else:
                    all_preds.append(np.zeros((0, 6), dtype=np.float32))

        # -------------------------------------------------------
        # SAFE loss extraction
        # -------------------------------------------------------
        total_val = _to_float(loss_dict.get("total", 0))
        bbox_val  = _to_float(loss_dict.get("bbox",  0))
        obj_val   = _to_float(loss_dict.get("obj",   0))
        cls_val   = _to_float(loss_dict.get("cls",   0))

        # Accumulate
        total_loss += total_val
        bbox_loss_total += bbox_val
        obj_loss_total += obj_val
        cls_loss_total += cls_val

        # -------------------------------------------------------
        # Final: extend per-image GT list
        # -------------------------------------------------------
        all_targets.extend(batch_targets_per_image)

        # Update progress bar using the safe float
        pbar.set_postfix({"Loss": f"{total_val:.3f}"})

    # ---- Averages (protect against zero-length loader) ----
    n_batches = len(loader) if len(loader) > 0 else 1
    avg_total = total_loss / n_batches
    avg_bbox = bbox_loss_total / n_batches
    avg_obj = obj_loss_total / n_batches
    avg_cls = cls_loss_total / n_batches

    # ---- Metrics ----
    metrics = {}
    if calculate_metrics and len(all_targets) > 0 and len(all_preds) > 0:
        map_50_95, map_50, map_75, per_class_ap = calculate_map(
            predictions=all_preds,
            targets=all_targets,
            num_classes=NUM_CLASSES,
            iou_thresholds=[0.5, 0.75]
        )
        metrics = {
            "mAP_50": map_50,
            "mAP_75": map_75,
            "mAP_50_95": map_50_95,
        }

        if writer:
            writer.add_scalar("Metrics/mAP_50", map_50, epoch)
            writer.add_scalar("Metrics/mAP_75", map_75, epoch)
            writer.add_scalar("Metrics/mAP_50_95", map_50_95, epoch)

    # ---- TensorBoard losses ----
    if writer:
        writer.add_scalar("Loss/val", avg_total, epoch)
        writer.add_scalar("Loss/val_bbox", avg_bbox, epoch)
        writer.add_scalar("Loss/val_obj", avg_obj, epoch)
        writer.add_scalar("Loss/val_cls", avg_cls, epoch)

    # ---- Plots every 20 epochs ----
    if epoch % 20 == 0 and len(all_targets) > 0:
        try:
            # fallback: if no preds, create empty preds per image
            if len(all_preds) == 0:
                all_preds = [np.zeros((0, 6), dtype=np.float32) for _ in range(len(all_targets))]

            # --------------------------------------------------
            # Confusion matrix (single-object-per-image)
            # --------------------------------------------------
            y_true_cm = []
            y_pred_cm = []

            for gt, pred in zip(all_targets, all_preds):
                # Ground truth class (first GT if exists)
                if isinstance(gt, np.ndarray) and gt.shape[0] > 0:
                    y_true_cm.append(int(gt[0, 0]))
                else:
                    y_true_cm.append(-1)

                # Predicted class (top prediction if exists)
                if isinstance(pred, np.ndarray) and pred.shape[0] > 0:
                    y_pred_cm.append(int(pred[0, 0]))
                else:
                    y_pred_cm.append(-1)

            # Remove samples where both GT and pred are empty
            y_true_clean = []
            y_pred_clean = []
            for t, p in zip(y_true_cm, y_pred_cm):
                if t == -1 and p == -1:
                    continue
                y_true_clean.append(t if t != -1 else 0)
                y_pred_clean.append(p if p != -1 else 0)

            plot_confusion_matrix(
                y_true=y_true_clean,
                y_pred=y_pred_clean,
                labels=CLASSES,
                run_dir=os.path.join(run_dir, "plots"),
                base_title=f"Confusion Matrix - Epoch {epoch}"
            )

            # --------------------------------------------------
            # F1-confidence curve
            # --------------------------------------------------
            plot_f1_confidence_curve(
                predictions=all_preds,
                targets=all_targets,
                class_names=CLASSES,
                run_dir=os.path.join(run_dir, "plots")
            )

            # --------------------------------------------------
            # Label heatmap
            # --------------------------------------------------
            if len(all_box_centers) > 0:
                plot_label_heatmap(
                    box_centers=all_box_centers,
                    run_dir=os.path.join(run_dir, "plots")
                )

            # --------------------------------------------------
            # TRUE Precision–Recall curves (IoU = 0.5)
            # --------------------------------------------------
            IOU_TH = 0.5
            img_size = 512  # adjust if your evaluation size differs
            confidences = np.linspace(0.0, 1.0, 100)

            precisions = []
            recalls = []
            eps = 1e-12

            for cls_id in range(NUM_CLASSES):
                prec_per_thr = []
                rec_per_thr = []

                for thr in confidences:
                    TP = FP = FN = 0

                    for preds_img, gts_img in zip(all_preds, all_targets):
                        # select predictions of this class above threshold
                        if isinstance(preds_img, np.ndarray) and preds_img.shape[0] > 0:
                            mask = (preds_img[:, 0].astype(int) == cls_id) & (preds_img[:, 1] >= thr)
                            preds_sel = preds_img[mask]
                        else:
                            preds_sel = np.zeros((0, 6), dtype=np.float32)

                        # select GTs of this class
                        if isinstance(gts_img, np.ndarray) and gts_img.shape[0] > 0:
                            gt_idx = np.where(gts_img[:, 0].astype(int) == cls_id)[0]
                        else:
                            gt_idx = np.array([], dtype=int)

                        # no GTs → all preds are FP
                        if gt_idx.size == 0:
                            FP += preds_sel.shape[0]
                            continue

                        gts_cls = gts_img[gt_idx]
                        cx = gts_cls[:, 1] * img_size
                        cy = gts_cls[:, 2] * img_size
                        w  = gts_cls[:, 3] * img_size
                        h  = gts_cls[:, 4] * img_size
                        gt_boxes = np.stack(
                            [cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1
                        ).astype(np.float32)

                        if preds_sel.shape[0] == 0:
                            FN += gt_boxes.shape[0]
                            continue

                        pred_boxes = preds_sel[:, 2:6].astype(np.float32)

                        ious = box_iou_batch(
                            torch.from_numpy(pred_boxes),
                            torch.from_numpy(gt_boxes)
                        ).cpu().numpy()

                        # sort preds by confidence (desc)
                        order = np.argsort(-preds_sel[:, 1])
                        ious = ious[order]

                        matched_gt = set()
                        for i in range(ious.shape[0]):
                            row = ious[i]
                            for m in matched_gt:
                                row[m] = -1.0
                            best_gt = int(np.argmax(row))
                            if row[best_gt] >= IOU_TH:
                                TP += 1
                                matched_gt.add(best_gt)
                            else:
                                FP += 1

                        FN += max(0, gt_boxes.shape[0] - len(matched_gt))

                    prec_per_thr.append(TP / (TP + FP + eps))
                    rec_per_thr.append(TP / (TP + FN + eps))

                precisions.append(np.array(prec_per_thr))
                recalls.append(np.array(rec_per_thr))

            save_precision_recall_curves(
                confidences=confidences,
                precisions=np.array(precisions),
                recalls=np.array(recalls),
                class_names=CLASSES,
                run_dir=os.path.join(run_dir, "plots")
            )

        except Exception as e:
            print(f"⚠️ Plotting error: {e}")


    print(f"  ⏱️ Validation time: {time.time() - start_time:.1f}s")
    return avg_total, avg_bbox, avg_cls, avg_obj, metrics




# =================== SAVE LOSS PLOT ===================
def save_loss_plot(train_losses, val_losses, run_dir, title="Training and Validation Loss"):
    """Save loss plot with smooth curves."""
    plt.figure(figsize=(12, 8))
    
    # Smooth the curves for better visualization
    window = max(1, len(train_losses) // 20)
    if window > 1 and len(train_losses) > window:
        kernel = np.ones(window) / window
        train_smooth = np.convolve(train_losses, kernel, mode='valid')
        val_smooth = np.convolve(val_losses, kernel, mode='valid')
        x_train = np.arange(window-1, len(train_losses))
        x_val = np.arange(window-1, len(val_losses))
    else:
        train_smooth = train_losses
        val_smooth = val_losses
        x_train = np.arange(len(train_losses))
        x_val = np.arange(len(val_losses))
    
    plt.plot(x_train, train_smooth, label='Train Loss', 
             color='blue', linewidth=2, alpha=0.8)
    plt.plot(x_val, val_smooth, label='Val Loss', 
             color='red', linewidth=2, alpha=0.8)
    
    # Plot original points as scatter
    plt.scatter(range(len(train_losses)), train_losses, 
                color='blue', alpha=0.3, s=10)
    plt.scatter(range(len(val_losses)), val_losses, 
                color='red', alpha=0.3, s=10)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add best validation loss annotation
    if val_losses:
        best_epoch = np.argmin(val_losses)
        best_loss = val_losses[best_epoch]
        plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
        plt.text(best_epoch, max(val_losses)*0.9, 
                f'Best: {best_loss:.4f} @ epoch {best_epoch+1}',
                fontsize=10, color='green')
    
    plt.tight_layout()
    
    plot_path = os.path.join(run_dir, "plots", "loss_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Loss plot saved to {plot_path}")

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
    
    # Create structured run directory under runs/detect/train/

    run_dir = Path(get_run_dir("detect/train"))
    print(f"📂 Run directory: {run_dir}")
    # Train directory structure
    (run_dir / "model").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "images").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    # =================== TRAINING HISTORY ===================
    history = {
        "train_box": [],
        "train_cls": [],
        "train_dfl": [],
        "val_box": [],
        "val_cls": [],
        "val_dfl": [],
        "precision": [],
        "recall": [],
        "map50": [],
        "map5095": []
    }


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
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if args.workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=detection_collate_fn,
            pin_memory=True,
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
    scaler = GradScaler()
    
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
    val_losses = []
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1:03d}/{args.epochs}")
        
        # Train
        train_loss, train_box, train_cls, train_dfl = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, args.accum_steps, scheduler, epoch, 
            args.warmup_epochs, writer, ema
        )
        train_losses.append(train_loss)

        history["train_box"].append(train_box)
        history["train_cls"].append(train_cls)
        history["train_dfl"].append(train_dfl)
        
        # Validate with EMA model if available
        val_model = ema.ema if ema is not None else model
        val_loss, val_box, val_cls, val_dfl, metrics = validate(
                        val_model, val_loader, criterion, device,
                        run_dir, epoch+1, writer, ema, args.calculate_map
                    )
        val_losses.append(val_loss)
        history["val_box"].append(val_box)
        history["val_cls"].append(val_cls)
        history["val_dfl"].append(val_dfl)

        history["map50"].append(metrics.get("mAP_50", 0.0))
        history["map5095"].append(metrics.get("mAP_50_95", 0.0))

        
        # Print metrics
        metric_str = ""
        if metrics:
            metric_str = f" | mAP@0.5: {metrics.get('mAP_50', 0):.4f}"
        
        print(f"  📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{metric_str}")
        print(f"  📈 LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save metrics to CSV
        csv_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]['lr']
        }
        if metrics:
            csv_data.update(metrics)
        save_results_csv(csv_data, run_dir)
        
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
                torch.save(checkpoint, run_dir / "model" / "best_mcu.pt")
                print(f"  💾 BEST mAP! Model saved (mAP@0.5: {current_metric:.4f})")
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
                torch.save(checkpoint, run_dir / "model" / "best_mcu.pt")
                print(f"  💾 BEST! Model saved (val_loss: {val_loss:.4f})")
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = run_dir / "weights" / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
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
            }, checkpoint_path)
            torch.save(
                torch.load(checkpoint_path),
                run_dir / "model" / f"checkpoint_epoch_{epoch+1}.pt"
            )
            print(f"  💾 Checkpoint saved at epoch {epoch+1}")
            
        # Save loss plot every 20 epochs
        if (epoch + 1) % 20 == 0:
            save_loss_plot(train_losses, val_losses, run_dir)
        
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
    torch.save(final_checkpoint, run_dir / "model" / "final_mcu.pt")
    
    # Save final loss plot
    save_loss_plot(train_losses, val_losses, run_dir, "Final Training and Validation Loss")
    plot_yolo_results(
    history,
    save_path=os.path.join(run_dir, "plots", "results.png")
)
    
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
    print(f"📊 Plots saved to: {run_dir}/plots/")
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