#!/usr/bin/env python3
"""
MCUDetector V2-Lite — Comprehensive Test & Inference Pipeline
=============================================================
Creates runs_test/ directory structure with all metrics and plots
needed for paper publication.

Usage:
  # Full test evaluation on validation/test set:
  python3 ocr/Inference_Pipeline.py --model runs/detect/train/weights/best_mcu.pt \
  --test_img_dir data/dataset_inference/images \
  --test_label_dir data/dataset_inference/labels \
  --conf_thresh 0.25 --nms_thresh 0.40 --batch_size 8 \
  --run_name "paper_final_results"

  # Single image inference:
  python test_pipeline.py --model runs/detect/train/weights/best_mcu.pt \
      --image path/to/image.jpg

  # Inference on folder of images (no labels):
  python test_pipeline.py --model runs/detect/train/weights/best_mcu.pt \
      --image_dir path/to/images/

Output structure:
  runs_test/
  ├── plots/
  │   ├── confusion_matrix.png
  │   ├── confusion_matrix_normalized.png
  │   ├── precision_recall_curve.png
  │   ├── f1_confidence_curve.png
  │   ├── precision_confidence_curve.png
  │   ├── recall_confidence_curve.png
  │   ├── label_heatmap.png
  │   └── label_scatter.png
  ├── images/
  │   ├── test_batch0_labels.jpg
  │   ├── test_batch0_pred.jpg
  │   └── ...
  ├── detections/           # Individual detection images
  │   ├── image1_det.jpg
  │   └── ...
  ├── logs/
  │   └── test_summary.txt
  └── results/
      ├── test_metrics.csv
      └── per_class_results.csv
"""

import os
import sys
import time
import json
import csv
import math
import argparse
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── project imports ──
from model import MCUDetector
from utils import (
    CLASSES, NUM_CLASSES, decode_predictions, calculate_map,
    compute_precision_recall_curves, save_precision_recall_curves,
)
from dataset import MCUDetectionDataset, detection_collate_fn, IMG_EXTS
from torch.utils.data import DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore", category=UserWarning)

IMG_SIZE = 512
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Color palette for 14 classes (BGR for cv2)
COLORS_BGR = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255),
]

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: MODEL COMPLEXITY — FLOPs, Parameters, Model Size
# ═══════════════════════════════════════════════════════════════════════

def count_parameters_detailed(model):
    """Count parameters by module type."""
    total = 0
    trainable = 0
    by_module = defaultdict(int)
    
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        
        # Categorize
        if "backbone" in name:
            by_module["Backbone"] += n
        elif "fpn" in name or "bifpn" in name:
            by_module["BiFPN Neck"] += n
        elif "head" in name:
            by_module["Detection Head"] += n
        else:
            by_module["Other"] += n
    
    return total, trainable, dict(by_module)


def estimate_flops(model, img_size=512, device="cpu"):
    """
    Estimate FLOPs using hooks on Conv2d and Linear layers.
    Works on any PyTorch version without external packages.
    """
    flops_list = []
    
    def conv_hook(module, inp, out):
        # FLOPs = 2 * Cout * Hout * Wout * Cin * Kh * Kw / groups
        batch = out.shape[0]
        out_channels = out.shape[1]
        out_h, out_w = out.shape[2], out.shape[3]
        
        in_channels = module.in_channels
        kh, kw = module.kernel_size
        groups = module.groups
        
        # Multiply-accumulate ops
        flops = 2 * out_channels * out_h * out_w * (in_channels // groups) * kh * kw
        if module.bias is not None:
            flops += out_channels * out_h * out_w
        flops_list.append(flops)
    
    def linear_hook(module, inp, out):
        flops = 2 * module.in_features * module.out_features
        if module.bias is not None:
            flops += module.out_features
        flops_list.append(flops)
    
    def bn_hook(module, inp, out):
        # BN: 2 * elements (normalize + scale-shift)
        flops_list.append(2 * out.numel())
    
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            hooks.append(m.register_forward_hook(bn_hook))
    
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        model(dummy)
    
    for h in hooks:
        h.remove()
    
    total_flops = sum(flops_list)
    return total_flops


def get_model_size_mb(model):
    """Get model size in MB (state_dict serialized)."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(model.state_dict(), tmp.name)
    size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
    os.unlink(tmp.name)
    return size_mb


def measure_inference_time(model, device, img_size=512, warmup=50, runs=200):
    """Measure average inference time in milliseconds."""
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    
    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
        "fps": float(1000.0 / np.mean(times)),
    }


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: VISUALIZATION — Batch images, Detection overlays
# ═══════════════════════════════════════════════════════════════════════

def draw_boxes_on_image(img_rgb, boxes, class_names, mode="labels", img_size=512):
    """
    Draw bounding boxes on image.
    
    Args:
        img_rgb: numpy array (H, W, 3) in RGB, float [0,1] or uint8 [0,255]
        boxes: 
          mode="labels": list/array of [cls, cx, cy, w, h] normalized
          mode="preds": list/array of [cls, conf, x1, y1, x2, y2] pixels
        class_names: list of class name strings
        mode: "labels" for ground truth, "preds" for predictions
    
    Returns:
        img_bgr: numpy array (H, W, 3) in BGR uint8
    """
    if img_rgb.dtype == np.float32 or img_rgb.dtype == np.float64:
        img_bgr = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    H, W = img_bgr.shape[:2]
    
    if boxes is None or (isinstance(boxes, np.ndarray) and boxes.size == 0):
        return img_bgr
    
    for box in boxes:
        if mode == "labels":
            # [cls, cx, cy, w, h] normalized
            cls_id = int(box[0])
            cx, cy, bw, bh = float(box[1]), float(box[2]), float(box[3]), float(box[4])
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            label_text = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
            conf_text = ""
        else:
            # [cls, conf, x1, y1, x2, y2] pixels
            cls_id = int(box[0])
            conf = float(box[1])
            x1, y1, x2, y2 = int(box[2]), int(box[3]), int(box[4]), int(box[5])
            label_text = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
            conf_text = f" {conf:.2f}"
        
        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        
        color = COLORS_BGR[cls_id % len(COLORS_BGR)]
        thickness = 2
        
        # Draw box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
        
        # Label background
        text = f"{label_text}{conf_text}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
        
        # Text background
        cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_bgr, text, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img_bgr


def save_batch_mosaic(images, targets_or_preds, class_names, save_path, 
                      mode="labels", max_images=16, img_size=512):
    """
    Create YOLOv8-style batch mosaic image.
    
    Args:
        images: tensor (B, C, H, W) or list of numpy arrays
        targets_or_preds: list of arrays per image
        class_names: list of class names
        save_path: output file path
        mode: "labels" or "preds"
        max_images: max images to show
    """
    if isinstance(images, torch.Tensor):
        B = min(images.shape[0], max_images)
        imgs_np = []
        for i in range(B):
            img = images[i].cpu()
            # Unnormalize
            mean = torch.tensor(MEAN).view(3, 1, 1)
            std = torch.tensor(STD).view(3, 1, 1)
            img = img * std + mean
            img = img.clamp(0, 1).permute(1, 2, 0).numpy()
            imgs_np.append(img)
    else:
        B = min(len(images), max_images)
        imgs_np = images[:B]
    
    # Grid layout
    cols = min(4, B)
    rows = math.ceil(B / cols)
    
    cell_h, cell_w = img_size, img_size
    mosaic = np.full((rows * cell_h, cols * cell_w, 3), 114, dtype=np.uint8)
    
    for i in range(B):
        row, col = divmod(i, cols)
        
        img_rgb = imgs_np[i]
        if img_rgb.shape[0] != cell_h or img_rgb.shape[1] != cell_w:
            if img_rgb.dtype == np.float32 or img_rgb.dtype == np.float64:
                img_rgb = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
            img_rgb = cv2.resize(img_rgb, (cell_w, cell_h))
            img_rgb = img_rgb.astype(np.float32) / 255.0
        
        boxes = targets_or_preds[i] if i < len(targets_or_preds) else None
        img_drawn = draw_boxes_on_image(img_rgb, boxes, class_names, mode=mode, img_size=img_size)
        
        y0, x0 = row * cell_h, col * cell_w
        mosaic[y0:y0 + cell_h, x0:x0 + cell_w] = img_drawn
    
    cv2.imwrite(str(save_path), mosaic)


def save_single_detection(img_path, predictions, class_names, save_path, img_size=512):
    """Save individual detection result image."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return
    
    h0, w0 = img_bgr.shape[:2]
    
    for box in predictions:
        cls_id = int(box[0])
        conf = float(box[1])
        # Predictions are in pixel coords relative to img_size
        x1 = int(box[2] / img_size * w0)
        y1 = int(box[3] / img_size * h0)
        x2 = int(box[4] / img_size * w0)
        y2 = int(box[5] / img_size * h0)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w0, x2), min(h0, y2)
        
        color = COLORS_BGR[cls_id % len(COLORS_BGR)]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        
        label = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        text = f"{label} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.6, 1)
        cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_bgr, text, (x1 + 2, y1 - 4), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imwrite(str(save_path), img_bgr)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: PLOT FUNCTIONS — Confusion matrix, PR, F1, Heatmap
# ═══════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, labels, save_dir, base_title="Confusion Matrix"):
    """Plot both raw and normalized confusion matrices."""
    from sklearn.metrics import confusion_matrix as sklearn_cm
    
    all_labels = sorted(set(y_true) | set(y_pred))
    cm = sklearn_cm(y_true, y_pred, labels=all_labels)
    
    # Filter to only labels that appear
    active = [i for i, idx in enumerate(all_labels) if idx < len(labels)]
    active_labels = [labels[all_labels[i]] for i in active]
    cm_active = cm[np.ix_(active, active)]
    
    # Raw confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_active, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(base_title, fontsize=14)
    plt.colorbar(im)
    
    tick_marks = np.arange(len(active_labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(active_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(active_labels, fontsize=8)
    
    # Numbers on cells
    thresh = cm_active.max() / 2.0
    for i in range(cm_active.shape[0]):
        for j in range(cm_active.shape[1]):
            ax.text(j, i, format(cm_active[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_active[i, j] > thresh else "black",
                    fontsize=7)
    
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Normalized
    cm_norm = cm_active.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_title(f"{base_title} Normalized", fontsize=14)
    plt.colorbar(im)
    
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(active_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(active_labels, fontsize=8)
    
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=7)
    
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix_normalized.png"), dpi=200, bbox_inches='tight')
    plt.close()


def plot_f1_confidence(all_preds, all_targets, class_names, save_dir, img_size=512):
    """Plot F1 vs Confidence threshold for each class."""
    conf_thresholds = np.linspace(0.0, 1.0, 101)
    num_classes = len(class_names)
    
    # Collect all predictions and targets
    all_p = []
    all_t = []
    for pred, target in zip(all_preds, all_targets):
        if isinstance(pred, np.ndarray) and pred.shape[0] > 0:
            all_p.append(pred)
        if isinstance(target, np.ndarray) and target.shape[0] > 0:
            # Convert target to pixel format: [cls, x1, y1, x2, y2]
            t = target.copy()
            cx, cy, w, h = t[:, 1]*img_size, t[:, 2]*img_size, t[:, 3]*img_size, t[:, 4]*img_size
            t_xyxy = np.stack([t[:, 0], cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=1)
            all_t.append(t_xyxy)
    
    if not all_p:
        return
    
    preds_cat = np.concatenate(all_p, axis=0)  # [cls, conf, x1, y1, x2, y2]
    targets_cat = np.concatenate(all_t, axis=0) if all_t else np.zeros((0, 5))
    
    f1_per_class = np.zeros((num_classes, len(conf_thresholds)))
    f1_all = np.zeros(len(conf_thresholds))
    
    for ci, thresh in enumerate(conf_thresholds):
        mask = preds_cat[:, 1] >= thresh
        filtered = preds_cat[mask]
        
        tp_total, fp_total, fn_total = 0, 0, 0
        
        for cls_id in range(num_classes):
            cls_preds = filtered[filtered[:, 0] == cls_id]
            cls_gts = targets_cat[targets_cat[:, 0] == cls_id] if targets_cat.shape[0] > 0 else np.zeros((0, 5))
            
            tp = min(len(cls_preds), len(cls_gts))  # simplified
            fp = max(0, len(cls_preds) - tp)
            fn = max(0, len(cls_gts) - tp)
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f1_per_class[cls_id, ci] = f1
            
            tp_total += tp
            fp_total += fp
            fn_total += fn
        
        p_all = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        r_all = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1_all[ci] = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    for cls_id in range(num_classes):
        ax.plot(conf_thresholds, f1_per_class[cls_id], linewidth=1, alpha=0.7,
                label=class_names[cls_id])
    
    best_idx = np.argmax(f1_all)
    best_f1 = f1_all[best_idx]
    best_conf = conf_thresholds[best_idx]
    ax.plot(conf_thresholds, f1_all, linewidth=3, color='navy',
            label=f'all classes {best_f1:.2f} at {best_conf:.3f}')
    
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("F1", fontsize=12)
    ax.set_title("F1-Confidence Curve", fontsize=14)
    ax.legend(fontsize=7, loc='lower left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "f1_confidence_curve.png"), dpi=200, bbox_inches='tight')
    plt.close()


def plot_label_distribution(all_targets, class_names, save_dir, img_size=512):
    """Plot label heatmap and scatter + class distribution bar chart."""
    centers = []
    class_counts = Counter()
    
    for target in all_targets:
        if isinstance(target, np.ndarray) and target.shape[0] > 0:
            for row in target:
                cls_id = int(row[0])
                cx, cy = float(row[1]), float(row[2])
                centers.append((cx, cy))
                class_counts[cls_id] += 1
    
    if not centers:
        return
    
    centers = np.array(centers)
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    heatmap, xedges, yedges = np.histogram2d(centers[:, 0], centers[:, 1], bins=50, range=[[0,1],[0,1]])
    ax.imshow(heatmap.T, origin='lower', cmap='YlOrRd', extent=[0,1,0,1], aspect='auto')
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized)")
    ax.set_title("Label Center Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "label_heatmap.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(centers[:, 0], centers[:, 1], s=3, alpha=0.5)
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized)")
    ax.set_title("Label Center Scatter")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "label_scatter.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Class distribution bar chart (EXTRA for paper)
    fig, ax = plt.subplots(figsize=(12, 6))
    classes_sorted = sorted(class_counts.keys())
    counts = [class_counts.get(c, 0) for c in classes_sorted]
    names = [class_names[c] if c < len(class_names) else str(c) for c in classes_sorted]
    
    bars = ax.bar(range(len(names)), counts, color=[
        f'#{COLORS_BGR[i % len(COLORS_BGR)][2]:02x}'
        f'{COLORS_BGR[i % len(COLORS_BGR)][1]:02x}'
        f'{COLORS_BGR[i % len(COLORS_BGR)][0]:02x}'
        for i in range(len(names))
    ])
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Number of Instances")
    ax.set_title("Test Set Class Distribution")
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_distribution.png"), dpi=200, bbox_inches='tight')
    plt.close()


def plot_per_class_ap_bar(per_class_ap, class_names, save_dir):
    """Plot per-class AP as horizontal bar chart (paper figure)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    aps = []
    names = []
    for cls_id in range(len(class_names)):
        ap = per_class_ap.get(cls_id, 0.0)
        aps.append(ap)
        names.append(class_names[cls_id])
    
    y_pos = np.arange(len(names))
    colors = ['#2ecc71' if ap > 0.95 else '#f39c12' if ap > 0.85 else '#e74c3c' for ap in aps]
    
    bars = ax.barh(y_pos, aps, color=colors, edgecolor='gray', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Average Precision (AP@0.5)", fontsize=12)
    ax.set_title("Per-Class AP@0.5", fontsize=14)
    ax.set_xlim(0, 1.05)
    ax.axvline(x=np.mean(aps), color='navy', linestyle='--', linewidth=1.5,
               label=f'mAP@0.5 = {np.mean(aps):.4f}')
    ax.legend(fontsize=10)
    
    for bar, ap in zip(bars, aps):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{ap:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_class_ap.png"), dpi=200, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: FULL TEST EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def run_full_test(model, test_loader, device, run_dir, class_names,
                  conf_thresh=0.05, nms_thresh=0.40, img_size=512):
    """
    Complete test evaluation with all metrics and plots.
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_box_centers = []
    pred_counter = Counter()
    
    # For batch visualization
    saved_batches = 0
    max_batches_to_save = 4
    
    images_dir = Path(run_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"RUNNING TEST EVALUATION")
    print(f"{'='*60}")
    print(f"  Conf threshold: {conf_thresh}")
    print(f"  NMS threshold:  {nms_thresh}")
    print(f"  Test batches:   {len(test_loader)}")
    print(f"  Image size:     {img_size}")
    print()
    
    def _to_decoder_format(scale_pred):
        if isinstance(scale_pred, (tuple, list)) and len(scale_pred) == 3:
            obj, cls, reg = scale_pred
            return (torch.cat([obj, cls], dim=1), reg)
        if isinstance(scale_pred, (tuple, list)) and len(scale_pred) == 2:
            return scale_pred
        raise ValueError("Unexpected pred format")
    
    total_images = 0
    start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(test_loader):
        images = images.to(device)
        B = images.shape[0]
        total_images += B
        
        # Forward
        with torch.no_grad():
            pred = model(images)
        
        # Decode predictions
        try:
            p3_dec = _to_decoder_format(pred[0])
            p4_dec = _to_decoder_format(pred[1])
            decoded = decode_predictions(
                p3_dec, p4_dec,
                conf_thresh=conf_thresh,
                nms_thresh=nms_thresh,
                img_size=img_size
            )
        except Exception as e:
            print(f"  ⚠️ Decode failed batch {batch_idx}: {e}")
            decoded = [np.zeros((0, 6), dtype=np.float32) for _ in range(B)]
        
        # Process each image in batch
        batch_targets = []
        for i, t in enumerate(targets):
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                gt_np = t.detach().cpu().numpy().astype(np.float32)
            else:
                gt_np = np.zeros((0, 5), dtype=np.float32)
            batch_targets.append(gt_np)
            
            # Collect box centers
            if gt_np.shape[0] > 0:
                all_box_centers.extend(gt_np[:, 1:3].tolist())
        
        # Align lengths
        while len(decoded) < len(batch_targets):
            decoded.append(np.zeros((0, 6), dtype=np.float32))
        decoded = decoded[:len(batch_targets)]
        
        for pb, gt in zip(decoded, batch_targets):
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
            
            if pb.shape[0] > 0:
                pred_counter.update([int(x[0]) for x in pb])
        
        # Save batch visualization (first N batches)
        if saved_batches < max_batches_to_save:
            try:
                # Labels mosaic
                save_batch_mosaic(
                    images, batch_targets, class_names,
                    images_dir / f"test_batch{saved_batches}_labels.jpg",
                    mode="labels", img_size=img_size
                )
                # Predictions mosaic
                save_batch_mosaic(
                    images, decoded, class_names,
                    images_dir / f"test_batch{saved_batches}_pred.jpg",
                    mode="preds", img_size=img_size
                )
                saved_batches += 1
            except Exception as e:
                print(f"  ⚠️ Batch visualization failed: {e}")
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches ({total_images} images)")
    
    total_time = time.time() - start_time
    print(f"\n  ✅ Processed {total_images} images in {total_time:.1f}s")
    
    # ── Calculate all metrics ──
    print(f"\n{'='*60}")
    print(f"COMPUTING METRICS")
    print(f"{'='*60}")
    
    # mAP computation
    map_50_95, map_50, map_75, per_class_ap = calculate_map(
        predictions=all_preds,
        targets=all_targets,
        num_classes=len(class_names),
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        epoch=999,  # High epoch to avoid warmup behavior
        img_size=img_size,
        conf_thresh=0.001
    )
    
    # Precision / Recall / F1 at conf_thresh
    tp, fp, fn = 0, 0, 0
    for pred, target in zip(all_preds, all_targets):
        n_pred = pred.shape[0] if isinstance(pred, np.ndarray) else 0
        n_gt = target.shape[0] if isinstance(target, np.ndarray) else 0
        matched = min(n_pred, n_gt)
        tp += matched
        fp += max(0, n_pred - matched)
        fn += max(0, n_gt - matched)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "mAP@0.5": map_50,
        "mAP@0.75": map_75,
        "mAP@0.5:0.95": map_50_95,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Total Predictions": sum(pred_counter.values()),
        "Total GT": sum(t.shape[0] for t in all_targets if isinstance(t, np.ndarray)),
        "Total Images": total_images,
    }
    
    # ── Generate all plots ──
    print(f"\n{'='*60}")
    print(f"GENERATING PLOTS")
    print(f"{'='*60}")
    
    plots_dir = Path(run_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    try:
        y_true, y_pred = [], []
        for gt, pred in zip(all_targets, all_preds):
            gt_ok = isinstance(gt, np.ndarray) and gt.shape[0] > 0
            pred_ok = isinstance(pred, np.ndarray) and pred.shape[0] > 0
            if not gt_ok and not pred_ok:
                continue
            gt_cls = int(gt[0, 0]) if gt_ok else len(class_names)
            pred_cls = int(pred[0, 0]) if pred_ok else len(class_names)
            y_true.append(gt_cls)
            y_pred.append(pred_cls)
        
        if y_true:
            plot_confusion_matrix(y_true, y_pred, list(class_names) + ["background"],
                                  str(plots_dir), "Confusion Matrix - Test")
            print("  ✅ Confusion matrix saved")
    except Exception as e:
        print(f"  ⚠️ Confusion matrix failed: {e}")
    
    # 2. PR Curves
    try:
        confidences, precisions_pr, recalls_pr = compute_precision_recall_curves(
            all_preds=all_preds,
            all_targets=all_targets,
            num_classes=len(class_names),
            img_size=img_size,
            adaptive_iou=True
        )
        save_precision_recall_curves(
            confidences=confidences,
            precisions=precisions_pr,
            recalls=recalls_pr,
            class_names=class_names,
            run_dir=str(plots_dir)
        )
        print("  ✅ Precision-Recall curves saved")
    except Exception as e:
        print(f"  ⚠️ PR curves failed: {e}")
    
    # 3. F1-Confidence
    try:
        plot_f1_confidence(all_preds, all_targets, class_names, str(plots_dir), img_size)
        print("  ✅ F1-Confidence curve saved")
    except Exception as e:
        print(f"  ⚠️ F1-Confidence failed: {e}")
    
    # 4. Label distribution + heatmap
    try:
        plot_label_distribution(all_targets, class_names, str(plots_dir), img_size)
        print("  ✅ Label distribution plots saved")
    except Exception as e:
        print(f"  ⚠️ Label distribution failed: {e}")
    
    # 5. Per-class AP bar chart
    try:
        plot_per_class_ap_bar(per_class_ap, class_names, str(plots_dir))
        print("  ✅ Per-class AP bar chart saved")
    except Exception as e:
        print(f"  ⚠️ Per-class AP bar failed: {e}")
    
    return metrics, per_class_ap


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: SINGLE IMAGE / FOLDER INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def run_single_inference(model, image_path, device, class_names, save_dir,
                         conf_thresh=0.25, nms_thresh=0.40, img_size=512):
    """Run inference on a single image and save detection."""
    model.eval()
    
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"  ❌ Cannot read: {image_path}")
        return []
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_float = img_resized.astype(np.float32) / 255.0
    
    # Apply normalization
    img_tensor = torch.from_numpy(img_float).permute(2, 0, 1)
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    img_tensor = normalize(img_tensor).unsqueeze(0).to(device)
    
    # Forward
    with torch.no_grad():
        pred = model(img_tensor)
    
    # Decode
    def _to_decoder_format(scale_pred):
        if isinstance(scale_pred, (tuple, list)) and len(scale_pred) == 3:
            obj, cls, reg = scale_pred
            return (torch.cat([obj, cls], dim=1), reg)
        return scale_pred
    
    p3_dec = _to_decoder_format(pred[0])
    p4_dec = _to_decoder_format(pred[1])
    decoded = decode_predictions(
        p3_dec, p4_dec,
        conf_thresh=conf_thresh,
        nms_thresh=nms_thresh,
        img_size=img_size
    )
    
    preds = decoded[0] if decoded else np.zeros((0, 6))
    
    # Save detection image
    save_path = Path(save_dir) / f"{Path(image_path).stem}_det.jpg"
    save_single_detection(image_path, preds, class_names, save_path, img_size)
    
    # Print results
    print(f"\n  🎯 {Path(image_path).name}: {len(preds)} detections")
    for p in preds:
        cls_id = int(p[0])
        conf = float(p[1])
        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        print(f"     {name:<40} conf={conf:.3f}")
    
    return preds


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MCUDetector V2-Lite Test Pipeline")
    parser.add_argument("--model", required=True, help="Path to best_mcu.pt checkpoint")
    parser.add_argument("--test_img_dir", type=str, default=None, help="Test images directory")
    parser.add_argument("--test_label_dir", type=str, default=None, help="Test labels directory")
    parser.add_argument("--image", type=str, default=None, help="Single image path for inference")
    parser.add_argument("--image_dir", type=str, default=None, help="Folder of images for inference")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms_thresh", type=float, default=0.40, help="NMS IoU threshold")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--img_size", type=int, default=512, help="Image size")
    parser.add_argument("--run_name", type=str, default="test", help="Run name for output directory")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}")
    print(f"  MCUDetector V2-Lite — Test & Inference Pipeline")
    print(f"{'═'*60}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # ── Create run directory ──
    run_dir = Path("runs_test") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "images").mkdir(exist_ok=True)
    (run_dir / "detections").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)
    
    class_names = list(CLASSES)
    
    # ── Load model ──
    print(f"\n  Loading model from: {args.model}")
    model = MCUDetector(num_classes=len(class_names)).to(device)
    
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        ckpt_epoch = checkpoint.get("epoch", "?")
        ckpt_map = checkpoint.get("best_map", "?")
        print(f"  Checkpoint epoch: {ckpt_epoch}, best_mAP: {ckpt_map}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"  ✅ Model loaded successfully")
    
    # ── Model complexity analysis ──
    print(f"\n{'='*60}")
    print(f"MODEL COMPLEXITY ANALYSIS")
    print(f"{'='*60}")
    
    total_params, trainable_params, params_by_module = count_parameters_detailed(model)
    print(f"  Total parameters:     {total_params:,} ({total_params/1e6:.3f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.3f}M)")
    print(f"\n  Parameter distribution:")
    for module_name, count in sorted(params_by_module.items(), key=lambda x: -x[1]):
        pct = count / total_params * 100
        print(f"    {module_name:<20} {count:>10,} ({pct:.1f}%)")
    
    # FLOPs
    try:
        flops = estimate_flops(model, args.img_size, device)
        gflops = flops / 1e9
        print(f"\n  FLOPs: {flops:,.0f} ({gflops:.2f} GFLOPs)")
    except Exception as e:
        gflops = -1
        print(f"\n  ⚠️ FLOPs estimation failed: {e}")
    
    # Model size
    model_size = get_model_size_mb(model)
    print(f"  Model size: {model_size:.2f} MB")
    
    # Inference speed
    print(f"\n  Measuring inference speed (warmup=50, runs=200)...")
    timing = measure_inference_time(model, device, args.img_size)
    print(f"  Inference time: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")
    print(f"  Throughput: {timing['fps']:.1f} FPS")
    print(f"  Min/Max: {timing['min_ms']:.2f} / {timing['max_ms']:.2f} ms")
    
    # ── Mode: Full test evaluation or single inference ──
    if args.test_img_dir and args.test_label_dir:
        # Full evaluation mode
        print(f"\n{'='*60}")
        print(f"LOADING TEST DATASET")
        print(f"{'='*60}")
        
        val_transform = transforms.Compose([
            transforms.Normalize(mean=MEAN, std=STD),
        ])
        
        test_dataset = MCUDetectionDataset(
            img_dir=args.test_img_dir,
            label_dir=args.test_label_dir,
            img_size=args.img_size,
            transform=val_transform,
            augment=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=detection_collate_fn,
            pin_memory=False
        )
        
        metrics, per_class_ap = run_full_test(
            model, test_loader, device, str(run_dir), class_names,
            conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, img_size=args.img_size
        )
        
        # ── Save comprehensive test summary ──
        summary_path = run_dir / "logs" / "test_summary.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("MCUDetector V2-Lite — TEST EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MODEL ARCHITECTURE:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Architecture:     RepViT-CSP + BiFPN + Decoupled Head\n")
            f.write(f"Parameters:       {total_params:,} ({total_params/1e6:.3f}M)\n")
            f.write(f"FLOPs:            {gflops:.2f} GFLOPs\n")
            f.write(f"Model Size:       {model_size:.2f} MB\n")
            f.write(f"Input Size:       {args.img_size}×{args.img_size}\n")
            f.write(f"Num Classes:      {len(class_names)}\n\n")
            
            f.write("INFERENCE SPEED:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Device:           {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})\n")
            f.write(f"Mean Latency:     {timing['mean_ms']:.2f} ms\n")
            f.write(f"Std Latency:      {timing['std_ms']:.2f} ms\n")
            f.write(f"Throughput:       {timing['fps']:.1f} FPS\n\n")
            
            f.write("DETECTION METRICS:\n")
            f.write("-" * 60 + "\n")
            for k, v in metrics.items():
                if isinstance(v, float):
                    f.write(f"{k:<20} {v:.4f}\n")
                else:
                    f.write(f"{k:<20} {v}\n")
            
            f.write(f"\nPER-CLASS AP@0.5:\n")
            f.write("-" * 60 + "\n")
            for cls_id in range(len(class_names)):
                ap = per_class_ap.get(cls_id, 0.0)
                f.write(f"  {class_names[cls_id]:<40} {ap:.4f}\n")
            
            f.write(f"\nPARAMETER DISTRIBUTION:\n")
            f.write("-" * 60 + "\n")
            for module_name, count in sorted(params_by_module.items(), key=lambda x: -x[1]):
                pct = count / total_params * 100
                f.write(f"  {module_name:<20} {count:>10,} ({pct:.1f}%)\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        # Save metrics CSV
        metrics_csv_path = run_dir / "results" / "test_metrics.csv"
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in metrics.items():
                writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else v])
            writer.writerow(["Parameters (M)", f"{total_params/1e6:.3f}"])
            writer.writerow(["GFLOPs", f"{gflops:.2f}"])
            writer.writerow(["Model Size (MB)", f"{model_size:.2f}"])
            writer.writerow(["Latency (ms)", f"{timing['mean_ms']:.2f}"])
            writer.writerow(["FPS", f"{timing['fps']:.1f}"])
        
        # Save per-class results CSV
        per_class_csv = run_dir / "results" / "per_class_results.csv"
        with open(per_class_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Class ID", "Class Name", "AP@0.5"])
            for cls_id in range(len(class_names)):
                ap = per_class_ap.get(cls_id, 0.0)
                writer.writerow([cls_id, class_names[cls_id], f"{ap:.6f}"])
        
        # Save config
        config = {
            "model_path": str(args.model),
            "test_img_dir": args.test_img_dir,
            "test_label_dir": args.test_label_dir,
            "conf_thresh": args.conf_thresh,
            "nms_thresh": args.nms_thresh,
            "img_size": args.img_size,
            "num_classes": len(class_names),
            "classes": class_names,
            "parameters": total_params,
            "gflops": gflops,
            "model_size_mb": model_size,
            "metrics": {k: float(v) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        with open(run_dir / "test_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Print final results
        print(f"\n{'═'*60}")
        print(f"  FINAL TEST RESULTS")
        print(f"{'═'*60}")
        print(f"  mAP@0.5:      {metrics['mAP@0.5']:.4f}")
        print(f"  mAP@0.75:     {metrics['mAP@0.75']:.4f}")
        print(f"  mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
        print(f"  Precision:    {metrics['Precision']:.4f}")
        print(f"  Recall:       {metrics['Recall']:.4f}")
        print(f"  F1:           {metrics['F1']:.4f}")
        print(f"")
        print(f"  Parameters:   {total_params/1e6:.3f}M")
        print(f"  FLOPs:        {gflops:.2f} GFLOPs")
        print(f"  Model Size:   {model_size:.2f} MB")
        print(f"  Latency:      {timing['mean_ms']:.2f} ms ({timing['fps']:.1f} FPS)")
        print(f"")
        print(f"  📁 Results saved to: {run_dir}")
        print(f"{'═'*60}")
    
    elif args.image:
        # Single image inference
        preds = run_single_inference(
            model, args.image, device, class_names,
            str(run_dir / "detections"),
            conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, img_size=args.img_size
        )
    
    elif args.image_dir:
        # Folder inference
        img_dir = Path(args.image_dir)
        image_files = []
        for ext in IMG_EXTS:
            image_files.extend(img_dir.glob(f"*{ext}"))
        image_files = sorted(image_files)
        
        print(f"\n  Found {len(image_files)} images in {img_dir}")
        
        all_results = {}
        for img_path in image_files:
            preds = run_single_inference(
                model, str(img_path), device, class_names,
                str(run_dir / "detections"),
                conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, img_size=args.img_size
            )
            all_results[img_path.name] = len(preds)
        
        print(f"\n  ✅ Processed {len(image_files)} images")
        print(f"  📁 Detections saved to: {run_dir / 'detections'}")
    
    else:
        print("\n  ❌ Please specify one of:")
        print("     --test_img_dir + --test_label_dir  (full evaluation)")
        print("     --image <path>                     (single image)")
        print("     --image_dir <path>                 (folder inference)")


if __name__ == "__main__":
    main()