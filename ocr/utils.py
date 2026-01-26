import math
import os
import re
import string
import cv2
from matplotlib.patches import Rectangle
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, precision_score, recall_score
import csv
import difflib
import itertools
import torch.nn as nn
from typing import List, Optional, Sequence, Tuple, Any
from pathlib import Path

def _pil_to_cv2(img: Image.Image) -> np.ndarray:#will check later used by ocr
    arr = np.array(img)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def _ensure_cv2_array(img: Any) -> np.ndarray:#will check later used by ocr
    if isinstance(img, Image.Image):
        return _pil_to_cv2(img)
    if isinstance(img, np.ndarray):
        return img
    raise TypeError("Unsupported image type")

#will check later used by ocr
CHARS = string.ascii_lowercase + string.digits + '_'
char2idx = {char: i for i, char in enumerate(CHARS)}
idx2char = {i: char for char, i in char2idx.items()}
BLANK_IDX = len(CHARS)

CLASSES = [#checked ok
    "8051",
    "ARDUINO_NANO_ATMEGA328P",
    "ARMCORTEXM3",
    "ARMCORTEXM7",
    "ESP32_DEVKIT",
    "NODEMCU_ESP8266",
    "RASPBERRY_PI_3B_PLUS"
    # "Arduino",
    # "Pico",
    # "RaspberryPi",
    # "Arduino Due",
    # "Arduino Leonardo",
    # "Arduino Mega 2560 -Black and Yellow-",
    # "Arduino Mega 2560 -Black-",
    # "Arduino Mega 2560 -Blue-",
    # "Arduino Uno -Black-",
    # "Arduino Uno -Green-",
    # "Arduino Uno Camera Shield",
    # "Arduino Uno R3",
    # "Arduino Uno WiFi Shield",
    # "Beaglebone Black",
    # "Raspberry Pi 1 B-",
    # "Raspberry Pi 3 B-",
    # "Raspberry Pi A-"
]

VALID_LABELS = [
    "8051",
    "arduinonanoatmega328p", "armcortexm3", "armcortexm7", "esp32devkit",
    "nodemcuesp8266", "raspberrypi3bplus" 
    # "arduino", "pico", "raspberrypi",
    # "arduinodue", "arduinoleonardo", "arduinomega2560blackandyellow",
    # "arduinomega2560black", "arduinomega2560blue", "arduinounoblack",
    # "arduinounogreen", "arduinounocamerashield", "arduinounor3",
    # "arduinounowifishield", "beagleboneblack", "raspberrypi1b",
    # "raspberrypi3b", "raspberrypia"
]

NUM_CLASSES = len(CLASSES)#checked ok

# =================== DETECTION DECODING ===================

def decode_predictions(pred_p3, pred_p4, pred_p5=None, conf_thresh=0.01, nms_thresh=0.45, img_size=512):
    # Unpack tuples from model output
    cls_p3, reg_p3 = pred_p3  # NEW: P3 scale
    cls_p4, reg_p4 = pred_p4
    # cls_p5, reg_p5 = pred_p5
    
    batch_size = cls_p4.shape[0]
    device = cls_p4.device
    
    all_predictions = []
    
    for i in range(batch_size):
        # Decode P3 (stride=4, highest resolution) - NEW
        boxes_p3 = decode_single_scale(
            cls_p3[i], reg_p3[i],
            stride=4, conf_thresh=conf_thresh,  # stride=4 for small objects
            img_size=img_size, device=device
        )
        
        # Decode P4 (stride=8, higher resolution)
        boxes_p4 = decode_single_scale(
            cls_p4[i], reg_p4[i],
            stride=8, conf_thresh=conf_thresh,
            img_size=img_size, device=device
        )
        
        # Decode P5 (stride=16, lower resolution)//will remove this later 
        # boxes_p5 = decode_single_scale(
        #     # cls_p5[i], reg_p5[i],
        #     stride=16, conf_thresh=conf_thresh,
        #     img_size=img_size, device=device
        # )
        
        # Combine boxes from all three scales - UPDATED
        all_boxes = []
        if len(boxes_p3) > 0:
            all_boxes.append(boxes_p3)
        if len(boxes_p4) > 0:
            all_boxes.append(boxes_p4)
        # if len(boxes_p5) > 0:
        #     all_boxes.append(boxes_p5)
        
        if len(all_boxes) > 0:
            boxes = torch.cat(all_boxes, dim=0)
        else:
            boxes = torch.zeros((0, 6), device=device)
        
        # --------------------------------------------------
        # 🔥 CRITICAL: LIMIT BOX COUNT BEFORE NMS (YOLOv8-style)
        # --------------------------------------------------
        MAX_DETECTIONS = 300  # YOLOv8 default (safe for 2080 Ti)

        if boxes.shape[0] > MAX_DETECTIONS:
            scores = boxes[:, 1]                 # confidence column
            topk = torch.topk(scores, MAX_DETECTIONS).indices
            boxes = boxes[topk]

        # --------------------------------------------------
        # NMS (CPU ONLY)
        # --------------------------------------------------
        if boxes.shape[0] > 0:
            boxes = boxes.detach().cpu()
            boxes = non_max_suppression(boxes, nms_thresh)
        
        all_predictions.append(boxes)
    
    return all_predictions


def decode_single_scale(cls_map, reg_map, stride, conf_thresh, img_size, device):#checked ok
    H, W = cls_map.shape[1:]
    
    # Extract objectness and class scores
    obj_scores = torch.sigmoid(cls_map[0:1, :, :])  # (1, H, W)
    cls_scores = torch.sigmoid(cls_map[1:, :, :])   # (num_classes, H, W)
    
    # Compute confidence: obj * cls
    conf = obj_scores * cls_scores  # (num_classes, H, W)
    
    # Get max confidence and class per grid cell
    max_conf, max_cls = conf.max(dim=0)  # (H, W)
    
    # Filter by confidence threshold
    mask = max_conf > conf_thresh
    
    if not mask.any():
        return torch.zeros((0, 6), device=device)
    
    # Get grid coordinates
    gy, gx = torch.where(mask)
    
    # Get regression outputs for selected cells
    dx = torch.sigmoid(reg_map[0, gy, gx])
    dy = torch.sigmoid(reg_map[1, gy, gx])
    dw = torch.exp(reg_map[2, gy, gx].clamp(-4.0, 4.0))
    dh = torch.exp(reg_map[3, gy, gx].clamp(-4.0, 4.0))
    
    # Compute box centers in pixel coordinates
    cx = (gx.float() + dx) * stride
    cy = (gy.float() + dy) * stride
    
    # Compute box dimensions
    w = dw * stride
    h = dh * stride
    
    # Convert to corner format [x1, y1, x2, y2]
    x1 = (cx - w / 2).clamp(0, img_size)
    y1 = (cy - h / 2).clamp(0, img_size)
    x2 = (cx + w / 2).clamp(0, img_size)
    y2 = (cy + h / 2).clamp(0, img_size)

    # --- ADD THIS ---
    eps = 1.0
    x2 = torch.max(x2, x1 + eps)
    y2 = torch.max(y2, y1 + eps)
    x2 = x2.clamp(0, img_size)
    y2 = y2.clamp(0, img_size)
    # ----------------
    
    # Get class IDs and confidences for selected cells
    class_ids = max_cls[mask].float()
    confidences = max_conf[mask]
    
    # Stack into output format: [class_id, confidence, x1, y1, x2, y2]
    result = torch.stack([class_ids, confidences, x1, y1, x2, y2], dim=1)
    #remove later
    if result.numel() > 0:
        print("DEBUG box area min:",
            ((result[:,4]-result[:,2]) * (result[:,5]-result[:,3])).min().item())

    
    return result


def non_max_suppression(boxes, iou_thresh):#checked ok
    """Apply NMS to boxes."""
    if len(boxes) == 0:
        return boxes
    boxes = boxes[boxes[:, 1].argsort(descending=True)]
    keep = []
    while len(boxes) > 0:
        box = boxes[0]
        keep.append(box.unsqueeze(0))
        if len(boxes) == 1:
            break
        ious = calculate_iou(box[2:], boxes[1:, 2:])
        mask = ious < iou_thresh
        boxes = boxes[1:][mask]
    return torch.cat(keep, dim=0) if keep else torch.zeros((0, 6), device=boxes.device)


def calculate_iou(box, boxes):#checked ok
    """Calculate IoU between one box and multiple boxes."""
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    inter_x1 = torch.max(box[0], boxes[:, 0])
    inter_y1 = torch.max(box[1], boxes[:, 1])
    inter_x2 = torch.min(box[2], boxes[:, 2])
    inter_y2 = torch.min(box[3], boxes[:, 3])
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    union_area = box_area + boxes_area - inter_area + 1e-6
    return inter_area / union_area


# =================== PROPER mAP CALCULATION ===================

def box_iou_batch(boxes1, boxes2):#checked ok
    # --- ADD THIS GUARD ---
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]),
            device=boxes1.device if boxes1.numel() else boxes2.device
        )

    # --- CHANGE: clamp widths/heights ---
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * \
            (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)

    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * \
            (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    # Expand dimensions for broadcasting
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)

    return iou

def compute_epoch_precision_recall(
    preds, targets,
    conf_thresh=0.01,#change after sanity
    iou_thresh=0.5,
    img_size=512
):
    """
    Compute dataset-level precision and recall with adaptive IoU for small objects.
    preds: List[np.ndarray] (N_i,6) -> cls, conf, x1,y1,x2,y2
    targets: List[np.ndarray] (M_i,5) -> cls, cx,cy,w,h
    """
    TP = FP = FN = 0
    eps = 1e-12

    for p_img, t_img in zip(preds, targets):

        # ---- Ground truth ----
        if t_img.shape[0] > 0:
            cx = t_img[:, 1] * img_size
            cy = t_img[:, 2] * img_size
            w  = t_img[:, 3] * img_size
            h  = t_img[:, 4] * img_size
            gt_boxes = np.stack(
                [cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1
            )
            gt_cls = t_img[:, 0].astype(int)
            
            # ← NEW: Adaptive IoU based on box size
            gt_areas = w * h
            adaptive_iou = np.where(
                gt_areas < 16 * 16,
                iou_thresh * 0.7,     # Very small: 0.35 if iou_thresh=0.5
                np.where(
                    gt_areas < 32 * 32,
                    iou_thresh * 0.85,  # Small: 0.425
                    iou_thresh          # Normal: 0.5
                )
            )
        else:
            gt_boxes = np.zeros((0, 4))
            gt_cls = np.array([])
            adaptive_iou = np.array([])

        # ---- Predictions ----
        if p_img.shape[0] > 0:
            mask = p_img[:, 1] >= conf_thresh
            p_img = p_img[mask]
        else:
            p_img = np.zeros((0, 6))

        if p_img.shape[0] == 0:
            FN += len(gt_boxes)
            continue

        pred_boxes = p_img[:, 2:6]
        pred_cls = p_img[:, 0].astype(int)

        if gt_boxes.shape[0] == 0:
            FP += len(pred_boxes)
            continue

        ious = box_iou_batch(
            torch.from_numpy(pred_boxes),
            torch.from_numpy(gt_boxes)
        ).numpy()

        matched_gt = set()
        for i in range(ious.shape[0]):
            # ← NEW: Use adaptive threshold for each GT box
            valid_matches = ious[i] >= adaptive_iou
            if not valid_matches.any():
                FP += 1
                continue
                
            # Among valid matches, find best IoU
            best = int(np.argmax(ious[i]))
            if (
                ious[i, best] >= adaptive_iou[best] and
                best not in matched_gt and
                pred_cls[i] == gt_cls[best]
            ):
                TP += 1
                matched_gt.add(best)
            else:
                FP += 1

        FN += max(0, len(gt_boxes) - len(matched_gt))

    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    return float(precision), float(recall)

def compute_ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):#checked ok

    # Ensure numpy arrays and 1D
    tp = np.asarray(tp).ravel().astype(float)
    conf = np.asarray(conf).ravel().astype(float)
    pred_cls = np.asarray(pred_cls).ravel()
    target_cls = np.asarray(target_cls).ravel()

    # quick guard
    if conf.size == 0 or tp.size == 0 or pred_cls.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # Remove NaNs/Infs from conf (and corresponding entries)
    valid_mask = np.isfinite(conf)
    if not valid_mask.all():
        tp = tp[valid_mask]
        pred_cls = pred_cls[valid_mask]
        conf = conf[valid_mask]

    # Sort by confidence (descending)
    order = np.argsort(-conf)
    tp = tp[order]
    conf = conf[order]
    pred_cls = pred_cls[order]

    # Unique classes present in targets (we compute AP only for classes that have GT)
    unique_classes = np.unique(target_cls)
    n_classes = len(unique_classes)
    ap = np.zeros(n_classes, dtype=float)

    p_curve_list = []
    r_curve_list = []

    for ci, c in enumerate(unique_classes):
        # Predictions that are predicted as class c (order is preserved => descending conf)
        idxs = (pred_cls == c)
        n_p = idxs.sum()
        n_gt = int((target_cls == c).sum())

        if n_p == 0 or n_gt == 0:
            # leave ap[ci] = 0 (or NaN if you prefer)
            continue

        tp_c = tp[idxs].astype(int)  # ensure binary
        fpc = (1 - tp_c).cumsum()
        tpc = tp_c.cumsum()

        recall = tpc / (n_gt + eps)
        precision = tpc / (tpc + fpc + eps)

        p_curve_list.append(precision)
        r_curve_list.append(recall)

        ap[ci] = compute_ap(recall, precision)

    # Concatenate curves if needed
    p_curve_all = np.concatenate(p_curve_list, 0) if p_curve_list else np.array([])
    r_curve_all = np.concatenate(r_curve_list, 0) if r_curve_list else np.array([])
    f1_curve_all = 2 * p_curve_all * r_curve_all / (p_curve_all + r_curve_all + eps) if p_curve_all.size else np.array([])

    return ap, p_curve_all, r_curve_all, f1_curve_all, unique_classes


def compute_ap(recall, precision, method='interp', eps=1e-12):#checked ok
    
    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)

    if recall.size == 0 or precision.size == 0 or recall.shape[0] != precision.shape[0]:
        return 0.0

    # Clip/nan-safe
    recall = np.nan_to_num(recall, nan=0.0, posinf=1.0, neginf=0.0)
    precision = np.nan_to_num(precision, nan=0.0, posinf=0.0, neginf=0.0)

    # Append sentinels
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Precision envelope (monotonic)
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    if method == 'interp':
        x = np.linspace(0.0, 1.0, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    else:  # continuous
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return float(ap + eps) if ap == 0.0 else float(ap)

def calculate_map(predictions, targets, num_classes,
                  iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                  img_size=512):
    """
    Calculate mAP with adaptive IoU thresholds for small objects.
    Small objects (<32x32) use relaxed IoU thresholds.
    """
    
    all_predictions = []
    all_targets = []

    # Collect and normalize per-image entries
    for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
        # ---- predictions ----
        if isinstance(pred, torch.Tensor):
            if pred.numel() == 0:
                pred = pred.cpu().numpy().reshape(0, 6)
            else:
                pred = pred.cpu().numpy()
        else:
            pred = np.array(pred).reshape(-1, 6) if len(pred) > 0 else np.zeros((0, 6))

        # ---- targets (YOLO cx,cy,w,h -> x1,y1,x2,y2) ----
        if isinstance(target, torch.Tensor):
            if target.numel() == 0:
                target = target.cpu().numpy().reshape(0, 5)
            else:
                target = target.cpu().numpy()
        else:
            target = np.array(target).reshape(-1, 5) if len(target) > 0 else np.zeros((0, 5))

        if target.shape[0] > 0:
            tboxes = np.zeros((target.shape[0], 5))
            tboxes[:, 0] = target[:, 0]
            cx = target[:, 1] * img_size
            cy = target[:, 2] * img_size
            w = target[:, 3] * img_size
            h = target[:, 4] * img_size
            tboxes[:, 1] = cx - w / 2
            tboxes[:, 2] = cy - h / 2
            tboxes[:, 3] = cx + w / 2
            tboxes[:, 4] = cy + h / 2
            target = tboxes

        # Add image index column
        if pred.shape[0] > 0:
            pred_with_img = np.column_stack([np.full(pred.shape[0], img_idx), pred])
            all_predictions.append(pred_with_img)
        if target.shape[0] > 0:
            target_with_img = np.column_stack([np.full(target.shape[0], img_idx), target])
            all_targets.append(target_with_img)

    if len(all_predictions) == 0:
        return 0.0, 0.0, 0.0, {}

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0) if len(all_targets) > 0 else np.zeros((0, 6))

    if all_targets.shape[0] == 0:
        return 0.0, 0.0, 0.0, {}

    aps = []
    tp_at_50 = None
    conf_global = all_predictions[:, 2]

    # ← NEW: Calculate target areas for adaptive thresholding
    target_areas = (all_targets[:, 3] - all_targets[:, 1]) * (all_targets[:, 4] - all_targets[:, 2])

    # Loop IoU thresholds
    for base_iou_thresh in iou_thresholds:
        Npred = len(all_predictions)
        tp = np.zeros(Npred, dtype=np.int32)
        fp = np.zeros(Npred, dtype=np.int32)

        # iterate images
        unique_imgs = np.unique(all_predictions[:, 0])
        for img_idx in unique_imgs:
            img_mask = np.where(all_predictions[:, 0] == img_idx)[0]
            pred_img = all_predictions[img_mask]
            target_img = all_targets[all_targets[:, 0] == img_idx]

            if target_img.shape[0] == 0:
                fp[img_mask] = 1
                continue

            # prepare boxes and classes
            pred_boxes = torch.from_numpy(pred_img[:, 3:7]).float()
            target_boxes = torch.from_numpy(target_img[:, 2:6]).float()
            pred_classes = pred_img[:, 1]
            target_classes = target_img[:, 1]

            # ← NEW: Calculate adaptive IoU thresholds based on GT box size
            gt_areas = (target_img[:, 3] - target_img[:, 1]) * (target_img[:, 4] - target_img[:, 2])
            adaptive_iou_thresholds = np.where(
                gt_areas < 16 * 16,           # Very small objects (<16x16)
                base_iou_thresh * 0.7,        # 30% relaxation
                np.where(
                    gt_areas < 32 * 32,       # Small objects (16-32)
                    base_iou_thresh * 0.85,   # 15% relaxation
                    base_iou_thresh           # Normal threshold
                )
            )

            # compute IoU matrix (P x G)
            iou_matrix = box_iou_batch(pred_boxes, target_boxes).numpy()

            matched_gt = set()
            for local_p in range(len(pred_img)):
                pred_cls = pred_classes[local_p]
                same_class_mask = (target_classes == pred_cls)
                if not same_class_mask.any():
                    fp[img_mask[local_p]] = 1
                    continue

                ious = iou_matrix[local_p, same_class_mask]
                if ious.size == 0:
                    fp[img_mask[local_p]] = 1
                    continue

                # ← NEW: Use adaptive threshold for each GT
                gt_local_indices = np.where(same_class_mask)[0]
                adaptive_thresholds = adaptive_iou_thresholds[gt_local_indices]
                
                # Find best match that exceeds its adaptive threshold
                valid_matches = ious >= adaptive_thresholds
                if not valid_matches.any():
                    fp[img_mask[local_p]] = 1
                    continue

                # Among valid matches, pick highest IoU
                best_local_idx = int(ious.argmax())
                if ious[best_local_idx] < adaptive_thresholds[best_local_idx]:
                    fp[img_mask[local_p]] = 1
                    continue

                gt_idx = int(gt_local_indices[best_local_idx])

                if gt_idx not in matched_gt:
                    tp[img_mask[local_p]] = 1
                    matched_gt.add(gt_idx)
                else:
                    fp[img_mask[local_p]] = 1

        # Sort by confidence descending for AP computation
        order = np.argsort(-conf_global)
        tp_sorted = tp[order]
        conf_sorted = conf_global[order]
        pred_cls_sorted = all_predictions[order, 1]

        # compute AP per class
        ap, _, _, _, unique_classes = compute_ap_per_class(
            tp=tp_sorted,
            conf=conf_sorted,
            pred_cls=pred_cls_sorted,
            target_cls=all_targets[:, 1]
        )

        aps.append(ap.mean() if ap.size else 0.0)

        # store TP at iou=0.5 for per-class breakdown
        if abs(base_iou_thresh - 0.5) < 1e-6:
            tp_at_50 = tp_sorted.copy()

    if len(aps) == 0:
        return 0.0, 0.0, 0.0, {}

    map_50_95 = float(np.mean(aps))
    map_50 = float(aps[0]) if len(aps) > 0 else 0.0
    map_75 = float(aps[iou_thresholds.index(0.75)]) if 0.75 in iou_thresholds else 0.0

    # compute per-class ap at IoU=0.5
    ap_per_class, _, _, _, unique_classes = compute_ap_per_class(
        tp=tp_at_50 if tp_at_50 is not None else np.array([]),
        conf=conf_global[np.argsort(-conf_global)] if conf_global.size else np.array([]),
        pred_cls=all_predictions[np.argsort(-conf_global), 1] if conf_global.size else np.array([]),
        target_cls=all_targets[:, 1] if all_targets.size else np.array([])
    )

    per_class_ap = {int(cls): float(ap) for cls, ap in zip(unique_classes, ap_per_class)} if ap_per_class.size else {}

    return map_50_95, map_50, map_75, per_class_ap


# -------------------- OCR UTILS --------------------

def denoise_image(image: Any, h: int = 10, hColor: int = 10, templateWindowSize: int = 7, searchWindowSize: int = 21) -> Any:
    img = _ensure_cv2_array(image)
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        out = cv2.fastNlMeansDenoising(img, None, h, templateWindowSize, searchWindowSize)
    else:
        out = cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)
    if isinstance(image, Image.Image):
        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)) if out.ndim == 3 else Image.fromarray(out)
    return out

def deskew_image(image: Any) -> Any:
    img = _ensure_cv2_array(image)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = 255 - thresh
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if isinstance(image, Image.Image):
        return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)) if rotated.ndim == 3 else Image.fromarray(rotated)
    return rotated

def encode_label(text: Any) -> List[int]:
    if not isinstance(text, str):
        text = str(text)
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '', text.lower())
    return [char2idx[c] for c in cleaned if c in char2idx]

def decode_argmax_output(logits: torch.Tensor, labels: Optional[Sequence[str]] = None, blank_index: int = BLANK_IDX) -> str:
    if logits is None:
        return ""
    if logits.device.type != 'cpu':
        logits = logits.cpu()
    if logits.dim() == 3:
        logits = logits[0]
    indices = torch.argmax(logits, dim=1).tolist()
    chars = []
    prev = None
    for idx in indices:
        if idx == prev:
            prev = idx
            continue
        prev = idx
        if idx == blank_index:
            continue
        if labels:
            if 0 <= idx < len(labels):
                chars.append(labels[idx])
        else:
            if idx in idx2char:
                chars.append(idx2char[idx])
    return post_process_prediction("".join(chars))

def decode_output(output: torch.Tensor) -> List[str]:
    if output is None:
        return []
    if output.device.type != 'cpu':
        output = output.cpu()
    if output.dim() == 3 and output.shape[0] != output.shape[1]:
        pred = torch.argmax(output, dim=2).permute(1, 0)
    else:
        pred = torch.argmax(output, dim=2)
    sequences = []
    for seq in pred:
        chars = []
        prev = -1
        for idx in seq.tolist():
            if idx != prev and idx != BLANK_IDX and idx in idx2char:
                chars.append(idx2char[idx])
            prev = idx
        raw = "".join(chars)
        collapsed = "".join([c for c, _ in itertools.groupby(raw)])
        sequences.append(post_process_prediction(collapsed))
    return sequences

def normalize_class_name(name: str) -> str:
    return name.lower().replace(' ', '').replace('-', '').replace('_', '')

def correct_ocr_text(text: str, valid_classes: Optional[Sequence[str]] = None, cutoff: float = 0.75) -> str:
    if valid_classes is None:
        valid_classes = CLASSES
    norm = re.sub(r'[^0-9a-z]', '', text.lower())
    candidates = [normalize_class_name(c) for c in valid_classes]
    matches = difflib.get_close_matches(norm, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else text

def post_process_prediction(text: str, valid_labels: Optional[Sequence[str]] = None, threshold: int = 70) -> str:
    if not text:
        return text
    if valid_labels is None:
        valid_labels = VALID_LABELS
    norm = re.sub(r'[^a-z0-9]', '', text.lower())
    if norm in valid_labels:
        return norm
    best_match = None
    best_score = 0
    for label in valid_labels:
        r = fuzz.ratio(norm, label)
        p = fuzz.partial_ratio(norm, label)
        t = fuzz.token_sort_ratio(norm, label)
        score = max(r, p, t)
        if score >= threshold and score > best_score:
            best_score = score
            best_match = label
    return best_match if best_match else norm

def calculate_cer(pred: str, target: str) -> float:
    if not target:
        return 1.0 if pred else 0.0
    dist = Levenshtein.distance(pred, target)
    return dist / max(1, len(target))

def calculate_wer(pred: str, target: str) -> float:
    pred_words = pred.split()
    target_words = target.split()
    if not target_words:
        return 1.0 if pred_words else 0.0
    dist = Levenshtein.distance(pred_words, target_words)
    return dist / max(1, len(target_words))

# -------------------- DETECTION METRICS --------------------

# runs/detect/
# ├── train/
# │   ├── weights/
# │   │   ├── best_mcu.pt
# │   │   └── final_mcu.pt
# │   └── plots/
# │       ├── confusion_matrix.png           # ← TRAIN data
# │       ├── confusion_matrix_normalized.png
# │       ├── F1_curve.png
# │       ├── P_curve.png, R_curve.png, PR_curve.png
# │       ├── label_heatmap.png
# │       └── results.png                    # ← Loss curves
# └── test/
#     └── plots/
#         ├── confusion_matrix.png           # ← VALIDATION data
#         ├── confusion_matrix_normalized.png
#         ├── F1_curve.png
#         ├── P_curve.png, R_curve.png, PR_curve.png
#         ├── label_heatmap.png
#         ├── val_batch0_labels.jpg          # ← NEW: Ground truth
#         ├── val_batch0_pred.jpg            # ← NEW: Predictions
#         ├── val_batch1_labels.jpg
#         └── val_batch1_pred.jpg


def compute_precision_recall_curves(
    all_preds: List[np.ndarray],
    all_targets: List[np.ndarray],
    num_classes: int,
    img_size: int = 512,
    iou_thresh: float = 0.5,
    adaptive_iou: bool = True
):
    
    confidences = np.linspace(0.0, 1.0, 100)
    precisions = []
    recalls = []
    eps = 1e-12

    for cls_id in range(num_classes):
        prec_per_thr = []
        rec_per_thr = []

        for thr in confidences:
            TP = FP = FN = 0

            for preds_img, gts_img in zip(all_preds, all_targets):
                # Select predictions of this class above threshold
                if isinstance(preds_img, np.ndarray) and preds_img.shape[0] > 0:
                    mask = (preds_img[:, 0].astype(int) == cls_id) & (preds_img[:, 1] >= thr)
                    preds_sel = preds_img[mask]
                else:
                    preds_sel = np.zeros((0, 6), dtype=np.float32)

                # Select GTs of this class
                if isinstance(gts_img, np.ndarray) and gts_img.shape[0] > 0:
                    gt_idx = np.where(gts_img[:, 0].astype(int) == cls_id)[0]
                else:
                    gt_idx = np.array([], dtype=int)

                # No GTs → all preds are FP
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

                # Adaptive IoU thresholds
                if adaptive_iou:
                    gt_areas = w * h
                    adaptive_iou_thresholds = np.where(
                        gt_areas < 16 * 16,
                        iou_thresh * 0.7,
                        np.where(
                            gt_areas < 32 * 32,
                            iou_thresh * 0.85,
                            iou_thresh
                        )
                    )
                else:
                    adaptive_iou_thresholds = np.full(len(gt_boxes), iou_thresh)

                ious = box_iou_batch(
                    torch.from_numpy(pred_boxes),
                    torch.from_numpy(gt_boxes)
                ).cpu().numpy()

                # Sort preds by confidence (desc)
                order = np.argsort(-preds_sel[:, 1])
                ious = ious[order]

                matched_gt = set()
                for i in range(ious.shape[0]):
                    row = ious[i].copy()
                    for m in matched_gt:
                        row[m] = -1.0
                    
                    best_gt = int(np.argmax(row))
                    
                    if row[best_gt] >= adaptive_iou_thresholds[best_gt]:
                        TP += 1
                        matched_gt.add(best_gt)
                    else:
                        FP += 1

                FN += max(0, gt_boxes.shape[0] - len(matched_gt))

            prec_per_thr.append(TP / (TP + FP + eps))
            rec_per_thr.append(TP / (TP + FN + eps))

        precisions.append(np.array(prec_per_thr))
        recalls.append(np.array(rec_per_thr))

    return confidences, np.array(precisions), np.array(recalls)

def save_precision_recall_curves(
    confidences: np.ndarray,
    precisions: np.ndarray,
    recalls: np.ndarray,
    class_names: List[str],
    run_dir: str
):
    
    try:
        # --------------------------------------------------
        # Resolve plots directory safely
        # --------------------------------------------------
        run_dir = os.path.normpath(run_dir)
        if os.path.basename(run_dir) == "plots":
            plots_dir = run_dir
        else:
            plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Guard: nothing to plot
        if confidences is None or len(confidences) == 0:
            return

        num_classes = len(class_names)

        # --------------------------------------------------
        # Precision vs Confidence
        # --------------------------------------------------
        plt.figure(figsize=(8, 6))
        for i in range(min(num_classes, precisions.shape[0])):
            plt.plot(confidences, precisions[i], label=class_names[i])
        plt.xlabel("Confidence")
        plt.ylabel("Precision")
        plt.title("Precision–Confidence Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "precision_confidence_curve.png"), dpi=150)
        plt.close()

        # --------------------------------------------------
        # Recall vs Confidence
        # --------------------------------------------------
        plt.figure(figsize=(8, 6))
        for i in range(min(num_classes, recalls.shape[0])):
            plt.plot(confidences, recalls[i], label=class_names[i])
        plt.xlabel("Confidence")
        plt.ylabel("Recall")
        plt.title("Recall–Confidence Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "recall_confidence_curve.png"), dpi=150)
        plt.close()

        # --------------------------------------------------
        # Precision vs Recall
        # --------------------------------------------------
        plt.figure(figsize=(8, 6))
        for i in range(min(num_classes, precisions.shape[0], recalls.shape[0])):
            plt.plot(recalls[i], precisions[i], label=class_names[i])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "precision_recall_curve.png"), dpi=150)
        plt.close()

    except Exception as e:
        print(f"⚠️ Plotting error (PR curves): {e}")


def plot_confusion_matrix(#checked ok
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[Sequence[str]] = None,
    run_dir: str = "runs/detect",
    base_title: str = "Confusion Matrix"
):
    os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Compute raw confusion matrix
    # ------------------------------------------------------------------
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(labels)) if labels else None
    )

    # ------------------------------------------------------------------
    # 2. Plot & save RAW confusion matrix
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, int(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(base_title)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix.png"))
    plt.close()

    # ------------------------------------------------------------------
    # 3. Normalize confusion matrix (row-wise)
    # ------------------------------------------------------------------
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    # ------------------------------------------------------------------
    # 4. Plot & save NORMALIZED confusion matrix
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

    thresh = cm_norm.max() / 2.0 if cm_norm.size else 0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black"
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{base_title} Normalized")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix_normalized.png"))
    plt.close()


def plot_f1_confidence_curve(
    predictions,        # List[np.ndarray], each (N,6): cls, conf, x1,y1,x2,y2
    targets,            # List[np.ndarray], each (M,5): cls, cx,cy,w,h
    class_names,
    run_dir,
    iou_thresh=0.5,
    img_size=512
):
    """
    Plot F1-Confidence curve with proper type handling for torch tensors.
    
    FIXED: Now handles both numpy arrays and torch tensors.
    """
    os.makedirs(run_dir, exist_ok=True)

    num_classes = len(class_names)
    confs = np.linspace(0.0, 1.0, 101)

    def iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    f1_curves = []
    plt.figure(figsize=(8, 6))

    for cls in range(num_classes):
        f1_vals = []

        for conf in confs:
            TP = FP = FN = 0

            for preds_img, gts_img in zip(predictions, targets):
                # ✅ FIX: Convert to numpy if needed
                if isinstance(preds_img, torch.Tensor):
                    preds_img = preds_img.cpu().numpy()
                if isinstance(gts_img, torch.Tensor):
                    gts_img = gts_img.cpu().numpy()
                
                # ✅ FIX: Handle empty arrays safely
                if not isinstance(preds_img, np.ndarray) or preds_img.shape[0] == 0:
                    if isinstance(gts_img, np.ndarray) and gts_img.shape[0] > 0:
                        gts_cls = gts_img[gts_img[:, 0] == cls]
                        FN += len(gts_cls)
                    continue
                
                if not isinstance(gts_img, np.ndarray) or gts_img.shape[0] == 0:
                    preds = preds_img[preds_img[:, 1] >= conf]
                    preds = preds[preds[:, 0] == cls]
                    FP += len(preds)
                    continue
                
                preds = preds_img[preds_img[:, 1] >= conf]
                preds = preds[preds[:, 0] == cls]

                gts_cls = gts_img[gts_img[:, 0] == cls]
                matched = set()

                for p in preds:
                    hit = False
                    for gi, gt in enumerate(gts_cls):
                        if gi in matched:
                            continue
                        cx, cy, w, h = gt[1:] * img_size
                        gt_box = [
                            cx - w/2, cy - h/2,
                            cx + w/2, cy + h/2
                        ]
                        if iou(p[2:], gt_box) >= iou_thresh:
                            TP += 1
                            matched.add(gi)
                            hit = True
                            break
                    if not hit:
                        FP += 1

                FN += max(0, len(gts_cls) - len(matched))

            prec = TP / (TP + FP + 1e-12)
            rec  = TP / (TP + FN + 1e-12)
            f1_vals.append(2 * prec * rec / (prec + rec + 1e-12))

        f1_vals = np.array(f1_vals)
        f1_curves.append(f1_vals)
        plt.plot(confs, f1_vals, label=class_names[cls], alpha=0.7)

    # ---- ALL CLASSES (macro) ----
    if len(f1_curves) > 0:
        f1_mean = np.mean(f1_curves, axis=0)
        best_i = np.argmax(f1_mean)

        plt.plot(
            confs, f1_mean,
            linewidth=3, color="blue",
            label=f"all classes {f1_mean[best_i]:.2f} at {confs[best_i]:.3f}"
        )

    plt.xlabel("Confidence")
    plt.ylabel("F1")
    plt.title("F1-Confidence Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(run_dir, "f1_confidence_curve.png"))
    plt.close()


def plot_label_heatmap(#work done
    box_centers: Sequence[Tuple[float, float]],
    run_dir: str,
    img_size: Tuple[int, int] = (640, 640),
    grid_size: Tuple[int, int] = (100, 100)
):
    """
    box_centers: list of (cx, cy) normalized coordinates
    Saves:
      - label_heatmap.png        (2D density)
      - label_scatter.png        (cx, cy scatter)
    """
    os.makedirs(run_dir, exist_ok=True)

    centers = np.array(box_centers, dtype=np.float32)
    if centers.size == 0:
        return

    cx = centers[:, 0]
    cy = centers[:, 1]

    # ======================================================
    # 1. HEATMAP (cx, cy density)
    # ======================================================
    heatmap = np.zeros(grid_size, dtype=np.float32)
    h_bins, w_bins = grid_size

    for x, y in zip(cx, cy):
        col = int(x * w_bins)
        row = int(y * h_bins)
        if 0 <= row < h_bins and 0 <= col < w_bins:
            heatmap[row, col] += 1

    heatmap = cv2.GaussianBlur(heatmap, (9, 9), 0)

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="hot", origin="lower")
    plt.colorbar(label="Density")
    plt.title("Label Center Heatmap (cx, cy)")
    plt.xlabel("x bins")
    plt.ylabel("y bins")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "label_heatmap.png"), dpi=150)
    plt.close()

    # ======================================================
    # 2. SCATTER PLOT (cx, cy)
    # ======================================================
    plt.figure(figsize=(6, 6))
    plt.scatter(cx, cy, s=8, alpha=0.5)
    plt.xlabel("cx (normalized)")
    plt.ylabel("cy (normalized)")
    plt.title("Label Center Scatter")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "label_scatter.png"), dpi=150)
    plt.close()


def get_run_dir(task: str) -> str:#checked
    """
    task:
      - detect/train
      - detect/test
    """
    run_dir = Path("runs") / task
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


# def plot_image_grid(images: List[np.ndarray], #will use later
#                     labels: List[List[Tuple]],
#                     preds: Optional[List[List[Tuple]]] = None,
#                     class_names: Optional[List[str]] = None,
#                     save_path: str = "runs/detect/grid.jpg",
#                     max_images: int = 16) -> None:
#     """
#     images: list of HxWxC numpy arrays (float in [0..1])
#     labels: per-image list of (cls, cx, cy, w, h) --- cx,cy,w,h normalized [0..1]
#     preds: same format as labels (optional)
#     """
#     n = min(len(images), max_images)
#     if n == 0:
#         return
#
#     cols = int(math.sqrt(n))
#     cols = max(1, cols)
#     rows = math.ceil(n / cols)
#
#     fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
#     if isinstance(axs, np.ndarray):
#         axs = axs.flatten()
#     else:
#         axs = [axs]
#
#     for i in range(n):
#         img = images[i]
#         if isinstance(img, torch.Tensor):
#             img = img.permute(1, 2, 0).cpu().numpy()
#         img = np.clip(img, 0.0, 1.0)
#
#         axs[i].imshow(img)
#         axs[i].axis('off')
#
#         h, w = img.shape[:2]
#
#         if labels and i < len(labels):
#             for cls, cx, cy, bw, bh in labels[i]:
#                 x1 = (cx - bw / 2) * w
#                 y1 = (cy - bh / 2) * h
#                 rect = Rectangle((x1, y1), bw * w, bh * h, linewidth=2,
#                                  edgecolor='green', facecolor='none')
#                 axs[i].add_patch(rect)
#                 if class_names:
#                     axs[i].text(x1, y1, class_names[int(cls)], color='green',
#                                 fontsize=8, verticalalignment='top')
#
#         if preds and i < len(preds):
#             for cls, cx, cy, bw, bh in preds[i]:
#                 x1 = (cx - bw / 2) * w
#                 y1 = (cy - bh / 2) * h
#                 rect = Rectangle((x1, y1), bw * w, bh * h, linewidth=1.5,
#                                  edgecolor='red', facecolor='none', linestyle='--')
#                 axs[i].add_patch(rect)
#                 if class_names:
#                     axs[i].text(x1, y1, class_names[int(cls)], color='red',
#                                 fontsize=8, verticalalignment='bottom')
#
#     # blank remaining axes
#     for j in range(n, len(axs)):
#         axs[j].axis('off')
#
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=200)
#     plt.close()


# -- log_predictions --------------------------------------------------------
# def log_predictions(images: torch.Tensor, #will use later on
#                     labels: List[torch.Tensor],
#                     preds: Optional[List[torch.Tensor]],
#                     run_dir: str,
#                     class_names: Optional[List[str]] = None,
#                     prefix: str = "val_batch",
#                     mean: Optional[Tuple[float, float, float]] = None,
#                     std: Optional[Tuple[float, float, float]] = None) -> None:
#     """
#     images: torch.Tensor (B, C, H, W) - possibly normalized with mean/std
#     labels: list of length B, each a tensor (N,5) in [cls, cx, cy, w, h] normalized coords
#     preds: list of length B, each a tensor (M,6) or (M,7) depending on decode format
#     mean/std: if images were normalized with transforms.Normalize, pass mean/std to unnormalize
#     Saves two images (labels, preds) under run_dir (e.g. run_dir/images/)
#     """
#     os.makedirs(run_dir, exist_ok=True)

#     B = min(images.shape[0], 8)
#     img_t = images[:B].detach().cpu().clone()  # (B,C,H,W)

#     # Unnormalize if mean/std given
#     if mean is not None and std is not None:
#         mean_t = torch.tensor(mean).view(1, -1, 1, 1)
#         std_t = torch.tensor(std).view(1, -1, 1, 1)
#         img_t = img_t * std_t + mean_t

#     # Clamp to [0,1]
#     img_t = img_t.clamp(0.0, 1.0)

#     def draw_boxes_on_tensor(img_tensor, boxes, color_bgr=(0, 255, 0)):
#         """
#         img_tensor: Tensor (C,H,W) in [0,1]
#         boxes: list/iterable of (cls, cx, cy, w, h) normalized
#         returns Tensor (C,H,W) in [0,1] with boxes drawn (RGB)
#         """
#         img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()  # H,W,3 RGB
#         img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

#         H, W = img_bgr.shape[:2]
#         for box in boxes:
#             cls, cx, cy, bw, bh = box
#             x1 = int((cx - bw / 2) * W)
#             y1 = int((cy - bh / 2) * H)
#             x2 = int((cx + bw / 2) * W)
#             y2 = int((cy + bh / 2) * H)

#             # clamp
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(W - 1, x2), min(H - 1, y2)

#             label = class_names[int(cls)] if (class_names and 0 <= int(cls) < len(class_names)) else str(int(cls))
#             cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, 2)

#             # text background
#             (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
#             cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color_bgr, -1)
#             cv2.putText(img_bgr, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
#                         (255, 255, 255), 1, cv2.LINE_AA)

#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         return torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

#     imgs_labels = []
#     imgs_preds = []
#     for i in range(B):
#         img = img_t[i]  # C,H,W
#         lbl_boxes = []
#         if labels and i < len(labels) and isinstance(labels[i], torch.Tensor) and labels[i].numel() > 0:
#             arr = labels[i].cpu().numpy()
#             lbl_boxes = [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in arr]
#         pred_boxes = []
#         if preds and i < len(preds) and isinstance(preds[i], torch.Tensor) and preds[i].numel() > 0:
#             # We expect either [cls, conf, x1,y1,x2,y2] or [cls, conf, cx,cy,w,h]
#             arr = preds[i].cpu().numpy()
#             H, W = img.shape[1], img.shape[2]
#             for row in arr:
#                 # detect format heuristically
#                 if row.shape[0] >= 6:
#                     # assume [cls, conf, x1, y1, x2, y2, ...] absolute pixels
#                     cls = int(row[0])
#                     x1, y1, x2, y2 = row[-4], row[-3], row[-2], row[-1]
#                     bw = (x2 - x1) / W
#                     bh = (y2 - y1) / H
#                     cx = (x1 + x2) / 2.0 / W
#                     cy = (y1 + y2) / 2.0 / H
#                     pred_boxes.append((cls, float(cx), float(cy), float(bw), float(bh)))
#                 elif row.shape[0] >= 5:
#                     # assume [cls, cx, cy, w, h] normalized
#                     cls = int(row[0])
#                     cx, cy, bw, bh = float(row[1]), float(row[2]), float(row[3]), float(row[4])
#                     pred_boxes.append((cls, cx, cy, bw, bh))
#         imgs_labels.append(draw_boxes_on_tensor(img, lbl_boxes, color_bgr=(0, 200, 0)))
#         imgs_preds.append(draw_boxes_on_tensor(img, pred_boxes, color_bgr=(0, 0, 200)))

#     # Save
#     vutils.save_image(torch.stack(imgs_labels), os.path.join(run_dir, f"{prefix}_labels.jpg"), nrow=4)
#     vutils.save_image(torch.stack(imgs_preds),  os.path.join(run_dir, f"{prefix}_preds.jpg"),  nrow=4)

def plot_yolo_results(history, save_path):# checked
    epochs = range(1, len(history["train_box"]) + 1)

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))

    # --- TRAIN ---
    axs[0, 0].plot(epochs, history["train_box"], label="results")
    axs[0, 0].set_title("train/box_loss")

    axs[0, 1].plot(epochs, history["train_cls"])
    axs[0, 1].set_title("train/cls_loss")

    axs[0, 2].plot(epochs, history["train_obj"])   # ✅ FIXED! Use obj
    axs[0, 2].set_title("train/obj_loss") 

    axs[0, 3].plot(epochs, history["precision"])
    axs[0, 3].set_ylim(0, 1)
    axs[0, 3].set_title("metrics/precision(B)")

    axs[0, 4].plot(epochs, history["recall"])
    axs[0, 4].set_ylim(0, 1)
    axs[0, 4].set_title("metrics/recall(B)")

    # --- VAL ---
    axs[1, 0].plot(epochs, history["val_box"])
    axs[1, 0].set_title("val/box_loss")

    axs[1, 1].plot(epochs, history["val_cls"])
    axs[1, 1].set_title("val/cls_loss")

    axs[1, 2].plot(epochs, history["val_obj"])     # ✅ FIXED! Use obj
    axs[1, 2].set_title("val/obj_loss") 

    axs[1, 3].plot(epochs, history["map50"])
    axs[1, 3].set_ylim(0, 1)
    axs[1, 3].set_title("metrics/mAP50(B)")

    axs[1, 4].plot(epochs, history["map5095"])
    axs[1, 4].set_ylim(0, 1)
    axs[1, 4].set_title("metrics/mAP50-95(B)")

    for ax in axs.flat:
        ax.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

class SiLU(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    def forward(self, x):
        return x * torch.sigmoid(x)
    
def print_gpu_memory():
    if not torch.cuda.is_available():
        print("GPU not available.")
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"GPU Memory: {allocated:.2f} MB allocated / {reserved:.2f} MB reserved")

def save_results_csv(metrics: dict, run_dir: str):
    """
    Save metrics to CSV with proper type conversion.
    
    FIXED: Now handles torch tensors and numpy arrays.
    """
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "results.csv")
    write_header = not os.path.exists(csv_path)
    keys = list(metrics.keys())
    
    def _to_csv_value(v):
        """Convert any value to CSV-safe type."""
        if isinstance(v, torch.Tensor):
            # Extract scalar from tensor
            return float(v.item()) if v.numel() == 1 else float(v.mean())
        if isinstance(v, (np.ndarray, np.generic)):
            # Convert numpy to float
            return float(v) if v.size == 1 else float(v.mean())
        if isinstance(v, (int, float, str, bool)):
            # Already CSV-safe
            return v
        # Fallback: convert to string
        return str(v)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(keys)
        writer.writerow([_to_csv_value(metrics[k]) for k in keys])

# =================== TEST SUMMARY ===================
def save_test_summary(metrics, save_path):
    """
    Save test evaluation summary to text file.
    
    Args:
        metrics: dict with keys like 'mAP_50', 'precision', 'recall', etc.
        save_path: path to save summary.txt
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TEST EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        # Main metrics
        f.write("DETECTION METRICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"Precision:        {metrics.get('precision', 0):.4f}\n")
        f.write(f"Recall:           {metrics.get('recall', 0):.4f}\n")
        f.write(f"mAP@0.5:          {metrics.get('mAP_50', 0):.4f}\n")
        f.write(f"mAP@0.75:         {metrics.get('mAP_75', 0):.4f}\n")
        f.write(f"mAP@0.5:0.95:     {metrics.get('mAP_50_95', 0):.4f}\n")
        
        # F1 score
        p = metrics.get('precision', 0)
        r = metrics.get('recall', 0)
        f1 = 2 * p * r / (p + r + 1e-12)
        f.write(f"F1 Score:         {f1:.4f}\n")
        
        # Loss breakdown
        f.write("\nLOSS BREAKDOWN:\n")
        f.write("-"*60 + "\n")
        f.write(f"Total Loss:       {metrics.get('loss', 0):.4f}\n")
        f.write(f"Box Loss:         {metrics.get('box_loss', 0):.4f}\n")
        f.write(f"Class Loss:       {metrics.get('cls_loss', 0):.4f}\n")
        f.write(f"DFL Loss:         {metrics.get('dfl_loss', 0):.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"📄 Test summary saved to: {save_path}")
