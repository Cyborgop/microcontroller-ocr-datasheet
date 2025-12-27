#!/usr/bin/env python3
"""
🎓 UNIFIED MCUDetector + CRNN OCR Training Pipeline
Trains custom MCUDetector (24 classes) → CRNN OCR in sequence
Auto-creates directories, backs up old runs, saves metrics for thesis comparison
NO ULTRALYTICS DEPENDENCY
"""

import os
import sys
import shutil
import torch
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

# Local imports
from dataset import OCRDataset, load_labels, collate_fn
from model import EnhancedCRNN
from utils import (
    decode_output, BLANK_IDX,
    calculate_cer, calculate_wer, char2idx, idx2char,
    normalize_label, post_process_prediction
)

# ============================================================================
# CONSTANTS & PATHS
# ============================================================================
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / 'data'
RUNS_DIR = DATA_DIR / 'runs' / 'detect'
BACKUP_DIR = Path('D:/microcontroller-ocr-datasheet/microcontroller-ocr-datasheet/runsyolov8old')
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def create_directories():
    """Create necessary directories if they don't exist."""
    print("\n" + "="*80)
    print("📁 CREATING DIRECTORIES")
    print("="*80)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {RUNS_DIR}")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {BACKUP_DIR}")

    return RUNS_DIR, BACKUP_DIR


def backup_old_runs(runs_dir, backup_dir):
    """
    Backup existing detector runs to external folder.
    Useful for comparing baseline vs. new model.
    """
    print("\n" + "="*80)
    print("💾 BACKING UP OLD RUNS")
    print("="*80)

    old_train_dir = runs_dir / 'train'
    if old_train_dir.exists():
        backup_path = backup_dir / f'train_backup_{TIMESTAMP}'
        shutil.copytree(old_train_dir, backup_path)
        print(f"✓ Backed up: {old_train_dir} → {backup_path}")

        shutil.rmtree(old_train_dir)
        print(f"✓ Removed old runs folder for clean training")
    else:
        print(f"ℹ️  No old runs found (first training)")


def print_gpu_memory():
    """Print GPU memory usage if available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"  GPU Memory: {allocated:.2f} MB allocated / {reserved:.2f} MB reserved")


def indices_to_string(indices):
    """Convert label indices to string for ground truth comparison."""
    chars = []
    for idx in indices:
        idx = idx.item() if hasattr(idx, 'item') else idx
        if idx != BLANK_IDX and idx in idx2char:
            chars.append(idx2char[idx])
    return ''.join(chars)

# ============================================================================
# DETECTOR TRAINING (STUB - USE YOUR MCUDetector + MCUDetectionLoss)
# ============================================================================
def train_detector(epochs=100, batch_size=8, img_size=512, device=None):
    """
    Train your custom MCUDetector on 24 microcontroller classes.

    TODO: implement full training using MCUDetector + MCUDetectionLoss.
    For now it just checks for an existing best_mcu.pt and uses it.
    """
    print("\n" + "="*80)
    print("🚀 Custom MCUDetector Training (24 Classes)")
    print("="*80)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    print(f"✓ Image size: {img_size}x{img_size}")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Epochs: {epochs}")

    weights_path = RUNS_DIR / 'train' / 'weights' / 'best_mcu.pt'
    print(f"\n📂 Looking for detector weights at: {weights_path}")

    if weights_path.exists():
        print(f"✓ Detector weights found! Using: {weights_path}")
        return str(weights_path)

    raise FileNotFoundError(
        f"Detector weights not found at {weights_path}. "
        f"Train MCUDetector separately and save weights here."
    )

# ============================================================================
# CRNN OCR TRAINING
# ============================================================================
def train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, args):
    """Train CRNN for one epoch."""
    model.train()
    total_loss = 0

    for batch_idx, (images, targets, input_lens, label_lens) in enumerate(train_loader):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        input_lens = input_lens.to(DEVICE)
        label_lens = label_lens.to(DEVICE)

        logits = model(images)
        log_probs = nn.functional.log_softmax(logits.permute(1, 0, 2), dim=2)
        loss = criterion(log_probs, targets, input_lens, label_lens)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx == 0 and args.debug:
            print(f"  [DEBUG] Logits shape: {logits.shape}")
            print(f"  [DEBUG] First 5 logits argmax: {torch.argmax(logits, dim=2)[0, :5]}")
            print(f"  [DEBUG] BLANK_IDX: {BLANK_IDX}")

        if args.debug_one_batch:
            break

    if DEVICE.type == 'cuda' and args.debug:
        print_gpu_memory()

    return total_loss / len(train_loader)


def validate(model, test_loader, criterion, DEVICE, args):
    """Validate CRNN model."""
    model.eval()
    val_loss = 0
    total_cer = 0
    total_wer = 0
    count = 0
    sample_predictions = []

    with torch.no_grad():
        for images, targets, input_lens, label_lens in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            input_lens = input_lens.to(DEVICE)
            label_lens = label_lens.to(DEVICE)

            logits = model(images)
            log_probs = nn.functional.log_softmax(logits.permute(1, 0, 2), dim=2)
            loss = criterion(log_probs, targets, input_lens, label_lens)
            val_loss += loss.item()

            preds = decode_output(logits)
            start = 0
            for i, L in enumerate(label_lens):
                gt_indices = targets[start:start+L].cpu().numpy()
                gt = indices_to_string(gt_indices)
                start += L

                corrected_pred = post_process_prediction(preds[i])
                pred_norm = normalize_label(corrected_pred)
                gt_norm = normalize_label(gt)

                total_cer += calculate_cer(pred_norm, gt_norm)
                total_wer += calculate_wer(pred_norm, gt_norm)
                count += 1

                if len(sample_predictions) < 5:
                    sample_predictions.append({
                        'prediction': corrected_pred,
                        'ground_truth': gt,
                        'normalized_pred': pred_norm,
                        'normalized_gt': gt_norm,
                        'cer': calculate_cer(pred_norm, gt_norm),
                        'wer': calculate_wer(pred_norm, gt_norm)
                    })

            if args.debug_one_batch:
                break

    avg_val_loss = val_loss / len(test_loader)
    avg_cer = total_cer / count if count > 0 else 0
    avg_wer = total_wer / count if count > 0 else 0

    return avg_val_loss, avg_cer, avg_wer, sample_predictions


def train_crnn(detector_weights_path, args):
    """
    Train CRNN OCR model with custom MCUDetector for chip detection.
    Saves to: best_crnn_model_{TIMESTAMP}.pth
    """
    print("\n" + "="*80)
    print("🚀 CRNN OCR Training (Using Custom MCUDetector Crops)")
    print("="*80)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {DEVICE}")
    print(f"✓ Detector weights: {detector_weights_path}")
    print(f"✓ BLANK_IDX: {BLANK_IDX}, num_classes: {args.num_classes}")
    print(f"✓ Character set: {len(char2idx)} unique characters")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.RandomAffine(degrees=3, translate=(0.01, 0.01), scale=(0.98, 1.02)),
        transforms.RandomRotation(degrees=2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Load data
    print(f"\n📂 Loading datasets...")
    train_labels = load_labels(args.train_labels)
    test_labels = load_labels(args.test_labels)
    print(f"✓ Train samples: {len(train_labels)}, Test samples: {len(test_labels)}")

    # Datasets using your MCUDetector via detector_weights
    train_dataset = OCRDataset(
        img_dir=args.train_dir,
        label_dict=train_labels,
        transform=train_transform,
        use_detection=True,
        detector_weights=detector_weights_path,
        device=str(DEVICE)
    )

    test_dataset = OCRDataset(
        img_dir=args.test_dir,
        label_dict=test_labels,
        transform=test_transform,
        use_detection=True,
        detector_weights=detector_weights_path,
        device=str(DEVICE)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == 'cuda')
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == 'cuda')
    )

    # Model setup
    print(f"\n🧠 Creating Enhanced CRNN model...")
    model = EnhancedCRNN(num_classes=args.num_classes).to(DEVICE)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params = count_parameters(model)
    print(f"✓ Model parameters: {total_params / 1e6:.2f}M")

    # Training setup
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=7,
        
        min_lr=1e-6
    )

    print("\n" + "="*80)
    print("🎯 STARTING CRNN TRAINING")
    print("="*80 + "\n")

    train_losses, val_losses, val_cers, val_wers = [], [], [], []
    best_val_loss = float('inf')
    no_improve_count = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, args)
        val_loss, val_cer, val_wer, sample_predictions = validate(
            model, test_loader, criterion, DEVICE, args
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_cers.append(val_cer)
        val_wers.append(val_wer)

        print(
            f"[Epoch {epoch+1:3d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"CER: {val_cer:.4f} | "
            f"WER: {val_wer:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'char2idx': char2idx,
                    'idx2char': idx2char,
                },
                args.save_path,
            )
            print(f"  ✓ Model saved to {args.save_path}")
        else:
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(
                    f"\n⏹️  Early stopping at epoch {epoch+1} "
                    f"(no improvement for {args.patience} epochs)"
                )
                break

        if args.debug and (epoch + 1) % 5 == 0:
            print("\n  📋 Sample Predictions:")
            for i, sample in enumerate(sample_predictions[:3]):
                print(
                    f"    [{i+1}] Pred: '{sample['prediction']}' "
                    f"| GT: '{sample['ground_truth']}'"
                )
            print()

        if args.debug_one_batch:
            break

    print("\n" + "="*80)
    print("✅ CRNN TRAINING COMPLETE")
    print("="*80)

    print(f"\n📊 Final Sample Predictions:")
    for i, sample in enumerate(sample_predictions):
        print(f"[{i+1}] Prediction: '{sample['prediction']}'")
        print(f"    Ground Truth: '{sample['ground_truth']}'")
        print(f"    CER: {sample['cer']:.4f} | WER: {sample['wer']:.4f}\n")

    epochs_range = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs_range, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs_range, val_cers, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('CER')
    axes[0, 1].set_title('Validation Character Error Rate')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs_range, val_wers, 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('WER')
    axes[1, 0].set_title('Validation Word Error Rate')
    axes[1, 0].grid(True, alpha=0.3)

    ax2 = axes[1, 1]
    ax2.plot(epochs_range, train_losses, 'b-', linewidth=1.5, alpha=0.7, label='Train Loss')
    ax2.plot(epochs_range, val_losses, 'r-', linewidth=1.5, alpha=0.7, label='Val Loss')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs_range, val_cers, 'g--', linewidth=1.5, alpha=0.7, label='CER')
    ax2_twin.plot(epochs_range, val_wers, 'orange', linestyle='--',
                  linewidth=1.5, alpha=0.7, label='WER')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss', color='b')
    ax2_twin.set_ylabel('Error Rate', color='g')
    ax2.set_title('All Metrics Overview')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='g')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    metrics_plot = f'crnn_training_metrics_{TIMESTAMP}.png'
    plt.savefig(metrics_plot, dpi=150)
    print(f"✓ Metrics plot saved to '{metrics_plot}'")
    plt.close()

    return args.save_path

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """Run complete custom MCUDetector + CRNN training pipeline."""

    parser = argparse.ArgumentParser(
        description='Unified custom MCUDetector + CRNN OCR Training Pipeline'
    )

    # Detector arguments
    parser.add_argument('--detector_epochs', type=int, default=50)
    parser.add_argument('--detector_batch', type=int, default=8)
    parser.add_argument('--detector_img_size', type=int, default=512)
    parser.add_argument(
        '--detector_model',
        type=str,
        default='data/runs/detect/train/weights/best_mcu.pt',
        help='Path to pre-trained MCUDetector weights'
    )
    parser.add_argument('--backup_runs', action='store_true')

    # CRNN arguments
    parser.add_argument('--train_dir', type=str, default='data/dataset_train/images/train')
    parser.add_argument('--test_dir', type=str, default='data/dataset_test/images/train')
    parser.add_argument('--train_labels', type=str, default='data/dataset_train/labels/train')
    parser.add_argument('--test_labels', type=str, default='data/dataset_test/labels/train')
    parser.add_argument('--img_height', type=int, default=32)
    parser.add_argument('--img_width', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=len(char2idx) + 1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_one_batch', action='store_true')

    # Pipeline control
    parser.add_argument('--skip_detector', action='store_true',
                        help='Skip detector training (use existing weights)')
    parser.add_argument('--skip_crnn', action='store_true',
                        help='Skip CRNN training (detector only)')

    args = parser.parse_args()
    args.save_path = f'best_crnn_model_{TIMESTAMP}.pth'

    create_directories()

    if args.backup_runs:
        backup_old_runs(RUNS_DIR, BACKUP_DIR)

    if not args.skip_detector:
        detector_weights = train_detector(
            epochs=args.detector_epochs,
            batch_size=args.detector_batch,
            img_size=args.detector_img_size,
            device=0 if torch.cuda.is_available() else 'cpu'
        )
    else:
        detector_weights = args.detector_model
        print(f"\n⏭️  Skipping detector training - using: {detector_weights}")

    if not args.skip_crnn:
        crnn_model = train_crnn(detector_weights, args)

        print("\n" + "="*80)
        print("🎓 TRAINING PIPELINE COMPLETE")
        print("="*80)
        print(f"\n📊 Output Files:")
        print(f"  ✓ Detector: {detector_weights}")
        print(f"  ✓ CRNN: {crnn_model}")
        print(f"  ✓ Metrics: crnn_training_metrics_{TIMESTAMP}.png")
        print(f"\n💾 Backup: {BACKUP_DIR}")
        print(f"\n🎯 Custom MCUDetector + CRNN ready for MTP Chapter 5!\n")
    else:
        print("\n⏭️  Skipping CRNN training - detector only run")

if __name__ == "__main__":
    main()
