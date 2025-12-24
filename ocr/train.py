#!/usr/bin/env python3
"""
🎓 MCUDetector Training Script (24 Classes)
Trains custom MCUDetector from scratch → saves best_mcu.pt for unified pipeline
Compatible with YOLO-style annotations (class x_center y_center width height)
"""

import os
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

# Local imports - ADJUST PATHS TO YOUR PROJECT STRUCTURE
from model import MCUDetector, MCUDetectionLoss  # Your custom classes
from utils import print_gpu_memory  # Optional

# ============================================================================
# CONSTANTS & PATHS
# ============================================================================
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / 'data'
RUNS_DIR = DATA_DIR / 'runs' / 'detect'
TRAIN_IMG_DIR = DATA_DIR / 'dataset_train' / 'images' / 'train'
TRAIN_LABEL_DIR = DATA_DIR / 'dataset_train' / 'labels' / 'train'
VAL_IMG_DIR = DATA_DIR / 'dataset_test' / 'images' / 'train'
VAL_LABEL_DIR = DATA_DIR / 'dataset_test' / 'labels' / 'train'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# 24 Classes (same as your dataset.py)
CLASS_NAMES = [
    "8051", "ARDUINO_NANO_ATMEGA328P", "ARMCORTEXM3", "ARMCORTEXM7", "ESP32_DEVKIT",
    "NODEMCU_ESP8266", "RASPBERRY_PI_3B_PLUS", "Arduino", "Pico", "RaspberryPi",
    "Arduino Due", "Arduino Leonardo", "Arduino Mega 2560 -Black and Yellow-",
    "Arduino Mega 2560 -Black-", "Arduino Mega 2560 -Blue-", "Arduino Uno -Black-",
    "Arduino Uno -Green-", "Arduino Uno Camera Shield", "Arduino Uno R3",
    "Arduino Uno WiFi Shield", "Beaglebone Black", "Raspberry Pi 1 B-",
    "Raspberry Pi 3 B-", "Raspberry Pi A-"
]
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================================
# COLLATE FUNCTION (module-level, not lambda)
# ============================================================================
def default_collate_fn(batch):
    """Default collate that keeps variable-length targets as list."""
    return batch

# ============================================================================
# DETECTION DATASET
# ============================================================================
class MCUDetectionDataset(Dataset):
    """YOLO-format dataset for MCUDetector training."""
    
    def __init__(self, img_dir, label_dir, img_size=512, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.transform = transform
        
        # Get all image-label pairs
        self.image_files = [f for f in self.img_dir.glob('*.jpg')]
        self.image_files.sort()
        
        print(f"📊 Found {len(self.image_files)} images in {img_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_dir / img_path.with_suffix('.txt').name
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        targets = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        targets.append([cls_id, x_center, y_center, width, height])
        
        # Convert to tensors
        if len(targets) > 0:
            targets = torch.tensor(targets, dtype=torch.float32)
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)
        
        # Resize for training
        h, w = image.shape[:2]
        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        image_resized = image_resized.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1)  # HWC → CHW
        
        # Normalize targets to resized image coordinates
        if len(targets) > 0:
            targets[:, 1] *= self.img_size / w  # x_center
            targets[:, 2] *= self.img_size / h  # y_center
            targets[:, 3] *= self.img_size / w  # width
            targets[:, 4] *= self.img_size / h  # height
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, targets

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    print(f"\nEpoch {epoch+1}/{epochs} - Training...")
    
    for batch_idx, batch in enumerate(train_loader):
        images, targets = zip(*batch)  # because collate_fn=default_collate_fn
        images = torch.stack(images).to(device)

        # Build targets_p4 / targets_p5 as list of (N, 6): [cls, x, y, w, h, conf]
        targets_p4, targets_p5 = [], []
        for t in targets:
            if t is None or t.numel() == 0:
                empty = torch.zeros((0, 6), device=device)
                targets_p4.append(empty)
                targets_p5.append(empty)
                continue
            t = t.to(device)  # (N, 5)
            conf = torch.ones((t.shape[0], 1), device=device)
            t6 = torch.cat([t, conf], dim=1)  # (N, 6)
            targets_p4.append(t6)
            targets_p5.append(t6)
        
        optimizer.zero_grad()
        
        # Forward pass - get P4 and P5 predictions
        (cls_p4, reg_p4), (cls_p5, reg_p5) = model(images)
        
        # Compute loss
        loss_dict = criterion(
            (cls_p4, reg_p4),
            (cls_p5, reg_p5),
            targets_p4,
            targets_p5
        )
        loss = loss_dict['total']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{num_batches} - Loss: {loss.item():.4f}")
    
    return total_loss / max(num_batches, 1)


def validate_detector(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in val_loader:
            images, targets = zip(*batch)
            images = torch.stack(images).to(device)

            # Build targets_p4 / targets_p5
            targets_p4, targets_p5 = [], []
            for t in targets:
                if t is None or t.numel() == 0:
                    empty = torch.zeros((0, 6), device=device)
                    targets_p4.append(empty)
                    targets_p5.append(empty)
                    continue
                t = t.to(device)
                conf = torch.ones((t.shape[0], 1), device=device)
                t6 = torch.cat([t, conf], dim=1)
                targets_p4.append(t6)
                targets_p5.append(t6)

            (cls_p4, reg_p4), (cls_p5, reg_p5) = model(images)
            loss_dict = criterion(
                (cls_p4, reg_p4),
                (cls_p5, reg_p5),
                targets_p4,
                targets_p5
            )
            loss = loss_dict['total']
            
            total_loss += loss.item()
    
    return total_loss / max(num_batches, 1)

# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train MCUDetector (24 classes)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader workers (0=no multiprocessing on Windows)')
    parser.add_argument('--debug', action='store_true', help='Debug mode (1 epoch)')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Create directories
    (RUNS_DIR / 'train' / 'weights').mkdir(parents=True, exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets & DataLoaders
    train_dataset = MCUDetectionDataset(
        img_dir=TRAIN_IMG_DIR,
        label_dir=TRAIN_LABEL_DIR,
        img_size=args.img_size,
        transform=transform
    )
    val_dataset = MCUDetectionDataset(
        img_dir=VAL_IMG_DIR,
        label_dir=VAL_LABEL_DIR,
        img_size=args.img_size,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=default_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=default_collate_fn
    )
    
    # Model & Loss
    print(f"\n🧠 Creating MCUDetector (24 classes)...")
    model = MCUDetector(num_classes=NUM_CLASSES).to(device)
    
    criterion = MCUDetectionLoss(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"📁 Weights will be saved to: {RUNS_DIR / 'train' / 'weights' / 'best_mcu.pt'}")
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    epochs_to_run = 1 if args.debug else args.epochs
    
    for epoch in range(epochs_to_run):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs_to_run
        )
        
        # Validate
        val_loss = validate_detector(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\n📊 Epoch {epoch+1}/{epochs_to_run}")
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'args': vars(args)
            }, RUNS_DIR / 'train' / 'weights' / 'best_mcu.pt')
            print(f"   ✓ BEST MODEL SAVED! Val Loss: {best_val_loss:.4f}")
        else:
            print(f"   No improvement (best: {best_val_loss:.4f})")
        
        if args.debug:
            break
    
    # Final plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses)
    plt.axhline(y=best_val_loss, color='r', linestyle='--', label=f'Best: {best_val_loss:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(RUNS_DIR / 'train' / f'mcudetector_metrics_{TIMESTAMP}.png', dpi=150)
    plt.close()
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"   Best model: {RUNS_DIR / 'train' / 'weights' / 'best_mcu.pt'}")
    print(f"   Metrics plot: mcudetector_metrics_{TIMESTAMP}.png")
    print(f"\n🎯 Now run unified pipeline: python train_debug.py")

if __name__ == "__main__":
    main()
