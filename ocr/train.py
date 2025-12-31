#!/usr/bin/env python3
"""
🎓 MCUDetector Training Script (24 Classes)
Trains custom MCUDetector from scratch → saves best_mcu.pt for unified pipeline
Compatible with YOLO-style annotations (class x_center y_center width height)
"""

# ============================================================================

# ===== CUDA MEMORY FIX: MUST BE BEFORE TORCH IMPORTS =====
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

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

# Local imports
from model import MCUDetector, MCUDetectionLoss
from utils import print_gpu_memory

# ===== Performance flags =====
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================================

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / 'data'
RUNS_DIR = DATA_DIR / 'runs' / 'detect'
TRAIN_IMG_DIR = DATA_DIR / 'dataset_train' / 'images' / 'train'
TRAIN_LABEL_DIR = DATA_DIR / 'dataset_train' / 'labels' / 'train'
VAL_IMG_DIR = DATA_DIR / 'dataset_test' / 'images' / 'train'
VAL_LABEL_DIR = DATA_DIR / 'dataset_test' / 'labels' / 'train'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

CLASS_NAMES = [...]
NUM_CLASSES = len(CLASS_NAMES)

def default_collate_fn(batch):
    return batch

# ============================================================================

class MCUDetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=512, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.transform = transform

        self.image_files = sorted(self.img_dir.glob('*.jpg'))
        print(f"📊 Found {len(self.image_files)} images in {img_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_dir / img_path.with_suffix('.txt').name

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        targets = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for ln in f:
                    cls, x, y, ww, hh = map(float, ln.strip().split()[:5])
                    targets.append([cls, x, y, ww, hh])

        targets = torch.tensor(targets, dtype=torch.float32) if len(targets) else torch.zeros((0, 5))
        image = cv2.resize(image, (self.img_size, self.img_size)).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        if len(targets):
            targets[:, 1] *= self.img_size / w
            targets[:, 2] *= self.img_size / h
            targets[:, 3] *= self.img_size / w
            targets[:, 4] *= self.img_size / h

        if self.transform:
            image = self.transform(image)

        return image, targets

# ============================================================================

from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    scaler = GradScaler()
    total_loss = 0
    num_batches = len(train_loader)

    print(f"\nEpoch {epoch+1}/{epochs} - Training...")

    for batch_idx, batch in enumerate(train_loader):
        images, targets = zip(*batch)
        images = torch.stack(images).to(device)

        targets_p4, targets_p5 = [], []
        for t in targets:
            if t is None or t.numel() == 0:
                targets_p4.append(torch.zeros((0,6), device=device))
                targets_p5.append(torch.zeros((0,6), device=device))
            else:
                t = t.to(device)
                conf = torch.ones((t.shape[0],1), device=device)
                t6 = torch.cat([t, conf], dim=1)
                targets_p4.append(t6)
                targets_p5.append(t6)

        optimizer.zero_grad()

        with autocast():
            (cls_p4, reg_p4), (cls_p5, reg_p5) = model(images)
            loss_dict = criterion((cls_p4,reg_p4),(cls_p5,reg_p5),targets_p4,targets_p5)
            loss = loss_dict['total']

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{num_batches} - Loss: {loss.item():.4f}")

    return total_loss / num_batches


def validate_detector(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            images, targets = zip(*batch)
            images = torch.stack(images).to(device)

            targets_p4, targets_p5 = [], []
            for t in targets:
                if t is None or t.numel() == 0:
                    targets_p4.append(torch.zeros((0,6), device=device))
                    targets_p5.append(torch.zeros((0,6), device=device))
                else:
                    t = t.to(device)
                    conf = torch.ones((t.shape[0],1), device=device)
                    t6 = torch.cat([t, conf], dim=1)
                    targets_p4.append(t6)
                    targets_p5.append(t6)

            with autocast():
                (cls_p4, reg_p4), (cls_p5, reg_p5) = model(images)
                loss_dict = criterion((cls_p4,reg_p4),(cls_p5,reg_p5),targets_p4,targets_p5)
                loss = loss_dict['total']

            total_loss += loss.item()

    return total_loss / num_batches

# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    (RUNS_DIR/'train'/'weights').mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_loader = DataLoader(
        MCUDetectionDataset(TRAIN_IMG_DIR,TRAIN_LABEL_DIR,args.img_size,transform),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate_fn
    )

    val_loader = DataLoader(
        MCUDetectionDataset(VAL_IMG_DIR,VAL_LABEL_DIR,args.img_size,transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate_fn
    )

    print("\n🧠 Creating MCUDetector...")
    model = MCUDetector(num_classes=NUM_CLASSES).to(device)
    criterion = MCUDetectionLoss(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',0.5,10)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # print_gpu_memory("After CUDA cache reset")

    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"📁 Saving to: {RUNS_DIR/'train'/'weights'/'best_mcu.pt'}")

    best_val_loss = float('inf')
    epochs_to_run = 1 if args.debug else args.epochs

    for epoch in range(epochs_to_run):
        train_loss = train_one_epoch(model,train_loader,criterion,optimizer,device,epoch,epochs_to_run)
        val_loss = validate_detector(model,val_loader,criterion,device)
        scheduler.step(val_loss)

        print(f"\n📊 Epoch {epoch+1}/{epochs_to_run}")
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch':epoch+1,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'val_loss':best_val_loss,
                'args':vars(args)
            }, RUNS_DIR/'train'/'weights'/'best_mcu.pt')
            print(f"   ✓ BEST MODEL SAVED!")
        else:
            print(f"   No improvement (best: {best_val_loss:.4f})")

        if args.debug:
            break

    print("\n✅ TRAINING COMPLETE!")

if __name__ == "__main__":
    main()
