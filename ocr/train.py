#!/usr/bin/env python3
"""
MCUDetector Training Script (Final Fixed)
YOLO-style training, scale-aware, AMP-safe, edge-stable
"""

# ================= CUDA MEMORY FIX =================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# ================= IMPORTS =================
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np

from model import MCUDetector, MCUDetectionLoss

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ================= PATHS =================
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = DATA_DIR / "runs" / "detect"

TRAIN_IMG_DIR = DATA_DIR / "dataset_train/images/train"
TRAIN_LABEL_DIR = DATA_DIR / "dataset_train/labels/train"
VAL_IMG_DIR   = DATA_DIR / "dataset_test/images/train"
VAL_LABEL_DIR = DATA_DIR / "dataset_test/labels/train"
NUM_CLASSES = 24

# ================= DATASET =================
class MCUDetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=512, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.transform = transform
        self.image_files = sorted(self.img_dir.glob("*.jpg"))
        print(f"📊 Found {len(self.image_files)} images in {img_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_dir / img_path.with_suffix(".txt").name

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        targets = []
        if label_path.exists():
            with open(label_path) as f:
                for ln in f:
                    cls, x, y, w, h = map(float, ln.split()[:5])
                    targets.append([cls, x, y, w, h])

        targets = (
            torch.tensor(targets, dtype=torch.float32)
            if len(targets)
            else torch.zeros((0, 5), dtype=torch.float32)
        )

        if self.transform:
            image = self.transform(image)

        return image, targets


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]  # list of per-image tensors
    return images, targets


# ================= TRAINING =================
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, accum_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        images, targets = batch
        images = images.to(device)

        targets_p4, targets_p5 = [], []
        for t in targets:
            if t.numel() == 0:
                targets_p4.append(torch.zeros((0, 6), device=device))
                targets_p5.append(torch.zeros((0, 6), device=device))
                continue

            t = t.to(device)
            conf = torch.ones((t.shape[0], 1), device=device)
            t6 = torch.cat([t, conf], dim=1)

            area = t[:, 3] * t[:, 4] * 512 * 512
            targets_p4.append(t6[area < 1024])   # < 32×32
            targets_p5.append(t6[area >= 1024])  # ≥ 32×32

        with autocast():
            pred = model(images)
            loss_dict = criterion(pred[0], pred[1], targets_p4, targets_p5)
            loss = loss_dict["total"] / accum_steps

        # DEBUG: first batch each epoch
        if i == 0:
            print(
                f"[DEBUG] bbox={loss_dict['bbox']:.4f} | "
                f"obj={loss_dict['obj']:.4f} | "
                f"cls={loss_dict['cls']:.4f}"
            )

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        images, targets = batch
        images = images.to(device)

        targets_p4, targets_p5 = [], []
        for t in targets:
            if t.numel() == 0:
                targets_p4.append(torch.zeros((0, 6), device=device))
                targets_p5.append(torch.zeros((0, 6), device=device))
                continue

            t = t.to(device)
            conf = torch.ones((t.shape[0], 1), device=device)
            t6 = torch.cat([t, conf], dim=1)

            area = t[:, 3] * t[:, 4] * 512 * 512
            targets_p4.append(t6[area < 1024])
            targets_p5.append(t6[area >= 1024])

        pred = model(images)
        loss = criterion(pred[0], pred[1], targets_p4, targets_p5)["total"]
        total_loss += loss.item()

    return total_loss / len(loader)


# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_loader = DataLoader(
        MCUDetectionDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        MCUDetectionDataset(VAL_IMG_DIR, VAL_LABEL_DIR, transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    model = MCUDetector(NUM_CLASSES).to(device)
    criterion = MCUDetectionLoss(NUM_CLASSES).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    scaler = GradScaler()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    best_loss = None
    accum_steps = 4

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, accum_steps
        )

        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step()  # ✅ correct for cosine warm restarts

        print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")

        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), RUNS_DIR / "best_mcu.pt")
            print("✓ Best model saved")

    print("✅ Training complete")


if __name__ == "__main__":
    main()
