#!/usr/bin/env python3
from torchvision import transforms
import torch
from model import MCUDetector, MCUDetectionLoss
from torch.utils.data import DataLoader
from pathlib import Path

# Paths (your setup)
DATA_DIR = Path.cwd() / "data"
VAL_IMG_DIR = DATA_DIR / "dataset_test/images/train"  # your val
VAL_LABEL_DIR = DATA_DIR / "dataset_test/labels/train"
NUM_CLASSES = 24
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load FINAL model
model = MCUDetector(NUM_CLASSES).to(device)
model.load_state_dict(torch.load(DATA_DIR / "runs/detect/best_mcu.pt", map_location=device))
model.eval()

criterion = MCUDetectionLoss(NUM_CLASSES).to(device)

# Your dataset/loader (reuse)
from train import MCUDetectionDataset, collate_fn  # copy from train.py
transform = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
val_dataset = MCUDetectionDataset(VAL_IMG_DIR, VAL_LABEL_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Final eval
total_loss = 0.0
with torch.no_grad():
    for batch in val_loader:
        images, targets = batch
        images = images.to(device)
        
        # Scale targets (copy from train)
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
        loss_dict = criterion(pred[0], pred[1], targets_p4, targets_p5)
        total_loss += loss_dict["total"].item()

print(f"🎯 FINAL BEST MODEL:")
print(f"Val Loss: {total_loss/len(val_loader):.4f}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Status: {'✅ PRODUCTION READY' if total_loss/len(val_loader) < 2.0 else '⚠️ REVIEW'}")
