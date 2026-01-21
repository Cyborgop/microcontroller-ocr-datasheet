#!/usr/bin/env python3
"""
sanity_overfit.py
Fast overfit sanity check for MCUDetector
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from pathlib import Path

from model import MCUDetector, MCUDetectionLoss
from train import MCUDetectionDataset, TRAIN_IMG_DIR, TRAIN_LABEL_DIR

# ---------------- CONFIG ----------------
NUM_IMAGES = 8
EPOCHS = 30
BATCH_SIZE = 2
LR = 1e-4
IMG_SIZE = 512
NUM_CLASSES = 24
# ----------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"🧪 Sanity overfit test on {NUM_IMAGES} images ({device})")

    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    full_dataset = MCUDetectionDataset(
        TRAIN_IMG_DIR,
        TRAIN_LABEL_DIR,
        img_size=IMG_SIZE,
        transform=transform
    )

    subset = Subset(full_dataset, list(range(NUM_IMAGES)))
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x
    )

    model = MCUDetector(NUM_CLASSES).to(device)
    criterion = MCUDetectionLoss(
        NUM_CLASSES,
        bbox_weight=0.05,
        obj_weight=1.0,
        cls_weight=0.5
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for batch in loader:
            images, targets = zip(*batch)
            images = torch.stack(images).to(device)

            # ---------- sanity targets ----------
            targets_p4, targets_p5 = [], []
            for t in targets:
                if t.numel() == 0:
                    empty = torch.zeros((0, 6), device=device)
                    targets_p4.append(empty)
                    targets_p5.append(empty)
                else:
                    t = t.to(device)
                    conf = torch.ones((t.shape[0], 1), device=device)
                    t6 = torch.cat([t, conf], dim=1)
                    targets_p4.append(t6)          # all to P4
                    targets_p5.append(
                        torch.zeros((0, 6), device=device)
                    )

            optimizer.zero_grad()
            pred_p4, pred_p5 = model(images)
            loss_dict = criterion(pred_p4, pred_p5,
                                   targets_p4, targets_p5)
            loss = loss_dict["total"]
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        print(f"Epoch {epoch+1:02d}/{EPOCHS}  loss={epoch_loss:.4f}")

    print("✅ Sanity overfit test finished")

if __name__ == "__main__":
    main()
