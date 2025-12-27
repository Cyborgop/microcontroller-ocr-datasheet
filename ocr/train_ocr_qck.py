#!/usr/bin/env python3
"""
⚡ Quick sanity check training for MCUDetector (single batch, few steps).

Usage:
    cd ocr
    python train_debug.py
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import MCUDetector, MCUDetectionLoss
from train import MCUDetectionDataset, TRAIN_IMG_DIR, TRAIN_LABEL_DIR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"⚡ Quick sanity train on single batch (device={device})")

    # Same normalization as main train.py
    transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Reuse full dataset class and paths but we’ll pull only first batch
    dataset = MCUDetectionDataset(
        img_dir=TRAIN_IMG_DIR,
        label_dir=TRAIN_LABEL_DIR,
        img_size=512,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x,  # same as in train.py
    )

    # Take a single batch
    batch = next(iter(loader))
    images, targets = zip(*batch)
    images = torch.stack(images).to(device)

    # Build targets_p4 / targets_p5 as in full training
    targets_p4, targets_p5 = [], []
    for t in targets:
        if t is None or t.numel() == 0:
            empty = torch.zeros((0, 6), device=device)
            targets_p4.append(empty)
            targets_p5.append(empty)
            continue
        t = t.to(device)  # (N, 5): [cls, x, y, w, h]
        conf = torch.ones((t.shape[0], 1), device=device)
        t6 = torch.cat([t, conf], dim=1)  # (N, 6)
        targets_p4.append(t6)
        targets_p5.append(t6)

    # Model, loss, optimizer
    model = MCUDetector(num_classes=24).to(device)
    criterion = MCUDetectionLoss(num_classes=24).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    steps = 5
    for step in range(steps):
        optimizer.zero_grad()

        (cls_p4, reg_p4), (cls_p5, reg_p5) = model(images)
        loss_dict = criterion(
            (cls_p4, reg_p4),
            (cls_p5, reg_p5),
            targets_p4,
            targets_p5
        )
        loss = loss_dict["total"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"step {step+1}/{steps}  loss={loss.item():.4f}")

    print("✅ Sanity check finished: forward, loss, backward all ran successfully.")


if __name__ == "__main__":
    main()
