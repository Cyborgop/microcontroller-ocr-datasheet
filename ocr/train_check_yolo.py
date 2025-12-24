#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path

from model import MCUDetector, MCUDetectionLoss
from train import MCUDetectionDataset, TRAIN_IMG_DIR, TRAIN_LABEL_DIR  # reuse dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"⚡ Quick sanity train on single batch (device={device})")

    transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = MCUDetectionDataset(
        img_dir=TRAIN_IMG_DIR,
        label_dir=TRAIN_LABEL_DIR,
        img_size=512,
        transform=transform,
    )

    # tiny loader, no shuffle, just to get first batch
    loader = DataLoader(dataset, batch_size=4, shuffle=False,
                        num_workers=0, collate_fn=lambda x: x)

    batch = next(iter(loader))        # one batch only
    images, targets = zip(*batch)
    images = torch.stack(images).to(device)

    # build targets_p4 / targets_p5 like in full train
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

    model = MCUDetector(num_classes=24).to(device)
    criterion = MCUDetectionLoss(num_classes=24).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    for step in range(5):  # just a few steps
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
        optimizer.step()
        print(f"step {step+1}/5  loss={loss.item():.4f}")

    print("✅ quick sanity check finished (model, loss, backward all OK).")

if __name__ == "__main__":
    main()
