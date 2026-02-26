import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path

from model import MCUDetector
from utils import decode_predictions, non_max_suppression, CLASSES
from dataset import MCUDetectionDataset

# ================= CONFIG =================
IMG_DIR = "data/dataset_test/images/train"
LABEL_DIR = "data/dataset_test/labels/train"
WEIGHTS = "runs/detect/train/best_mcu.pt"   # adjust if needed
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_IMAGES = 5
CONF_THRESH = 0.2
IOU_THRESH = 0.5
# =========================================


def plot_boxes(img, gt_boxes, pred_boxes):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)
    ax.axis("off")

    # GT: green
    for x1, y1, x2, y2, cls in gt_boxes:
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                              fill=False, edgecolor="g", linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1, f"GT:{CLASSES[cls]}", color="g")

    # Pred: red
    for x1, y1, x2, y2, cls, score in pred_boxes:
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                              fill=False, edgecolor="r", linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y2, f"P:{CLASSES[cls]}:{score:.2f}", color="r")

    plt.show()


def main():
    # Load dataset
    dataset = MCUDetectionDataset(
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        img_size=IMG_SIZE,
        transform=None
    )

    # Load model
    model = MCUDetector(num_classes=len(CLASSES)).to(DEVICE)
    ckpt = torch.load(WEIGHTS, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval()

    for idx in range(NUM_IMAGES):
        image, targets = dataset[idx]

        # GT boxes
        H, W = IMG_SIZE, IMG_SIZE
        gt_boxes = []
        for t in targets:
            cls, cx, cy, w, h = t.tolist()
            x1 = (cx - w/2) * W
            y1 = (cy - h/2) * H
            x2 = (cx + w/2) * W
            y2 = (cy + h/2) * H
            gt_boxes.append([x1, y1, x2, y2, int(cls)])

        # Inference
        inp = image.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            preds = model(inp)

        detections = decode_predictions(preds, IMG_SIZE)[0]
        detections = non_max_suppression(
            detections,
            conf_thres=CONF_THRESH,
            iou_thres=IOU_THRESH
        )

        pred_boxes = []
        if len(detections):
            for det in detections:
                cls, conf, x1, y1, x2, y2 = det.tolist()
                pred_boxes.append([x1, y1, x2, y2, int(cls), conf])

        # Image for display
        img_np = image.permute(1, 2, 0).cpu().numpy()

        print(f"\nImage {idx}: GT={len(gt_boxes)} Pred={len(pred_boxes)}")
        plot_boxes(img_np, gt_boxes, pred_boxes)


if __name__ == "__main__":
    main()