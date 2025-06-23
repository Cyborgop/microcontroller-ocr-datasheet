import os
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from ultralytics import YOLO
import cv2
from PIL import Image
from utils import calculate_cer

# Enhanced modules
from dataset import OCRDataset, load_labels, collate_fn
from model import EnhancedCRNN
from utils import (
    decode_output, BLANK_IDX, post_process_prediction,
    MetricsTracker, preprocess_crop_for_ocr
)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Train YOLO+CRNN OCR model')
parser.add_argument('--train_dir', type=str, default='data/dataset_train/images/train',
                    help='Directory of training images')  # Directory of train images [1]
parser.add_argument('--test_dir', type=str, default='data/dataset_test/images/train',
                    help='Directory of test images')   # Directory of val/test images [1]
parser.add_argument('--train_labels', type=str, default='data/dataset_train/labels/train',
                     help='Folder with YOLO .txt label files for training')  # YOLO labels dir [2]
parser.add_argument('--test_labels', type=str, default='data/dataset_test/labels/train',
                    help='Folder with YOLO .txt label files for testing')   # YOLO labels dir [2]
parser.add_argument('--yolo_model', type=str,
    default='data/runs/detect/train2/weights/best.pt',
    help='Path to trained YOLOv8 weights')

parser.add_argument('--img_height', type=int, default=32,
                    help='Input image height for CRNN')  # CRNN input height [4]
parser.add_argument('--img_width', type=int, default=128,
                    help='Input image width for CRNN')
parser.add_argument('--num_classes', type=int, default=38,
                    help='Number of character classes including blank')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for DataLoader')
parser.add_argument('--epochs', type=int, default=100,
                    help='Maximum number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate for optimizer')
parser.add_argument('--patience', type=int, default=15,
                    help='Early stopping patience in epochs')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate in CRNN')
parser.add_argument('--save_path', type=str, default='best_model.pth',
                    help='File path to save the best model')
args = parser.parse_args()

# --- Logging Setup ---
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Device Selection ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Augmentation Pipeline ---
transform = transforms.Compose([
    transforms.Resize((args.img_height, args.img_width)),
    transforms.RandomAffine(degrees=3, translate=(0.01,0.01),
                            scale=(0.98,1.02), shear=1),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --- Load Labels (Directory or File) ---
train_labels = load_labels(args.train_labels)
test_labels  = load_labels(args.test_labels)

# --- Datasets & DataLoaders ---
train_dataset = OCRDataset(
    img_dir=args.train_dir,
    label_dict=train_labels,
    transform=transform,
    use_detection=True,
    yolo_model_path=args.yolo_model
)
test_dataset = OCRDataset(
    img_dir=args.test_dir,
    label_dict=test_labels,
    transform=transform,
    use_detection=True,
    yolo_model_path=args.yolo_model
)
train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,
                          batch_size=args.batch_size,
                          shuffle=False,
                          collate_fn=collate_fn)

# --- Model, Loss, Optimizer, Scheduler ---
model = EnhancedCRNN(
    img_height=args.img_height,
    num_channels=1,
    num_classes=args.num_classes,
    dropout=args.dropout
).to(DEVICE)
criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=7
)

# --- Metrics Tracker for CER/WER ---
metrics = MetricsTracker()

# --- Training Loop Definitions ---
def train_one_epoch():
    model.train()
    total_loss = 0
    for images, targets, input_lens, label_lens in train_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        input_lens, label_lens = input_lens.to(DEVICE), label_lens.to(DEVICE)
        logits = model(images)
        log_probs = nn.functional.log_softmax(logits.permute(1,0,2), dim=2)
        loss = criterion(log_probs, targets, input_lens, label_lens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate():
    model.eval()
    val_loss, total_cer, count = 0, 0, 0
    with torch.no_grad():
        for images, targets, input_lens, label_lens in test_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            log_probs = nn.functional.log_softmax(
                model(images).permute(1,0,2), dim=2)
            val_loss += criterion(log_probs, targets,
                                  input_lens.to(DEVICE),
                                  label_lens.to(DEVICE)).item()
            preds = decode_output(log_probs)
            start = 0
            for i, L in enumerate(label_lens):
                gt = ''.join([chr(t+97) if t<26 else str(t-26)
                              for t in targets[start:start+L].cpu().numpy()])
                start += L
                total_cer += calculate_cer(preds[i], gt)
                count += 1
    return val_loss / len(test_loader), (total_cer / count if count else 0)

# --- Main Training Routine ---
best_val, no_improve = float('inf'), 0
for epoch in range(args.epochs):
    train_loss = train_one_epoch()
    val_loss, val_cer = validate()
    scheduler.step(val_loss)
    logging.info(
        f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
        f"Val Loss={val_loss:.4f}, CER={val_cer:.4f}"
    )
    print(
        f"[{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f}  "
        f"Val Loss: {val_loss:.4f}  CER: {val_cer:.4f}"
    )
    if val_loss < best_val:
        torch.save(model.state_dict(), args.save_path)
        best_val, no_improve = val_loss, 0
    else:
        no_improve += 1
        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# --- Sample Inference on Test Batch ---
model.eval()
with torch.no_grad():
    images, targets, input_lens, label_lens = next(iter(test_loader))
    logits = model(images.to(DEVICE)).permute(1,0,2)
    preds = decode_output(logits)
    corrected = [post_process_prediction(p) for p in preds]
    print("\nSample Predictions vs Ground Truth:")
    start = 0
    for i, L in enumerate(label_lens):
        gt = ''.join([chr(t+97) if t<26 else str(t-26)
                      for t in targets[start:start+L].cpu().numpy()])
        start += L
        print(f"Pred: {corrected[i]} | GT: {gt}")
        if i >= 4: break
