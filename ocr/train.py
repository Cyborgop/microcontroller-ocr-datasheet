import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CRNN
from dataset import OCRDataset, load_labels, collate_fn
from utils import decode_output, BLANK_IDX
import argparse
import logging
import matplotlib.pyplot as plt
from rapidfuzz.distance import Levenshtein

# --- Configuration with argparse ---
parser = argparse.ArgumentParser(description='Train CRNN OCR model')
parser.add_argument('--img_height', type=int, default=32)
parser.add_argument('--img_width', type=int, default=128)
parser.add_argument('--num_classes', type=int, default=37)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('--save_path', type=str, default='best_model.pth')
args = parser.parse_args()

# Setup logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Gaussian Noise Transform
class AddGaussianNoise(object):
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * 0.05

# Data augmentation pipeline
transform = transforms.Compose([
    transforms.Resize((args.img_height, args.img_width)),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
        shear=5
    ),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2
    ),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    AddGaussianNoise(),
    transforms.Normalize([0.5], [0.5])
])

# Load labels
train_labels = load_labels(r"D:/microcontroller-ocr-datasheet/microcontroller-ocr-datasheet/data/train_labels.txt")
test_labels = load_labels(r"D:/microcontroller-ocr-datasheet/microcontroller-ocr-datasheet/data/test_labels.txt")

# Datasets and loaders
train_dataset = OCRDataset(r"D:/microcontroller-ocr-datasheet/microcontroller-ocr-datasheet/data/train", train_labels, transform=transform)
test_dataset = OCRDataset(r"D:/microcontroller-ocr-datasheet/microcontroller-ocr-datasheet/data/test", test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

# Model, criterion, optimizer
model = CRNN(img_height=args.img_height, num_channels=1, num_classes=args.num_classes).to(DEVICE)
criterion = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Utility function to calculate CER
def cer(pred, gt):
    return Levenshtein.distance(pred, gt) / max(1, len(gt))

# Training function with checkpointing, early stopping, validation metrics
def train():
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for images, targets, input_lengths, label_lengths in train_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            input_lengths = input_lengths.to(DEVICE)
            label_lengths = label_lengths.to(DEVICE)

            logits = model(images)  # (B, W, C)
            logits = logits.permute(1, 0, 2)  # (W, B, C)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            loss = criterion(log_probs, targets, input_lengths, label_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        total_cer = 0
        total_samples = 0
        with torch.no_grad():
            for images, targets, input_lengths, label_lengths in test_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                input_lengths = input_lengths.to(DEVICE)
                label_lengths = label_lengths.to(DEVICE)

                logits = model(images)
                logits = logits.permute(1, 0, 2)
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)
                loss = criterion(log_probs, targets, input_lengths, label_lengths)
                val_loss += loss.item()

                preds = decode_output(logits)

                # CER calculation
                start = 0
                for i, length in enumerate(label_lengths):
                    gt = ''.join([chr(t + ord('a')) if t < 26 else str(t - 26) for t in targets[start:start+length].cpu().numpy()])
                    start += length
                    total_cer += cer(preds[i], gt)
                    total_samples += 1

        avg_val_loss = val_loss / len(test_loader)
        avg_cer = total_cer / total_samples if total_samples > 0 else 0

        logging.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, CER: {avg_cer:.4f}")
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, CER: {avg_cer:.4f}")

        # Checkpointing
        if avg_val_loss < best_loss:
            torch.save(model.state_dict(), args.save_path)
            best_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        # Step scheduler
        scheduler.step(avg_val_loss)

# Function to print final sample predictions after training
def print_final_predictions():
    model.eval()
    with torch.no_grad():
        for images, targets, input_lengths, label_lengths in test_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            input_lengths = input_lengths.to(DEVICE)
            label_lengths = label_lengths.to(DEVICE)

            logits = model(images)
            logits = logits.permute(1, 0, 2)
            preds = decode_output(logits)
            print("\nFinal sample predictions vs ground truth:")
            start = 0
            for i, length in enumerate(label_lengths):
                gt = ''.join([chr(t + ord('a')) if t < 26 else str(t - 26) for t in targets[start:start+length].cpu().numpy()])
                print(f"Pred: {preds[i]} | GT: {gt}")
                start += length
                if i >= 4:  # Print only first 5 samples
                    break
            break  # Only the first batch

# Visualize augmentations before training (optional)
# visualize_augmentations(train_dataset, num_images=5)

# Run training and print final predictions
if __name__ == '__main__':
    train()
    print_final_predictions()
