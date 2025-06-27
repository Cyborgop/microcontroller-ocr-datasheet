import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import OCRDataset, load_labels, collate_fn
from model import EnhancedCRNN
from utils import (
    decode_output, BLANK_IDX, post_process_prediction,
    calculate_cer, calculate_wer, char2idx, idx2char,
    normalize_label, correct_ocr_text
)

def indices_to_string(indices):
    chars = []
    for idx in indices:
        idx = idx.item() if hasattr(idx, 'item') else idx
        if idx != BLANK_IDX and idx in idx2char:
            chars.append(idx2char[idx])
    return ''.join(chars)

def train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, args):
    model.train()
    total_loss = 0
    debug_printed = False
    for batch_idx, (images, targets, input_lens, label_lens) in enumerate(train_loader):
        images, targets = images.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        input_lens, label_lens = input_lens.to(DEVICE), label_lens.to(DEVICE)
        logits = model(images)
        log_probs = nn.functional.log_softmax(logits.permute(1, 0, 2), dim=2)
        loss = criterion(log_probs, targets, input_lens, label_lens)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        # Debug: Print logits and decoded predictions for first batch
        if not debug_printed:
            print("\n[DEBUG] First batch logits argmax (per time step):")
            print(torch.argmax(logits, dim=2))
            print("[DEBUG] BLANK_IDX:", BLANK_IDX)
            preds = decode_output(logits)
            print("[DEBUG] Decoded predictions for first batch:")
            for i, pred in enumerate(preds):
                print(f"  Pred {i}: '{pred}'")
            debug_printed = True

        if args.debug_one_batch:
            break
    return total_loss / len(train_loader)

def validate(model, test_loader, criterion, DEVICE, args):
    model.eval()
    val_loss, total_cer, total_wer, count = 0, 0, 0, 0
    sample_predictions = []
    debug_printed = False
    with torch.no_grad():
        for images, targets, input_lens, label_lens in test_loader:
            images, targets = images.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            input_lens, label_lens = input_lens.to(DEVICE), label_lens.to(DEVICE)
            logits = model(images)
            log_probs = nn.functional.log_softmax(logits.permute(1, 0, 2), dim=2)
            loss = criterion(log_probs, targets, input_lens, label_lens)
            val_loss += loss.item()
            preds = decode_output(logits)
            start = 0
            for i, L in enumerate(label_lens):
                gt_indices = targets[start:start+L].cpu().numpy()
                gt = indices_to_string(gt_indices)
                start += L
                corrected_pred = correct_ocr_text(preds[i])
                pred_norm = normalize_label(corrected_pred)
                gt_norm = normalize_label(gt)
                total_cer += calculate_cer(pred_norm, gt_norm)
                total_wer += calculate_wer(pred_norm, gt_norm)
                count += 1
                if len(sample_predictions) < 5:
                    sample_predictions.append((corrected_pred, gt))

            # Debug: Print logits and decoded predictions for first batch
            if not debug_printed:
                print("\n[DEBUG] Validation batch logits argmax (per time step):")
                print(torch.argmax(logits, dim=2))
                print("[DEBUG] Decoded predictions for validation batch:")
                for i, pred in enumerate(preds[:5]):
                    print(f"  Pred {i}: '{pred}'")
                debug_printed = True

            if args.debug_one_batch:
                break

    return val_loss / len(test_loader), (total_cer / count if count else 0), (total_wer / count if count else 0), sample_predictions

def main():
    parser = argparse.ArgumentParser(description='Train YOLO+CRNN OCR model (Debug Mode)')
    parser.add_argument('--train_dir', type=str, default='data/dataset_train/images/train')
    parser.add_argument('--test_dir', type=str, default='data/dataset_test/images/train')
    parser.add_argument('--train_labels', type=str, default='data/dataset_train/labels/train')
    parser.add_argument('--test_labels', type=str, default='data/dataset_test/labels/train')
    parser.add_argument('--yolo_model', type=str, default='data/runs/detect/train2/weights/best.pt')
    parser.add_argument('--img_height', type=int, default=32)
    parser.add_argument('--img_width', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=len(char2idx)+1)
    parser.add_argument('--batch_size', type=int, default=1)  # For overfit/debug, use 1
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--debug_one_batch', action='store_true', help='Overfit a single batch for debugging')
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    print(f"BLANK_IDX: {BLANK_IDX}, num_classes: {args.num_classes}")

    # For overfit/debug: use only one sample for train/val
    train_labels = load_labels(args.train_labels)
    test_labels = load_labels(args.test_labels)
    # Use only first sample for overfit
    if args.debug_one_batch:
        train_labels = {k: train_labels[k] for k in list(train_labels)[:1]}
        test_labels = {k: test_labels[k] for k in list(test_labels)[:1]}
        print("[DEBUG] Overfitting a single batch (first image/label)")

    train_transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = OCRDataset(args.train_dir, train_labels, train_transform, True, args.yolo_model)
    test_dataset = OCRDataset(args.test_dir, test_labels, test_transform, True, args.yolo_model)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    model = EnhancedCRNN(args.img_height, 1, args.num_classes, args.dropout, False).to(DEVICE)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)

    print("Starting training (Debug Mode)...")
    best_val, no_improve = float('inf'), 0
    N = 10
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, args)
        val_loss, val_cer, val_wer, sample_predictions = validate(model, test_loader, criterion, DEVICE, args)
        scheduler.step(val_loss)
        print(f"[{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f}  "
              f"Val Loss: {val_loss:.4f}  CER: {val_cer:.4f}  WER: {val_wer:.4f}")

        if val_loss < best_val:
            torch.save(model.state_dict(), args.save_path)
            best_val, no_improve = val_loss, 0
            print(f"Model saved to {args.save_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % N == 0:
            print("\nSample Predictions vs Ground Truth (Debug):")
            for pred, gt in sample_predictions:
                print(f"Pred: '{pred}' | GT: '{gt}'")

        if args.debug_one_batch:
            break

    final_sample_predictions = sample_predictions
    print("Training complete.")
    print("\nSample Predictions vs Ground Truth (Final):")
    for pred, gt in final_sample_predictions:
        print(f"Pred: '{pred}' | GT: '{gt}'")

    # Additional diagnostics: print model and loss config
    print("\n[DEBUG] Model architecture summary:")
    print(model)
    print("\n[DEBUG] CTCLoss blank index:", BLANK_IDX)
    print("[DEBUG] Label encoding example (first train label):")
    if train_labels:
        first_label = next(iter(train_labels.values()))
        print("  Label string:", first_label)
        print("  Encoded:", [char2idx[c] for c in first_label if c in char2idx])

if __name__ == "__main__":
    main()
