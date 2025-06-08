import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CRNN
from dataset import OCRDataset, load_labels, collate_fn
from utils import decode_output, BLANK_IDX

# --- Configuration ---
IMG_HEIGHT = 32
IMG_WIDTH = 128
NUM_CLASSES = 37
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_labels = load_labels(r"G:/My Drive/MTP/ocr_dataset/train_labels.txt")
test_labels = load_labels(r"G:/My Drive/MTP/ocr_dataset/test_labels.txt")

train_dataset = OCRDataset(r"G:/My Drive/MTP/ocr_dataset/train", train_labels, transform=transform)
test_dataset = OCRDataset(r"G:/My Drive/MTP/ocr_dataset/test", test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = CRNN(img_height=IMG_HEIGHT, num_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
criterion = torch.nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train():
    model.train()
    for epoch in range(EPOCHS):
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

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

def test():
    model.eval()
    with torch.no_grad():
        for images, _, _, _ in test_loader:
            images = images.to(DEVICE)
            output = model(images)  # (B, W, C)
            preds = output.permute(1, 0, 2)
            decoded = decode_output(preds)
            print("Sample Predictions:", decoded[:5])
            break

if __name__ == '__main__':
    train()
    test()
