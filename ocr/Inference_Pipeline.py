# A. Inference Pipeline (Detection + OCR)
# Example script to run detection + OCR on new images:

# python
from ultralytics import YOLO
from model import EnhancedCRNN
from utils import decode_output, preprocess_crop_for_ocr
import torch
from torchvision import transforms
from PIL import Image
import cv2
import os

# Load YOLO detector and CRNN model
yolo_detector = YOLO("data/runs/detect/train2/weights/best.pt")
crnn_model = EnhancedCRNN(img_height=32, num_channels=1, num_classes=38)
crnn_model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
crnn_model.eval()

# Image preprocessing for CRNN
ocr_transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def detect_and_ocr(image_path):
    # Detection
    results = yolo_detector(image_path)
    image = cv2.imread(image_path)
    outputs = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        crop = image[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_pil = Image.fromarray(crop)
        crop_tensor = ocr_transform(crop_pil).unsqueeze(0)
        with torch.no_grad():
            logits = crnn_model(crop_tensor)
            pred = decode_output(logits)
        outputs.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "ocr_text": pred[0]
        })
    return outputs

# Example usage
image_path = "Untitled-design-96.jpg"
results = detect_and_ocr(image_path)
print(results)