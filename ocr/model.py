import torch
import torch.nn as nn
from PIL import Image
import cv2
import numpy as np


class BidirectionalLSTM(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, dropout=0.1):
        super().__init__()
        # Fix dropout warning by setting num_layers > 1
        self.rnn = nn.LSTM(
            in_features, hidden_size, num_layers=2,  # Changed from 1 to 2
            bidirectional=True, batch_first=True, 
            dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.embedding = nn.Linear(hidden_size * 2, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.embedding(x)
        return x

class EnhancedCRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes, dropout=0.1, use_attention=False):
        super().__init__()
        assert img_height == 32, "Expected image height 32"
        self.use_attention = use_attention
        
        # Simplified CNN for better convergence
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1), (2,1)),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1), (2,1)),
            nn.Dropout2d(dropout)
        )
        
        # Reduced complexity for small dataset
        self.rnn1 = BidirectionalLSTM(512, 256, 256, dropout=dropout)
        self.rnn2 = BidirectionalLSTM(256, 128, num_classes, dropout=dropout)
        
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=num_classes, num_heads=1, dropout=dropout, batch_first=True
            )

    def forward(self, x):
        conv = self.cnn(x)  # (B, C, H, W)
        b, c, h, w = conv.size()
        if h == 1:
            conv = conv.squeeze(2)
        else:
            conv = conv.mean(2)
        conv = conv.permute(0, 2, 1)  # (B, W, C)
        
        rnn_out1 = self.rnn1(conv)
        rnn_out2 = self.rnn2(rnn_out1)
        
        if self.use_attention:
            attended_out, _ = self.attention(rnn_out2, rnn_out2, rnn_out2)
            return attended_out
        return rnn_out2

# --- Detector Wrapper for Ultralytics YOLO ---
class UltralyticsYOLOWrapper:
    """
    Wraps Ultralytics YOLO to provide a .predict(image) interface.
    """
    def __init__(self, yolo_model_path):
        from ultralytics import YOLO
        self.model = YOLO(yolo_model_path)

    def predict(self, image):
        results = self.model(image)
        detections = []
        if hasattr(results, 'boxes'):
            boxes = results.boxes
        elif len(results) > 0 and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
        else:
            boxes = []
        for box in boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            confidence = float(box.conf.cpu().numpy())
            class_id = int(box.cls.cpu().numpy())
            detections.append([x1, y1, x2, y2, confidence, class_id])
        return detections

# --- Custom YOLO Detector Example ---
class CustomYOLODetector:
    """
    Example custom YOLO detector class.
    Replace the model and postprocessing with your actual YOLO implementation.
    """
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        # For demonstration: a dummy CNN backbone
        # Replace with your actual YOLO model loading code
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*32*32, 1000),  # assuming input 256x256
            nn.ReLU(),
            nn.Linear(1000, 6*6*25)    # 6x6 grid, 25 outputs per grid cell (example)
        ).to(self.device)
        self.model.eval()

        # If you have a real model, load weights here:
        # if model_path:
        #     self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def preprocess(self, image):
        img = cv2.resize(image, (256, 256))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)
        return img_tensor

    def postprocess(self, outputs, conf_threshold=0.5):
        # Dummy postprocessing: generate fake detections for demonstration
        # Replace this with actual YOLO output decoding
        detections = [
            [50, 50, 150, 150, 0.9, 1],    # x1, y1, x2, y2, confidence, class_id
            [120, 120, 220, 220, 0.75, 3]
        ]
        filtered = [det for det in detections if det[4] >= conf_threshold]
        return filtered

    def predict(self, image):
        input_tensor = self.preprocess(image)
        with torch.no_grad():
            outputs = self.model(input_tensor)
        detections = self.postprocess(outputs)
        return detections

# --- Detector-Agnostic Pipeline ---
class YOLOCRNNPipeline(nn.Module):
    """
    Pipeline combining any YOLO-like detector (with .predict(image)) and CRNN recognition.
    """
    def __init__(self, detector, crnn_model, device='cuda'):
        super().__init__()
        self.detector = detector  # Any detector with .predict(image)
        self.ocr_model = crnn_model
        self.device = device

    def forward(self, image_path, ocr_transform=None):
        results = []
        image = cv2.imread(image_path)
        detections = self.detector.predict(image)
        for det in detections:
            x1, y1, x2, y2, confidence, class_id = det
            crop = image[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop_pil = Image.fromarray(crop)
            if ocr_transform:
                crop_tensor = ocr_transform(crop_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    self.ocr_model.eval()
                    logits = self.ocr_model(crop_tensor)
                    from utils import decode_output
                    ocr_text = decode_output(logits.permute(1, 0, 2))[0]
            else:
                ocr_text = "No OCR transform provided"
            results.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'class_id': class_id,
                'ocr_text': ocr_text
            })
        return results

# --- Usage Example (Uncomment for testing) ---

# from torchvision import transforms

# # For Ultralytics YOLO
# detector = UltralyticsYOLOWrapper("data/runs/detect/train2/weights/best.pt")
# # For your custom YOLO, use:
# # detector = CustomYOLODetector(model_path='path/to/custom_yolo.pth', device='cpu')

# # Load CRNN model
# crnn_model = EnhancedCRNN(img_height=32, num_channels=1, num_classes=38)
# crnn_model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
# crnn_model.eval()

# ocr_transform = transforms.Compose([
#     transforms.Resize((32, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])

# pipeline = YOLOCRNNPipeline(detector, crnn_model, device='cpu')
# results = pipeline("Untitled-design-96.jpg", ocr_transform=ocr_transform)
# print(results)