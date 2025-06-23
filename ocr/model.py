import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import cv2
from PIL import Image

class BidirectionalLSTM(nn.Module):
    """Enhanced Bidirectional LSTM with dropout and layer normalization"""
    def __init__(self, in_features, hidden_size, out_features, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(in_features, hidden_size, 
                          bidirectional=True, 
                          batch_first=True,
                          dropout=dropout if hidden_size > 1 else 0)
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
    """Enhanced CRNN with improved architecture for microcontroller recognition"""
    def __init__(self, img_height, num_channels, num_classes, dropout=0.1):
        super().__init__()
        assert img_height == 32, "Expected image height 32"
        
        # Enhanced CNN with batch normalization and dropout
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1), (2,1)),
            
            # Block 5
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Block 6
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1), (2,1)),
            
            nn.Dropout2d(dropout)
        )
        
        # Enhanced RNN with attention mechanism
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256, dropout=dropout),
            BidirectionalLSTM(256, 256, num_classes, dropout=dropout)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=num_classes, 
                                              num_heads=1, 
                                              dropout=dropout,
                                              batch_first=True)

    def forward(self, x):
        # CNN feature extraction
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        
        # Handle different height dimensions
        if h == 1:
            conv = conv.squeeze(2)
        else:
            conv = conv.mean(2)
        
        # Reshape for RNN: (batch, width, channels)
        conv = conv.permute(0, 2, 1)
        
        # RNN processing
        rnn_out = self.rnn(conv)
        
        # Apply attention (optional, can be disabled for simpler model)
        # attended_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
        
        return rnn_out  # or attended_out if using attention

# Keep original CRNN for compatibility
class CRNN(nn.Module):
    """Original CRNN model for backward compatibility"""
    def __init__(self, img_height, num_channels, num_classes):
        super().__init__()
        assert img_height == 32, "Expected image height 32"
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1), (2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2,1), (2,1))
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, num_classes)
        )

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1 or h == 2, f"Expected height=1 or 2, got {h}"
        conv = conv.squeeze(2) if h == 1 else conv.mean(2)
        conv = conv.permute(0, 2, 1)
        out = self.rnn(conv)
        return out

class YOLOCRNNPipeline(nn.Module):
    """Complete pipeline combining YOLO detection and CRNN recognition"""
    def __init__(self, yolo_model_path, crnn_model, device='cuda'):
        super().__init__()
        self.detector = YOLO(yolo_model_path)
        self.ocr_model = crnn_model
        self.device = device
        
    def forward(self, image_path, ocr_transform=None):
        """Process full PCB image and return detections with OCR results"""
        results = []
        
        # Get detections
        detections = self.detector(image_path, verbose=False)
        
        if len(detections) > 0 and len(detections[0].boxes) > 0:
            import cv2
            image = cv2.imread(image_path)
            
            for box in detections[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Extract and process crop
                crop = image[y1:y2, x1:x2]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop_pil = Image.fromarray(crop)
                
                if ocr_transform:
                    crop_tensor = ocr_transform(crop_pil).unsqueeze(0).to(self.device)
                    
                    # OCR recognition
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
