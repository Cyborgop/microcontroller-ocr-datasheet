import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
# from utils import decode_output  # Uncomment when utils ready

class BidirectionalLSTM(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            in_features, hidden_size, num_layers=2,  
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

class EnhancedCRNN(nn.Module):  # KEEP - Your solid OCR
    def __init__(self, img_height, num_channels, num_classes, dropout=0.1, use_attention=False):
        super().__init__()
        assert img_height == 32, "Expected image height 32"
        self.use_attention = use_attention
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
        
        self.rnn1 = BidirectionalLSTM(512, 256, 256, dropout=dropout)
        self.rnn2 = BidirectionalLSTM(256, 128, num_classes, dropout=dropout)
        
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=num_classes, num_heads=1, dropout=dropout, batch_first=True
            )

    def forward(self, x):
        conv = self.cnn(x) 
        b, c, h, w = conv.size()
        if h == 1:
            conv = conv.squeeze(2)
        else:
            conv = conv.mean(2)
        conv = conv.permute(0, 2, 1)  
        
        rnn_out1 = self.rnn1(conv)
        rnn_out2 = self.rnn2(rnn_out1)
        
        if self.use_attention:
            attended_out, _ = self.attention(rnn_out2, rnn_out2, rnn_out2)
            return attended_out
        return rnn_out2

# ========== MCU-TUNED CUSTOM YOLO (NEW) ==========
class CustomYOLOv8_MCU(nn.Module):
    def __init__(self, chip_size=32):
        super().__init__()
        self.chip_size = chip_size
        self.stride = 8
        
        # MCU-specific lightweight backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),   # P3/8
            self._mcu_c2f(32, 32),
            nn.Conv2d(32, 64, 3, 2, 1),  # P4/16
            self._mcu_c2f(64, 64),
            nn.Conv2d(64, 128, 3, 2, 1), # P5/32 ← MCU sweet spot
            self._mcu_c2f(128, 128),
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(64, 1, 1)  # Single-class MCU heatmap
        )
    
    def _mcu_c2f(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch//2, 1),
            nn.Conv2d(out_ch//2, out_ch//2, 3, groups=out_ch//2, padding=1),  # Depthwise
            nn.Conv2d(out_ch//2, out_ch, 1),
            nn.SiLU()
        )
    
    def forward(self, x):
        return self.head(self.backbone(x))  # [B,1,H/32,W/32]

class MCUYOLODetector:  # MCU Pipeline Interface
    def __init__(self, model_path=None, device='cpu', chip_size=32):
        self.device = device
        self.chip_size = chip_size
        self.model = CustomYOLOv8_MCU(chip_size).to(device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
    
    def predict(self, image):
        # MCU-optimized inference pipeline
        orig_h, orig_w = image.shape[:2]
        
        # Letterbox preserving aspect ratio
        scale = min(224/orig_w, 224/orig_h)
        new_w, new_h = int(orig_w*scale), int(orig_h*scale)
        resized = cv2.resize(image, (new_w, new_h))
        pad_w, pad_h = (224-new_w)//2, (224-new_h)//2
        img_pad = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, 114)
        
        # Normalize + tensor
        img_tensor = torch.tensor(img_pad.transpose(2,0,1)/255.).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            heatmap = self.model(img_tensor)[0,0].cpu().numpy()  # 28x28
        
        # MCU chip post-processing
        detections = self._postprocess(heatmap)
        return self._denormalize(detections, scale, pad_w, pad_h, orig_w, orig_h)
    
    def _postprocess(self, heatmap, conf_thresh=0.5):
        h, w = heatmap.shape
        detections = []
        for i in range(h):
            for j in range(w):
                if heatmap[i,j] > conf_thresh:
                    cx, cy = j*self.model.stride + self.chip_size//2, i*self.model.stride + self.chip_size//2
                    detections.append([cx-self.chip_size//2, cy-self.chip_size//2, 
                                     cx+self.chip_size//2, cy+self.chip_size//2, heatmap[i,j], 0])
        return detections[:5]  # Top 5 MCUs
    
    def _denormalize(self, detections, scale, pad_w, pad_h, orig_w, orig_h):
        return [[int(x/scale)-pad_w, int(y/scale)-pad_h, int(x2/scale)-pad_w, int(y2/scale)-pad_h, conf, cls] 
                for x,y,x2,y2,conf,cls in detections]

# ========== SIMPLIFIED PIPELINE (KEEP CRNN) ==========
class YOLOCRNNPipeline(nn.Module):
    def __init__(self, detector, crnn_model, device='cpu'):
        super().__init__()
        self.detector = detector
        self.ocr_model = crnn_model.to(device)
        self.device = device
    
    def forward(self, image_path, ocr_transform=None):
        results = []
        image = cv2.imread(image_path)
        if image is None:
            return results
            
        detections = self.detector.predict(image)
        
        for det in detections:
            x1, y1, x2, y2, confidence, class_id = map(int, det[:6])
            crop = image[y1:y2, x1:x2]
            if crop.size == 0: continue
                
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop_pil = Image.fromarray(crop_gray)
            
            if ocr_transform:
                try:
                    crop_tensor = ocr_transform(crop_pil).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        self.ocr_model.eval()
                        logits = self.ocr_model(crop_tensor)
                        # Simple argmax decode (replace with utils.decode_output when ready)
                        pred = torch.argmax(logits, dim=2).squeeze().cpu().numpy()
                        ocr_text = ''.join([str(c) for c in pred if c != 0])[:10]  # Top 10 chars
                except:
                    ocr_text = "OCR Error"
            else:
                ocr_text = "No transform"
                
            results.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'class_id': class_id,
                'ocr_text': ocr_text
            })
        return results

# ========== DEBUGGING (MCU TUNED) ==========
if __name__ == "__main__":
    # MCU Detector (your new research model)
    mcu_detector = MCUYOLODetector(device='cpu')  # Train first!
    
    # CRNN (your existing model - KEEP)
    crnn_model = EnhancedCRNN(img_height=32, num_channels=1, num_classes=38)
    # crnn_model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))  # Uncomment when ready
    crnn_model.eval()
    
    ocr_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # MCU Pipeline
    pipeline = YOLOCRNNPipeline(mcu_detector, crnn_model, device='cpu')
    results = pipeline("Untitled-design-96.jpg", ocr_transform=ocr_transform)
    print("MCU Pipeline Results:", results)
