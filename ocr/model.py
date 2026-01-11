import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from PIL import Image
from torchvision import transforms
from utils import SiLU

# =============================================================================
# CORE BUILDING BLOCKS
# =============================================================================

class RepDWConvRARM(nn.Module):
    def __init__(self, channels, stride=1, high_res=True):
        super().__init__()
        self.high_res = high_res and stride == 1
        self.dw3 = nn.Conv2d(
            channels, channels, 3,
            stride=stride, padding=1,
            groups=channels, bias=False
        )
        if self.high_res:
            self.dw_dilated = nn.Conv2d(
                channels, channels, 3,
                padding=2, dilation=2,
                groups=channels, bias=False
            )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.dw3(x)
        if self.high_res:
            out = out + self.dw_dilated(x)
        return self.bn(out)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, high_res=True, use_se=False):
        super().__init__()
        self.use_res = in_ch == out_ch and stride == 1
        self.token_mixer = RepDWConvRARM(in_ch, stride, high_res)

        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, in_ch // 4, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(in_ch // 4, in_ch, 1),
                nn.Sigmoid()
            )

        hidden = in_ch * 2
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        identity = x
        x = self.token_mixer(x)
        if self.use_se:
            x = x * self.se(x)
        x = self.channel_mixer(x)
        if self.use_res:
            x = x + identity
        return x


class BottleneckCSPBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks=1, ratio=0.25):
        super().__init__()
        hidden = int(out_ch * ratio)
        self.cv1 = nn.Conv2d(in_ch, hidden, 1, bias=False)
        self.blocks = nn.Sequential(*[
            DepthwiseSeparableConv(hidden, hidden, high_res=False)
            for _ in range(n_blocks)
        ])
        self.cv2 = nn.Conv2d(hidden * 2, out_ch, 1, bias=False)

    def forward(self, x):
        y = self.cv1(x)
        z = self.blocks(y)
        return self.cv2(torch.cat([y, z], dim=1))


class FPNModule(nn.Module):
    def __init__(self, ch_p4, ch_p5):
        super().__init__()
        self.lat_p4 = nn.Conv2d(ch_p4, 256, 1, bias=False)
        self.lat_p5 = nn.Conv2d(ch_p5, 256, 1, bias=False)
        self.refine_p4 = DepthwiseSeparableConv(256, 256, high_res=False)
        self.refine_p5 = DepthwiseSeparableConv(256, 256, high_res=False)
        self.down_p4 = DepthwiseSeparableConv(256, 256, stride=2)

    def forward(self, p4, p5):
        p4_lat = self.lat_p4(p4)
        p5_lat = self.lat_p5(p5)
        p5_up = F.interpolate(p5_lat, scale_factor=2, mode="nearest")
        p4_out = self.refine_p4(0.5 * (p4_lat + p5_up))
        p4_down = self.down_p4(p4_out)
        p5_out = self.refine_p5(0.5 * (p5_lat + p4_down))
        return p4_out, p5_out


# =============================================================================
# BACKBONE
# =============================================================================

class MCUDetectorBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            SiLU(inplace=True)
        )
        self.p2 = DepthwiseSeparableConv(32, 32)
        self.p3_down = DepthwiseSeparableConv(32, 64, stride=2)
        self.p3 = nn.Sequential(*[BottleneckCSPBlock(64, 64, 2) for _ in range(2)])
        self.p4_down = DepthwiseSeparableConv(64, 128, stride=2)
        self.p4 = nn.Sequential(*[BottleneckCSPBlock(128, 128, 2) for _ in range(2)])
        self.p5_down = DepthwiseSeparableConv(128, 256, stride=2)
        self.p5 = nn.Sequential(*[BottleneckCSPBlock(256, 256, 2) for _ in range(2)])

    def forward(self, x):
        x = self.stem(x)
        x = self.p2(x)
        x = self.p3(self.p3_down(x))
        p4 = self.p4(self.p4_down(x))
        p5 = self.p5(self.p5_down(p4))
        return p4, p5


# =============================================================================
# HEAD
# =============================================================================

class MCUDetectionHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.p4_cls = nn.Sequential(
            DepthwiseSeparableConv(256, 128),
            nn.Conv2d(128, 1 + num_classes, 1)
        )
        self.p4_reg = nn.Sequential(
            DepthwiseSeparableConv(256, 128),
            nn.Conv2d(128, 4, 1)
        )
        self.p5_cls = nn.Sequential(
            DepthwiseSeparableConv(256, 256),
            nn.Conv2d(256, 1 + num_classes, 1)
        )
        self.p5_reg = nn.Sequential(
            DepthwiseSeparableConv(256, 256),
            nn.Conv2d(256, 4, 1)
        )

    def forward(self, p4, p5):
        return (self.p4_cls(p4), self.p4_reg(p4)), (self.p5_cls(p5), self.p5_reg(p5))


class MCUDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = MCUDetectorBackbone()
        self.fpn = FPNModule(128, 256)
        self.head = MCUDetectionHead(num_classes)

    def forward(self, x):
        p4, p5 = self.backbone(x)
        p4, p5 = self.fpn(p4, p5)
        return self.head(p4, p5)


# =============================================================================
# LOSS (CRITICAL FIX APPLIED HERE)
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        p = torch.sigmoid(pred)
        pt = p * target + (1 - p) * (1 - target)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

class MCUDetectionLoss(nn.Module):
    def __init__(self, num_classes, bbox_weight=2.0, obj_weight=1.0, cls_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.bbox_weight = bbox_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss()

    def forward(self, pred_p4, pred_p5, t4, t5):
        lb = lo = lc = 0.0
        n = 0
        for pred, tgt in [(pred_p4, t4), (pred_p5, t5)]:
            b, o, c, k = self._scale_loss(pred, tgt)
            lb += b; lo += o; lc += c; n += k

        if n > 0:
            lb /= n
            lc /= n
        lo /= max(n, 1)

        total = self.bbox_weight * lb + self.obj_weight * lo + self.cls_weight * lc
        return {"total": total, "bbox": lb.item(), "obj": lo.item(), "cls": lc.item()}

    def _scale_loss(self, pred, targets):
        cls_p, reg_p = pred
        B, C, H, W = cls_p.shape
        lb = lo = lc = 0.0
        n = 0
        obj_map = torch.zeros_like(cls_p[:, :1])

        for b_idx, t in enumerate(targets):
            if t.numel() == 0: 
                continue
            tx, ty = t[:,1]*W, t[:,2]*H
            tw, th = t[:,3]*W, t[:,4]*H
            cls_ids = t[:,0].long()

            for i in range(t.shape[0]):
                gx = int(tx[i].clamp(0, W-1))
                gy = int(ty[i].clamp(0, H-1))

                # ✅ FIX1: Clamp BEFORE exp
                dx = torch.sigmoid(reg_p[b_idx, 0, gy, gx])
                dy = torch.sigmoid(reg_p[b_idx, 1, gy, gx])
                dw = torch.exp(reg_p[b_idx, 2, gy, gx].clamp(-4.0, 4.0))
                dh = torch.exp(reg_p[b_idx, 3, gy, gx].clamp(-4.0, 4.0))

                px, py = gx + dx, gy + dy

                # ✅ FIX2: torch.stack keeps gradients
                pred_box = torch.stack([
                    px - dw / 2, py - dh / 2,
                    px + dw / 2, py + dh / 2
                ])
                tgt_box = torch.stack([
                    tx[i] - tw[i] / 2, ty[i] - th[i] / 2,
                    tx[i] + tw[i] / 2, ty[i] + th[i] / 2
                ])

                lb += F.smooth_l1_loss(pred_box, tgt_box)
                
                # Objectness
                obj_map[b_idx, 0, gy, gx] = 1.0
                lo += self.bce(cls_p[b_idx, :1, gy, gx], torch.ones_like(cls_p[b_idx, :1, gy, gx]))
                
                # Class
                onehot = torch.zeros(self.num_classes, device=cls_p.device)
                onehot[cls_ids[i]] = 1.0
                lc += self.focal(cls_p[b_idx, 1:, gy, gx].unsqueeze(0), onehot.unsqueeze(0))
                n += 1

        # ✅ FIX3: Safer background (0.05 vs 0.1)
        bg_mask = (obj_map == 0)
        if bg_mask.any():
            lo += 0.05 * self.bce(cls_p[:, :1][bg_mask], torch.zeros_like(cls_p[:, :1][bg_mask]))

        return lb, lo, lc, n







# ============================================================================
# ======================== OCR MODEL ==========================================
# ============================================================================

class BidirectionalLSTM(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(in_features, hidden_features, 2, 
                           bidirectional=True, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_features * 2)
        self.fc = nn.Linear(hidden_features * 2, out_features)

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.fc(self.norm(x))


class EnhancedCRNN(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        
        # RNN for sequence modeling
        self.rnn1 = BidirectionalLSTM(512, 256, 256)
        self.rnn2 = BidirectionalLSTM(256, 128, num_classes)

    def forward(self, x):
        features = self.cnn(x)  # (B, 512, 2, 32)
        features = features.squeeze(2).permute(0, 2, 1)  # (B, 32, 512)
        
        seq = self.rnn1(features)  # (B, 32, 256)
        logits = self.rnn2(seq)  # (B, 32, num_classes)
        
        return logits


# ============================================================================
# ======================== DETECTION + OCR PIPELINE ==========================
# ============================================================================

class MCUDetectionOCRPipeline(nn.Module):
    """End-to-end pipeline: detection + OCR."""
    def __init__(self, detector, ocr, device="cpu"):
        super().__init__()
        self.detector = detector.to(device)
        self.ocr = ocr.to(device)
        self.device = device
        
        # OCR preprocessing
        self.ocr_transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # For OCR: mapping from character indices to ASCII
        # Adjust based on your character set during training
        self.char_map = {
            i: chr(32 + i) for i in range(95)  # ASCII 32-126
        }

    @torch.no_grad()
    def detect(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float() / 255.0
            if image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
        
        image = image.unsqueeze(0).to(self.device)
        
        pred_p4, pred_p5 = self.detector(image)
        
        # Decode predictions (simplified)
        boxes = self._decode_predictions(pred_p4, pred_p5, stride_p4=8, stride_p5=16)
        
        return boxes

    def _ocr_crop(self, crop):
        h, w = crop.shape[:2]
        
        # Safety check: OCR assumes sufficiently resolved text
        if h < 24 or w < 80:
            return "TOO_FAR_FOR_OCR"
        
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        
        # Preprocess
        img_pil = Image.fromarray(gray)
        img_tensor = self.ocr_transform(img_pil).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.ocr(img_tensor)  # (1, T, num_classes)
            pred_chars = torch.argmax(logits, dim=2)[0]  # (T,)
        
        # Decode to string
        text = ""
        for char_idx in pred_chars:
            char_idx = char_idx.item()
            if char_idx in self.char_map:
                text += self.char_map[char_idx]
        
        return text.strip()

    def _decode_predictions(self, pred_p4, pred_p5, stride_p4=8, stride_p5=16):
        """
        Decode model predictions to bounding boxes.
        
        PLACEHOLDER: Implement proper decoding with NMS
        """
        boxes = []
        # This is simplified - implement full decoding + NMS for production
        return boxes


# ============================================================================
# ======================== QUANTIZATION WRAPPER ==============================
# ============================================================================

class QuantizableMCUDetector(nn.Module):
    """Quantization-ready wrapper for int8 edge deployment."""
    def __init__(self, num_classes=24):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = MCUDetector(num_classes)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        pred_p4, pred_p5 = self.model(x)
        pred_p4 = (self.dequant(pred_p4[0]), self.dequant(pred_p4[1]))
        pred_p5 = (self.dequant(pred_p5[0]), self.dequant(pred_p5[1]))
        return pred_p4, pred_p5


# ============================================================================
# ======================== DISTILLATION FRAMEWORK ============================
# ============================================================================

class DistillationLoss(nn.Module):
    """Knowledge distillation loss for model compression."""
    def __init__(self, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_outputs, teacher_outputs, hard_loss):
        """
        Args:
            student_outputs: predictions from student model
            teacher_outputs: predictions from teacher model
            hard_loss: standard detection loss value
        
        Returns:
            total_loss: weighted combination of hard and soft targets
        """
        # Simplified soft target loss
        student_cls, _ = student_outputs
        teacher_cls, _ = teacher_outputs
        
        soft_loss = self.kl_div(
            F.log_softmax(student_cls / self.temperature, dim=1),
            F.softmax(teacher_cls / self.temperature, dim=1)
        )
        
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss


# ============================================================================
# ======================== UTILITY FUNCTIONS ==================================
# ============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# ======================== TEST ===============================================
# ============================================================================

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Device: {device}\n")
    
#     # Test detector
#     print("=" * 60)
#     print("DETECTOR TEST")
#     print("=" * 60)
#     detector = MCUDetector(num_classes=24).to(device)
#     print(f"Model size: {detector.get_model_size()}")
#     print(f"Trainable params: {count_parameters(detector) / 1e6:.2f}M")
    
#     dummy_input = torch.randn(1, 3, 512, 512).to(device)
#     (cls_p4, reg_p4), (cls_p5, reg_p5) = detector(dummy_input)
    
#     print(f"\nP4 output shapes:")
#     print(f"  Classification: {cls_p4.shape}")
#     print(f"  Regression: {reg_p4.shape}")
#     print(f"\nP5 output shapes:")
#     print(f"  Classification: {cls_p5.shape}")
#     print(f"  Regression: {reg_p5.shape}")
    
#     # Test loss function
#     print("\n" + "=" * 60)
#     print("LOSS FUNCTION TEST")
#     print("=" * 60)
#     loss_fn = MCUDetectionLoss(num_classes=24)
    
#     # Create dummy targets
#     targets_p4 = [torch.randn(3, 6)]  # 3 objects
#     targets_p5 = [torch.randn(2, 6)]  # 2 objects
    
#     loss_dict = loss_fn(
#         (cls_p4, reg_p4),
#         (cls_p5, reg_p5),
#         targets_p4,
#         targets_p5
#     )
    
#     print("Loss components:")
#     for key, val in loss_dict.items():
#         if isinstance(val, float):
#             print(f"  {key}: {val:.4f}")
#         else:
#             print(f"  {key}: {val.item():.4f}")
    
#     # Test OCR
#     print("\n" + "=" * 60)
#     print("OCR MODEL TEST")
#     print("=" * 60)
#     ocr = EnhancedCRNN(num_classes=38).to(device)
#     dummy_ocr = torch.randn(2, 1, 32, 128).to(device)
#     ocr_out = ocr(dummy_ocr)
#     print(f"OCR output shape: {ocr_out.shape}")
    
#     # Test quantization wrapper
#     print("\n" + "=" * 60)
#     print("QUANTIZATION TEST")
#     print("=" * 60)
#     qdetector = QuantizableMCUDetector(num_classes=24).to(device)
#     print(f"Quantizable model created successfully")
    
#     print("\n✅ All tests passed!")