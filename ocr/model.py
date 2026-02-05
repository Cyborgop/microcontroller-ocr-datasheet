import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Optional
from utils import SiLU
from utils import decode_argmax_output
import torchvision
from utils import CHARS, BLANK_IDX, idx2char, char2idx


# =============================================================================
# CORE BUILDING BLOCKS
# =============================================================================


# ---------- helpers ----------
def _fuse_conv_bn(conv, bn):
    """Return fused conv kernel and bias tensors for conv (possibly bias=False) and bn.
    conv: nn.Conv2d
    bn:   nn.BatchNorm2d
    returns (kernel, bias) as tensors.
    """
    # conv.weight: [out_channels, in_channels/groups, k, k]
    W = conv.weight.detach()
    if conv.bias is not None:
        conv_bias = conv.bias.detach()
    else:
        conv_bias = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)

    # BN parameters
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    running_mean = bn.running_mean.detach()
    running_var = bn.running_var.detach()
    eps = bn.eps

    std = torch.sqrt(running_var + eps)

    # reshape for broadcasting: gamma/std -> shape [out_ch, 1, 1, 1]
    scale = (gamma / std).reshape(-1, 1, 1, 1)

    W_fused = W * scale
    b_fused = beta - (gamma * running_mean) / std + (conv_bias * (gamma / std))

    return W_fused, b_fused


def _pad_1x1_to_3x3_tensor(k1x1):
    """Pad depthwise 1x1 kernel tensor into 3x3 centered kernel.
    Input k1x1 shape: [C, 1, 1, 1] or [C, 1, 1, 1]
    Output shape: [C, 1, 3, 3] with center = original.
    """
    C = k1x1.shape[0]
    device = k1x1.device
    dtype = k1x1.dtype
    k3 = torch.zeros((C, 1, 3, 3), dtype=dtype, device=device)
    k3[:, :, 1, 1] = k1x1[:, :, 0, 0]
    return k3


# ---------- Rep-style depthwise conv module (train-time multi-branch) ----------
class RepDWConvSR(nn.Module):
    """
    Rep-style depthwise block (training-time: 3x3 DW conv + 1x1 DW conv + optional identity-BN).
    Call .fuse() before inference to collapse branches into a single depthwise conv.
    """
    def __init__(self, channels, stride=1, use_identity_bn=True):
        super().__init__()
        self.channels = channels
        self.stride = stride

        # 3x3 depthwise branch (conv + bn)
        self.dw3 = nn.Conv2d(
            channels, channels, kernel_size=3,
            stride=stride, padding=1, groups=channels, bias=False
        )
        self.bn3 = nn.BatchNorm2d(channels)

        # 1x1 depthwise branch — IMPORTANT: use same stride so spatial sizes match
        self.dw1 = nn.Conv2d(
            channels, channels, kernel_size=1,
            stride=stride, padding=0, groups=channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        # identity branch only valid when stride == 1
        self.use_identity_bn = use_identity_bn and (stride == 1)
        if self.use_identity_bn:
            self.id_bn = nn.BatchNorm2d(channels)
        else:
            self.id_bn = None

        self.fused = False

    def forward(self, x):
        if self.fused:
            return self.dw3(x)

        out = self.bn3(self.dw3(x)) + self.bn1(self.dw1(x))
        if self.use_identity_bn:
            out = out + self.id_bn(x)
        return out

    def fuse(self):
        """Fuse all branches and BNs into a single depthwise 3x3 conv (replaces self.dw3)."""
        if self.fused:
            return

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # fuse dw3 + bn3
        k3, b3 = _fuse_conv_bn(self.dw3, self.bn3)  # shapes [C,1,3,3], [C]

        # fuse dw1 + bn1 -> pad to 3x3
        k1, b1 = _fuse_conv_bn(self.dw1, self.bn1)  # k1 shape [C,1,1,1]
        k1_p = _pad_1x1_to_3x3_tensor(k1)

        # identity BN -> convert to kernel + bias equivalent
        if self.use_identity_bn:
            # Identity conv kernel 1x1 is delta: center=1
            C = self.channels
            id_kernel_1x1 = torch.ones((C, 1, 1, 1), dtype=dtype, device=device)
            # create a fake conv to fold with id_bn semantics using helper math manually:
            # fold id_bn: k_id = id_kernel * (gamma/std), b_id = beta - gamma*run_mean/std
            gamma = self.id_bn.weight.detach()
            beta = self.id_bn.bias.detach()
            running_mean = self.id_bn.running_mean.detach()
            running_var = self.id_bn.running_var.detach()
            std = torch.sqrt(running_var + self.id_bn.eps)
            scale = (gamma / std).reshape(-1, 1, 1, 1)
            k_id_1x1 = id_kernel_1x1 * scale
            b_id = beta - (gamma * running_mean) / std
            k_id_3x3 = _pad_1x1_to_3x3_tensor(k_id_1x1)
        else:
            k_id_3x3 = torch.zeros_like(k3)
            b_id = torch.zeros_like(b3)

        # sum kernels and biases
        k_sum = k3 + k1_p + k_id_3x3
        b_sum = b3 + b1 + b_id

        # create new conv (3x3 depthwise) with bias True
        fused_conv = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=self.stride,
                               padding=1, groups=self.channels, bias=True)
        fused_conv.weight.data.copy_(k_sum)
        fused_conv.bias.data.copy_(b_sum)

        # replace modules
        self.dw3 = fused_conv.to(device)
        # delete unused params to save memory (optional)
        self.dw1 = None
        self.bn1 = None
        self.bn3 = None
        self.id_bn = None
        self.fused = True

# ---------- RepViT Block using the RepDWConvSR ----------
class RepvitBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, high_res=True, use_se=False, expansion=2):
        """
        in_ch -> input channels
        out_ch -> output channels
        stride -> stride for token mixer (depthwise)
        high_res -> (no direct effect here — kept for API compatibility)
        use_se -> whether to use squeeze-excitation (recommend True only for low-res stages)
        expansion -> channel expansion factor for channel mixer (paper used 2)
        """
        super().__init__()
        self.use_res = (in_ch == out_ch and stride == 1)
        # token mixer: Rep-style DW block
        self.token_mixer = RepDWConvSR(in_ch, stride=stride, use_identity_bn=True)

        # SE (optional). Recommend enabling only for later stages (low resolution)
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, max(1, in_ch // 4), 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(1, in_ch // 4), in_ch, 1),
                nn.Sigmoid()
            )

        hidden = in_ch * expansion  # expansion ratio; paper uses 2
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.token_mixer(x)      # depthwise token mixing
        if self.use_se:
            x = x * self.se(x)
        x = self.channel_mixer(x)    # channel mixing (pointwise)
        if self.use_res:
            x = x + identity
        return self.act(x)

    def fuse(self):
        """Fuse the token_mixer (multi-branch) into a single conv for inference.
        Also fuse channel_mixer BNs into convs where applicable (optional).
        """
        # fuse token mixer
        self.token_mixer.fuse()

        # optionally fold channel_mixer batchnorms into conv weights for inference
        # (simple approach: leave channel_mixer as-is; for max speed you can implement BN folding similarly)
        # If you want, I can add BN folding for channel_mixer in a follow-up.

        # mark block fused (token_mixer.fused flag exists)



class BottleneckCSPBlock(nn.Module):
    """Cross Stage Partial Bottleneck Block."""
    def __init__(self, in_ch, out_ch, n_blocks=1, ratio=0.25):
        super().__init__()
        hidden = max(1, int(out_ch * ratio))
        self.cv1 = nn.Conv2d(in_ch, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act1 = SiLU(inplace=True)
        self.blocks = nn.Sequential(*[
            RepvitBlock(
                hidden,
                hidden,
                stride=1,
                high_res=False,
                use_se=False
            )
            for _ in range(n_blocks)
        ])
        self.cv2 = nn.Conv2d(hidden * 2, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = SiLU(inplace=True)

    def forward(self, x):
        y = self.act1(self.bn1(self.cv1(x)))
        z = self.blocks(y)
        out = self.cv2(torch.cat([y, z], dim=1))
        out = self.bn2(out)
        return self.act2(out)


# =============================================================================
# FPN MODULE
# =============================================================================

class FPNModule(nn.Module):
    """Feature Pyramid Network (P3 + P4 only, P5 ignored)."""
    def __init__(self, ch_p3, ch_p4, ch_p5):
        super().__init__()
        fpn_ch = 192  # or 128

        self.lat_p3 = nn.Conv2d(ch_p3, fpn_ch, 1, bias=False)
        self.lat_p4 = nn.Conv2d(ch_p4, fpn_ch, 1, bias=False)

        self.refine_p3 = RepvitBlock(fpn_ch, fpn_ch, high_res=False)
        self.refine_p4 = RepvitBlock(fpn_ch, fpn_ch, high_res=False)

    def forward(self, p3, p4, p5=None):
        p3_lat = self.lat_p3(p3)
        p4_lat = self.lat_p4(p4)

        p4_out = self.refine_p4(p4_lat)

        # 🔑 CRITICAL FIX: match spatial size explicitly
        p4_up = F.interpolate(
            p4_out,
            size=p3_lat.shape[-2:],   # (H, W) of p3
            mode="nearest"
        )

        p3_out = self.refine_p3(p3_lat + p4_up)
        return p3_out, p4_out


# =============================================================================
# BACKBONE
# =============================================================================

class MCUDetectorBackbone(nn.Module):
    """Lightweight backbone for MCU detection.

    Produces:
      - p3: stride=4, channels=64
      - p4: stride=8, channels=128
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),  # /2
            nn.BatchNorm2d(24),
            SiLU(inplace=True),
            nn.Conv2d(24, 32, 3, stride=2, padding=1, bias=False), # /4 total
            nn.BatchNorm2d(32),
            SiLU(inplace=True),
        )
        # p2: keeps stride = 4
        self.p2 = RepvitBlock(32, 32)

        # p3_down: **no spatial downsample** so p3 will have stride = 4
        self.p3_down = RepvitBlock(32, 64, stride=1)

        # p3: use fewer blocks for mobile if desired; current uses 2 CSP repeats
        self.p3 = nn.Sequential(*[BottleneckCSPBlock(64, 64, n_blocks=2) for _ in range(2)])

        # p4_down: downsample here => p4 stride = 8
        self.p4_down = RepvitBlock(64, 128, stride=2)

        self.p4 = nn.Sequential(*[BottleneckCSPBlock(128, 128, n_blocks=2) for _ in range(2)])

        # helpful metadata for consistency elsewhere
        self.out_channels = {"p3": 64, "p4": 128}
        self.out_strides = {"p3": 4, "p4": 8}

    def forward(self, x):
        x = self.stem(x)     # stride = 4
        x = self.p2(x)       # stride = 4
        p3 = self.p3(self.p3_down(x))    # p3: stride = 4, 64 ch
        p4 = self.p4(self.p4_down(p3))   # p4: stride = 8, 128 ch
        return p3, p4


# =============================================================================
# DETECTION HEAD
# =============================================================================

class MCUDetectionHead(nn.Module):
    def __init__(self, num_classes, num_anchors=1, fpn_ch=192, head_ch=128):
        """
        fpn_ch: number of channels produced by FPN (e.g. 192)
        head_ch: internal channel width inside head (e.g. 128)
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # P3 heads (expect input channels == fpn_ch)
        self.p3_cls = nn.Sequential(
            RepvitBlock(fpn_ch, head_ch),
            nn.Conv2d(head_ch, num_anchors * (1 + num_classes), 1)
        )
        self.p3_reg = nn.Sequential(
            RepvitBlock(fpn_ch, head_ch),
            nn.Conv2d(head_ch, num_anchors * 4, 1)
        )

        # P4 heads
        self.p4_cls = nn.Sequential(
            RepvitBlock(fpn_ch, head_ch),
            nn.Conv2d(head_ch, num_anchors * (1 + num_classes), 1)
        )
        self.p4_reg = nn.Sequential(
            RepvitBlock(fpn_ch, head_ch),
            nn.Conv2d(head_ch, num_anchors * 4, 1)
        )

        # P5 intentionally disabled
        self.p5_cls = None
        self.p5_reg = None

    def forward(self, p3, p4, p5=None):
        p3_cls = self.p3_cls(p3)
        p3_reg = self.p3_reg(p3)

        p4_cls = self.p4_cls(p4)
        p4_reg = self.p4_reg(p4)

        return ((p3_cls, p3_reg), (p4_cls, p4_reg))




class MCUDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = MCUDetectorBackbone()
        self.fpn = FPNModule(
            ch_p3=self.backbone.out_channels["p3"],
            ch_p4=self.backbone.out_channels["p4"],
            ch_p5=None
        )
        self.head = MCUDetectionHead(
            num_classes=num_classes,
            fpn_ch=192,
            head_ch=128
        )

        self.num_classes = num_classes
        print("DEBUG[model] num_classes:", self.num_classes)

    def forward(self, x):
        p3, p4 = self.backbone(x)
        p3, p4 = self.fpn(p3, p4)
        return self.head(p3, p4)


# =============================================================================
# LOSS FUNCTIONS (WITH CRITICAL FIXES FROM SECOND CODE)
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        p = torch.sigmoid(pred)
        pt = p * target + (1 - p) * (1 - target)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


class MCUDetectionLoss(nn.Module):
    """Multi-task loss for detection with critical fixes."""
    def __init__(self, num_classes, bbox_weight=2.0, obj_weight=1.0, cls_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.bbox_weight = bbox_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss()

    def forward(self, pred_p3, pred_p4, t3, t4):  # ← ADD pred_p3, t3
        lb = lo = lc = 0.0
        n = 0
        
        # ← ADD P3 to the loop
        for pred, targets in ((pred_p3, t3), (pred_p4, t4)):
            b, o, c, k = self._scale_loss(pred, targets)
            lb += b
            lo += o
            lc += c
            n += k

        if n > 0:
            lb /= n
            lc /= n
        lo /= max(n, 1)

        total = self.bbox_weight * lb + self.obj_weight * lo + self.cls_weight * lc
        return {
            "total": total,
            "bbox": lb.item() if isinstance(lb, torch.Tensor) else float(lb),
            "obj": lo.item() if isinstance(lo, torch.Tensor) else float(lo),
            "cls": lc.item() if isinstance(lc, torch.Tensor) else float(lc)
        }

    def _scale_loss(self, pred, targets):
        """Calculate loss at a specific scale."""
        cls_p, reg_p = pred
        B, C, H, W = cls_p.shape
        device = cls_p.device
        
        lb = torch.tensor(0.0, device=device)
        lo = torch.tensor(0.0, device=device)
        lc = torch.tensor(0.0, device=device)
        n = 0
        
        obj_map = torch.zeros((B, 1, H, W), device=device)

        for b_idx, t in enumerate(targets):
            if t.numel() == 0:
                continue

            # Normalize target coordinates
            tx = t[:, 1] * W
            ty = t[:, 2] * H
            tw = t[:, 3] * W
            th = t[:, 4] * H
            cls_ids = t[:, 0].long()

            for i in range(t.shape[0]):
                # Grid coordinates
                gx = int(tx[i].clamp(0, W - 1))
                gy = int(ty[i].clamp(0, H - 1))

                # ✅ CRITICAL FIX: Clamp BEFORE exponent to prevent explosion
                dx = torch.sigmoid(reg_p[b_idx, 0, gy, gx])
                dy = torch.sigmoid(reg_p[b_idx, 1, gy, gx])
                dw = torch.exp(reg_p[b_idx, 2, gy, gx].clamp(-4.0, 4.0))
                dh = torch.exp(reg_p[b_idx, 3, gy, gx].clamp(-4.0, 4.0))

                px = gx + dx
                py = gy + dy

                # ✅ CRITICAL FIX: Use torch.stack to preserve gradients
                pred_box = torch.stack([
                    px - dw / 2, py - dh / 2,
                    px + dw / 2, py + dh / 2
                ])
                tgt_box = torch.stack([
                    tx[i] - tw[i] / 2, ty[i] - th[i] / 2,
                    tx[i] + tw[i] / 2, ty[i] + th[i] / 2
                ])

                # Bounding box loss
                lb = lb + F.smooth_l1_loss(pred_box, tgt_box)

                # Objectness loss
                obj_map[b_idx, 0, gy, gx] = 1.0
                lo = lo + self.bce(cls_p[b_idx, :1, gy, gx], 
                                  torch.ones_like(cls_p[b_idx, :1, gy, gx]))

                # Classification loss
                onehot = torch.zeros(self.num_classes, device=device)
                onehot[cls_ids[i]] = 1.0
                lc = lc + self.focal(cls_p[b_idx, 1:, gy, gx].unsqueeze(0), 
                                    onehot.unsqueeze(0))
                n += 1

        # ✅ CRITICAL FIX: Safer background weighting (0.05 vs 0.1)
        bg_mask = (obj_map == 0)
        if bg_mask.any():
            lo = lo + 0.05 * self.bce(cls_p[:, :1][bg_mask], 
                                     torch.zeros_like(cls_p[:, :1][bg_mask]))

        return lb, lo, lc, n


# =============================================================================
# OCR MODEL
# =============================================================================

class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM for sequence modeling."""
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            in_features, hidden_features, 2,
            bidirectional=True, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_features * 2)
        self.fc = nn.Linear(hidden_features * 2, out_features)

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.fc(self.norm(x))


class EnhancedCRNN(nn.Module):
    """CRNN model for text recognition."""
    def __init__(self, num_classes=38):
        super().__init__()
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.InstanceNorm2d(64),  # ✅ FIXED
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.InstanceNorm2d(128),  # ✅ FIXED
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.InstanceNorm2d(256),  # ✅ FIXED
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.InstanceNorm2d(512),  # ✅ FIXED
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
        logits = self.rnn2(seq)    # (B, 32, num_classes)
        
        return logits


# =============================================================================
# DETECTION + OCR PIPELINE
# =============================================================================

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
        
        # Character mapping
        self.CHARS = CHARS
        self.BLANK_IDX = BLANK_IDX
        self.idx2char = idx2char
        self.char2idx = char2idx

    @torch.no_grad()
    def detect(self, image):
        """Run detection on input image."""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float() / 255.0
            if image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
        
        image = image.unsqueeze(0).to(self.device)
        pred_p4, pred_p5 = self.detector(image)
        boxes = self._decode_predictions(pred_p4, pred_p5, stride_p4=8, stride_p5=16)
        return boxes

    def _ocr_crop(self, crop):
        """Run OCR on a cropped region."""
        h, w = crop.shape[:2]
        
        # Safety check: OCR assumes sufficiently resolved text
        if h < 10 or w < 20:
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
            logits = self.ocr(img_tensor)
            text = decode_argmax_output(logits)
        
        return text.strip()

    def _decode_predictions(self, pred_p4, pred_p5, stride_p4=8, stride_p5=16):
        """
        Return one highest-confidence bounding box from p4 or p5.
        Later, extend to multi-box + NMS if needed.
        """
        def extract_box(cls_map, reg_map, stride):
            obj = torch.sigmoid(cls_map[0, :1])
            _, idx = obj.view(-1).max(0)
            H, W = obj.shape[1:]
            gy, gx = divmod(idx.item(), W)

            dx = torch.sigmoid(reg_map[0, 0, gy, gx])
            dy = torch.sigmoid(reg_map[0, 1, gy, gx])
            bw = torch.exp(reg_map[0, 2, gy, gx]).clamp(20, 300)
            bh = torch.exp(reg_map[0, 3, gy, gx]).clamp(20, 300)

            cx = (gx + dx) * stride
            cy = (gy + dy) * stride
            x1 = int(max(0, cx - bw / 2))
            y1 = int(max(0, cy - bh / 2))
            x2 = int(min(stride * W, cx + bw / 2))
            y2 = int(min(stride * H, cy + bh / 2))

            return [x1, y1, x2, y2], obj[0, gy, gx].item()

        box4, score4 = extract_box(*pred_p4, stride_p4)
        box5, score5 = extract_box(*pred_p5, stride_p5)
        return [box4] if score4 > score5 else [box5]


# =============================================================================
# QUANTIZATION WRAPPER
# =============================================================================

class QuantizableMCUDetector(nn.Module):
    """Quantization-ready wrapper for int8 edge deployment."""
    def __init__(self, num_classes=7):
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


# =============================================================================
# DISTILLATION FRAMEWORK
# =============================================================================

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
        student_cls, _ = student_outputs
        teacher_cls, _ = teacher_outputs
        
        soft_loss = self.kl_div(
            F.log_softmax(student_cls / self.temperature, dim=1),
            F.softmax(teacher_cls / self.temperature, dim=1)
        )
        
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return total_loss





