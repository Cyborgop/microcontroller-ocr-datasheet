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

import math
# =============================================================================
# CORE BUILDING BLOCKS
# =============================================================================


# ---------- helpers ----------
def _fuse_conv_bn(conv, bn):# checked okay
    # Conv–BN fusion is used only for inference optimization, not for training or fixing metric issues; 
    # it mathematically merges a Conv2d layer and its following BatchNorm into a single Conv with modified weights 
    # and bias so that the output remains exactly the same, but the forward pass becomes faster and more memory-efficient
    # by removing the BatchNorm operation. It is mainly useful for deployment on edge or mobile devices 
    # (like Rep-style architectures such as RepVGG or RepViT in inference mode), and it does not improve accuracy, mAP, precision, 
    # or recall—so if you are debugging zero metrics during training, Conv–BN fusion is not related to that problem.
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


def _pad_1x1_to_3x3_tensor(k1x1):#checked ok
    # This function is used in structural re-parameterization (as in RepViT/RepVGG) to convert a 1×1 depthwise 
    # convolution kernel into a 3×3 kernel by placing the original weight at the center and padding the rest with zeros, 
    # so that it can be mathematically merged with a 3×3 depthwise convolution during inference. Since different kernel sizes 
    # cannot be directly added, this padding step ensures all branches have the same spatial size before fusion into a single
    # efficient 3×3 convolution. It is required only for inference-time model simplification and does not affect training 
    # behavior or mAP metrics.
    C = k1x1.shape[0]
    device = k1x1.device
    dtype = k1x1.dtype
    k3 = torch.zeros((C, 1, 3, 3), dtype=dtype, device=device)
    k3[:, :, 1, 1] = k1x1[:, :, 0, 0]
    return k3


# ---------- Rep-style depthwise conv module (train-time multi-branch) ----------
class RepDWConvSR(nn.Module):#checked fine
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

    def fuse(self):#checked fine
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
class RepvitBlock(nn.Module):#checked fine 
    def __init__(self, in_ch, out_ch, stride=1, high_res=True, use_se=True, expansion=2):
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
        self.token_mixer = RepDWConvSR(in_ch, stride=stride, use_identity_bn=self.use_res)

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
        x = self.act(x) 
        if self.use_se:
            x = x * self.se(x)
        x = self.channel_mixer(x)    # channel mixing (pointwise)
        if self.use_res:
            x = x + identity
        return self.act(x)

    def fuse(self):#checked fine
        """Fuse the token_mixer (multi-branch) into a single conv for inference.
        Also fuse channel_mixer BNs into convs where applicable (optional).
        """
        # fuse token mixer
        self.token_mixer.fuse()

        # optionally fold channel_mixer batchnorms into conv weights for inference
        # (simple approach: leave channel_mixer as-is; for max speed you can implement BN folding similarly)
        # If you want, I can add BN folding for channel_mixer in a follow-up.

        # mark block fused (token_mixer.fused flag exists)


class BottleneckCSPBlock(nn.Module):#checked fine but changed from original
    """
    Cross Stage Partial Network Block
    Exact implementation from CSPNet paper (Figure 2b)
    
    RECOMMENDED CONFIGURATION (γ=0.25):
    - 11% computation reduction
    - +0.8% higher accuracy
    """
    def __init__(self, in_ch, out_ch, n_blocks=1, ratio=0.25,use_se=False):
        super().__init__()
        
        # ============= CORE CSP CONCEPT =============
        # Split channels into TWO parts - γ=0.25 (25% direct, 75% through blocks)
        self.part1_chnls = int(in_ch * ratio)   # 25% channels - direct path (no computation)
        self.part2_chnls = in_ch - self.part1_chnls  # 75% channels - through blocks
        
        # PART 2: Process through blocks
        hidden = max(1, int(self.part2_chnls * 0.5))  # Bottleneck
        
        # Entry conv for part2
        self.cv1 = nn.Conv2d(self.part2_chnls, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act1 = nn.SiLU(inplace=True)
        
        # Main processing blocks (RepViT blocks)
        self.blocks = nn.Sequential(*[
            RepvitBlock(hidden, hidden, stride=1, use_se=use_se)
            for _ in range(n_blocks)
        ])
        
        # Exit conv for part2
        self.cv2 = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.act2 = nn.SiLU(inplace=True)
        
        # ============= FUSION =============
        # Concatenate PART1 (untouched) + processed PART2
        fusion_ch = self.part1_chnls + hidden
        
        # Final transition layer
        self.cv3 = nn.Conv2d(fusion_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.act3 = nn.SiLU(inplace=True)
        
    def forward(self, x):
        # ============= 1. SPLIT =============
        part1 = x[:, :self.part1_chnls, :, :]   # 25% channels - NO COMPUTATION
        part2 = x[:, self.part1_chnls:, :, :]   # 75% channels - through blocks
        
        # ============= 2. PROCESS ONLY PART2 =============
        part2 = self.act1(self.bn1(self.cv1(part2)))
        part2 = self.blocks(part2)
        part2 = self.act2(self.bn2(self.cv2(part2)))
        
        # ============= 3. CONCATENATE =============
        out = torch.cat([part1, part2], dim=1)
        
        # ============= 4. TRANSITION =============
        out = self.act3(self.bn3(self.cv3(out)))
        
        return out

# =============================================================================
# FPN MODULE
# =============================================================================

class FPNModule(nn.Module):#checkedok i will change it later to bifpn
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

class MCUDetectorBackbone(nn.Module):#checked okay
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
        self.p3 = nn.Sequential(*[BottleneckCSPBlock(64, 64, n_blocks=2, use_se=True) for _ in range(2)])

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

class MCUDetectionHead(nn.Module):#checked okay

    def __init__(
        self,
        num_classes,
        num_anchors=1,
        fpn_ch=192,
        head_ch=128,
        prior_prob=0.01
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_anchors = int(num_anchors)

        # =======================
        # P3 HEAD (stride = 4)
        # =======================
        self.p3_refine = RepvitBlock(fpn_ch, head_ch)

        self.p3_obj = nn.Conv2d(head_ch, num_anchors * 1, kernel_size=1)
        self.p3_cls = nn.Conv2d(head_ch, num_anchors * num_classes, kernel_size=1)
        self.p3_reg = nn.Conv2d(head_ch, num_anchors * 4, kernel_size=1)

        # =======================
        # P4 HEAD (stride = 8)
        # =======================
        self.p4_refine = RepvitBlock(fpn_ch, head_ch)

        self.p4_obj = nn.Conv2d(head_ch, num_anchors * 1, kernel_size=1)
        self.p4_cls = nn.Conv2d(head_ch, num_anchors * num_classes, kernel_size=1)
        self.p4_reg = nn.Conv2d(head_ch, num_anchors * 4, kernel_size=1)

        self._init_weights(prior_prob)

    # --------------------------------------------------
    # Weight initialization
    # --------------------------------------------------
    def _init_weights(self, prior_prob):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        
        # Objectness + Classification bias initialization (CRITICAL)
        if prior_prob is not None and prior_prob > 0:
            bias = float(-torch.log(torch.tensor((1.0 - prior_prob) / prior_prob)))

            # objectness
            nn.init.constant_(self.p3_obj.bias, bias)
            nn.init.constant_(self.p4_obj.bias, bias)

            # 🔥 classification (THIS WAS MISSING)
            nn.init.constant_(self.p3_cls.bias, bias)
            nn.init.constant_(self.p4_cls.bias, bias)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, p3, p4, p5=None):
        # ----- P3 -----
        p3_feat = self.p3_refine(p3)
        p3_out = (
            self.p3_obj(p3_feat),
            self.p3_cls(p3_feat),
            self.p3_reg(p3_feat),
        )

        # ----- P4 -----
        p4_feat = self.p4_refine(p4)
        p4_out = (
            self.p4_obj(p4_feat),
            self.p4_cls(p4_feat),
            self.p4_reg(p4_feat),
        )

        return p3_out, p4_out

    # --------------------------------------------------
    # Optional: decoder compatibility helper
    # --------------------------------------------------
    def to_decoder_format(self, preds):
        (p3_obj, p3_cls, p3_reg), (p4_obj, p4_cls, p4_reg) = preds

        p3_cls_comb = torch.cat([p3_obj, p3_cls], dim=1)
        p4_cls_comb = torch.cat([p4_obj, p4_cls], dim=1)

        return ((p3_cls_comb, p3_reg),
                (p4_cls_comb, p4_reg))




class MCUDetector(nn.Module):#checked okay
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
        if hasattr(torch, "cuda"): pass
        # sanity
        assert isinstance(self.num_classes, int) and self.num_classes > 0, "num_classes must be positive int"
        print("DEBUG[model] num_classes:", self.num_classes)

    def forward(self, x):
        p3, p4 = self.backbone(x)
        p3, p4 = self.fpn(p3, p4)
        return self.head(p3, p4)


# =============================================================================
# LOSS FUNCTIONS (WITH CRITICAL FIXES FROM SECOND CODE)
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        mod = (1 - p_t) ** self.gamma

        # 🔑 CRITICAL FIX
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha * targets
        else:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * mod * ce
        return loss.mean()





class MCUDetectionLoss(nn.Module):
    """
    Drop-in, robust MCUDetectionLoss.

    - Same API: forward(pred_p3, pred_p4, t3, t4)
    - Returns dict: { "total": tensor(requires_grad), "bbox": detached, "obj": detached, "cls": detached }
    - Default: CIoU box loss (1 - ciou). Falls back to smooth-l1 if needed.
    - Positive objectness target kept = 1.0 (compatible with your existing training).
    """

    def __init__(
        self,
        num_classes,
        bbox_weight=2.0,
        obj_weight=1.0,
        cls_weight=2.0,
        anchors=1,
        use_ciou: bool = True,       # use CIoU for bbox regression (recommended)
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bbox_weight = float(bbox_weight)
        self.obj_weight = float(obj_weight)
        self.cls_weight = float(cls_weight)
        self.anchors = anchors
        self.use_ciou = use_ciou

        # expects you already have a FocalLoss implementation in scope (as before)
        # inverse-frequency weights (example — replace with your real stats)
        class_weights = torch.tensor(
            [1.2, 1.1, 0.4, 1.0, 1.0, 1.3, 1.4],
            dtype=torch.float32
        )
        assert len(class_weights) == num_classes
        self.register_buffer("cls_alpha", class_weights)
        self.focal = FocalLoss(alpha=self.cls_alpha, gamma=2.0)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred_p3, pred_p4, t3, t4):
        """
        pred_p3/pred_p4: tuples (obj_logits, cls_logits, reg_preds)
          - obj_logits: (B, A, 1, H, W) OR (B, A, H, W) flattened variant accepted
          - cls_logits: (B, A*num_classes, H, W) OR (B, A, C, H, W) depending on head
          - reg_preds: (B, A*4, H, W) OR (B, A, 4, H, W)
        t3, t4: list of length B, each either tensor (N,5) or empty tensor (0,5) in YOLO format
        """
        device = pred_p3[0].device

        # graph-safe scalar anchor (zero tensor on same device)
        graph_anchor = pred_p3[0].sum() * 0.0 + pred_p4[0].sum() * 0.0

        total_bbox = graph_anchor.clone()
        total_obj  = graph_anchor.clone()
        total_cls  = graph_anchor.clone()
        npos = 0  # positive count (int)

        for pred, targets in ((pred_p3, t3), (pred_p4, t4)):
            b, o, c, k = self._scale_loss(pred, targets)
            total_bbox = total_bbox + b
            total_obj  = total_obj  + o
            total_cls  = total_cls  + c
            npos += int(k)

        if npos > 0:
            inv = 1.0 / float(npos)
            total_bbox = total_bbox * inv
            total_obj  = total_obj  * inv
            total_cls  = total_cls  * inv

        total = (
            self.bbox_weight * total_bbox +
            self.obj_weight  * total_obj +
            self.cls_weight  * total_cls
        )

        return {
            "total": total,                 # scalar tensor, requires_grad=True
            "bbox": total_bbox.detach(),
            "obj":  total_obj.detach(),
            "cls":  total_cls.detach(),
        }

    # ------------------- helpers for IoU / CIoU -------------------
    @staticmethod
    def _box_area(box):
        # box: tensor [x1, y1, x2, y2]
        w = (box[2] - box[0]).clamp(min=0.0)
        h = (box[3] - box[1]).clamp(min=0.0)
        return w * h

    @staticmethod
    def _box_iou(box1, box2):
        # scalar IoU between two 1D tensors [4]
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])

        inter_w = (x2 - x1).clamp(min=0.0)
        inter_h = (y2 - y1).clamp(min=0.0)
        inter = inter_w * inter_h

        area1 = MCUDetectionLoss._box_area(box1)
        area2 = MCUDetectionLoss._box_area(box2)
        union = area1 + area2 - inter + 1e-7
        return (inter / union).clamp(min=0.0, max=1.0)

    @staticmethod
    def _bbox_ciou(pred_box, tgt_box):
        """
        CIoU (scalar) between two boxes in same coordinate units.
        pred_box, tgt_box: tensors [x1,y1,x2,y2]
        """
        # IoU
        iou = MCUDetectionLoss._box_iou(pred_box, tgt_box)

        # centers
        px = (pred_box[0] + pred_box[2]) / 2.0
        py = (pred_box[1] + pred_box[3]) / 2.0
        gx = (tgt_box[0] + tgt_box[2]) / 2.0
        gy = (tgt_box[1] + tgt_box[3]) / 2.0

        center_dist2 = (px - gx) ** 2 + (py - gy) ** 2

        enc_x1 = torch.min(pred_box[0], tgt_box[0])
        enc_y1 = torch.min(pred_box[1], tgt_box[1])
        enc_x2 = torch.max(pred_box[2], tgt_box[2])
        enc_y2 = torch.max(pred_box[3], tgt_box[3])
        c2 = ((enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2).clamp(min=1e-6)

        w_pred = (pred_box[2] - pred_box[0]).clamp(min=1e-6)
        h_pred = (pred_box[3] - pred_box[1]).clamp(min=1e-6)
        w_tgt  = (tgt_box[2] - tgt_box[0]).clamp(min=1e-6)
        h_tgt  = (tgt_box[3] - tgt_box[1]).clamp(min=1e-6)

        v = (4.0 / (math.pi ** 2)) * (torch.atan(w_tgt / h_tgt) - torch.atan(w_pred / h_pred)) ** 2
        # alpha uses no grad flow through denominator (as common)
        with torch.no_grad():
            alpha = v / (1.0 - iou + v + 1e-7)

        ciou = iou - (center_dist2 / c2) - alpha * v
        # clamp to [-1,1]
        return ciou.clamp(min=-1.0, max=1.0)

    # ------------------- per-scale loss -------------------
    def _scale_loss(self, pred, targets):
        """
        pred: tuple (obj_logits, cls_logits, reg_preds)
        targets: list length B, each tensor (N,5) in YOLO [cls,cx,cy,w,h] normalized to [0..1] (or empty)
        returns: bbox_loss_tensor, obj_loss_tensor, cls_loss_tensor, npos (int)
        """
        obj_logits, cls_logits, reg_preds = pred
        device = obj_logits.device

        # canonical shapes (allow flexibility if channels flattened)
        B = obj_logits.shape[0]
        # If obj_logits originally (B, A, H, W) or (B, A, 1, H, W) or (B, A, H, W) flattened:
        # We try to reshape accommodating both.
        # Expect channel dimension equals anchors (A)
        # We'll infer H,W by last two dims
        if obj_logits.dim() == 4:  # (B, A, H, W)
            Bc, A, H, W = obj_logits.shape
            obj = obj_logits.view(B, A, 1, H, W)
        elif obj_logits.dim() == 5:  # (B, A, 1, H, W)
            Bc, A, one, H, W = obj_logits.shape
            obj = obj_logits.view(Bc, A, 1, H, W)
        else:
            # fallback: assume (B, A, H, W)
            A = self.anchors
            H, W = obj_logits.shape[-2], obj_logits.shape[-1]
            obj = obj_logits.view(B, A, 1, H, W)

        # classification channels: either (B, A*num_classes, H, W) or (B, A, C, H, W)
        if cls_logits.dim() == 4:  # (B, A*num_classes, H, W)
            _, cchan, _, _ = cls_logits.shape
            assert cchan % self.num_classes == 0 or cchan == self.num_classes
            A_cls = cchan // self.num_classes if cchan // self.num_classes > 0 else 1
            cls = cls_logits.view(B, A_cls, self.num_classes, H, W)
        elif cls_logits.dim() == 5:
            cls = cls_logits.view(B, A, self.num_classes, H, W)
        else:
            cls = cls_logits.view(B, A, self.num_classes, H, W)

        # regression channels: either (B, A*4, H, W) or (B, A, 4, H, W)
        if reg_preds.dim() == 4:
            rchan = reg_preds.shape[1]
            A_reg = rchan // 4 if rchan // 4 > 0 else 1
            reg = reg_preds.view(B, A_reg, 4, H, W)
        elif reg_preds.dim() == 5:
            reg = reg_preds.view(B, A, 4, H, W)
        else:
            reg = reg_preds.view(B, A, 4, H, W)

        # init scalar-zero tensors (graph-safe)
        zero = obj_logits.sum() * 0.0
        bbox_loss = zero.clone()
        obj_loss  = zero.clone()
        cls_loss  = zero.clone()
        npos = 0

        # obj target placeholders
        obj_target = torch.zeros_like(obj, device=device)

        # iterate batch
        for b_idx in range(B):
            t = targets[b_idx]
            if t is None or t.numel() == 0:
                continue

            # scale targets to grid coords (grid units)
            tx = t[:, 1] * W  # x center in grid coords
            ty = t[:, 2] * H
            tw = t[:, 3] * W
            th = t[:, 4] * H
            cls_ids = t[:, 0].long()

            for i in range(t.shape[0]):
                # grid cell assignment
                gx = int(tx[i].clamp(0, W - 1))
                gy = int(ty[i].clamp(0, H - 1))
                a = 0  # single anchor (kept as 0 here to match your head)

                # decode predicted regression at that cell
                dx = torch.sigmoid(reg[b_idx, a, 0, gy, gx])
                dy = torch.sigmoid(reg[b_idx, a, 1, gy, gx])
                dw = torch.exp(reg[b_idx, a, 2, gy, gx].clamp(-4.0, 4.0))
                dh = torch.exp(reg[b_idx, a, 3, gy, gx].clamp(-4.0, 4.0))

                px = (gx + dx)    # center x in grid units
                py = (gy + dy)    # center y in grid units
                pw = dw           # width in grid units (as per your encoding)
                ph = dh           # height in grid units

                # construct pred and target boxes in grid units (xyxy)
                pred_box = torch.stack([
                    px - pw / 2.0, py - ph / 2.0,
                    px + pw / 2.0, py + ph / 2.0
                ]).to(device)

                tgt_box = torch.stack([
                    tx[i] - tw[i] / 2.0, ty[i] - th[i] / 2.0,
                    tx[i] + tw[i] / 2.0, ty[i] + th[i] / 2.0
                ]).to(device)

                # bbox regression: CIoU or Smooth-L1
                if self.use_ciou:
                    ciou_val = MCUDetectionLoss._bbox_ciou(pred_box, tgt_box)
                    # bbox loss = (1 - ciou)
                    bbox_loss = bbox_loss + (1.0 - ciou_val)
                else:
                    bbox_loss = bbox_loss + F.smooth_l1_loss(pred_box, tgt_box)

                # objectness: positive target = 1.0 (keeps compatibility)
                obj_target[b_idx, a, 0, gy, gx] = 1.0
                pos_logit = obj[b_idx, a, 0, gy, gx].view(1)
                obj_loss = obj_loss + self.bce(pos_logit, torch.ones_like(pos_logit, device=device)).sum()

                # classification: focal (expects (N, C) vs (N, C) one-hot)
                if cls_ids[i] < self.num_classes:
                    onehot = torch.zeros(self.num_classes, device=device)
                    onehot[cls_ids[i]] = 1.0
                    cls_pred = cls[b_idx, a, :, gy, gx].unsqueeze(0)  # (1, C)
                    cls_loss = cls_loss + self.focal(cls_pred, onehot.unsqueeze(0))

                npos += 1

        # background hard negatives: pick top-k highest BCE losses among negatives
        if npos > 0:
            bg_mask = (obj_target == 0)
            if bg_mask.any():
                neg_logits = obj[bg_mask]
                bg_losses = self.bce(neg_logits, torch.zeros_like(neg_logits))
                num_hard = min(npos * 3, int(bg_losses.numel()))
                if num_hard > 0:
                    # topk returns values sorted; sum them
                    obj_loss = obj_loss + bg_losses.view(-1).topk(num_hard)[0].sum()

        # return scalar tensors and npos
        return bbox_loss, obj_loss, cls_loss, npos



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





