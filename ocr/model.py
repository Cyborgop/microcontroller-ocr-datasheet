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
        self.act1 = SiLU(inplace=True)
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
        self.act2 = SiLU(inplace=True)
        self.res_scale = 0.5

    def forward(self, x):
        identity = x
        x = self.token_mixer(x)      # depthwise token mixing
        x = self.act1(x) 
        if self.use_se:
            x = x * self.se(x)
        x = self.channel_mixer(x)    # channel mixing (pointwise)
        if self.use_res:
            x = identity + self.res_scale * x
        return self.act2(x)

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
    def __init__(self, in_ch, out_ch, n_blocks=2, ratio=0.25,use_se=False):
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
        
        blocks = []
        for i in range(n_blocks):
            blocks.append(
                RepvitBlock(
                    hidden, hidden,
                    stride=1,
                    use_se=(use_se and i % 2 == 0)
                )
            )
        self.blocks = nn.Sequential(*blocks)
        
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
class BiFPNModule(nn.Module):
    """
    BiFPN with separate refinement blocks AND BatchNorm for stability
    """
    def __init__(self, ch_p3, ch_p4, fpn_ch=128):
        super().__init__()
        self.eps = 1e-4
        
        # Lateral projections
        self.lat_p3 = nn.Conv2d(ch_p3, fpn_ch, 1, bias=False)
        self.lat_p4 = nn.Conv2d(ch_p4, fpn_ch, 1, bias=False)
        
        # Learnable fusion weights
        self.w_td = nn.Parameter(torch.ones(2))
        self.w_bu = nn.Parameter(torch.ones(2))
        
        # TOP-DOWN PATH
        self.td_refine_p3 = RepvitBlock(fpn_ch, fpn_ch, use_se=False)
        
        # BOTTOM-UP PATH (with BatchNorm for stability ✅)
        self.bu_down = nn.Conv2d(fpn_ch, fpn_ch, 3, stride=2, padding=1, bias=False)
        self.bu_bn = nn.BatchNorm2d(fpn_ch)  # From first version ✅
        self.bu_refine_p4 = RepvitBlock(fpn_ch, fpn_ch, use_se=False)
        
        # Use YOUR proven weight init ✅
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)  # Your proven method
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, p3, p4):
        p3_lat = self.lat_p3(p3)
        p4_lat = self.lat_p4(p4)
        
        # Fast normalized fusion
        w_td = F.relu(self.w_td)
        w_td = w_td / (w_td.sum() + self.eps)
        
        w_bu = F.relu(self.w_bu)
        w_bu = w_bu / (w_bu.sum() + self.eps)
        
        # TOP-DOWN
        p4_up = F.interpolate(p4_lat, size=p3_lat.shape[-2:], mode='nearest')
        p3_td = self.td_refine_p3(w_td[0] * p3_lat + w_td[1] * p4_up)
        
        # BOTTOM-UP (with BatchNorm)
        p3_down = self.bu_bn(self.bu_down(p3_td))
        p4_out = self.bu_refine_p4(w_bu[0] * p4_lat + w_bu[1] * p3_down)
        
        return p3_td, p4_out



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
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),  # /2
            nn.BatchNorm2d(32),
            SiLU(inplace=True),
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),  # /4 total
            nn.BatchNorm2d(48),
            SiLU(inplace=True),   
        )
        # p2: keeps stride = 4
        self.p2 = RepvitBlock(48, 48, use_se=True)              # stride=4, 48ch

        # p3_down: **no spatial downsample** so p3 will have stride = 4
        self.p3_down = RepvitBlock(48, 96, stride=1)  # stride=4, 96ch  ← INCREASED

        # p3: use fewer blocks for mobile if desired; current uses 2 CSP repeats
        self.p3 = nn.Sequential(
            BottleneckCSPBlock(96, 96, n_blocks=2, use_se=True),   # Block 0: WITH SE
            BottleneckCSPBlock(96, 96, n_blocks=2, use_se=False),  # Block 1: NO SE
        )

        # p4_down: downsample here => p4 stride = 8
        self.p4_down = RepvitBlock(96, 192, stride=2)  # stride=8, 192ch ← INCREASED

        self.p4 = nn.Sequential(
            BottleneckCSPBlock(192, 192, n_blocks=2, use_se=False), # Block 0: NO SE (low-res)
            BottleneckCSPBlock(192, 192, n_blocks=2, use_se=False), # Block 1: NO SE
        )

        # helpful metadata for consistency elsewhere
        self.out_channels = {"p3": 96, "p4": 192}
        self.out_strides = {"p3": 4, "p4": 8}
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)     # stride = 4
        x = self.p2(x)       # stride = 4
        p3 = self.p3(self.p3_down(x))    # p3: stride = 4, 96 ch
        p4 = self.p4(self.p4_down(p3))   # p4: stride = 8, 192 ch
        return p3, p4

# =============================================================================
# DECOUPLED DETECTION HEAD (YOLOX-style)
# =============================================================================
class DecoupledScaleHead(nn.Module):
    """
    Decoupled per-scale head (classification & regression separated),
    but named to match your existing MCUDetectionHead conventions.
    Returns: (obj_logits, cls_logits, reg_preds)
    """
    def __init__(self, fpn_ch, head_ch, num_classes, num_anchors=1, prior_prob=0.01):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_anchors = int(num_anchors)

        # Shared stem (refine)
        self.refine = RepvitBlock(fpn_ch, head_ch, use_se=False)

        # Classification branch
        self.cls_branch = nn.Sequential(
            RepvitBlock(head_ch, head_ch, use_se=False),
            nn.Conv2d(head_ch, num_anchors * num_classes, kernel_size=1)
        )

        # Regression branch
        self.reg_branch = nn.Sequential(
            RepvitBlock(head_ch, head_ch, use_se=False),
            nn.Conv2d(head_ch, num_anchors * 4, kernel_size=1)
        )

        # Objectness conv (kept separate as single 1-channel output)
        self.obj_conv = nn.Conv2d(head_ch, num_anchors * 1, kernel_size=1)

        # weight init / prior bias
        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        if prior_prob is not None and prior_prob > 0:
            bias = float(-torch.log(torch.tensor((1.0 - prior_prob) / prior_prob)))
            # classification bias (last conv in cls_branch)
            nn.init.constant_(self.cls_branch[-1].bias, bias)
            # objectness bias
            nn.init.constant_(self.obj_conv.bias, bias)

    def forward(self, x):
        feat = self.refine(x)            # (B, head_ch, H, W)
        obj = self.obj_conv(feat)        # (B, 1 or A, H, W)
        cls = self.cls_branch(feat)      # (B, num_classes or A*num_classes, H, W)
        reg = self.reg_branch(feat)      # (B, 4 or A*4, H, W)
        return obj, cls, reg


class MCUDetectionHead(nn.Module):
    """
    Decoupled multi-scale detection head with naming consistent with your original head.
    Each scale returns (obj_logits, cls_logits, reg_preds) to match older code.
    """
    def __init__(self, num_classes, num_anchors=1, fpn_ch=192, head_ch=128, prior_prob=0.01):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_anchors = int(num_anchors)

        # P3 head (stride=4)
        self.p3_refine = DecoupledScaleHead(fpn_ch=fpn_ch, head_ch=head_ch,
                                            num_classes=num_classes, num_anchors=num_anchors,
                                            prior_prob=prior_prob)

        # P4 head (stride=8)
        self.p4_refine = DecoupledScaleHead(fpn_ch=fpn_ch, head_ch=head_ch,
                                            num_classes=num_classes, num_anchors=num_anchors,
                                            prior_prob=prior_prob)

    def forward(self, p3, p4, p5=None):
        # Keep the exact return shape as your original head:
        # p3_out = (p3_obj, p3_cls, p3_reg)
        p3_out = self.p3_refine(p3)
        p4_out = self.p4_refine(p4)
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




class MCUDetector(nn.Module):
    """
    MCUDetector (V2-compatible):
    - RepViT-style backbone
    - Lightweight BiFPN
    - Decoupled YOLOX-style heads
    - SAME external interface as old MCUDetector
    """
    def __init__(self, num_classes):
        super().__init__()
        assert isinstance(num_classes, int) and num_classes > 0
        self.num_classes = num_classes

        # Backbone (your latest stable one)
        self.backbone = MCUDetectorBackbone()

        # Neck (BiFPN)
        self.fpn = BiFPNModule(
            ch_p3=self.backbone.out_channels["p3"],
            ch_p4=self.backbone.out_channels["p4"],
            fpn_ch=128
        )

        # Head (decoupled, but API-compatible)
        self.head = MCUDetectionHead(
            num_classes=num_classes,
            num_anchors=1,
            fpn_ch=128,
            head_ch=128,
            prior_prob=0.01
        )

        print(
            f"[MCUDetector V2] num_classes={num_classes}, "
            f"backbone_ch={self.backbone.out_channels}, "
            f"params={self.count_params():.2f}M"
        )

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6

    def forward(self, x):
        # Backbone
        p3, p4 = self.backbone(x)

        # Neck
        p3, p4 = self.fpn(p3, p4)

        # Head
        # returns: (p3_obj, p3_cls, p3_reg), (p4_obj, p4_cls, p4_reg)
        return self.head(p3, p4)


# =============================================================================
# LOSS FUNCTIONS (WITH CRITICAL FIXES FROM SECOND CODE)
# =============================================================================

class FocalLoss(nn.Module):
    """
    Numerically STABLE Focal Loss.
    Drop-in replacement for your original FocalLoss.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # ---- 1. Clamp logits to avoid overflow ----
        logits = logits.clamp(-10, 10)

        # ---- 2. Stable probabilities ----
        p = torch.sigmoid(logits)
        p = p.clamp(1e-7, 1.0 - 1e-7)

        # ---- 3. BCE with logits (stable) ----
        ce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        ce = ce.clamp(0.0, 100.0)

        # ---- 4. Focal modulation ----
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        mod = (1.0 - p_t).pow(self.gamma)

        # ---- 5. Alpha balancing ----
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        else:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        loss = alpha_t * mod * ce
        loss = loss.clamp(0.0, 100.0)

        # ---- 6. Reduction ----
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class SimOTALiteAssigner:
    """
    Simplified OTA (Optimal Transport Assignment) for lightweight detectors.
    
    Assigns top-k grid cells per GT based on center distance.
    
    FIX: When multiple GTs claim the same cell, the CLOSEST GT wins
    (distance-based priority instead of last-writer-wins).
    """
    def __init__(self, topk=9):
        self.topk = topk
    
    def assign(self, grid_h, grid_w, stride, gt_boxes, gt_cls, device):
        """
        Args:
            grid_h, grid_w: Feature map size
            stride: Feature stride (4 or 8)
            gt_boxes: (N, 4) normalized boxes [cx, cy, w, h]
            gt_cls: (N,) class indices
            device: torch device
            
        Returns:
            pos_mask: (H, W) bool tensor
            target_boxes: (H, W, 4) regression targets
            target_cls: (H, W) class targets (-1 for negatives)
        """
        num_gt = gt_boxes.shape[0]
        
        if num_gt == 0:
            return (torch.zeros(grid_h, grid_w, dtype=torch.bool, device=device),
                    torch.zeros(grid_h, grid_w, 4, device=device),
                    torch.full((grid_h, grid_w), -1, dtype=torch.long, device=device))
        
        # Create grid centers (in grid coordinates)
        yv, xv = torch.meshgrid(
            torch.arange(grid_h, device=device),
            torch.arange(grid_w, device=device),
            indexing='ij'
        )
        grid_xy = torch.stack([xv, yv], dim=-1).float() + 0.5  # (H, W, 2)
        
        # GT centers in grid coordinates
        gt_cx = gt_boxes[:, 0] * grid_w
        gt_cy = gt_boxes[:, 1] * grid_h
        gt_centers = torch.stack([gt_cx, gt_cy], dim=-1)  # (N, 2)
        
        # Distance from each grid cell to each GT: (H, W, N)
        dist = (grid_xy.unsqueeze(2) - gt_centers.view(1, 1, num_gt, 2)).pow(2).sum(-1).sqrt()
        
        # =====================================================================
        # FIX: Track best (closest) GT per cell to resolve conflicts
        # =====================================================================
        # best_dist[y,x] = distance to the GT that currently owns cell (y,x)
        best_dist = torch.full((grid_h, grid_w), float('inf'), device=device)
        
        # Initialize outputs
        pos_mask = torch.zeros(grid_h, grid_w, dtype=torch.bool, device=device)
        target_boxes = torch.zeros(grid_h, grid_w, 4, device=device)
        target_cls = torch.full((grid_h, grid_w), -1, dtype=torch.long, device=device)
        
        # For each GT, find topk closest grid cells
        for gt_idx in range(num_gt):
            gt_dist = dist[:, :, gt_idx]  # (H, W)
            
            # Flatten and get topk
            flat_dist = gt_dist.view(-1)
            k = min(self.topk, flat_dist.numel())
            topk_vals, topk_indices = flat_dist.topk(k, largest=False)
            
            topk_y = topk_indices // grid_w
            topk_x = topk_indices % grid_w
            
            # Only assign if this GT is closer than the current owner
            for j in range(k):
                y, x = topk_y[j], topk_x[j]
                d = topk_vals[j]
                if d < best_dist[y, x]:
                    best_dist[y, x] = d
                    pos_mask[y, x] = True
                    target_boxes[y, x] = gt_boxes[gt_idx]
                    target_cls[y, x] = gt_cls[gt_idx]
        
        return pos_mask, target_boxes, target_cls



class MCUDetectionLoss(nn.Module):
    """
    MCUDetectionLoss (FINAL FIXED).

    ✔ SimOTALiteAssigner with distance-based conflict resolution
    ✔ VECTORIZED bbox/cls computation (no per-pixel Python loop)
    ✔ Separate obj normalization (by total cells, not npos)
    ✔ Stable FocalLoss
    ✔ CIoU regression (normalized boxes)
    ✔ Same forward() signature & return dict keys
    """

    def __init__(
        self,
        num_classes,
        bbox_weight=1.0,
        obj_weight=1.0,
        cls_weight=1.0,
        topk=9,
        focal_gamma=2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bbox_weight = float(bbox_weight)
        self.obj_weight = float(obj_weight)
        self.cls_weight = float(cls_weight)

        self.assigner = SimOTALiteAssigner(topk=topk)
        self.focal = FocalLoss(alpha=0.25, gamma=focal_gamma, reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    # --------------------------------------------------
    # VECTORIZED CIoU on normalized cx,cy,w,h
    # --------------------------------------------------
    @staticmethod
    def _bbox_ciou_batch(pred_boxes, tgt_boxes):
        """
        Vectorized CIoU for N box pairs.
        pred_boxes: (N, 4) as [cx, cy, w, h] normalized
        tgt_boxes:  (N, 4) as [cx, cy, w, h] normalized
        Returns: (N,) CIoU values clamped to [-1, 1]
        """
        px, py, pw, ph = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        tx, ty, tw, th = tgt_boxes[:, 0], tgt_boxes[:, 1], tgt_boxes[:, 2], tgt_boxes[:, 3]

        # to x1y1x2y2
        px1, py1 = px - pw / 2, py - ph / 2
        px2, py2 = px + pw / 2, py + ph / 2
        tx1, ty1 = tx - tw / 2, ty - th / 2
        tx2, ty2 = tx + tw / 2, ty + th / 2

        inter_x1 = torch.max(px1, tx1)
        inter_y1 = torch.max(py1, ty1)
        inter_x2 = torch.min(px2, tx2)
        inter_y2 = torch.min(py2, ty2)

        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area_p = pw * ph
        area_t = tw * th
        union = area_p + area_t - inter + 1e-7
        iou = inter / union

        center_dist = (px - tx) ** 2 + (py - ty) ** 2

        enc_x1 = torch.min(px1, tx1)
        enc_y1 = torch.min(py1, ty1)
        enc_x2 = torch.max(px2, tx2)
        enc_y2 = torch.max(py2, ty2)
        c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

        v = (4 / (math.pi ** 2)) * (
            torch.atan(tw / (th + 1e-7)) -
            torch.atan(pw / (ph + 1e-7))
        ) ** 2

        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)

        ciou = iou - center_dist / c2 - alpha * v
        return ciou.clamp(-1, 1)

    # --------------------------------------------------
    # Forward (same API)
    # --------------------------------------------------
    def forward(self, pred_p3, pred_p4, targets_p3, targets_p4):
        device = pred_p3[0].device

        zero = pred_p3[0].sum() * 0.0
        total_bbox = zero.clone()
        total_obj  = zero.clone()
        total_cls  = zero.clone()
        npos = 0
        total_cells = 0

        for pred, targets, stride in (
            (pred_p3, targets_p3, 4),
            (pred_p4, targets_p4, 8),
        ):
            b, o, c, k, n_cells = self._scale_loss(pred, targets, stride, device)
            total_bbox += b
            total_obj  += o
            total_cls  += c
            npos += k
            total_cells += n_cells

        # =====================================================================
        # FIX: Separate normalization for obj vs bbox/cls
        # bbox and cls: normalize by number of positive samples
        # obj: normalize by total grid cells (since it's summed over ALL cells)
        # =====================================================================
        if npos > 0:
            inv = 1.0 / float(npos)
            total_bbox *= inv
            total_cls  *= inv

        if total_cells > 0:
            total_obj *= (1.0 / float(total_cells))

        total = (
            self.bbox_weight * total_bbox +
            self.obj_weight  * total_obj +
            self.cls_weight  * total_cls
        )

        return {
            "total": total,
            "bbox": total_bbox.detach(),
            "obj":  total_obj.detach(),
            "cls":  total_cls.detach(),
        }

    # --------------------------------------------------
    # Per-scale loss (FULLY VECTORIZED)
    # --------------------------------------------------
    def _scale_loss(self, pred, targets, stride, device):
        obj_pred, cls_pred, reg_pred = pred
        B, _, H, W = obj_pred.shape

        bbox_loss = obj_pred.sum() * 0.0
        obj_loss  = obj_pred.sum() * 0.0
        cls_loss  = obj_pred.sum() * 0.0
        npos = 0
        total_cells = B * H * W

        obj_target = torch.zeros_like(obj_pred)

        for b in range(B):
            t = targets[b]
            if t is None or t.numel() == 0:
                continue

            gt_cls = t[:, 0].long()
            gt_boxes = t[:, 1:5]

            pos_mask, tgt_boxes, tgt_cls = self.assigner.assign(
                H, W, stride, gt_boxes, gt_cls, device
            )

            ys, xs = torch.where(pos_mask)
            if ys.numel() == 0:
                continue

            n = ys.numel()
            npos += n
            obj_target[b, 0, ys, xs] = 1.0

            # =========================================================
            # VECTORIZED regression decode + CIoU
            # =========================================================
            dx = torch.sigmoid(reg_pred[b, 0, ys, xs])       # (N,)
            dy = torch.sigmoid(reg_pred[b, 1, ys, xs])       # (N,)
            dw = reg_pred[b, 2, ys, xs].clamp(-4, 4).exp()   # (N,)
            dh = reg_pred[b, 3, ys, xs].clamp(-4, 4).exp()   # (N,)

            pred_boxes = torch.stack([
                (xs.float() + dx) / W,
                (ys.float() + dy) / H,
                dw / W,
                dh / H
            ], dim=1)  # (N, 4) as [cx, cy, w, h] normalized

            tgt = tgt_boxes[ys, xs]  # (N, 4) as [cx, cy, w, h] normalized

            ciou = self._bbox_ciou_batch(pred_boxes, tgt)  # (N,)
            bbox_loss = bbox_loss + (1.0 - ciou).sum()

            # =========================================================
            # VECTORIZED classification loss
            # =========================================================
            cls_indices = tgt_cls[ys, xs]  # (N,)
            valid = (cls_indices >= 0) & (cls_indices < self.num_classes)
            
            if valid.any():
                n_valid = valid.sum().item()
                cls_targets_oh = torch.zeros(n_valid, self.num_classes, device=device)
                cls_targets_oh[torch.arange(n_valid, device=device), cls_indices[valid]] = 1.0
                
                # Gather cls logits: cls_pred is (B, num_classes, H, W)
                cls_logits = cls_pred[b, :, ys[valid], xs[valid]].T  # (N_valid, num_classes)
                
                cls_loss = cls_loss + self.focal(cls_logits, cls_targets_oh).sum()

        # Objectness loss: BCE over ALL cells (positives + negatives)
        obj_loss = self.bce(obj_pred, obj_target).sum()

        return bbox_loss, obj_loss, cls_loss, npos, total_cells



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





