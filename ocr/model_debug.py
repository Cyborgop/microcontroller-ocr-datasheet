#!/usr/bin/env python3
"""
model_debug.py — MCU Detector V2-Lite (OPTIMIZED)
=============================================================
Updated for new optimized architecture:
  - Streamlined backbone (no P2, single CSP per stage)
  - Lightweight BiFPN (96ch, depthwise downsampling)
  - Decoupled heads with DW-separable branches
  - Dropout2d(0.1) in cls branch
  - Label smoothing support
  - 14 classes, ~0.38M params
"""

import os, sys, shutil, traceback
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from model import (
    MCUDetector, MCUDetectionLoss, MCUDetectorBackbone,
    BiFPNModule, MCUDetectionHead, DecoupledScaleHead,
    RepvitBlock, BottleneckCSPBlock, RepDWConvSR
)
from utils import NUM_CLASSES, CLASSES

# ---- CONFIG ----
IMG_DIR = Path("data/dataset_train/images/train")
LBL_DIR = Path("data/dataset_train/labels/train")
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_FAIL_DIR = Path("debug_failures")
BATCH_TEST_SIZE = 2


# ---- HELPERS ----
def reset_dir(p):
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def read_image(p):
    img = cv2.imread(str(p))
    if img is None: raise RuntimeError(f"cv2.imread failed: {p}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) \
        .permute(2,0,1).float().div(255).unsqueeze(0).to(DEVICE)
    return t, img

def load_labels(p):
    if not p.exists(): return torch.zeros((0,5))
    rows = []
    for l in p.read_text().splitlines():
        if l.strip(): rows.append(list(map(float, l.split()[:5])))
    t = torch.tensor(rows) if rows else torch.zeros((0,5))
    if t.numel():
        assert torch.all((t[:,1:] >= 0) & (t[:,1:] <= 1)), "Label not normalized"
        assert torch.all(t[:,3:] > 0), "Zero/neg bbox size"
    return t

def feature_ok(x, name, threshold=1e-6):
    mean = x.abs().mean().item()
    std = x.std().item()
    if std < threshold and mean < threshold:
        print(f"   ⚠️ {name} low-energy (mean={mean:.2e}, std={std:.2e})")
        return False
    print(f"   ✅ {name} energy OK (mean={mean:.2e}, std={std:.2e})")
    return True

def check_weight_init(module, name):
    has_bad = False
    for n, p in module.named_parameters():
        if 'bn' in n or 'bias' in n: continue
        if p.dim() >= 2:
            if p.std().item() < 1e-6:
                print(f"   ❌ {name}.{n} poorly init (std={p.std().item():.2e})")
                has_bad = True
            if p.abs().max().item() > 10:
                print(f"   ⚠️ {name}.{n} large values (max={p.abs().max().item():.2f})")
    return not has_bad


# ================= V2-LITE ARCHITECTURE CHECKS =================

def check_backbone(model, img_t):
    """Test backbone produces correct feature maps (V2-Lite)."""
    print("\n🔍 BACKBONE CHECK (V2-Lite)")
    model.eval()
    with torch.no_grad():
        p3, p4 = model.backbone(img_t)

    expected_ch = model.backbone.out_channels
    B = img_t.shape[0]

    # P3: stride=4 → 512/4 = 128
    assert p3.shape == (B, expected_ch["p3"], IMG_SIZE//4, IMG_SIZE//4), \
        f"P3 shape wrong: {p3.shape} vs expected (B, {expected_ch['p3']}, 128, 128)"
    feature_ok(p3, "P3")

    # P4: stride=8 → 512/8 = 64
    assert p4.shape == (B, expected_ch["p4"], IMG_SIZE//8, IMG_SIZE//8), \
        f"P4 shape wrong: {p4.shape} vs expected (B, {expected_ch['p4']}, 64, 64)"
    feature_ok(p4, "P4")

    print(f"   ✅ Backbone OK: P3={tuple(p3.shape)}, P4={tuple(p4.shape)}")
    return p3, p4


def check_bifpn(model, p3, p4):
    """Test BiFPN produces correct fused features (V2-Lite: 96ch, depthwise)."""
    print("\n🔍 BiFPN CHECK (96ch, depthwise)")
    model.eval()
    with torch.no_grad():
        p3_out, p4_out = model.fpn(p3, p4)

    # BiFPN output should be fpn_ch=96
    fpn_ch = model.fpn.lat_p3.out_channels  # should be 96
    B = p3.shape[0]

    assert p3_out.shape == (B, fpn_ch, IMG_SIZE//4, IMG_SIZE//4), \
        f"BiFPN P3 shape wrong: {p3_out.shape} (expected {B},{fpn_ch},{IMG_SIZE//4},{IMG_SIZE//4})"
    assert p4_out.shape == (B, fpn_ch, IMG_SIZE//8, IMG_SIZE//8), \
        f"BiFPN P4 shape wrong: {p4_out.shape} (expected {B},{fpn_ch},{IMG_SIZE//8},{IMG_SIZE//8})"

    feature_ok(p3_out, "BiFPN_P3")
    feature_ok(p4_out, "BiFPN_P4")

    # Check learnable fusion weights
    w_td = F.relu(model.fpn.w_td)
    w_bu = F.relu(model.fpn.w_bu)
    print(f"   📊 TD weights: {w_td.data.cpu().tolist()}")
    print(f"   📊 BU weights: {w_bu.data.cpu().tolist()}")

    # Weights should not be all zero
    assert w_td.sum() > 0, "TD fusion weights all zero!"
    assert w_bu.sum() > 0, "BU fusion weights all zero!"

    # Check that bu_down is depthwise (groups == in_channels)
    assert model.fpn.bu_down.groups == fpn_ch, \
        f"bu_down should be depthwise (groups={fpn_ch}), got groups={model.fpn.bu_down.groups}"

    print(f"   ✅ BiFPN OK (depthwise downsampling: {model.fpn.bu_down.weight.shape} params)")
    print(f"   ✅ BiFPN OK: P3={tuple(p3_out.shape)}, P4={tuple(p4_out.shape)}")
    return p3_out, p4_out


def check_decoupled_head(model, p3_fpn, p4_fpn):
    """Test decoupled detection head (V2-Lite: DW+PW branches)."""
    print("\n🔍 DECOUPLED HEAD CHECK (DW+PW lightweight)")
    model.eval()
    B = p3_fpn.shape[0]
    
    # Get head_ch from model
    head_ch = model.head.p3_refine.refine.token_mixer.channels

    with torch.no_grad():
        p3_out, p4_out = model.head(p3_fpn, p4_fpn)

    for name, out, spatial in [("P3", p3_out, IMG_SIZE//4), ("P4", p4_out, IMG_SIZE//8)]:
        assert isinstance(out, (tuple, list)) and len(out) == 3, \
            f"{name} head must return (obj, cls, reg), got {type(out)} len={len(out) if isinstance(out, (tuple,list)) else '?'}"

        obj, cls, reg = out
        assert obj.shape == (B, 1, spatial, spatial), \
            f"{name} obj shape wrong: {obj.shape} vs ({B}, 1, {spatial}, {spatial})"
        assert cls.shape == (B, NUM_CLASSES, spatial, spatial), \
            f"{name} cls shape wrong: {cls.shape} vs ({B}, {NUM_CLASSES}, {spatial}, {spatial})"
        assert reg.shape == (B, 4, spatial, spatial), \
            f"{name} reg shape wrong: {reg.shape} vs ({B}, 4, {spatial}, {spatial})"

        assert torch.isfinite(obj).all(), f"{name} obj has NaN/Inf"
        assert torch.isfinite(cls).all(), f"{name} cls has NaN/Inf"
        assert torch.isfinite(reg).all(), f"{name} reg has NaN/Inf"

        feature_ok(obj, f"{name}_obj")
        feature_ok(cls, f"{name}_cls")
        feature_ok(reg, f"{name}_reg")

        # Get the head module for this scale
        head_module = model.head.p3_refine if name == "P3" else model.head.p4_refine
        
        # Check that branches are lightweight (DW + PW)
        cls_conv1 = head_module.cls_branch[0]
        cls_conv2 = head_module.cls_branch[-1]
        assert cls_conv1.groups == head_ch, f"{name} cls first conv should be depthwise (groups={head_ch})"
        assert cls_conv2.kernel_size == (1,1), f"{name} cls last conv should be 1x1"

    # ========== DROPOUT CHECK ==========
    print("\n   🔬 Checking for dropout in classification branches:")
    for head_name, head_module in [("P3", model.head.p3_refine), ("P4", model.head.p4_refine)]:
        has_dropout = False
        for name, module in head_module.cls_branch.named_modules():
            if isinstance(module, nn.Dropout2d):
                has_dropout = True
                print(f"      ✅ {head_name} has {module.__class__.__name__} with p={module.p}")
                break
        if not has_dropout:
            print(f"      ❌ {head_name} MISSING dropout in classification branch!")
    # ========== END DROPOUT CHECK ==========

    # Check prior_prob bias initialization
    for head_name, head_module in [("P3", model.head.p3_refine), ("P4", model.head.p4_refine)]:
        cls_bias = head_module.cls_branch[-1].bias
        obj_bias = head_module.obj_conv.bias
        if cls_bias is not None:
            expected_bias = -4.595  # -log((1-0.01)/0.01)
            actual_mean = cls_bias.mean().item()
            print(f"   📊 {head_name} cls_bias mean: {actual_mean:.3f} (expect ~{expected_bias:.3f})")
            if abs(actual_mean - expected_bias) > 1.0:
                print(f"   ⚠️ {head_name} cls bias seems off — may cause high false positive rate")

    print(f"   ✅ Decoupled Head OK (lightweight branches)")
    return p3_out, p4_out


def check_loss_function(model, loss_fn, img_t, labels):
    """Test loss computation with real labels (with label smoothing check)."""
    print("\n🔍 LOSS FUNCTION CHECK")
    model.train()

    # ===== AMP-SAFE CHECK: Test under autocast =====
    from torch.cuda.amp import autocast
    use_amp = (img_t.device.type == "cuda")

    with autocast(enabled=use_amp):
        outputs = model(img_t)
    B = img_t.shape[0]

    # Split targets by area
    targets_p3, targets_p4 = [], []
    H, W = img_t.shape[-2], img_t.shape[-1]
    area_threshold = 0.02 * H * W

    for b in range(B):
        if isinstance(labels, list):
            t = labels[b].to(DEVICE) if labels[b].numel() > 0 else torch.zeros((0, 5), device=DEVICE)
        else:
            t = labels.to(DEVICE) if labels.numel() > 0 else torch.zeros((0, 5), device=DEVICE)

        if t.numel() == 0:
            targets_p3.append(torch.zeros((0, 5), device=DEVICE))
            targets_p4.append(torch.zeros((0, 5), device=DEVICE))
            continue

        areas = (t[:, 3] * W) * (t[:, 4] * H)
        small = areas < area_threshold
        targets_p3.append(t[small] if small.any() else t)
        targets_p4.append(t[~small] if (~small).any() else t)

    loss_dict = loss_fn(outputs[0], outputs[1], targets_p3, targets_p4)

    # Verify keys
    for key in ["total", "bbox", "obj", "cls"]:
        assert key in loss_dict, f"Loss dict missing '{key}'"

    total = loss_dict["total"]
    assert isinstance(total, torch.Tensor), "total must be tensor"
    assert total.requires_grad, "total must require grad"
    assert torch.isfinite(total), f"total is NaN/Inf: {total.item()}"

    # ✅ KEY CHECK: cls_loss must NOT be negative (old bug)
    cls_val = loss_dict["cls"].item()
    assert cls_val >= 0, f"❌ cls_loss is NEGATIVE ({cls_val:.4f}) — FocalLoss bug NOT fixed!"

    # ✅ LABEL SMOOTHING CHECK
    if hasattr(loss_fn, 'label_smoothing') and loss_fn.label_smoothing > 0:
        print(f"   🔬 Label smoothing enabled: {loss_fn.label_smoothing}")
        if cls_val < 1e-6 and targets_p3[0].numel() > 0:
            print(f"   ⚠️ cls_loss suspiciously low ({cls_val:.6f}) with label smoothing")
    else:
        print(f"   ⚠️ Label smoothing NOT enabled")

    # ✅ KEY CHECK: total loss should be positive and reasonable
    total_val = total.item()
    assert total_val >= 0, f"❌ total loss is NEGATIVE ({total_val:.4f})"
    assert total_val < 1000, f"⚠️ total loss suspiciously high ({total_val:.4f})"

    # ✅ AMP-SPECIFIC: Test loss under autocast
    if img_t.device.type == "cuda":
        print("   🔬 Testing loss under AMP autocast (FP16)...")
        model.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            amp_outputs = model(img_t)
            amp_loss_dict = loss_fn(amp_outputs[0], amp_outputs[1], targets_p3, targets_p4)
        amp_total = amp_loss_dict["total"]
        assert torch.isfinite(amp_total), f"❌ AMP loss is NaN/Inf!"
        assert amp_loss_dict["cls"].item() >= 0, f"❌ AMP cls_loss NEGATIVE"
        print(f"   ✅ AMP loss OK: total={amp_total.item():.4f}")

    print(f"   📊 Loss values: total={total_val:.4f}, bbox={loss_dict['bbox'].item():.4f}, "
          f"obj={loss_dict['obj'].item():.4f}, cls={cls_val:.4f}")

    # Backward check
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients produced"

    # Check for NaN gradients
    nan_grads = sum(1 for g in grads if not torch.isfinite(g).all())
    if nan_grads > 0:
        print(f"   ⚠️ {nan_grads} parameters have NaN/Inf gradients")
    else:
        print(f"   ✅ All gradients finite ({len(grads)} params with grad)")

    print(f"   ✅ Loss function OK")


def check_all_weights_v2(model):
    """Check weight initialization for V2-Lite architecture."""
    print("\n" + "="*60)
    print("📊 WEIGHT INITIALIZATION CHECK (V2-Lite)")
    print("="*60)

    all_ok = True
    stats = {}

    def _check(name, module):
        count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        stats[name] = count
        return check_weight_init(module, name)

    # Backbone (V2-Lite: no p2, p3_expand instead of p3_down)
    print("\n🔍 Backbone (V2-Lite):")
    all_ok &= _check("backbone.stem", model.backbone.stem)
    all_ok &= _check("backbone.p3_expand", model.backbone.p3_expand)
    all_ok &= _check("backbone.p3", model.backbone.p3)
    all_ok &= _check("backbone.p4_down", model.backbone.p4_down)
    all_ok &= _check("backbone.p4", model.backbone.p4)

    # BiFPN (96ch, depthwise)
    print("\n🔍 BiFPN (96ch, depthwise):")
    all_ok &= _check("fpn.lat_p3", model.fpn.lat_p3)
    all_ok &= _check("fpn.lat_p4", model.fpn.lat_p4)
    all_ok &= _check("fpn.td_refine_p3", model.fpn.td_refine_p3)
    all_ok &= _check("fpn.bu_refine_p4", model.fpn.bu_refine_p4)
    all_ok &= _check("fpn.bu_down", model.fpn.bu_down)  # Should be depthwise

    # Decoupled Heads (lightweight DW+PW)
    print("\n🔍 Decoupled Heads (lightweight):")
    for scale_name, scale_head in [("p3_refine", model.head.p3_refine),
                                     ("p4_refine", model.head.p4_refine)]:
        all_ok &= _check(f"head.{scale_name}.refine", scale_head.refine)
        all_ok &= _check(f"head.{scale_name}.cls_branch", scale_head.cls_branch)
        all_ok &= _check(f"head.{scale_name}.reg_branch", scale_head.reg_branch)
        all_ok &= _check(f"head.{scale_name}.obj_conv", scale_head.obj_conv)

    # Summary
    print("\n📊 Parameter Summary (V2-Lite):")
    total = 0
    for name, count in stats.items():
        print(f"   {name:40s}: {count:>8,}")
        total += count
    print(f"   {'TOTAL':40s}: {total:>8,} ({total/1e6:.3f}M)")

    if all_ok:
        print("\n✅ All weights properly initialized")
    else:
        print("\n⚠️ Some weights need review")
    return all_ok


def check_batch_mode(model, loss_fn, img_t, labels):
    """Test with batch_size=2 to catch indexing bugs."""
    print("\n🔍 BATCH MODE CHECK (B=2)")

    # Duplicate image with flip
    img_batch = torch.cat([img_t, img_t.flip(-1)], dim=0)

    if isinstance(labels, list):
        labels_batch = labels * 2  # repeat
    elif labels.numel() > 0:
        labels2 = labels.clone()
        labels2[:, 1] = (1.0 - labels2[:, 1]).clamp(0, 1)
        labels_batch = [labels.to(DEVICE), labels2.to(DEVICE)]
    else:
        labels_batch = [torch.zeros((0, 5), device=DEVICE), torch.zeros((0, 5), device=DEVICE)]

    model.train()
    model.zero_grad(set_to_none=True)

    outputs = model(img_batch)

    # Split targets
    H, W = IMG_SIZE, IMG_SIZE
    area_threshold = 0.02 * H * W
    targets_p3, targets_p4 = [], []
    for t in labels_batch:
        if t.numel() == 0:
            targets_p3.append(torch.zeros((0, 5), device=DEVICE))
            targets_p4.append(torch.zeros((0, 5), device=DEVICE))
            continue
        t = t.to(DEVICE)
        areas = (t[:, 3] * W) * (t[:, 4] * H)
        small = areas < area_threshold
        targets_p3.append(t[small] if small.any() else t)
        targets_p4.append(t[~small] if (~small).any() else t)

    loss_dict = loss_fn(outputs[0], outputs[1], targets_p3, targets_p4)
    loss = loss_dict["total"]
    assert torch.isfinite(loss), "Batch loss is NaN/Inf"
    loss.backward()

    grads = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"   📊 Batch loss: {loss.item():.4f}, grads: {grads} params")
    print(f"   ✅ Batch mode OK")


# ================= MAIN =================
def main():
    print("\n" + "="*60)
    print("🚀 MCU DETECTOR V2-Lite — OPTIMIZED MODEL DEBUG")
    print(f"   NUM_CLASSES = {NUM_CLASSES}")
    print(f"   Classes: {', '.join(CLASSES[:5])}...")
    print("="*60)

    reset_dir(SAVE_FAIL_DIR)

    imgs = sorted([p for p in IMG_DIR.iterdir()
                   if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    assert imgs, f"No images found in {IMG_DIR}"

    model = MCUDetector(num_classes=NUM_CLASSES).to(DEVICE)

    # ✅ Updated loss constructor with label smoothing
    loss_fn = MCUDetectionLoss(
        num_classes=NUM_CLASSES,
        bbox_weight=1.0,
        obj_weight=2.0,      # Boosted for training (will be set in train.py)
        cls_weight=5.0,      # Boosted for training (will be set in train.py)
        topk=9,
        focal_gamma=2.0,
        label_smoothing=0.05,  # Enabled for training
    ).to(DEVICE)

    fails = 0

    # 1. Weight init check
    try:
        check_all_weights_v2(model)
    except Exception as e:
        fails += 1
        print(f"❌ Weight check failed: {e}")
        traceback.print_exc()

    # 2. Per-component forward checks
    img_p = imgs[0]
    lbl_p = LBL_DIR / (img_p.stem + ".txt")
    img_t, _ = read_image(img_p)
    labels = load_labels(lbl_p)

    try:
        p3, p4 = check_backbone(model, img_t)
        p3_fpn, p4_fpn = check_bifpn(model, p3, p4)
        check_decoupled_head(model, p3_fpn, p4_fpn)
    except Exception as e:
        fails += 1
        print(f"❌ Forward check failed: {e}")
        traceback.print_exc()

    # 3. Loss function check (with real labels and label smoothing)
    try:
        check_loss_function(model, loss_fn, img_t, labels)
    except Exception as e:
        fails += 1
        print(f"❌ Loss check failed: {e}")
        traceback.print_exc()

    # 4. Batch mode
    try:
        check_batch_mode(model, loss_fn, img_t, labels)
    except Exception as e:
        fails += 1
        print(f"❌ Batch check failed: {e}")
        traceback.print_exc()

    # 5. Multi-image test
    tested = 0
    for img_p in tqdm(imgs[:5], desc="🔬 Testing images"):
        lbl_p = LBL_DIR / (img_p.stem + ".txt")
        try:
            img_t, _ = read_image(img_p)
            labels = load_labels(lbl_p)
            model.train()
            model.zero_grad(set_to_none=True)
            out = model(img_t)

            # Quick loss check
            t3 = [labels.to(DEVICE) if labels.numel() > 0 else torch.zeros((0,5), device=DEVICE)]
            t4 = [labels.to(DEVICE) if labels.numel() > 0 else torch.zeros((0,5), device=DEVICE)]
            ld = loss_fn(out[0], out[1], t3, t4)
            assert torch.isfinite(ld["total"]), f"Loss NaN/Inf for {img_p.name}"
            assert ld["cls"].item() >= 0, f"Negative cls_loss for {img_p.name}"
            ld["total"].backward()
            tested += 1
        except Exception as e:
            fails += 1
            (SAVE_FAIL_DIR / f"{img_p.stem}.txt").write_text(traceback.format_exc())
            print(f"❌ {img_p.name}: {e}")

    # Summary
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print(f"   Images tested: {tested}")
    print(f"   Failures: {fails}")
    print("="*60)

    if fails == 0:
        print("\n🎉 MODEL V2-Lite FULLY VERIFIED — SAFE TO TRAIN 🎉\n")
    else:
        print(f"\n❌ {fails} FAILURES — CHECK debug_failures/\n")

    return fails == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)