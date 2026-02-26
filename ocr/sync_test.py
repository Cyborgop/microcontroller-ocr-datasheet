#!/usr/bin/env python3
"""
sync_test.py — V2 System Synchronization Test
(utils × model × dataset × train)

PASS = training-safe
FAIL = DO NOT TRAIN
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import torch

IMG_SIZE = 512
BATCH = 2
SEED = 42


def die(msg):
    print("\n" + "=" * 70)
    print(f"❌ SYSTEM SYNC FAILURE: {msg}")
    print("=" * 70)
    sys.exit(1)


def ok(msg):
    print(f"✔ {msg}")


def seed_all(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_all()


# ======================================================
# [1/10] IMPORTS
# ======================================================
print("\n[1/10] IMPORT CHECK")
try:
    import utils
    import model
    import dataset
    import train
except Exception as e:
    traceback.print_exc()
    die(f"Import failed: {e}")

ok("utils / model / dataset / train imported")


# ======================================================
# [2/10] GLOBAL CONTRACT
# ======================================================
print("\n[2/10] GLOBAL CONTRACT")

if not hasattr(utils, "CLASSES"):
    die("utils.CLASSES missing")

NUM_CLASSES = len(utils.CLASSES)
if NUM_CLASSES <= 0:
    die("NUM_CLASSES must be > 0")

if NUM_CLASSES != 14:
    print(f"⚠️ Expected 14 classes, got {NUM_CLASSES}")

ok(f"NUM_CLASSES = {NUM_CLASSES}")
print(f"   Classes: {utils.CLASSES[:5]}...")


# ======================================================
# [3/10] V2 ARCHITECTURE CONTRACT
# ======================================================
print("\n[3/10] V2 ARCHITECTURE CONTRACT")

# Check that key V2 classes exist
for cls_name in ["MCUDetector", "MCUDetectionLoss", "BiFPNModule",
                  "MCUDetectionHead", "DecoupledScaleHead", "SimOTALiteAssigner"]:
    if not hasattr(model, cls_name):
        die(f"model.{cls_name} missing — is this V2?")

ok("All V2 architecture classes present")


# ======================================================
# [4/10] SYNTHETIC BATCH
# ======================================================
print("\n[4/10] SYNTHETIC BATCH")

images = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE)
targets = [
    torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]], dtype=torch.float32)
    for _ in range(BATCH)
]

ok(f"images: {tuple(images.shape)}")


# ======================================================
# [5/10] MODEL FORWARD (INFERENCE CONTRACT)
# ======================================================
print("\n[5/10] MODEL FORWARD")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = model.MCUDetector(NUM_CLASSES).to(device)
net.eval()

with torch.no_grad():
    outputs = net(images.to(device))

if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
    die("Model must return (P3, P4)")

for i, scale in enumerate(outputs):
    if not isinstance(scale, (tuple, list)) or len(scale) != 3:
        die(f"P{3+i} must return (obj, cls, reg)")

    obj, cls, reg = scale

    if obj.shape[0] != BATCH:
        die(f"P{3+i} obj batch mismatch")
    if obj.shape[1] != 1:
        die(f"P{3+i} obj channels must be 1, got {obj.shape[1]}")
    if cls.shape[1] != NUM_CLASSES:
        die(f"P{3+i} cls channels must be {NUM_CLASSES}, got {cls.shape[1]}")
    if reg.shape[1] != 4:
        die(f"P{3+i} reg channels must be 4, got {reg.shape[1]}")

    ok(
        f"P{3+i} OK | "
        f"obj {tuple(obj.shape)}, "
        f"cls {tuple(cls.shape)}, "
        f"reg {tuple(reg.shape)}"
    )


# ======================================================
# [6/10] LOSS + BACKWARD (TRAINING CONTRACT)
# ======================================================
print("\n[6/10] LOSS + BACKWARD")

# V2 loss constructor
criterion = model.MCUDetectionLoss(
    num_classes=NUM_CLASSES,
    bbox_weight=1.0,
    obj_weight=1.0,
    cls_weight=1.0,
    topk=9,
    focal_gamma=2.0,
).to(device)

net.train()
net.zero_grad(set_to_none=True)

# Must re-run forward (no_grad outputs can't backprop)
outputs = net(images.to(device))

# Split targets using train.py helper if available
try:
    t3, t4 = train._split_targets_by_area(targets, IMG_SIZE, IMG_SIZE, device, warmup=False)
except Exception:
    # Fallback
    t3, t4 = [], []
    for t in targets:
        area = t[:, 3] * t[:, 4] * IMG_SIZE * IMG_SIZE
        area_thresh = 0.02 * IMG_SIZE * IMG_SIZE
        t3.append(t[area < area_thresh].to(device) if (area < area_thresh).any() else t.to(device))
        t4.append(t[area >= area_thresh].to(device) if (area >= area_thresh).any() else t.to(device))

loss_dict = criterion(outputs[0], outputs[1], t3, t4)

if "total" not in loss_dict:
    die("Loss dict missing 'total'")

loss = loss_dict["total"]

if not loss.requires_grad:
    die("Loss does not require grad")

loss.backward()

if not torch.isfinite(loss):
    die("Loss is NaN / Inf")

# CRITICAL: cls_loss must not be negative
cls_val = loss_dict["cls"].item()
if cls_val < 0:
    die(f"cls_loss is NEGATIVE ({cls_val:.4f}) — FocalLoss bug!")

has_grad = any(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in net.parameters()
)

if not has_grad:
    die("Backward ran but no gradients produced")

ok(f"Loss backward OK | total={float(loss):.6f}, cls={cls_val:.6f} (non-negative ✔)")


# ======================================================
# [7/10] DECODE SEMANTICS
# ======================================================
print("\n[7/10] DECODE SEMANTICS")

# V2: model returns (obj, cls, reg) tuples — decode expects (cls_combined, reg) format
def to_decoder(pred):
    obj, cls, reg = pred
    return (torch.cat([obj, cls], dim=1), reg)

preds = utils.decode_predictions(
    pred_p3=to_decoder(outputs[0]),
    pred_p4=to_decoder(outputs[1]),
    conf_thresh=0.01,
    nms_thresh=0.45,
    img_size=IMG_SIZE
)

if not isinstance(preds, list):
    die("decode_predictions must return list")

ok("decode_predictions OK")


# ======================================================
# [8/10] EMA + FUSION
# ======================================================
print("\n[8/10] EMA + FUSION")

ema = train.ModelEMA(net)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

net.train()
net.zero_grad(set_to_none=True)

x = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE).to(device)
out = net(x)

# Valid scalar loss
ema_loss = out[0][0].mean() + out[1][0].mean()
ema_loss.backward()
optimizer.step()

ema.update(net)

diff = sum(
    (p1 - p2).abs().sum()
    for p1, p2 in zip(net.parameters(), ema.ema.parameters())
)

if diff == 0:
    die("EMA did not update")

net.eval()
out_before = net(x)
train.fuse_repv_blocks(net)
out_after = net(x)

if out_before[0][0].shape != out_after[0][0].shape:
    die("Fusion changed output shape")

ok("EMA + fusion OK")


# ======================================================
# [9/10] V2 LOSS BOUNDS CHECK
# ======================================================
print("\n[9/10] V2 LOSS BOUNDS CHECK")

# Run loss with multiple classes to verify no negatives
net2 = model.MCUDetector(NUM_CLASSES).to(device)
net2.train()

multi_targets = [
    torch.tensor([[0, 0.3, 0.3, 0.05, 0.05],
                   [5, 0.7, 0.7, 0.08, 0.08],
                   [13, 0.2, 0.8, 0.1, 0.1]], dtype=torch.float32),
    torch.tensor([[2, 0.5, 0.5, 0.3, 0.3],
                   [10, 0.6, 0.4, 0.15, 0.15]], dtype=torch.float32),
]

imgs2 = torch.rand(2, 3, IMG_SIZE, IMG_SIZE).to(device)
out2 = net2(imgs2)

try:
    mt3, mt4 = train._split_targets_by_area(multi_targets, IMG_SIZE, IMG_SIZE, device, warmup=False)
except Exception:
    mt3 = [t.to(device) for t in multi_targets]
    mt4 = [t.to(device) for t in multi_targets]

ld = criterion(out2[0], out2[1], mt3, mt4)

for key in ["total", "bbox", "obj", "cls"]:
    val = ld[key].item()
    if val < 0:
        die(f"{key} is NEGATIVE ({val:.4f}) with multi-class targets")
    ok(f"{key}={val:.4f} (non-negative ✔)")


# ======================================================
# [10/10] TRAIN.PY FUNCTION CONTRACTS
# ======================================================
print("\n[10/10] TRAIN.PY FUNCTION CONTRACTS")

# Check key functions exist
for fn_name in ["train_one_epoch", "validate", "ModelEMA", "EarlyStopping",
                 "fuse_repv_blocks", "get_eval_model", "_split_targets_by_area",
                 "save_loss_plot", "find_optimal_lr"]:
    if not hasattr(train, fn_name):
        die(f"train.{fn_name} missing")

ok("All train.py functions present")


# ======================================================
# VERDICT
# ======================================================
print("\n" + "=" * 70)
print("✅ V2 SYSTEM SYNC PASSED — SAFE TO TRAIN")
print("=" * 70 + "\n")