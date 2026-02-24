#!/usr/bin/env python3
"""
system_sync_debug.py

FINAL STRICT SYSTEM SYNCHRONIZATION TEST
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


# ======================================================
# helpers
# ======================================================
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
# imports
# ======================================================
print("\n[1/9] IMPORT CHECK")
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
# global contracts
# ======================================================
print("\n[2/9] GLOBAL CONTRACT")

if not hasattr(utils, "CLASSES"):
    die("utils.CLASSES missing")

NUM_CLASSES = len(utils.CLASSES)
if NUM_CLASSES <= 0:
    die("NUM_CLASSES must be > 0")

ok(f"NUM_CLASSES = {NUM_CLASSES}")


# ======================================================
# dataset sanity (NO IO)
# ======================================================
print("\n[3/9] DATASET SANITY")

if not hasattr(dataset, "MCUDetectionDataset"):
    die("MCUDetectionDataset class missing")

ok("MCUDetectionDataset class available (instantiation skipped)")


# ======================================================
# synthetic batch
# ======================================================
print("\n[4/9] SYNTHETIC BATCH")

images = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE)
targets = [
    torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]], dtype=torch.float32)
    for _ in range(BATCH)
]

ok(f"images: {tuple(images.shape)}")


# ======================================================
# model forward (INFERENCE CONTRACT)
# ======================================================
print("\n[5/9] MODEL FORWARD")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = model.MCUDetector(NUM_CLASSES).to(device)
net.eval()

with torch.no_grad():
    outputs = net(images.to(device))

if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
    die("Model must return (P3, P4)")

for i, scale in enumerate(outputs):
    if not isinstance(scale, (tuple, list)) or len(scale) != 3:
        die(f"P{i} must return (obj, cls, reg)")

    obj, cls, reg = scale

    if obj.shape[0] != BATCH:
        die(f"P{i} obj batch mismatch")
    if cls.shape[0] != BATCH:
        die(f"P{i} cls batch mismatch")
    if reg.shape[0] != BATCH:
        die(f"P{i} reg batch mismatch")

    if reg.shape[1] < 4:
        die(f"P{i} reg channels < 4")

    ok(
        f"P{i} OK | "
        f"obj {tuple(obj.shape)}, "
        f"cls {tuple(cls.shape)}, "
        f"reg {tuple(reg.shape)}"
    )


# ======================================================
# loss + backward (TRAINING CONTRACT)
# ======================================================
print("\n[6/9] LOSS + BACKWARD")

criterion = model.MCUDetectionLoss(NUM_CLASSES).to(device)

net.train()                     # ✅ enable gradients
net.zero_grad(set_to_none=True)

# 🔁 MUST re-run forward (never reuse no_grad outputs)
outputs = net(images.to(device))

# Scale-aware target split (same logic as train.py)
t3, t4 = [], []
for t in targets:
    area = t[:, 3] * t[:, 4] * IMG_SIZE * IMG_SIZE
    t3.append(t[area < 512].to(device))
    t4.append(t[(area >= 512) & (area < 1024)].to(device))

loss_dict = criterion(outputs[0], outputs[1], t3, t4)

if "total" not in loss_dict:
    die("Loss dict missing 'total'")

loss = loss_dict["total"]

if not loss.requires_grad:
    die("Loss does not require grad")

loss.backward()

if not torch.isfinite(loss):
    die("Loss is NaN / Inf")

has_grad = any(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in net.parameters()
)

if not has_grad:
    die("Backward ran but no gradients produced")

ok(f"Loss backward OK | loss={float(loss):.6f}")


# ======================================================
# decode semantics
# ======================================================
print("\n[7/9] DECODE SEMANTICS")

preds = utils.decode_predictions(
    pred_p3=outputs[0],
    pred_p4=outputs[1],
    conf_thresh=0.01,
    nms_thresh=0.45,
    img_size=IMG_SIZE
)

if not isinstance(preds, list):
    die("decode_predictions must return list")

ok("decode_predictions OK")


# ======================================================
# EMA + fusion (SAFE)
# ======================================================
print("\n[8/9] EMA + FUSION")

ema = train.ModelEMA(net)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

net.train()
net.zero_grad(set_to_none=True)

x = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE).to(device)
out = net(x)

# ✅ VALID scalar loss (NOT indexing tensors blindly)
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
# final verdict
# ======================================================
print("\n" + "=" * 70)
print("✅ SYSTEM SYNC PASSED — SAFE TO TRAIN")
print("=" * 70 + "\n")
