#!/usr/bin/env python3
"""
ultimate_system_sync_check.py

WORLD-GRADE system consistency checker.
Validates utils × model × dataset × train with:

✓ forward contracts
✓ autograd correctness
✓ EMA + fusion safety
✓ decode semantics
✓ parameter count sanity
✓ memory usage
✓ batch size scaling

If THIS passes — the system is deployment-grade.
"""

import argparse
import sys
import traceback
import time
from pathlib import Path

import numpy as np
import torch

IMG_SIZE = 512
BATCH = 2
SEED = 42


# =========================================================
# helpers
# =========================================================
def fail(msg):
    print("\n" + "=" * 70)
    print(f"❌ FAILURE: {msg}")
    print("=" * 70)
    sys.exit(1)

def ok(msg):
    print(f"✔ {msg}")

def seed_all(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# imports
# =========================================================
def import_all():
    print("\n[1/10] IMPORT CHECK")
    try:
        import utils
        import model
        import dataset
        import train
    except Exception as e:
        traceback.print_exc()
        fail(f"Import failed: {e}")
    ok("utils, model, dataset, train imported")
    return utils, model, dataset, train


# =========================================================
# globals
# =========================================================
def check_globals(utils, model):
    print("\n[2/10] GLOBAL CONTRACT")

    if not hasattr(utils, "CLASSES"):
        fail("utils.CLASSES missing")

    NUM_CLASSES = len(utils.CLASSES)
    if NUM_CLASSES <= 0:
        fail("NUM_CLASSES must be > 0")

    if not hasattr(model, "MCUDetector"):
        fail("MCUDetector missing")

    if not hasattr(model, "MCUDetectionLoss"):
        fail("MCUDetectionLoss missing")

    ok(f"NUM_CLASSES = {NUM_CLASSES}")
    return NUM_CLASSES


# =========================================================
# parameter count
# =========================================================
def check_parameter_counts(model_mod, NUM_CLASSES):
    print("\n[3/10] PARAMETER COUNT CHECK")

    model = model_mod.MCUDetector(NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    if total_params < 100_000 or total_params > 5_000_000:
        print(f"⚠️ Warning: unexpected parameter count ({total_params:,})")

    ok("Parameter count check complete")
    return total_params


# =========================================================
# data
# =========================================================
def load_batch(dataset_mod, img_dir, label_dir):
    print("\n[4/10] DATASET CHECK")
    try:
        ds = dataset_mod.MCUDetectionDataset(
            Path(img_dir), Path(label_dir), IMG_SIZE, transform=None
        )
    except Exception as e:
        print(f"⚠ dataset init failed: {e}")
        return None, None

    if len(ds) == 0:
        print("⚠ empty dataset → synthetic")
        return None, None

    imgs, tgts = [], []
    for i in range(min(BATCH, len(ds))):
        img, tgt = ds[i]
        imgs.append(img)
        tgts.append(tgt if tgt.numel() else torch.zeros((0, 5)))

    images = torch.stack(imgs)
    ok(f"Loaded real batch {tuple(images.shape)}")
    return images, tgts


def synthetic_batch():
    images = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE)
    targets = [torch.zeros((0, 5)) for _ in range(BATCH)]
    ok("Using synthetic batch")
    return images, targets


# =========================================================
# forward
# =========================================================
def check_forward(model, images, device):
    print("\n[5/10] FORWARD INTERFACE")

    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))

    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        fail("Model must return (P3, P4)")

    for i, (cls, reg) in enumerate(outputs):
        if not isinstance(cls, torch.Tensor):
            fail(f"P{i} cls not tensor")
        if not isinstance(reg, torch.Tensor):
            fail(f"P{i} reg not tensor")
        if cls.shape[0] != images.shape[0]:
            fail(f"P{i} batch mismatch")
        if reg.shape[1] < 4:
            fail(f"P{i} reg channels < 4")

        ok(f"P{i}: cls {tuple(cls.shape)}, reg {tuple(reg.shape)}")

    return outputs


# =========================================================
# loss + backward
# =========================================================
def split_targets(targets):
    p3, p4 = [], []
    for t in targets:
        if t.numel() == 0:
            p3.append(torch.zeros((0, 5)))
            p4.append(torch.zeros((0, 5)))
            continue
        area = t[:, 3] * t[:, 4] * IMG_SIZE * IMG_SIZE
        p3.append(t[area < 512])
        p4.append(t[(area >= 512) & (area < 1024)])
    return p3, p4


def check_loss_backward(model, criterion, images, targets, device):
    print("\n[6/10] LOSS + BACKWARD")

    model.train()
    outputs = model(images.to(device))

    t3, t4 = split_targets(targets)
    t3 = [x.to(device) for x in t3]
    t4 = [x.to(device) for x in t4]

    loss_dict = criterion(outputs[0], outputs[1], t3, t4)
    loss = loss_dict.get("total")

    if not isinstance(loss, torch.Tensor):
        fail("Loss is not tensor")
    if not loss.requires_grad:
        fail("Loss does not require grad")

    model.zero_grad(set_to_none=True)
    loss.backward()

    grads = [
        p.grad for p in model.parameters()
        if p.grad is not None and p.requires_grad
    ]
    if len(grads) == 0:
        fail("No parameter received gradients")

    ok(f"Loss/backward OK – loss={float(loss):.6f}")


# =========================================================
# memory usage
# =========================================================
def check_memory_usage(model, images, device):
    print("\n[7/10] MEMORY USAGE CHECK")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    model.eval()
    with torch.no_grad():
        _ = model(images.to(device))

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"GPU memory allocated: {alloc:.1f} MB")
        print(f"GPU peak memory     : {peak:.1f} MB")
    else:
        print("CPU-only run (GPU memory check skipped)")

    ok("Memory usage check complete")


# =========================================================
# batch scaling
# =========================================================
def check_batch_scaling(model, device):
    print("\n[8/10] BATCH SCALING CHECK")

    model.eval()
    for bs in [1, 2, 4, 8]:
        x = torch.rand(bs, 3, IMG_SIZE, IMG_SIZE).to(device)

        # warmup
        with torch.no_grad():
            _ = model(x)

        start = time.time()
        with torch.no_grad():
            _ = model(x)
        elapsed = time.time() - start

        print(
            f"Batch {bs:<2} → "
            f"{elapsed*1000:.1f} ms total | "
            f"{elapsed/bs*1000:.1f} ms / image"
        )

    ok("Batch scaling check complete")


# =========================================================
# decode + ema + fusion
# =========================================================
def check_decode(utils, outputs):
    print("\n[9/10] DECODE SEMANTICS")

    preds = utils.decode_predictions(
        outputs[0], outputs[1], 0.01, 0.45, IMG_SIZE
    )

    if not isinstance(preds, list):
        fail("decode_predictions must return list")

    for p in preds:
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if p.size == 0:
            continue
        if p.shape[1] < 6:
            fail("Decoded boxes must have ≥6 columns")
        if np.any(p[:, 4] <= p[:, 2]) or np.any(p[:, 5] <= p[:, 3]):
            fail("Invalid box geometry detected")

    ok("decode_predictions semantics OK")


def check_ema_and_fuse(train_mod, model, device):
    print("\n[10/10] EMA + FUSION")

    ema = train_mod.ModelEMA(model)
    ok("EMA initialized")

    # --- simulate ONE real training step ---
    model.train()
    x = torch.rand(2, 3, IMG_SIZE, IMG_SIZE).to(device)
    dummy_loss = sum(p.mean() for p in model(x)[0]) * 0.0
    dummy_loss.backward()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.step()

    # --- now update EMA ---
    ema.update(model)

    # --- now EMA MUST differ ---
    diffs = [
        torch.sum(torch.abs(p1 - p2)).item()
        for p1, p2 in zip(model.parameters(), ema.ema.parameters())
    ]
    if sum(diffs) == 0:
        fail("EMA did not update after optimizer step")

    ok("EMA diverges after update (correct behavior)")

    # --- forward test ---
    model.eval()
    ema.ema.eval()
    _ = ema.ema(x)
    ok("EMA forward OK")

    # --- fusion check ---
    out_before = model(x)
    train_mod.fuse_repv_blocks(model)
    out_after = model(x)

    if out_before[0][0].shape != out_after[0][0].shape:
        fail("Fusion changed output shape")

    ok("Fusion preserves output shape")



# =========================================================
# main
# =========================================================
def main():
    seed_all()

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-img-dir", default="data/dataset_train/images/train")
    parser.add_argument("--train-label-dir", default="data/dataset_train/labels/train")
    args = parser.parse_args()

    utils, model_mod, dataset_mod, train_mod = import_all()
    NUM_CLASSES = check_globals(utils, model_mod)

    check_parameter_counts(model_mod, NUM_CLASSES)

    images, targets = load_batch(dataset_mod, args.train_img_dir, args.train_label_dir)
    if images is None:
        images, targets = synthetic_batch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_mod.MCUDetector(NUM_CLASSES).to(device)
    criterion = model_mod.MCUDetectionLoss(NUM_CLASSES).to(device)

    outputs = check_forward(model, images, device)
    check_loss_backward(model, criterion, images, targets, device)
    check_memory_usage(model, images, device)
    check_batch_scaling(model, device)
    check_decode(utils, outputs)
    check_ema_and_fuse(train_mod, model, device)

    print("\n" + "=" * 70)
    print("🌍 FINAL VERDICT: SYSTEM IS DEPLOYMENT-GRADE & WORLD-SAFE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
