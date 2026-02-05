# ================================================================
# FINAL MODEL DEBUG LOCK — MCU DETECTOR (P3 + P4)
# ================================================================

import os, sys, shutil, traceback
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import NUM_CLASSES
sys.path.insert(0, os.path.dirname(__file__))
from model import MCUDetector, MCUDetectionLoss

# ---------------- CONFIG ----------------
IMG_DIR = Path("data/dataset_train/images/train")
LBL_DIR = Path("data/dataset_train/labels/train")
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_FAIL_DIR = Path("debug_failures")

CHECK_GRADIENTS      = True
CHECK_FEATURE_ENERGY = True   # WARN only (never assert)
CHECK_OBJ_STATS      = True
CHECK_BBOX_SANITY    = True
# ---------------------------------------


# ================= UTILS =================
def reset_dir(p):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def read_image(p):
    img = cv2.imread(str(p))
    if img is None:
        raise RuntimeError("cv2.imread failed")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) \
        .permute(2,0,1).float().div(255).unsqueeze(0).to(DEVICE)
    return t, img


def load_labels(p):
    if not p.exists():
        return torch.zeros((0,5))
    rows = []
    for l in p.read_text().splitlines():
        if l.strip():
            rows.append(list(map(float, l.split()[:5])))
    t = torch.tensor(rows) if rows else torch.zeros((0,5))
    if t.numel():
        assert torch.all((t[:,1:] >= 0) & (t[:,1:] <= 1)), "Label not normalized"
        assert torch.all(t[:,3:] > 0), "Zero/neg bbox size"
    return t


def feature_ok(x, name):
    mean = x.abs().mean().item()
    std  = x.std().item()
    if std < 1e-5 and mean < 1e-5:
        print(f"[WARN] {name} low-energy (init / BN expected)")
        return False
    return True


# ================= CORE DEBUG =================
def debug_single(model, loss_fn, img_t, labels):
    with torch.enable_grad():
        # ---- forward ----
        p3, p4 = model.backbone(img_t)
        f3, f4 = model.fpn(p3, p4)
        (p3_cls, p3_reg), (p4_cls, p4_reg) = model.head(f3, f4)

        # ---- SHAPE LOCK (HARD FAIL) ----
        assert f3.shape[-2:] == (IMG_SIZE // 4, IMG_SIZE // 4), "Bad P3 spatial"
        assert f4.shape[-2:] == (IMG_SIZE // 8, IMG_SIZE // 8), "Bad P4 spatial"
        assert f3.shape[-1] == 2 * f4.shape[-1], "P3/P4 ratio broken"

        # ---- FEATURE ENERGY (WARN ONLY) ----
        if CHECK_FEATURE_ENERGY:
            feature_ok(f3, "P3")
            feature_ok(f4, "P4")

        # ---- OBJECTNESS LOGIT SANITY ----
        if CHECK_OBJ_STATS:
            o3 = p3_cls[:,0].mean().item()
            o4 = p4_cls[:,0].mean().item()
            if abs(o3) > 5 or abs(o4) > 5:
                print(f"[WARN] objectness bias P3={o3:.2f}, P4={o4:.2f}")

        # ---- LOSS (HARD FAIL ON NaN/Inf) ----
        t3 = [labels.to(DEVICE)]
        t4 = [labels.to(DEVICE)]
        loss = loss_fn((p3_cls, p3_reg), (p4_cls, p4_reg), t3, t4)

        assert torch.isfinite(loss["total"]), "Loss is NaN/Inf"

        # ---- BACKWARD / GRADIENT CHECK ----
        if CHECK_GRADIENTS:
            model.zero_grad(set_to_none=True)
            loss["total"].backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"Bad grad in {n}"

        # ---- BBOX PARAM SANITY ----
        if CHECK_BBOX_SANITY and labels.numel():
            gx = int(labels[0,1] * f3.shape[-1])
            gy = int(labels[0,2] * f3.shape[-2])

            dx = torch.sigmoid(p3_reg[0,0,gy,gx])
            dy = torch.sigmoid(p3_reg[0,1,gy,gx])
            dw = torch.exp(p3_reg[0,2,gy,gx].clamp(-4,4))
            dh = torch.exp(p3_reg[0,3,gy,gx].clamp(-4,4))

            assert dw > 0 and dh > 0, "Invalid bbox size"

        return True


# ================= MAIN =================
def main():
    reset_dir(SAVE_FAIL_DIR)

    imgs = sorted([p for p in IMG_DIR.iterdir()
                   if p.suffix.lower() in (".jpg",".png",".jpeg")])
    assert imgs, "No images found"

    model = MCUDetector(NUM_CLASSES).to(DEVICE).eval()
    loss_fn = MCUDetectionLoss(NUM_CLASSES)

    fails = 0

    for img_p in tqdm(imgs, desc="LOCK CHECK"):
        lbl_p = LBL_DIR / (img_p.stem + ".txt")
        try:
            img_t, img_np = read_image(img_p)
            labels = load_labels(lbl_p)
            debug_single(model, loss_fn, img_t, labels)

        except Exception as e:
            fails += 1
            err = traceback.format_exc()
            cv2.imwrite(str(SAVE_FAIL_DIR / f"{img_p.stem}.jpg"), img_np)
            (SAVE_FAIL_DIR / f"{img_p.stem}.txt").write_text(err)
            print("[FAIL]", img_p.name, e)

    print("\n=========== SUMMARY ===========")
    print("Images checked :", len(imgs))
    print("Failures       :", fails)

    if fails == 0:
        print("✅ MODEL FULLY LOCKED — SAFE TO TRAIN")
    else:
        print("❌ DEBUG REQUIRED — CHECK debug_failures/")


if __name__ == "__main__":
    main()
