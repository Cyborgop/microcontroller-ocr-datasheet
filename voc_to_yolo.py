#!/usr/bin/env python3
"""Convert Pascal VOC2007+VOC2012 to YOLO format for MCUDetector training.

Layout (matches MCUDetectionDataset expectations):
    data/voc_train/images/train/<year>_<id>.jpg
    data/voc_train/labels/train/<year>_<id>.txt
    data/voc_test/images/train/<year>_<id>.jpg
    data/voc_test/labels/train/<year>_<id>.txt

Splits:
    TRAIN = VOC2007 trainval + VOC2012 trainval  (skip difficult=1)
    VAL   = VOC2007 test                          (KEEP difficult; VOC07 protocol)

Filenames are prefixed with year to avoid VOC07/VOC12 ID collisions.
"""
from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):  # minimal fallback
        return it

CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

PROJ = Path(__file__).resolve().parent
VOC_ROOT = PROJ / "pascal-voc" / "VOCdevkit"

OUT_TRAIN_IMG = PROJ / "data" / "voc_train" / "images" / "train"
OUT_TRAIN_LBL = PROJ / "data" / "voc_train" / "labels" / "train"
OUT_VAL_IMG   = PROJ / "data" / "voc_test"  / "images" / "train"
OUT_VAL_LBL   = PROJ / "data" / "voc_test"  / "labels" / "train"


def parse_voc_xml(xml_path: Path, skip_difficult: bool):
    """Return (W, H, [(cls_idx, cx, cy, w, h)]) with normalized coords, or None if unreadable."""
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, OSError):
        return None
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return None
    W = float(size.findtext("width") or 0)
    H = float(size.findtext("height") or 0)
    if W <= 0 or H <= 0:
        return None

    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLS2IDX:
            continue
        difficult = int(obj.findtext("difficult") or 0)
        if skip_difficult and difficult == 1:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.findtext("xmin") or 0)
        ymin = float(bb.findtext("ymin") or 0)
        xmax = float(bb.findtext("xmax") or 0)
        ymax = float(bb.findtext("ymax") or 0)
        # VOC is 1-indexed; clamp into [0, W/H]
        xmin = max(0.0, min(W, xmin - 1.0))
        ymin = max(0.0, min(H, ymin - 1.0))
        xmax = max(0.0, min(W, xmax - 1.0))
        ymax = max(0.0, min(H, ymax - 1.0))
        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 0 or bh <= 0:
            continue
        cx = (xmin + xmax) / 2.0 / W
        cy = (ymin + ymax) / 2.0 / H
        nw = bw / W
        nh = bh / H
        # Final normalized clamp
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        boxes.append((CLS2IDX[name], cx, cy, nw, nh))
    return W, H, boxes


def convert_split(year: str, image_set: str, out_img: Path, out_lbl: Path,
                  skip_difficult: bool) -> tuple[int, int, int]:
    """Convert one VOC split. Returns (images_written, labels_written, objects_written)."""
    voc = VOC_ROOT / f"VOC{year}"
    split_file = voc / "ImageSets" / "Main" / f"{image_set}.txt"
    if not split_file.exists():
        print(f"  [skip] {split_file} not found", file=sys.stderr)
        return 0, 0, 0
    ids = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]
    n_img = n_lbl = n_obj = 0
    for img_id in tqdm(ids, desc=f"VOC{year}/{image_set}", unit="img"):
        xml_p = voc / "Annotations" / f"{img_id}.xml"
        jpg_p = voc / "JPEGImages" / f"{img_id}.jpg"
        if not xml_p.exists() or not jpg_p.exists():
            continue
        parsed = parse_voc_xml(xml_p, skip_difficult=skip_difficult)
        if parsed is None:
            continue
        _, _, boxes = parsed

        out_name = f"{year}_{img_id}"
        # Always copy image (a "background" image with zero boxes is still useful for negatives)
        shutil.copy2(jpg_p, out_img / f"{out_name}.jpg")
        n_img += 1

        with open(out_lbl / f"{out_name}.txt", "w") as f:
            for cls_idx, cx, cy, w, h in boxes:
                f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                n_obj += 1
        if boxes:
            n_lbl += 1
    return n_img, n_lbl, n_obj


def main():
    if not VOC_ROOT.exists():
        print(f"ERROR: VOC root not found at {VOC_ROOT}", file=sys.stderr)
        sys.exit(1)

    for d in (OUT_TRAIN_IMG, OUT_TRAIN_LBL, OUT_VAL_IMG, OUT_VAL_LBL):
        d.mkdir(parents=True, exist_ok=True)

    print(f"VOC root: {VOC_ROOT}")
    print(f"Train out: {OUT_TRAIN_IMG.parent.parent}")
    print(f"Val out:   {OUT_VAL_IMG.parent.parent}")
    print(f"Classes ({len(CLASSES)}): {CLASSES}\n")

    totals = {"train_img": 0, "train_lbl": 0, "train_obj": 0,
              "val_img": 0,   "val_lbl": 0,   "val_obj": 0}

    # ---------- TRAIN: VOC2007 trainval + VOC2012 trainval (skip difficult) ----------
    print("== TRAIN ==")
    for year in ("2007", "2012"):
        if not (VOC_ROOT / f"VOC{year}").exists():
            print(f"  [warn] VOC{year} directory missing — skipping its trainval contribution")
            continue
        ni, nl, no = convert_split(year, "trainval", OUT_TRAIN_IMG, OUT_TRAIN_LBL, skip_difficult=True)
        totals["train_img"] += ni
        totals["train_lbl"] += nl
        totals["train_obj"] += no
        print(f"  VOC{year}/trainval: {ni} images, {nl} non-empty labels, {no} objects")

    # ---------- VAL: VOC2007 test (KEEP difficult per VOC07 protocol) ----------
    print("\n== VAL ==")
    if (VOC_ROOT / "VOC2007").exists():
        ni, nl, no = convert_split("2007", "test", OUT_VAL_IMG, OUT_VAL_LBL, skip_difficult=False)
        totals["val_img"] += ni
        totals["val_lbl"] += nl
        totals["val_obj"] += no
        print(f"  VOC2007/test: {ni} images, {nl} non-empty labels, {no} objects")
    else:
        print("  [warn] VOC2007 directory missing — no val data written")

    print("\n== SUMMARY ==")
    print(f"Train: {totals['train_img']} images, {totals['train_obj']} objects")
    print(f"Val:   {totals['val_img']} images, {totals['val_obj']} objects")
    print(f"Expected ~16551 train, ~4952 val (full VOC07+12)")


if __name__ == "__main__":
    main()
