#!/usr/bin/env python3
"""Convert VOC2007+2012 to YOLO format matching MCUDetector's dataset layout."""
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from tqdm import tqdm

CLASSES = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
           "chair","cow","diningtable","dog","horse","motorbike","person",
           "pottedplant","sheep","sofa","train","tvmonitor"]
cls2idx = {c: i for i, c in enumerate(CLASSES)}

PROJ = Path(__file__).resolve().parent
VOC_RAW = PROJ / "data" / "voc_raw" / "VOCdevkit"

# Match MCUDetector's expected layout: data/<split>/images/train, data/<split>/labels/train
OUT_TRAIN_IMG = PROJ / "data" / "voc_train" / "images" / "train"
OUT_TRAIN_LBL = PROJ / "data" / "voc_train" / "labels" / "train"
OUT_VAL_IMG   = PROJ / "data" / "voc_test"  / "images" / "train"
OUT_VAL_LBL   = PROJ / "data" / "voc_test"  / "labels" / "train"

for d in (OUT_TRAIN_IMG, OUT_TRAIN_LBL, OUT_VAL_IMG, OUT_VAL_LBL):
    d.mkdir(parents=True, exist_ok=True)

def convert_split(voc_year, image_set, out_img, out_lbl, skip_difficult=True):
    voc = VOC_RAW / f"VOC{voc_year}"
    ids = (voc / "ImageSets" / "Main" / f"{image_set}.txt").read_text().split()
    n_ok, n_skip = 0, 0
    for img_id in tqdm(ids, desc=f"VOC{voc_year}/{image_set}"):
        xml_p = voc / "Annotations" / f"{img_id}.xml"
        img_p = voc / "JPEGImages" / f"{img_id}.jpg"
        if not xml_p.exists() or not img_p.exists():
            n_skip += 1; continue

        root = ET.parse(xml_p).getroot()
        W = float(root.find("size/width").text)
        H = float(root.find("size/height").text)
        lines = []
        for obj in root.findall("object"):
            diff = obj.find("difficult")
            if skip_difficult and diff is not None and int(diff.text) == 1:
                continue
            name = obj.find("name").text.strip()
            if name not in cls2idx:
                continue
            b = obj.find("bndbox")
            x1 = float(b.find("xmin").text); y1 = float(b.find("ymin").text)
            x2 = float(b.find("xmax").text); y2 = float(b.find("ymax").text)
            cx = ((x1 + x2) / 2.0) / W
            cy = ((y1 + y2) / 2.0) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            if bw <= 0 or bh <= 0: continue
            cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
            bw = min(bw, 1.0); bh = min(bh, 1.0)
            lines.append(f"{cls2idx[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Use unique names: VOC07 and VOC12 have ID collisions
        out_name = f"{voc_year}_{img_id}"
        shutil.copy(img_p, out_img / f"{out_name}.jpg")
        (out_lbl / f"{out_name}.txt").write_text("\n".join(lines))
        n_ok += 1
    print(f"  {voc_year}/{image_set}: wrote {n_ok}, skipped {n_skip}")

# TRAIN = VOC07 trainval + VOC12 trainval  (drop difficult)
convert_split("2007", "trainval", OUT_TRAIN_IMG, OUT_TRAIN_LBL, skip_difficult=True)
convert_split("2012", "trainval", OUT_TRAIN_IMG, OUT_TRAIN_LBL, skip_difficult=True)

# VAL = VOC07 test  (KEEP difficult — needed for VOC07 protocol comparability)
convert_split("2007", "test",     OUT_VAL_IMG,   OUT_VAL_LBL,   skip_difficult=False)

print("\nDONE")
print(f"  Train: {len(list(OUT_TRAIN_IMG.glob('*.jpg')))} images, {len(list(OUT_TRAIN_LBL.glob('*.txt')))} labels")
print(f"  Val:   {len(list(OUT_VAL_IMG.glob('*.jpg')))} images, {len(list(OUT_VAL_LBL.glob('*.txt')))} labels")