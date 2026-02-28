import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from typing import Optional, Sequence


# IMPORTANT: adjust import if filename differs
from dataset import MCUDetectionDataset   # dataset.py
from utils import NUM_CLASSES, CLASSES


# ================= CONFIG =================
IMG_DIR = "data/dataset_train/images/train"
LABEL_DIR = "data/dataset_train/labels/train"
IMG_SIZE = 512
NUM_SAMPLES_TO_CHECK = 3

EXPECTED_NUM_CLASSES = NUM_CLASSES   # 🔴 MUST MATCH MODEL
MIN_BOX_PIXELS = 2         # minimum box size in pixels
# =========================================


# -----------------------------------------------------------------------------
# GLOBAL CONSISTENCY CHECKS
# -----------------------------------------------------------------------------
# TEMP collate for debugging only
def detection_collate_fn_debug(batch):
    images = []
    targets = []

    for idx, sample in enumerate(batch):
        if sample is None:
            print(f"[collate] sample {idx} is None -> skip")
            continue
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            print(f"[collate] sample {idx} wrong format -> skip: {type(sample)}")
            continue
        img, tgt = sample
        if not isinstance(img, torch.Tensor):
            print(f"[collate] sample {idx} img not tensor -> skip")
            continue
        if tgt is None or not isinstance(tgt, torch.Tensor):
            tgt = torch.zeros((0, 5), dtype=torch.float32)
        images.append(img)
        targets.append(tgt)

    if len(images) == 0:
        raise RuntimeError("Empty batch encountered in detection_collate_fn_debug. This indicates a dataset/sampler bug.")

    # Debug prints: per-sample shapes and classes
    print(f"[collate] batch_size={len(images)}")
    for j, t in enumerate(targets):
        if t.numel() == 0:
            print(f"  sample {j}: target_shape=(0,5), num_boxes=0, classes=[]")
        else:
            try:
                classes = sorted(set(int(x) for x in t[:, 0].tolist()))
            except Exception:
                classes = "ERR"
            print(f"  sample {j}: target_shape={tuple(t.shape)}, num_boxes={t.shape[0]}, classes={classes}")

    return torch.stack(images, 0), targets


def verify_image_label_pairs(img_dir, label_dir):
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)

    img_names = {p.stem for p in img_dir.glob("*.jpg")} | {p.stem for p in img_dir.glob("*.png")} | {p.stem for p in img_dir.glob("*.jpeg")} | {p.stem for p in img_dir.glob("*.bmp")}
    lbl_names = {p.stem for p in label_dir.glob("*.txt")}

    missing_labels = img_names - lbl_names
    orphan_labels = lbl_names - img_names

    if missing_labels:
        print(f"⚠️ Images without labels found: {len(missing_labels)}")
        if len(missing_labels) <= 10:
            print(f"   Missing labels for: {list(missing_labels)}")
    else:
        print("✅ All images have corresponding labels")

    if orphan_labels:
        print(f"⚠️ Labels without images found: {len(orphan_labels)}")
        if len(orphan_labels) <= 10:
            print(f"   Orphan labels for: {list(orphan_labels)}")

    print("✅ Image–label pairing check complete")


def verify_num_classes(label_dir):
    label_dir = Path(label_dir)
    classes_found = set()

    for lf in label_dir.glob("*.txt"):
        with open(lf, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                cls = int(float(line.split()[0]))
                classes_found.add(cls)

    assert classes_found, "❌ No class labels found in dataset"

    max_cls = max(classes_found)

    print("\n===== CLASS CONSISTENCY CHECK =====")
    print("Classes found:", sorted(classes_found))
    print("Max class id:", max_cls)
    print("EXPECTED_NUM_CLASSES:", EXPECTED_NUM_CLASSES)

    assert max_cls < EXPECTED_NUM_CLASSES, (
        f"❌ Dataset contains class id {max_cls}, "
        f"but EXPECTED_NUM_CLASSES={EXPECTED_NUM_CLASSES}"
    )
    assert set(range(len(CLASSES))) == set(range(NUM_CLASSES)), "Mismatch between CLASSES and expected label indices"

    print("✅ Class count is consistent")
    print("==================================\n")
    


# -----------------------------------------------------------------------------
# MAIN DATASET DEBUG
# -----------------------------------------------------------------------------


def debug_dataset():
    print("\n🔍 DATASET DEBUG STARTED\n")

    verify_image_label_pairs(IMG_DIR, LABEL_DIR)
    verify_num_classes(LABEL_DIR)
    
    def check_label_index_mapping(labels_dir: str, expected_num_classes: int, class_names: Optional[Sequence[str]] = None):
        # quick sanity prints
        inst = dataset_label_stats(labels_dir)
        imglvl = image_level_counts(labels_dir)
        print("\n📊 Image-level class counts:", dict(sorted(imglvl.items())))
        print("\n📊 Instance counts per class:", dict(sorted(inst.items())))
        # ensure class ids are contiguous 0..N-1
        ids = sorted(set(list(inst.keys()) + list(imglvl.keys())))
        if ids != list(range(min(ids), max(ids)+1)):
            print("⚠️ label ids not contiguous, ids found:", ids)
        if max(ids) >= expected_num_classes:
            raise AssertionError(f"label id {max(ids)} >= EXPECTED_NUM_CLASSES ({expected_num_classes})")
        if class_names:
            # optional name-check: ensure mapping length matches
            assert len(class_names) == expected_num_classes, "class_names length mismatch"
    
    def image_level_counts(labels_dir):
        img_counts = Counter()
        for f in Path(labels_dir).glob("**/*.txt"):
            classes = set()
            for L in f.read_text().splitlines():
                if L.strip():
                    classes.add(int(L.split()[0]))
            for c in classes:
                img_counts[c] += 1
        return img_counts

    def dataset_label_stats(labels_dir):
        from collections import Counter
        counts = Counter()
        files = list(Path(labels_dir).glob("**/*.txt"))
        for f in files:
            for line in f.read_text().strip().splitlines():
                if not line: continue
                cls = int(line.split()[0])
                counts[cls] += 1
        return counts
    
    check_label_index_mapping(LABEL_DIR, EXPECTED_NUM_CLASSES, class_names=None)
    print("✅ label-index mapping sanity-checked")

    print("\n===== INSTANCE COUNT PER CLASS =====")
    class_instance_counts = dataset_label_stats(LABEL_DIR)
    for cls_id in range(EXPECTED_NUM_CLASSES):
        count = class_instance_counts.get(cls_id, 0)
        class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Class_{cls_id}"
        print(f"  Class {cls_id:2d} ({class_name:30s}): {count:4d} instances")
    print("====================================\n")

    # Test without augmentation first
    print("\n🔍 Testing WITHOUT augmentation:")
    dataset_no_aug = MCUDetectionDataset(
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        img_size=IMG_SIZE,
        transform=None,
        augment=False
    )
    
    print(f"Dataset size (no aug): {len(dataset_no_aug)}")
    test_collate(dataset_no_aug, batch_size=4, num_batches=2)

    # Test WITH augmentation
    print("\n🔍 Testing WITH augmentation:")
    dataset_aug = MCUDetectionDataset(
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        img_size=IMG_SIZE,
        transform=None,
        augment=True
    )
    
    print(f"Dataset size (with aug): {len(dataset_aug)}")
    test_collate(dataset_aug, batch_size=4, num_batches=2)

    # Use non-augmented version for detailed stats (augmented would show different coords each time)
    dataset = dataset_no_aug

    # ---------- GLOBAL STATS ----------
    empty_images = 0
    cls_counter = Counter()
    multi_object_images = 0
    max_objects_per_image = 0

    for i in range(len(dataset)):  # Limit to 1000 for speed
        _, targets = dataset[i]
        if targets.numel() == 0:
            empty_images += 1
        else:
            num_boxes = targets.shape[0]
            if num_boxes > 1:
                multi_object_images += 1
            max_objects_per_image = max(max_objects_per_image, num_boxes)
            for c in targets[:, 0].tolist():
                cls_counter[int(c)] += 1

    empty_ratio = empty_images / min(len(dataset), 1000)

    print("\n===== DATASET STATISTICS =====")
    print(f"Total images checked: {min(len(dataset), 1000)}")
    print(f"Empty images: {empty_images} ({empty_ratio:.2%})")
    print(f"Images with multiple objects: {multi_object_images}")
    print(f"Max objects in one image: {max_objects_per_image}")
    
    if empty_ratio < 0.6:
        print(f"✅ Empty images ratio OK ({empty_ratio:.2%} < 60%)")
    else:
        print(f"⚠️ High empty images ratio ({empty_ratio:.2%})")

    print("\n📊 Class distribution:")
    for c in range(EXPECTED_NUM_CLASSES):
        count = cls_counter.get(c, 0)
        class_name = CLASSES[c] if c < len(CLASSES) else f"Class_{c}"
        print(f"  Class {c:2d} ({class_name:30s}): {count:4d} instances")

    if cls_counter:
        imbalance = max(cls_counter.values()) / max(1, min(cls_counter.values()))
        print(f"\n📊 Imbalance ratio (max/min): {imbalance:.2f}")
        if imbalance < 10:
            print("✅ Dataset is well-balanced")
        else:
            print("⚠️ Dataset has significant imbalance")

    print("================================\n")

    # ---------- PER-SAMPLE CHECKS ----------
    for idx in range(min(NUM_SAMPLES_TO_CHECK, len(dataset))):
        image, targets = dataset[idx]

        print(f"\n--- Sample {idx} ---")
        print(f"Image path: {dataset.image_files[idx]}")

        # ----- Image checks -----
        assert image.shape == (3, IMG_SIZE, IMG_SIZE), f"❌ Image shape mismatch: {image.shape}"
        assert image.dtype == torch.float32, f"❌ Image dtype not float32: {image.dtype}"
        assert 0.0 <= image.min() and image.max() <= 1.0, f"❌ Image not normalized: [{image.min():.3f}, {image.max():.3f}]"

        print(f"Image OK: {image.shape}, range=[{image.min():.3f}, {image.max():.3f}]")

        # ----- Target checks -----
        assert targets.ndim == 2 and targets.shape[1] == 5, f"❌ Invalid target shape: {targets.shape}"

        if targets.numel() > 0:
            cls = targets[:, 0]
            boxes = targets[:, 1:5]

            # class validity
            assert torch.all(cls >= 0), "❌ Negative class id"
            assert torch.all(cls < EXPECTED_NUM_CLASSES), f"❌ Class id overflow: {cls}"

            # normalization
            assert torch.all(boxes >= 0) and torch.all(boxes <= 1), f"❌ Box not normalized: {boxes}"

            # geometry
            w = boxes[:, 2]
            h = boxes[:, 3]

            assert torch.all(w > 0), "❌ Zero / negative box width"
            assert torch.all(h > 0), "❌ Zero / negative box height"

            min_box = MIN_BOX_PIXELS / IMG_SIZE
            assert torch.all(w >= min_box), f"❌ Box width too small: {w}"
            assert torch.all(h >= min_box), f"❌ Box height too small: {h}"

            # corners inside image
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2

            assert torch.all(x1 >= -0.01) and torch.all(y1 >= -0.01), f"❌ Box corner < 0: {x1}, {y1}"
            assert torch.all(x2 <= 1.01) and torch.all(y2 <= 1.01), f"❌ Box corner > 1: {x2}, {y2}"

            # stride compatibility (P3=4, P4=8)
            gx3 = (boxes[:, 0] * IMG_SIZE / 4).long()
            gy3 = (boxes[:, 1] * IMG_SIZE / 4).long()
            gx4 = (boxes[:, 0] * IMG_SIZE / 8).long()
            gy4 = (boxes[:, 1] * IMG_SIZE / 8).long()

            assert torch.all(gx3 >= 0) and torch.all(gy3 >= 0), "❌ P3 grid error"
            assert torch.all(gx4 >= 0) and torch.all(gy4 >= 0), "❌ P4 grid error"

            print(f"Targets OK: {targets.shape}, classes={torch.unique(cls).tolist()}")

            # Show class names
            class_names_found = []
            for c in torch.unique(cls).tolist():
                if int(c) < len(CLASSES):
                    class_names_found.append(f"{int(c)}:{CLASSES[int(c)]}")
                else:
                    class_names_found.append(str(int(c)))
            print(f"Classes found: {', '.join(class_names_found)}")

        else:
            print("No targets (empty image)")

        # ----- Visualization -----
        img_vis = image.permute(1, 2, 0).numpy()

        plt.figure(figsize=(8, 8))
        plt.imshow(img_vis)
        plt.axis("off")

        if targets.numel() > 0:
            for t in targets:
                cls_id, x, y, w, h = t.tolist()
                H, W, _ = img_vis.shape

                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H

                # Get class name
                class_name = CLASSES[int(cls_id)] if int(cls_id) < len(CLASSES) else f"Class_{int(cls_id)}"
                
                plt.gca().add_patch(
                    plt.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
                )
                plt.text(x1, y1-5, class_name, color='red', fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        plt.title(f"Sample {idx}")
        plt.tight_layout()
        plt.show()

    print("\n✅ DATASET DEBUG COMPLETED — DATASET IS LOCK-READY 🔒\n")


def test_collate(dataset, batch_size=4, num_batches=3):
    from torch.utils.data import DataLoader
    print("\n--- Running collate debug test ---")
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    collate_fn=detection_collate_fn_debug, num_workers=0)  # use num_workers=0 for clean prints
    it = iter(dl)
    for b in range(num_batches):
        try:
            images, targets = next(it)
            # Summary for training-style expectation:
            classes = []
            per_sample_counts = []
            for t in targets:
                per_sample_counts.append(int(t.shape[0]))
                if t.numel() > 0:
                    classes += [int(x) for x in t[:, 0].tolist()]
            print(f"[test] Batch {b}: unique_classes_in_batch={sorted(set(classes))}, per_sample_counts={per_sample_counts}")
            
            # Quick augmentation check - compare first image with and without aug
            if b == 0:
                print(f"   Image stats: mean={images[0].mean():.3f}, std={images[0].std():.3f}")
        except StopIteration:
            break
    print("--- collate test done ---\n")


if __name__ == "__main__":
    debug_dataset()