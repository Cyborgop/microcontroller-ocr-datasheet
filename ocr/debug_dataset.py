import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from typing import Optional, Sequence


# IMPORTANT: adjust import if filename differs
from dataset import MCUDetectionDataset   # dataset.py
from utils import NUM_CLASSES, CLASSES


# ================= CONFIG =================
IMG_DIR = "data/dataset_test/images/train"
LABEL_DIR = "data/dataset_test/labels/train"
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

    img_names = {p.stem for p in img_dir.glob("*.jpg")}
    lbl_names = {p.stem for p in label_dir.glob("*.txt")}

    missing_labels = img_names - lbl_names
    orphan_labels = lbl_names - img_names

    assert not missing_labels, (
        f"❌ Images without labels found: {list(missing_labels)[:5]}"
    )
    assert not orphan_labels, (
        f"❌ Labels without images found: {list(orphan_labels)[:5]}"
    )

    print("✅ Image–label pairing is consistent")


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
        print("\nImage-level class counts:", imglvl)
        print("\nInstance counts per class:", inst)
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

    print("Image-level class counts:", image_level_counts(LABEL_DIR))
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
        print(f"  Class {cls_id}: {class_instance_counts.get(cls_id, 0)} instances")
    print("====================================\n")

    dataset = MCUDetectionDataset(
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        img_size=IMG_SIZE,
        transform=None
    )

    print("Dataset size:", len(dataset))

    # ---------- RUN COLLATE DEBUG TEST ----------
    # This will print per-sample shapes and classes for a few batches
    test_collate(dataset, batch_size=4, num_batches=4)


    # ---------- GLOBAL STATS ----------
    empty_images = 0
    cls_counter = Counter()

    for i in range(len(dataset)):
        _, targets = dataset[i]
        if targets.numel() == 0:
            empty_images += 1
        else:
            for c in targets[:, 0].tolist():
                cls_counter[int(c)] += 1

    empty_ratio = empty_images / len(dataset)

    print("\n===== DATASET STATISTICS =====")
    print(f"Empty images: {empty_images}/{len(dataset)} ({empty_ratio:.2%})")
    assert empty_ratio < 0.6, "❌ Too many empty images (>60%)"

    print("\nClass distribution:")
    for c in range(EXPECTED_NUM_CLASSES):
        print(f"  Class {c}: {cls_counter.get(c, 0)}")

    if cls_counter:
        imbalance = max(cls_counter.values()) / max(1, min(cls_counter.values()))
        print("Imbalance ratio:", imbalance)

    print("================================\n")

    # ---------- PER-SAMPLE CHECKS ----------
    for idx in range(NUM_SAMPLES_TO_CHECK):
        image, targets = dataset[idx]

        print(f"\n--- Sample {idx} ---")

        # ----- Image checks -----
        assert image.shape == (3, IMG_SIZE, IMG_SIZE), "❌ Image shape mismatch"
        assert image.dtype == torch.float32, "❌ Image dtype not float32"
        assert 0.0 <= image.min() and image.max() <= 1.0, "❌ Image not normalized"

        print("Image OK:", image.shape, image.min().item(), image.max().item())

        # ----- Target checks -----
        assert targets.ndim == 2 and targets.shape[1] == 5, "❌ Invalid target shape"

        if targets.numel() > 0:
            cls = targets[:, 0]
            boxes = targets[:, 1:5]

            # class validity
            assert torch.all(cls >= 0), "❌ Negative class id"
            assert torch.all(cls < EXPECTED_NUM_CLASSES), "❌ Class id overflow"

            # normalization
            assert torch.all(boxes >= 0) and torch.all(boxes <= 1), "❌ Box not normalized"

            # geometry
            w = boxes[:, 2]
            h = boxes[:, 3]

            assert torch.all(w > 0), "❌ Zero / negative box width"
            assert torch.all(h > 0), "❌ Zero / negative box height"

            min_box = MIN_BOX_PIXELS / IMG_SIZE
            assert torch.all(w >= min_box), "❌ Box width too small"
            assert torch.all(h >= min_box), "❌ Box height too small"

            # corners inside image
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2

            assert torch.all(x1 >= 0) and torch.all(y1 >= 0), "❌ Box corner < 0"
            assert torch.all(x2 <= 1) and torch.all(y2 <= 1), "❌ Box corner > 1"

            # stride compatibility (P3=4, P4=8)
            gx3 = (boxes[:, 0] * IMG_SIZE / 4).long()
            gy3 = (boxes[:, 1] * IMG_SIZE / 4).long()
            gx4 = (boxes[:, 0] * IMG_SIZE / 8).long()
            gy4 = (boxes[:, 1] * IMG_SIZE / 8).long()

            assert torch.all(gx3 >= 0) and torch.all(gy3 >= 0), "❌ P3 grid error"
            assert torch.all(gx4 >= 0) and torch.all(gy4 >= 0), "❌ P4 grid error"

            print("Targets OK:", targets.shape)

        # ----- Visualization -----
        img_vis = image.permute(1, 2, 0).numpy()

        plt.figure(figsize=(4, 4))
        plt.imshow(img_vis)
        plt.axis("off")

        for t in targets:
            _, x, y, w, h = t.tolist()
            H, W, _ = img_vis.shape

            x1 = (x - w / 2) * W
            y1 = (y - h / 2) * H
            x2 = (x + w / 2) * W
            y2 = (y + h / 2) * H

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

        plt.title(f"Sample {idx}")
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
        except StopIteration:
            break
    print("--- collate test done ---\n")


if __name__ == "__main__":
    debug_dataset()