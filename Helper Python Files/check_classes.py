import glob
import os

def check_classes(label_dir, name):
    classes = set()
    for f in glob.glob(os.path.join(label_dir, "*.txt")):
        with open(f) as file:
            for line in file:
                parts = line.strip().split()
                if parts:
                    classes.add(int(float(parts[0])))
    print(f"{name}: classes {sorted(classes)}")
    print(f"{name}: {len(classes)} unique classes")
    return classes

train_cls = check_classes("data/dataset_train/labels/train", "TRAIN")
val_cls = check_classes("data/dataset_test/labels/train", "VAL")

if train_cls != val_cls:
    print(f"\n❌ MISMATCH! Missing in val: {train_cls - val_cls}")
    print(f"❌ MISMATCH! Extra in val: {val_cls - train_cls}")