import os
import csv
from collections import defaultdict

LABEL_DIR = r"D:\microcontroller-ocr-datasheet\microcontroller-ocr-datasheet\data\dataset_test\labels\train"
OUTPUT_CSV = r"D:\microcontroller-ocr-datasheet\microcontroller-ocr-datasheet\Helper Python Files\yolo_classes_summary_test.csv"

class_box_count = defaultdict(int)
class_image_set = defaultdict(set)

for file in os.listdir(LABEL_DIR):
    if not file.endswith(".txt"):
        continue

    img_name = file.replace(".txt", "")
    file_path = os.path.join(LABEL_DIR, file)

    with open(file_path, "r") as f:   # ✅ FIX IS HERE
        for line in f:
            if not line.strip():
                continue

            class_id = int(line.split()[0])
            class_box_count[class_id] += 1
            class_image_set[class_id].add(img_name)

with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["class_id", "num_boxes", "num_images"])

    for cid in sorted(class_box_count):
        writer.writerow([cid, class_box_count[cid], len(class_image_set[cid])])

print("✅ CSV saved at:", OUTPUT_CSV)
print("📌 Classes found:", sorted(class_box_count.keys()))
