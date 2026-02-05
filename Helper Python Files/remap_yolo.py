from pathlib import Path

# ================= CONFIG =================
LABEL_DIR = Path(
    r"D:\microcontroller-ocr-datasheet\microcontroller-ocr-datasheet\data"
    r"\dataset_test\labels\train"
)

CLASS_ID_START = 7   # from your YAML
NUM_CLASSES = 17     # 23 - 7 + 1
# ========================================

assert LABEL_DIR.exists(), f"❌ Label dir not found: {LABEL_DIR}"

print(f"🔧 Remapping labels in: {LABEL_DIR}")

for txt in LABEL_DIR.glob("*.txt"):
    new_lines = []

    with open(txt, "r") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split()
            raw_cls = int(float(parts[0]))

            # Drop legacy classes 0–6
            if raw_cls < CLASS_ID_START:
                continue

            new_cls = raw_cls - CLASS_ID_START

            if not (0 <= new_cls < NUM_CLASSES):
                raise ValueError(
                    f"❌ Invalid class id {raw_cls} in {txt.name}"
                )

            parts[0] = str(new_cls)
            new_lines.append(" ".join(parts))

    # Overwrite file safely
    with open(txt, "w") as f:
        f.write("\n".join(new_lines) + "\n")

print("✅ DONE — All train labels remapped (7–23 → 0–16)")
