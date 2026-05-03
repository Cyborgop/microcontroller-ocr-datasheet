# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end system for detecting microcontroller boards in images, performing OCR on chip surface markings, and retrieving datasheet information from MongoDB. Detects 14 board/chip variants (Arduino, Raspberry Pi, ARM Cortex, STM32, Teensy).

## Commands

**Run the API server:**
```bash
python api/main.py
# Listens on http://0.0.0.0:8000 with hot reload
```

**Train the detection model:**
```bash
python ocr/train.py --epochs 200 --batch_size 8 --lr 5e-4 --workers 8 --img_size 512
# --find_lr: run LR finder first (outputs to runs/detect/lr_finder/)
# --use_ema: enable EMA (decay 0.9999, kicks in at epoch 15)
# --calculate_map: compute mAP during validation (expensive)
# --accum_steps N: gradient accumulation (default 2)
# --train_img_dir / --train_label_dir / --val_img_dir / --val_label_dir: override data paths
```

**Run inference/evaluation:**
```bash
python ocr/Inference_Pipeline.py \
  --model runs/detect/train/weights/best_mcu.pt \
  --test_img_dir data/dataset_inference/images \
  --test_label_dir data/dataset_inference/labels \
  --conf_thresh 0.25 --nms_thresh 0.40 \
  --run_name my_run   # outputs to runs_test/<run_name>/{plots,images,detections,results}
# Single image: --image path/to/img.jpg
# Folder without labels: --image_dir path/to/folder/
```

**Benchmark model latency:**
```bash
python ocr/benchmark_latency.py
```

**Convert VIA annotations to YOLO format:**
```bash
python "Helper Python Files/yolo_converter.py"
```

**Validate dataset before training:**
```bash
python ocr/debug_dataset.py
# Checks image-label pairing, class ID validity, box geometry; visualizes 3 samples
```

**Environment variables** (set in `.env` or shell):
- `MONGO_URL` — MongoDB connection string (default: `mongodb://localhost:27017`)
- `API_KEY` — Header auth key (default: `dev-key`)
- `YOLO_WEIGHTS` — Path to YOLO `.pt` weights file
- `CRNN_WEIGHTS` — Path to CRNN `.pt` weights file
- `CONF_THRESH` — Detection confidence threshold (default: `0.85`)

## Architecture

**Inference pipeline** (`POST /recognize-and-resolve`):
1. Image uploaded → decoded with OpenCV
2. **MCUDetector** (`ocr/model.py`): RepVit backbone → BiFPN neck → DecoupledScaleHead; outputs bounding boxes + class (14 board types)
3. Crops are deskewed, denoised, resized to 128×32
4. **EnhancedCRNN** (`ocr/model.py`): Conv layers + LSTM + CTC decode → raw text
5. Text is cleaned/corrected with fuzzy matching (`rapidfuzz`)
6. MongoDB lookup: exact → partial → vendor-variant → prefix-relaxed fallback
7. Response: detections list with `datasheet_found` flag and metadata

**Detection model** (`ocr/model.py`):
- Backbone: RepVit blocks + BottleneckCSP
- Neck: Bidirectional FPN (multi-scale P3/P4/P5)
- Head: Decoupled per-scale heads
- Loss: SimOTA assignment + FocalLoss (cls) + CIoU (bbox)
- Training: AdamW with differential LR (2× for classifier head), CosineAnnealingLR + warmup, optional EMA

**OCR model** (`ocr/model.py`):
- Input: 32×128 grayscale, normalized to `[-1, 1]`
- Character set: lowercase letters + digits + `_` (38 tokens)
- Output: CTC-decoded string

**Experimental architectures** (not used in production API):
- `ocr/model_repvit.py` — RepVit backbone variant
- `ocr/model_star.py` — StarNet backbone
- `ocr/new_simam_model.py` — SimAM attention variant

**API** (`api/main.py`): FastAPI + Motor (async MongoDB). Key endpoints:
- `POST /recognize` — detections only
- `POST /recognize-and-resolve` — detections + datasheet lookup
- `GET /datasheet/{part_number}`, `GET /search/`, `GET /datasheets/`

**MongoDB schema** (`microcontrollers.datasheets` collection):
```
part_number, manufacturer, core, flash, ram, max_clock, datasheet_url, features
```

**Dataset**: YOLO format, 14 classes. Training outputs land in `runs/detect/train/`.

> **Warning**: The checked-in `data/data.yaml` lists only 7 classes and is outdated. The authoritative 14-class mapping is hardcoded in `ocr/utils.py`. Any class ID ≥ 14 in a label file causes an immediate assertion error at dataset load.

Required directory structure for training:
```
data/dataset_train/images/train/   ← training images
data/dataset_train/labels/train/   ← training labels (.txt, YOLO normalized)
data/dataset_test/images/train/    ← val images (note: "train" suffix is required)
data/dataset_test/labels/train/    ← val labels
```
Use `--train_img_dir`/`--val_img_dir` flags to override. Class indices:
```
0  Arduino Due                    7  Arduino Uno Camera Shield
1  Arduino Leonardo               8  ARM Cortex M0+
2  Arduino Mega 2560 (Blk+Yel)   9  ARM Cortex M3
3  Arduino Mega 2560 (Black)     10  ARM Cortex M4
4  Arduino Mega 2560 (Blue)      11  STM32 Discovery
5  Arduino Uno (Black)           12  Teensy 3.1
6  Arduino Uno (Green)           13  Raspberry Pi A+
```

## Key Files

| File | Role |
|------|------|
| `ocr/model.py` | MCUDetector + EnhancedCRNN model definitions |
| `ocr/train.py` | Full training loop (EMA, LR finder, early stopping) |
| `ocr/dataset.py` | `MCUDetectionDataset` — YOLO label loading |
| `ocr/utils.py` | mAP/metrics, image preprocessing (deskew, denoise) |
| `ocr/Inference_Pipeline.py` | Evaluation harness with metric plots |
| `api/main.py` | FastAPI server — all inference + DB logic |
| `data.yaml` | YOLO dataset config (paths + class list) |
