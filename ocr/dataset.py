import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from utils import deskew_image, denoise_image
from model import MCUDetector   # your custom detector


class OCRDataset(Dataset):
    """Enhanced OCR Dataset with support for detection-based crops and preprocessing."""
    def __init__(self, img_dir, label_dict, transform=None,
                 use_detection=True, detector_weights=None, device=None):
        self.img_dir = img_dir
        self.label_dict = label_dict
        self.image_names = list(label_dict.keys())
        self.transform = transform
        self.use_detection = use_detection

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.detector = None
        if self.use_detection and detector_weights:
            # Load your custom MCUDetector
            self.detector = MCUDetector(num_classes=24).to(self.device)
            state = torch.load(detector_weights, map_location=self.device)
            # Support both plain state_dict and checkpoint dict
            if isinstance(state, dict) and 'model_state_dict' in state:
                self.detector.load_state_dict(state['model_state_dict'])
            else:
                self.detector.load_state_dict(state)
            self.detector.eval()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            if self.use_detection and self.detector is not None:
                image = self._get_detected_crop(img_path)
            else:
                image = Image.open(img_path).convert("L")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        # Preprocessing
        image = deskew_image(image)
        image = denoise_image(image)

        if self.transform:
            image = self.transform(image)

        label = self.label_dict[img_name]
        return image, label

    def _get_detected_crop(self, img_path):
        """Extract microcontroller crop using custom MCUDetector (simple decoding)."""
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            # Fallback to full image if read fails
            return Image.open(img_path).convert("L")

        h_orig, w_orig = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(image_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)

        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, 3, 512, 512)

        with torch.no_grad():
            (cls_p4, reg_p4), (cls_p5, reg_p5) = self.detector(img_tensor)

        # ---------------------------------------------------------------------
        # Very simple decoding: pick highest scoring cell in P5 feature map.
        # This is NOT full YOLO decoding + NMS, but enough to get a crop.
        # ---------------------------------------------------------------------
        # cls_p5 shape: (B, 1 + num_classes, H, W)
        obj_logits = cls_p5[0, :1, :, :]          # (1, H, W)
        cls_logits = cls_p5[0, 1:, :, :]          # (C, H, W)

        obj_scores = torch.sigmoid(obj_logits)    # (1, H, W)
        cls_scores = torch.sigmoid(cls_logits)    # (C, H, W)
        max_cls_scores, _ = cls_scores.max(dim=0, keepdim=True)  # (1, H, W)

        scores = obj_scores * max_cls_scores      # (1, H, W)
        _, max_idx = scores.view(-1).max(0)
        max_idx = max_idx.item()

        H, W = scores.shape[1], scores.shape[2]
        gy, gx = divmod(max_idx, W)

        stride = 16  # P5 stride
        cx = (gx + 0.5) * stride
        cy = (gy + 0.5) * stride
        bw = 80
        bh = 80

        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(512, cx + bw / 2))
        y2 = int(min(512, cy + bh / 2))

        # Map back to original image coordinates
        x1 = int(x1 / 512 * w_orig)
        x2 = int(x2 / 512 * w_orig)
        y1 = int(y1 / 512 * h_orig)
        y2 = int(y2 / 512 * h_orig)

        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w_orig, x2 + pad)
        y2 = min(h_orig, y2 + pad)

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            # Fallback to full image
            return Image.open(img_path).convert("L")

        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_pil = Image.fromarray(crop_gray)
        crop_pil = crop_pil.resize((128, 32), Image.BILINEAR)
        return crop_pil


# ============================================================================
# CHANGE #1: Updated to 24 CLASSES
# ============================================================================
def load_labels(label_path):
    """Load labels from YOLO annotation files - now supports 24 classes."""
    CLASS_NAMES = [
        "8051",                                 # 0
        "ARDUINO_NANO_ATMEGA328P",              # 1
        "ARMCORTEXM3",                          # 2
        "ARMCORTEXM7",                          # 3
        "ESP32_DEVKIT",                         # 4
        "NODEMCU_ESP8266",                      # 5
        "RASPBERRY_PI_3B_PLUS",                 # 6
        "Arduino",                              # 7
        "Pico",                                 # 8
        "RaspberryPi",                          # 9
        "Arduino Due",                          # 10
        "Arduino Leonardo",                     # 11
        "Arduino Mega 2560 -Black and Yellow-", # 12
        "Arduino Mega 2560 -Black-",            # 13
        "Arduino Mega 2560 -Blue-",             # 14
        "Arduino Uno -Black-",                  # 15
        "Arduino Uno -Green-",                  # 16
        "Arduino Uno Camera Shield",            # 17
        "Arduino Uno R3",                       # 18
        "Arduino Uno WiFi Shield",              # 19
        "Beaglebone Black",                     # 20
        "Raspberry Pi 1 B-",                    # 21
        "Raspberry Pi 3 B-",                    # 22
        "Raspberry Pi A-"                       # 23
    ]

    label_dict = {}
    if os.path.isdir(label_path):
        for fname in os.listdir(label_path):
            if not fname.endswith('.txt'):
                continue
            img_name = fname.replace('.txt', '.jpg')
            with open(os.path.join(label_path, fname), 'r', encoding='utf-8') as f:
                line = f.readline().strip()
                if line:
                    class_id = int(line.split()[0])
                    if class_id < len(CLASS_NAMES):
                        label_dict[img_name] = CLASS_NAMES[class_id]
                    else:
                        print(f"Warning: class_id {class_id} out of range for {img_name}")
                        label_dict[img_name] = CLASS_NAMES[0]
    else:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) >= 2:
                    name = parts[0]
                    text = parts[1]
                    label_dict[name] = text.lower()
    return label_dict


# ============================================================================
# CHANGE #2: Updated collate_fn to handle variable-length sequences better
# ============================================================================
def collate_fn(batch):
    """Enhanced collate function with better error handling."""
    from utils import encode_label

    images, texts = zip(*batch)
    images = torch.stack(images)
    encoded_texts = []

    for text in texts:
        try:
            encoded = torch.tensor(encode_label(text), dtype=torch.long)
            if len(encoded) == 0:
                encoded = torch.tensor([0], dtype=torch.long)
            encoded_texts.append(encoded)
        except Exception as e:
            print(f"Warning: Error encoding text '{text}': {e}")
            encoded_texts.append(torch.tensor([0], dtype=torch.long))

    label_lengths = torch.tensor([len(t) for t in encoded_texts], dtype=torch.long)
    targets = torch.cat(encoded_texts)

    # Input lengths based on image width (128 pixels / 4 for CNN stride)
    input_lengths = torch.full(
        size=(len(images),),
        fill_value=images.shape[-1] // 4,  # 128 // 4 = 32
        dtype=torch.long
    )

    return images, targets, input_lengths, label_lengths
