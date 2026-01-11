import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from utils import deskew_image, denoise_image
from model import MCUDetector   # your custom detector


class OCRDataset(Dataset):
    """
    OCR Dataset with OPTIONAL detector-based cropping.
    Detector is FROZEN and used only for inference.
    """

    def __init__(
        self,
        img_dir,
        label_dict,
        transform=None,
        use_detection=True,
        detector_weights=None,
        device=None,
    ):
        self.img_dir = img_dir
        self.label_dict = label_dict
        self.image_names = list(label_dict.keys())
        self.transform = transform
        self.use_detection = use_detection

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.detector = None
        if self.use_detection and detector_weights is not None:
            self.detector = MCUDetector(num_classes=24).to(self.device)
            state = torch.load(detector_weights, map_location=self.device)
            self.detector.load_state_dict(
                state["model_state_dict"] if isinstance(state, dict) else state
            )
            self.detector.eval()

            # 🔒 CRITICAL: Freeze detector completely
            for p in self.detector.parameters():
                p.requires_grad = False

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
            raise RuntimeError(f"Failed to load {img_path}: {e}")

        # OCR preprocessing (order is important)
        image = deskew_image(image)
        image = denoise_image(image)

        if self.transform:
            image = self.transform(image)

        label = self.label_dict[img_name]
        return image, label

    @torch.no_grad()
    def _get_detected_crop(self, img_path):
        """Detector-guided crop using objectness + regression."""
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            return Image.open(img_path).convert("L")

        h0, w0 = image_bgr.shape[:2]

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512))

        img_tensor = (
            torch.from_numpy(img_resized)
            .float()
            .div(255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        (cls_p4, reg_p4), (cls_p5, reg_p5) = self.detector(img_tensor)

        # ---- Choose best scale (P4 or P5) ----
        s4 = torch.sigmoid(cls_p4[:, :1]).max()
        s5 = torch.sigmoid(cls_p5[:, :1]).max()

        if s4 > s5:
            cls_map, reg_map, stride = cls_p4, reg_p4, 8
        else:
            cls_map, reg_map, stride = cls_p5, reg_p5, 16

        obj = torch.sigmoid(cls_map[0, :1])
        _, idx = obj.view(-1).max(0)
        H, W = obj.shape[1:]
        gy, gx = divmod(idx.item(), W)

        # ---- Decode box ----
        dx = torch.sigmoid(reg_map[0, 0, gy, gx])
        dy = torch.sigmoid(reg_map[0, 1, gy, gx])
        bw = torch.exp(reg_map[0, 2, gy, gx]).clamp(20, 300)
        bh = torch.exp(reg_map[0, 3, gy, gx]).clamp(20, 300)

        cx = (gx + dx) * stride
        cy = (gy + dy) * stride

        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(512, cx + bw / 2))
        y2 = int(min(512, cy + bh / 2))

        # ---- Map back to original resolution ----
        x1 = int(x1 / 512 * w0)
        x2 = int(x2 / 512 * w0)
        y1 = int(y1 / 512 * h0)
        y2 = int(y2 / 512 * h0)

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
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
