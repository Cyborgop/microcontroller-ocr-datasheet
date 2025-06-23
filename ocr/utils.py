import string
import torch
import re
import numpy as np
from rapidfuzz import process, fuzz
from PIL import Image
import cv2
import difflib

# --- Character Mapping ---
CHARS = string.ascii_lowercase + string.digits + '_'
char2idx = {char: i for i, char in enumerate(CHARS)}
idx2char = {i: char for char, i in char2idx.items()}
BLANK_IDX = len(CHARS)

CLASSES = [
    "8051",
    "arduino_nano_atmega328p",
    "armcortexm3",
    "armcortexm7",
    "esp32devkit",
    "nodemcuesp8266",
    "raspberry_pi_3b_plus"
]

def correct_ocr_text(ocr_text, valid_classes=CLASSES, cutoff=0.6):
    """
    Map OCR output to the closest valid class name using fuzzy matching.
    Returns the best match if above cutoff, else returns the original text.
    """
    matches = difflib.get_close_matches(ocr_text, valid_classes, n=1, cutoff=cutoff)
    return matches[0] if matches else ocr_text

# Valid microcontroller labels for post-processing
VALID_LABELS = [
    "armcortexm3",
    "armcortexm7", 
    "8051",
    "raspberrypi3bplus",
    "arduinononoatmega328p",
    "esp32devkit",
    "nodemcuesp8266"
]

def normalize_label(label):
    return label.replace('_', '').lower()

# --- Denoise function ---
def denoise_image(pil_img):
    img = np.array(pil_img)
    denoised = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return Image.fromarray(denoised)

# --- Advanced Deskew Function ---
def deskew_image(pil_img):
    """
    Deskews a PIL image (grayscale or color) using OpenCV's minAreaRect.
    Returns a PIL image of the same mode.
    """
    img = np.array(pil_img)
    is_color = False
    if img.ndim == 3 and img.shape[2] == 3:
        is_color = True
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img if img.ndim == 2 else img[:, :, 0]

    # Threshold to binary and invert for minAreaRect
    _, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = 255 - img_bin

    coords = np.column_stack(np.where(img_bin > 0))
    if coords.shape[0] == 0:
        return pil_img  # No foreground pixels

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    if is_color:
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated)
    else:
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated)

# --- Label Encoding/Decoding ---
def encode_label(text):
    """Enhanced label encoding with better error handling."""
    if not isinstance(text, str):
        text = str(text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9_]', '', text.lower())
    encoded = [char2idx[c] for c in cleaned_text if c in char2idx]
    return encoded

def decode_output(output):
    """Enhanced CTC decoding with post-processing and batch support."""
    if output.dim() == 3 and output.shape[0] != output.shape[1]:
        pred_indices = torch.argmax(output, dim=2).permute(1, 0)
    else:
        pred_indices = torch.argmax(output, dim=2)
    decoded = []
    for seq in pred_indices:
        chars = []
        prev = -1
        for idx in seq:
            idx = idx.item()
            if idx != prev and idx != BLANK_IDX and idx in idx2char:
                chars.append(idx2char[idx])
            prev = idx
        raw_text = ''.join(chars)
        processed_text = post_process_prediction(raw_text)
        decoded.append(processed_text)
    return decoded

# --- Post-Processing ---
def post_process_prediction(text, valid_labels=VALID_LABELS, threshold=70):
    """
    Post-process OCR predictions using fuzzy matching with multiple metrics.
    """
    if not text:
        return text
    if text in valid_labels:
        return text
    best_match = None
    best_score = 0
    for label in valid_labels:
        ratio_score = fuzz.ratio(text, label)
        partial_score = fuzz.partial_ratio(text, label)
        token_score = fuzz.token_sort_ratio(text, label)
        max_score = max(ratio_score, partial_score, token_score)
        if max_score > best_score and max_score >= threshold:
            best_score = max_score
            best_match = label
    return best_match if best_match else text

# --- Metrics ---
def calculate_cer(pred, target):
    """Calculate Character Error Rate."""
    from rapidfuzz.distance import Levenshtein
    if not target:
        return 1.0 if pred else 0.0
    return Levenshtein.distance(pred, target) / len(target)

def calculate_wer(pred, target):
    """Calculate Word Error Rate."""
    pred_words = pred.split()
    target_words = target.split()
    if not target_words:
        return 1.0 if pred_words else 0.0
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(pred_words, target_words) / len(target_words)

# --- Preprocessing for OCR Crops ---
def preprocess_crop_for_ocr(crop_image, target_height=32, target_width=128):
    """
    Preprocess detected crop for OCR input:
    - Converts to grayscale
    - Gaussian blur
    - Adaptive threshold
    - Morphological clean-up
    - Resize with aspect ratio and pad to target size
    """
    # Convert PIL to numpy if needed
    if hasattr(crop_image, 'convert'):
        crop_array = np.array(crop_image.convert('L'))
    else:
        crop_array = crop_image

    # 1. Gaussian blur to reduce noise
    crop_array = cv2.GaussianBlur(crop_array, (3, 3), 0)
    # 2. Adaptive thresholding for better text contrast
    crop_array = cv2.adaptiveThreshold(crop_array, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    # 3. Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    crop_array = cv2.morphologyEx(crop_array, cv2.MORPH_CLOSE, kernel)
    # 4. Resize while maintaining aspect ratio
    h, w = crop_array.shape
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    crop_array = cv2.resize(crop_array, (new_w, new_h))
    # 5. Pad to target size
    pad_h = target_height - new_h
    pad_w = target_width - new_w
    crop_array = np.pad(
        crop_array,
        ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)),
        mode='constant', constant_values=255
    )
    return crop_array

# --- Metrics Tracker Class ---
class MetricsTracker:
    """Track training and validation metrics."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.total_loss = 0.0
        self.total_cer = 0.0
        self.total_wer = 0.0
        self.total_samples = 0
        self.predictions = []
        self.targets = []
    def update(self, loss, predictions, targets):
        self.total_loss += loss
        self.total_samples += len(predictions)
        for pred, target in zip(predictions, targets):
            self.total_cer += calculate_cer(pred, target)
            self.total_wer += calculate_wer(pred, target)
            self.predictions.append(pred)
            self.targets.append(target)
    def get_metrics(self):
        if self.total_samples == 0:
            return {'loss': 0, 'cer': 0, 'wer': 0}
        return {
            'loss': self.total_loss / self.total_samples,
            'cer': self.total_cer / self.total_samples,
            'wer': self.total_wer / self.total_samples
        }
