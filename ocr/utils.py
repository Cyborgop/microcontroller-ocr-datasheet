import string
import torch
import re
import numpy as np
from rapidfuzz import fuzz
from PIL import Image
import cv2
import difflib
import itertools

CHARS = string.ascii_lowercase + string.digits + '_'
char2idx = {char: i for i, char in enumerate(CHARS)}
idx2char = {i: char for char, i in char2idx.items()}
BLANK_IDX = len(CHARS)

CLASSES = [
    "8051",
    "ARDUINO_NANO_ATMEGA328P",
    "ARMCORTEXM3",
    "ARMCORTEXM7",
    "ESP32_DEVKIT",
    "NODEMCU_ESP8266",
    "RASPBERRY_PI_3B_PLUS",
    "Arduino",
    "Pico",
    "RaspberryPi",
    "Arduino Due",
    "Arduino Leonardo",
    "Arduino Mega 2560 -Black and Yellow-",
    "Arduino Mega 2560 -Black-",
    "Arduino Mega 2560 -Blue-",
    "Arduino Uno -Black-",
    "Arduino Uno -Green-",
    "Arduino Uno Camera Shield",
    "Arduino Uno R3",
    "Arduino Uno WiFi Shield",
    "Beaglebone Black",
    "Raspberry Pi 1 B-",
    "Raspberry Pi 3 B-",
    "Raspberry Pi A-"
]

VALID_LABELS = [
    "8051",
    "arduinonanoatmega328p", "armcortexm3", "armcortexm7", "esp32devkit",
    "nodemcuesp8266", "raspberrypi3bplus", "arduino", "pico", "raspberrypi",
    "arduinodue", "arduinoleonardo", "arduinomega2560blackandyellow",
    "arduinomega2560black", "arduinomega2560blue", "arduinounoblack",
    "arduinounogreen", "arduinounocamerashield", "arduinounor3",
    "arduinounowifishield", "beagleboneblack", "raspberrypi1b",
    "raspberrypi3b", "raspberrypia"
]

def normalize_class_name(name):
    name = name.lower().replace(' ', '').replace('-', '').replace('_', '')
    return name

def correct_ocr_text(text, valid_classes=CLASSES, cutoff=0.75):
    norm = re.sub(r'[^0-9a-z]', '', text.lower())
    matches = difflib.get_close_matches(norm, [normalize_class_name(c) for c in valid_classes], n=1, cutoff=cutoff)
    return matches[0] if matches else text

def normalize_label(label):
    return label.replace('_', '').lower()

def denoise_image(pil_img):
    img = np.array(pil_img)
    denoised = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return Image.fromarray(denoised)

def deskew_image(pil_img):
    img = np.array(pil_img)
    is_color = False
    if img.ndim == 3 and img.shape[2] == 3:
        is_color = True
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img if img.ndim == 2 else img[:, :, 0]

    _, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = 255 - img_bin

    coords = np.column_stack(np.where(img_bin > 0))
    if coords.shape[0] == 0:
        return pil_img 

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

def decode_argmax_output(logits):
    pred_chars = torch.argmax(logits, dim=2)[0]
    text = ""
    for char_idx in pred_chars:
        char_idx = char_idx.item()
        if char_idx in idx2char:
            text += idx2char[char_idx]
    raw_text = text.strip()
    return post_process_prediction(raw_text)

def encode_label(text):
    if not isinstance(text, str):
        text = str(text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9_]', '', text.lower())
    encoded = [char2idx[c] for c in cleaned_text if c in char2idx]
    return encoded

def decode_output(output):
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
        collapsed_text = ''.join([c for c, _ in itertools.groupby(raw_text)])
        processed_text = post_process_prediction(collapsed_text)
        decoded.append(processed_text)
    return decoded

def post_process_prediction(text, valid_labels=VALID_LABELS, threshold=70):
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

def calculate_cer(pred, target):
    from rapidfuzz.distance import Levenshtein
    if not target:
        return 1.0 if pred else 0.0
    return Levenshtein.distance(pred, target) / len(target)

def calculate_wer(pred, target):
    pred_words = pred.split()
    target_words = target.split()
    if not target_words:
        return 1.0 if pred_words else 0.0
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(pred_words, target_words) / len(target_words)

# def preprocess_crop_for_ocr(crop_image, target_height=32, target_width=128):
#     """
#     Pre
def print_gpu_memory():
    """Print current GPU memory usage (MB)."""
    if not torch.cuda.is_available():
        print("GPU not available.")
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"GPU Memory: {allocated:.2f} MB allocated / {reserved:.2f} MB reserved")
