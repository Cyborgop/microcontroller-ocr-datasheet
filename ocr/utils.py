import string
import torch
import re
from rapidfuzz import process, fuzz

# --- Character Mapping ---
CHARS = string.ascii_lowercase + string.digits + '_'  # Added underscore for better microcontroller names
char2idx = {char: i for i, char in enumerate(CHARS)}
idx2char = {i: char for char, i in char2idx.items()}
BLANK_IDX = len(CHARS)

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

def encode_label(text):
    """Enhanced label encoding with better error handling"""
    if not isinstance(text, str):
        text = str(text)
    
    # Clean text: remove special characters except underscore
    cleaned_text = re.sub(r'[^a-zA-Z0-9_]', '', text.lower())
    
    # Encode only valid characters
    encoded = []
    for c in cleaned_text:
        if c in char2idx:
            encoded.append(char2idx[c])
    
    return encoded

def decode_output(output):
    """Enhanced CTC decoding with post-processing"""
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
        
        # Join characters and apply post-processing
        raw_text = ''.join(chars)
        processed_text = post_process_prediction(raw_text)
        decoded.append(processed_text)
    
    return decoded

def post_process_prediction(text, valid_labels=VALID_LABELS, threshold=70):
    """Post-process OCR predictions using fuzzy matching"""
    if not text:
        return text
    
    # Try exact match first
    if text in valid_labels:
        return text
    
    # Fuzzy matching with multiple algorithms
    best_match = None
    best_score = 0
    
    for label in valid_labels:
        # Try different similarity metrics
        ratio_score = fuzz.ratio(text, label)
        partial_score = fuzz.partial_ratio(text, label)
        token_score = fuzz.token_sort_ratio(text, label)
        
        # Take the maximum score
        max_score = max(ratio_score, partial_score, token_score)
        
        if max_score > best_score and max_score >= threshold:
            best_score = max_score
            best_match = label
    
    return best_match if best_match else text

def calculate_cer(pred, target):
    """Calculate Character Error Rate"""
    from rapidfuzz.distance import Levenshtein
    if not target:
        return 1.0 if pred else 0.0
    return Levenshtein.distance(pred, target) / len(target)

def calculate_wer(pred, target):
    """Calculate Word Error Rate"""
    pred_words = pred.split()
    target_words = target.split()
    
    if not target_words:
        return 1.0 if pred_words else 0.0
    
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(pred_words, target_words) / len(target_words)

def preprocess_crop_for_ocr(crop_image, target_height=32, target_width=128):
    """Preprocess detected crop for OCR input"""
    import cv2
    import numpy as np
    
    # Convert PIL to numpy if needed
    if hasattr(crop_image, 'convert'):
        crop_array = np.array(crop_image.convert('L'))
    else:
        crop_array = crop_image
    
    # Apply image enhancement
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
    crop_array = np.pad(crop_array, 
                       ((pad_h//2, pad_h - pad_h//2), 
                        (pad_w//2, pad_w - pad_w//2)), 
                       mode='constant', constant_values=255)
    
    return crop_array

class MetricsTracker:
    """Track training and validation metrics"""
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
