import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from ultralytics import YOLO

class OCRDataset(Dataset):
    """Enhanced OCR Dataset with support for detection-based crops"""
    def __init__(self, img_dir, label_dict, transform=None, use_detection=False, yolo_model_path=None):
        self.img_dir = img_dir
        self.label_dict = label_dict
        self.image_names = list(label_dict.keys())
        self.transform = transform
        self.use_detection = use_detection
        self.detector = None
        
        # Load YOLO model if detection is enabled
        if use_detection and yolo_model_path:
            self.detector = YOLO(yolo_model_path)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            if self.use_detection and self.detector:
                # Use YOLO detection to crop microcontroller region
                image = self._get_detected_crop(img_path)
            else:
                # Standard image loading for pre-cropped images
                image = Image.open(img_path).convert("L")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
            
        if self.transform:
            image = self.transform(image)
            
        label = self.label_dict[img_name]
        return image, label

    def _get_detected_crop(self, img_path):
        """Extract microcontroller crop using YOLO detection"""
        # Run YOLO detection
        results = self.detector(img_path, verbose=False)
        
        # Load original image
        image = cv2.imread(img_path)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get the most confident detection
            best_box = results[0].boxes[0]
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
            
            # Crop the detected region with some padding
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            crop = image[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            return Image.fromarray(crop)
        else:
            # Fallback to full image if no detection
            return Image.open(img_path).convert("L")

class DetectionOCRDataset(Dataset):
    """Dataset for processing full PCB images with multiple microcontrollers"""
    def __init__(self, img_dir, yolo_model_path, ocr_transform=None):
        self.img_dir = img_dir
        self.detector = YOLO(yolo_model_path)
        self.ocr_transform = ocr_transform
        self.image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Get all detections from the image
        results = self.detector(img_path, verbose=False)
        
        crops = []
        detection_info = []
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            image = cv2.imread(img_path)
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Extract crop
                crop = image[y1:y2, x1:x2]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop_pil = Image.fromarray(crop)
                
                if self.ocr_transform:
                    crop_pil = self.ocr_transform(crop_pil)
                
                crops.append(crop_pil)
                detection_info.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id
                })
        
        return crops, detection_info, os.path.basename(img_path)

def load_labels(label_path):
    CLASS_NAMES = [
        "8051",
        "ARDUINO_NANO_ATMEGA328P",
        "ARMCORTEXM3",
        "ARMCORTEXM7",
        "ESP32_DEVKIT",
        "NODEMCU_ESP8266",
        "RASPBERRY_PI_3B_PLUS"
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
                    label_dict[img_name] = CLASS_NAMES[class_id]
    else:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) >= 2:
                    name = parts[0]
                    text = parts[1]
                    label_dict[name] = text.lower()
    return label_dict



def collate_fn(batch):
    """Enhanced collate function with better error handling"""
    from utils import encode_label
    
    images, texts = zip(*batch)
    images = torch.stack(images)
    
    # Encode labels with error handling
    encoded_texts = []
    for text in texts:
        try:
            encoded = torch.tensor(encode_label(text), dtype=torch.long)
            if len(encoded) == 0:  # Handle empty encodings
                encoded = torch.tensor([0], dtype=torch.long)
            encoded_texts.append(encoded)
        except Exception as e:
            print(f"Warning: Error encoding text '{text}': {e}")
            encoded_texts.append(torch.tensor([0], dtype=torch.long))
    
    label_lengths = torch.tensor([len(t) for t in encoded_texts], dtype=torch.long)
    targets = torch.cat(encoded_texts)
    
    # More accurate input length calculation
    # Assuming final feature map width is input_width // 4 after pooling
    input_lengths = torch.full(size=(len(images),), 
                              fill_value=images.shape[-1] // 4, 
                              dtype=torch.long)
    
    return images, targets, input_lengths, label_lengths

def detection_collate_fn(batch):
    """Collate function for detection-based processing"""
    all_crops = []
    all_detection_info = []
    all_filenames = []
    
    for crops, detection_info, filename in batch:
        all_crops.extend(crops)
        all_detection_info.extend(detection_info)
        all_filenames.extend([filename] * len(crops))
    
    if all_crops:
        crops_tensor = torch.stack(all_crops)
        return crops_tensor, all_detection_info, all_filenames
    else:
        return torch.empty(0), [], []
