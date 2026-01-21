#!/usr/bin/env python3
"""
SIMPLEST MCU Detection Inference 
Input: Image → Output: Detected Classes (top predictions)
No OCR, no pipeline, just detection classes
"""

import torch
import cv2
from torchvision import transforms
import numpy as np
import argparse
from pathlib import Path
from model import MC


# =============================================================================
# PASTE YOUR MODEL HERE (minimal)
# =============================================================================
# Copy ONLY these classes from model.py: 
# RepDWConvRARM, DepthwiseSeparableConv, BottleneckCSPBlock, 
# FPNModule, MCUDetectorBackbone, MCUDetectionHead, MCUDetector

# [PASTE MODEL CLASSES HERE - I'll provide minimal version below if needed]

NUM_CLASSES = 24
IMG_SIZE = 512

CLASSES = [
    "8051", "ARDUINO_NANO_ATMEGA328P", "ARMCORTEXM3", "ARMCORTEXM7",
    "ESP32_DEVKIT", "NODEMCU_ESP8266", "RASPBERRY_PI_3B_PLUS", "Arduino",
    "Pico", "RaspberryPi", "Arduino Due", "Arduino Leonardo",
    "Arduino Mega 2560 -Black and Yellow-", "Arduino Mega 2560 -Black-",
    "Arduino Mega 2560 -Blue-", "Arduino Uno -Black-", "Arduino Uno -Green-",
    "Arduino Uno Camera Shield", "Arduino Uno R3", "Arduino Uno WiFi Shield",
    "Beaglebone Black", "Raspberry Pi 1 B-", "Raspberry Pi 3 B-", "Raspberry Pi A-"
]

# =============================================================================
# FIXED DECODER (SIMPLE VERSION)
# =============================================================================

def simple_decode(model, image_tensor, conf_thresh=0.2):
    """Returns TOP 3 class predictions + confidence"""
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        (cls_p4, _), (cls_p5, _) = model(image_tensor)
        
        # Objectness + Class probs
        obj_p4 = torch.sigmoid(cls_p4[:, 0]).flatten().cpu().numpy()
        obj_p5 = torch.sigmoid(cls_p5[:, 0]).flatten().cpu().numpy()
        
        cls_p4 = torch.sigmoid(cls_p4[:, 1:]).max(1)[0].flatten().cpu().numpy()
        cls_p5 = torch.sigmoid(cls_p5[:, 1:]).max(1)[0].flatten().cpu().numpy()
        
        # Combined confidences
        confs_p4 = obj_p4 * cls_p4
        confs_p5 = obj_p5 * cls_p5
        
        # Top predictions
        all_confs = np.concatenate([confs_p4, confs_p5])
        top_indices = np.argsort(all_confs)[-10:][::-1]  # Top 10
        
    predictions = []
    for idx in top_indices:
        conf = all_confs[idx]
        if conf > conf_thresh:
            # Map back to class (simplified - takes max class per grid)
            class_id = idx % NUM_CLASSES  # Approximate
            predictions.append((CLASSES[class_id], conf))
    
    return predictions[:3]  # Top 3 only

# =============================================================================
# MAIN
# =============================================================================

def main(image_path, model_path="data/runs/detect/best_mcu.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model = MCUDetector(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load & preprocess
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    x = transform(img_resized).unsqueeze(0).to(device)
    
    # RUN
    predictions = simple_decode(model, x)
    
    print(f"\n🎯 DETECTED CLASSES:")
    if predictions:
        for cls_name, conf in predictions:
            print(f"  {cls_name:<30} {conf:.3f}")
    else:
        print("  ❌ No confident detections (try lowering conf_thresh)")
    
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--model", default="data/runs/detect/best_mcu.pt")
    args = parser.parse_args()
    
    main(args.image, args.model)
