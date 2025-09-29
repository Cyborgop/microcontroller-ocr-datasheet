from ultralytics import YOLO
from model import EnhancedCRNN
from utils import decode_output, deskew_image, denoise_image, correct_ocr_text
import torch
from torchvision import transforms
from PIL import Image
import cv2
import os

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    yolo_detector = YOLO("data/runs/detect/train2/weights/best.pt")
    crnn_model = EnhancedCRNN(img_height=32, num_channels=1, num_classes=38)
    
    # Check if model file exists
    if os.path.exists("best_model.pth"):
        crnn_model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
        print("Loaded trained model weights")
    else:
        print("Warning: No trained model found, using random weights")
    
    crnn_model.to(DEVICE)
    crnn_model.eval()

    # OCR preprocessing
    ocr_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def detect_and_ocr(image_path):
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found!")
            return []
        
        results = yolo_detector(image_path)
        image = cv2.imread(image_path)
        outputs = []
        
        if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                
                # Extract and preprocess crop
                crop = image[y1:y2, x1:x2]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop_pil = Image.fromarray(crop)
                
                # Apply preprocessing
                crop_pil = deskew_image(crop_pil)
                crop_pil = denoise_image(crop_pil)
                crop_pil = crop_pil.resize((128, 32), Image.BILINEAR)
                
                # OCR inference
                crop_tensor = ocr_transform(crop_pil).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = crnn_model(crop_tensor)
                    pred = decode_output(logits)
                
                # Post-process
                corrected_pred = correct_ocr_text(pred[0] if pred else "")
                outputs.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "ocr_text": corrected_pred,
                    "raw_pred": pred[0] if pred else ""
                })
        else:
            print("No detections found")
        
        return outputs

    # # Test inference
    image_path = "dronebot.jpg"
    results = detect_and_ocr(image_path)
    print("Results:", results)

if __name__ == "__main__":
    main()
