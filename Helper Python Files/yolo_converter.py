import csv
import json
import os
import shutil
from PIL import Image
import yaml

def parse_via_shape_attributes(shape_str):
    """Parse VIA shape attributes from JSON string"""
    try:
        shape_data = json.loads(shape_str)
        return shape_data
    except:
        return None

def parse_via_region_attributes(attr_str):
    """Parse VIA region attributes from JSON string"""
    try:
        attr_data = json.loads(attr_str)
        return attr_data
    except:
        return None

def get_bbox_from_shape(shape_data, img_width, img_height):
    """Convert VIA shape to normalized YOLO bounding box"""
    shape_name = shape_data.get('name', '')
    
    if shape_name == 'rect':
        # Rectangle: x, y, width, height
        x = shape_data.get('x', 0)
        y = shape_data.get('y', 0)
        width = shape_data.get('width', 0)
        height = shape_data.get('height', 0)
        
        # Convert to center coordinates
        x_center = x + width / 2
        y_center = y + height / 2
        
    elif shape_name == 'polygon' or shape_name == 'polyline':
        # Polygon/Polyline: extract all points to get bounding box
        all_points_x = shape_data.get('all_points_x', [])
        all_points_y = shape_data.get('all_points_y', [])
        
        if not all_points_x or not all_points_y:
            return None
            
        # Get bounding box from all points
        min_x = min(all_points_x)
        max_x = max(all_points_x)
        min_y = min(all_points_y)
        max_y = max(all_points_y)
        
        x_center = (min_x + max_x) / 2
        y_center = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y
        
    elif shape_name == 'circle':
        # Circle: cx, cy, r
        cx = shape_data.get('cx', 0)
        cy = shape_data.get('cy', 0)
        r = shape_data.get('r', 0)
        
        x_center = cx
        y_center = cy
        width = r * 2
        height = r * 2
        
    else:
        print(f"Unknown shape type: {shape_name}")
        return None
    
    # Normalize coordinates (0-1)
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    # Ensure coordinates are within valid range
    x_center_norm = max(0, min(1, x_center_norm))
    y_center_norm = max(0, min(1, y_center_norm))
    width_norm = max(0, min(1, width_norm))
    height_norm = max(0, min(1, height_norm))
    
    return x_center_norm, y_center_norm, width_norm, height_norm

def convert_via_to_yolo(csv_file, images_folder, output_folder):
    """Convert VIA CSV annotations to YOLO format"""
    
    # Define class names (update these based on your classes)
    class_names = [
        '8051',
        'ARDUINO_NANO_ATMEGA328P',
        'ARMCORTEXM3', 
        'ARMCORTEXM7',
        'ESP32_DEVKIT',
        'NODEMCU_ESP8266',
        'RASPBERRY_PI_3B_PLUS'
    ]
    
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    # Create output directories
    os.makedirs(os.path.join(output_folder, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'labels', 'train'), exist_ok=True)
    
    annotations_count = 0
    processed_images = 0
    class_distribution = {name: 0 for name in class_names}
    
    # Read CSV file
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            filename = row['filename']
            shape_attributes = row['region_shape_attributes']
            region_attributes = row['region_attributes']
            
            # Skip empty annotations
            if not shape_attributes or shape_attributes == '{}':
                continue
                
            # Parse shape and region data
            shape_data = parse_via_shape_attributes(shape_attributes)
            attr_data = parse_via_region_attributes(region_attributes)
            
            if not shape_data or not attr_data:
                continue
            
            # Get class label
            class_label = None
            if 'CLASS_LABELS' in attr_data:
                class_label = attr_data['CLASS_LABELS']
            elif 'class' in attr_data:
                class_label = attr_data['class']
            elif 'label' in attr_data:
                class_label = attr_data['label']
            
            if not class_label or class_label not in class_to_id:
                print(f"Unknown class label: {class_label} in {filename}")
                continue
            
            class_id = class_to_id[class_label]
            class_distribution[class_label] += 1
            
            # Check if image exists
            image_path = os.path.join(images_folder, filename)
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            
            # Get image dimensions
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")
                continue
            
            # Convert shape to bounding box
            bbox = get_bbox_from_shape(shape_data, img_width, img_height)
            if not bbox:
                continue
            
            x_center, y_center, width, height = bbox
            
            # Copy image to output folder
            output_image_path = os.path.join(output_folder, 'images', 'train', filename)
            if not os.path.exists(output_image_path):
                shutil.copy2(image_path, output_image_path)
                processed_images += 1
            
            # Create/append to label file
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(output_folder, 'labels', 'train', label_filename)
            
            with open(label_path, 'a') as label_file:
                label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            annotations_count += 1
    
    # Create data.yaml
    data_yaml = {
        'path': output_folder,
        'train': 'images/train',
        'val': 'images/train',  # Using same for now, you can split later
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open(os.path.join(output_folder, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    # Create classes.txt
    with open(os.path.join(output_folder, 'classes.txt'), 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print(f"\nConversion completed!")
    print(f"Processed {processed_images} images")
    print(f"Created {annotations_count} annotations")
    print(f"\nClass distribution:")
    for class_name, count in class_distribution.items():
        print(f"  {class_name}: {count}")
    
    return processed_images, annotations_count

def main():
    """Main function to convert VIA annotations to YOLO format"""
    
    print("VIA to YOLO Converter")
    print("====================")
    
    # Check if required files and folders exist
    required_files = ['train.csv']
    required_folders = ['train']
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    for folder in required_folders:
        if not os.path.exists(folder):
            missing_files.append(folder)
    
    if missing_files:
        print(f"Error: Missing required files/folders: {missing_files}")
        print("Make sure you have:")
        print("- train.csv (VIA annotation file)")
        print("- train/ folder (with training images)")
        return
    
    # Convert training data
    print("\nConverting training data...")
    train_images, train_annotations = convert_via_to_yolo(
        'train.csv', 
        'train', 
        'dataset_train'
    )
    
    # Convert test data if available
    if os.path.exists('test.csv') and os.path.exists('test'):
        print("\nConverting test data...")
        test_images, test_annotations = convert_via_to_yolo(
            'test.csv', 
            'test', 
            'dataset_test'
        )
    else:
        print("\nNo test.csv or test/ folder found, skipping test data conversion.")
    
    print("\n" + "="*50)
    print("CONVERSION COMPLETE!")
    print("="*50)
    print("\nGenerated folders:")
    print("- dataset_train/ (training data in YOLO format)")
    if os.path.exists('test.csv') and os.path.exists('test'):
        print("- dataset_test/ (test data in YOLO format)")
    
    print("\nTo start training with YOLOv8:")
    print("yolo detect train data=dataset_train/data.yaml model=yolov8s.pt epochs=100 imgsz=640")
    
    print("\nTo start training with YOLOv5:")
    print("python train.py --data dataset_train/data.yaml --weights yolov5s.pt --epochs=100")

if __name__ == "__main__":
    main()
