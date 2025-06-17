import os

# Update this to your dataset root
dataset_root = r'D:\microcontroller-ocr-datasheet\microcontroller-ocr-datasheet\data'

def create_label_file(img_dir, label_file):
    img_dir_full = os.path.join(dataset_root, img_dir)
    label_file_full = os.path.join(dataset_root, label_file)
    with open(label_file_full, 'w') as f:
        for img_name in sorted(os.listdir(img_dir_full)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Enter label for {img_name} (only A-Z, 0-9):")
                label = input().strip().upper()
                label = ''.join([c for c in label if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'])
                f.write(f"{img_name} {label}\n")
    print(f"Labels saved to {label_file_full}")

# Create label files interactively
create_label_file('train', 'train_labels.txt')
create_label_file('test', 'test_labels.txt')