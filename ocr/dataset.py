import os
from PIL import Image
from torch.utils.data import Dataset
import torch

# --- Dataset ---
class OCRDataset(Dataset):
    def __init__(self, img_dir, label_dict, transform=None):
        self.img_dir = img_dir
        self.label_dict = label_dict
        self.image_names = list(label_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        # Robust error handling for missing/corrupted images
        try:
            image = Image.open(img_path).convert("L")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
        if self.transform:
            image = self.transform(image)
        label = self.label_dict[img_name]
        return image, label

def load_labels(label_file):
    label_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) >= 2:
                name = parts[0]
                text = parts[1]
                label_dict[name] = text.lower()
    return label_dict

def collate_fn(batch):
    from utils import encode_label  # Import here to avoid circular import
    images, texts = zip(*batch)
    images = torch.stack(images)
    encoded_texts = [torch.tensor(encode_label(t), dtype=torch.long) for t in texts]
    label_lengths = torch.tensor([len(t) for t in encoded_texts], dtype=torch.long)
    targets = torch.cat(encoded_texts)
    # Dynamically compute input_lengths based on model output width if needed
    input_lengths = torch.full(size=(len(images),), fill_value=images.shape[-1] // 4, dtype=torch.long)
    return images, targets, input_lengths, label_lengths
