import string
import torch
# --- Character Mapping ---
CHARS = string.ascii_lowercase + string.digits
char2idx = {char: i for i, char in enumerate(CHARS)}
idx2char = {i: char for char, i in char2idx.items()}
BLANK_IDX = len(CHARS)  # 36

# --- Utility Functions ---
def encode_label(text):
    return [char2idx[c] for c in text.lower() if c in char2idx]

def decode_output(output):
    pred_indices = torch.argmax(output, dim=2).permute(1, 0)  # (B, T)
    decoded = []
    for seq in pred_indices:
        chars = []
        prev = -1
        for idx in seq:
            idx = idx.item()
            if idx != prev and idx != BLANK_IDX:
                chars.append(idx2char.get(idx, ''))
            prev = idx
        decoded.append(''.join(chars))
    return decoded