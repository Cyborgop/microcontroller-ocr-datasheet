import torch.nn as nn
# --- Model Definition ---
class BidirectionalLSTM(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.rnn = nn.LSTM(in_features, hidden_size, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(hidden_size * 2, out_features)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.embedding(x)
        return x

class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes):
        super().__init__()
        assert img_height == 32, "Expected image height 32"
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1), (2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2,1), (2,1))
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, num_classes)
        )

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1 or h == 2, f"Expected height=1 or 2, got {h}"
        conv = conv.squeeze(2) if h == 1 else conv.mean(2)
        conv = conv.permute(0, 2, 1)
        out = self.rnn(conv)
        return out