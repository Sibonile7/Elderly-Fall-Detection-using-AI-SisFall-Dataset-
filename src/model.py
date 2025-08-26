import torch
import torch.nn as nn

class CNNGRUFall(nn.Module):
    """
    Simple, fast model for windowed IMU:
    - 1D CNN feature extractor
    - Bi-GRU temporal modeling
    - Average pooling over time -> classifier
    Input: (B, C, T), C=9 by default
    """
    def __init__(self, in_ch=9, n_classes=2):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),    nn.ReLU(), nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),    nn.ReLU(), nn.BatchNorm1d(96),
            nn.MaxPool1d(2),
        )
        self.gru = nn.GRU(
            input_size=96, hidden_size=96, num_layers=1,
            batch_first=True, bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Linear(192, 96), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(96, n_classes)
        )

    def forward(self, x):
        # x: (B, C, T)
        z = self.fe(x)            # (B, 96, T')
        z = z.transpose(1, 2)     # (B, T', 96)
        z, _ = self.gru(z)        # (B, T', 192)
        z = z.mean(dim=1)         # temporal average pooling
        out = self.head(z)        # (B, n_classes)
        return out
