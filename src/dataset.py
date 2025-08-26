import numpy as np
import torch
from torch.utils.data import Dataset

class SisFallWindows(Dataset):
    """
    Loads windowed SisFall data saved by prepare_sisfall.py:
      X_windows.npy : shape (N, T, C)  e.g., (N, 256, 9)
      y_labels.npy  : shape (N,)
    Returns tensors shaped (C, T) for PyTorch 1D convs.
    """
    def __init__(self, x_path, y_path, idx=None, dtype=torch.float32):
        self.X = np.load(x_path, mmap_mode='r')   # (N, T, C)
        self.y = np.load(y_path)                  # (N,)
        if idx is None:
            self.idx = np.arange(self.y.shape[0])
        else:
            self.idx = np.asarray(idx)
        self.dtype = dtype

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, i):
        j = self.idx[i]
        x = self.X[j]                               # (T, C)
        x = torch.from_numpy(x).to(self.dtype).transpose(0, 1)  # -> (C, T)
        y = int(self.y[j])
        return x, y
