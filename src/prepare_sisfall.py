import re
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path('data/raw/SisFall')
OUT_DIR = Path('data/processed')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Patterns: Fxx = Fall, Dxx = Daily activity
FALL_PAT = re.compile(r'(?:^|[^A-Za-z])F\d{2}(?:[^A-Za-z]|$)', re.IGNORECASE)
ADL_PAT  = re.compile(r'(?:^|[^A-Za-z])D\d{2}(?:[^A-Za-z]|$)', re.IGNORECASE)

WINDOW = 256   # ~1.28s at 200 Hz
STRIDE = 64    # ~0.32s hop

import re

def read_trial(path):
    data = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Split on commas/semicolons/whitespace
            parts = re.split(r"[,\s;]+", s)
            # Filter numeric tokens and convert to float
            nums = []
            for x in parts:
                if x == "":
                    continue
                try:
                    nums.append(float(x))
                except ValueError:
                    pass
            if len(nums) < 6:
                continue
            # Keep first 9 channels, pad to 9
            nums = nums[:9] + [0.0] * max(0, 9 - len(nums[:9]))
            data.append(nums)
    return np.asarray(data, dtype=np.float32)

def windows_from_signal(sig, window=WINDOW, stride=STRIDE):
    n = sig.shape[0]
    if n < window:
        return np.empty((0, window, sig.shape[1]), dtype=np.float32)
    idxs = np.arange(0, n - window + 1, stride)
    out = np.stack([sig[i:i+window] for i in idxs])
    return out

def infer_label_from_path(p: Path):
    s = str(p)
    if FALL_PAT.search(s):
        return 1
    if ADL_PAT.search(s):
        return 0
    return 1 if '/F' in s or '\\F' in s else 0

def main():
    files = sorted(glob.glob(str(RAW_DIR / '**' / '*.txt'), recursive=True))
    X_list, y_list, meta = [], [], []
    for fp in files:
        arr = read_trial(fp)
        if arr.size == 0:
            continue
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0) + 1e-6
        arr = (arr - mu) / sd
        win = windows_from_signal(arr)
        if win.shape[0] == 0:
            continue
        label = infer_label_from_path(Path(fp))
        y = np.full((win.shape[0],), label, dtype=np.int64)
        X_list.append(win)
        y_list.append(y)
        meta += [(fp, i) for i in range(win.shape[0])]
    if not X_list:
        print("No data found! Did you put SisFall files into data/raw/SisFall/?")
        return
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    np.save(OUT_DIR / 'X_windows.npy', X)
    np.save(OUT_DIR / 'y_labels.npy', y)
    pd.DataFrame(meta, columns=['filepath', 'win_idx']).to_parquet(OUT_DIR / 'meta.parquet')
    print('Saved:', X.shape, y.shape)

if __name__ == '__main__':
    main()
