# src/split_subjects.py
import re, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

X_PATH = Path('data/processed/X_windows.npy')
Y_PATH = Path('data/processed/y_labels.npy')
META   = Path('data/processed/meta.parquet')
OUTDIR = Path('data/processed'); OUTDIR.mkdir(parents=True, exist_ok=True)

# Load meta to get filepaths per window
meta = pd.read_parquet(META)  # columns: filepath, win_idx

def subject_id(fp: str) -> str:
    s = fp.replace('\\','/')
    m = re.search(r'/(S[AE]\d{2})/', s)
    return m.group(1) if m else 'UNK'

subjects = meta['filepath'].map(subject_id).values
n = len(subjects)
idx_all = np.arange(n)

# 60/20/20 split by SUBJECT
gss1 = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
train_idx, temp_idx = next(gss1.split(idx_all, groups=subjects))
gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=43)  # split the remaining half -> val/test
val_idx, test_idx = next(gss2.split(temp_idx, groups=subjects[temp_idx]))

# Map val/test back to absolute indices
val_idx  = temp_idx[val_idx]
test_idx = temp_idx[test_idx]

# Save to disk
np.save(OUTDIR/'idx_train.npy', train_idx)
np.save(OUTDIR/'idx_val.npy',   val_idx)
np.save(OUTDIR/'idx_test.npy',  test_idx)

# Small report
def uniq_subs(idxs): return sorted(set(subjects[idxs].tolist()))
print('Subjects (train):', uniq_subs(train_idx))
print('Subjects (val):  ', uniq_subs(val_idx))
print('Subjects (test): ', uniq_subs(test_idx))
print('Sizes -> train/val/test:', len(train_idx), len(val_idx), len(test_idx))
