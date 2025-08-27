# src/eval_test.py
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import SisFallWindows
from model import CNNGRUFall

ROOT = Path(__file__).resolve().parents[1]          # project root
PROC = ROOT / 'data' / 'processed'
X_PATH   = PROC / 'X_windows.npy'
Y_PATH   = PROC / 'y_labels.npy'
IDX_TEST = PROC / 'idx_test.npy'
MODEL    = ROOT / 'models' / 'best_model.pt'        # change to best_model_leaky.pt if needed
OUT_CM   = ROOT / 'confusion_matrix_test.png'

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

def main():
    PROC.mkdir(parents=True, exist_ok=True)

    te_idx = np.load(IDX_TEST)
    te_ds  = SisFallWindows(str(X_PATH), str(Y_PATH), idx=te_idx)
    te_dl  = DataLoader(te_ds, batch_size=256, shuffle=False, num_workers=2)

    m = CNNGRUFall(in_ch=9, n_classes=2).to(DEVICE)
    state = torch.load(MODEL, map_location=DEVICE)
    m.load_state_dict(state['state_dict'])
    m.eval()

    yt, yp, yprob = [], [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(DEVICE)
            logits = m(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            yprob.append(probs[:, 1])
            yp.append(preds)
            yt.append(yb.numpy())
    yt = np.concatenate(yt); yp = np.concatenate(yp); yprob = np.concatenate(yprob)

    # Save arrays for visualization
    y_test_path = PROC / 'y_test.npy'
    y_pred_path = PROC / 'y_pred.npy'
    y_prob_path = PROC / 'y_prob.npy'
    np.save(y_test_path, yt)
    np.save(y_pred_path, yp)
    np.save(y_prob_path, yprob)
    print(f"Saved: {y_test_path}")
    print(f"Saved: {y_pred_path}")
    print(f"Saved: {y_prob_path}")

    # Quick metrics + confusion matrix image
    print('\nTEST report:')
    print(classification_report(yt, yp, digits=3))
    acc = accuracy_score(yt, yp); print(f'TEST Accuracy: {acc:.3f}')

    cm = confusion_matrix(yt, yp); print('TEST Confusion Matrix:\n', cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ADL','Fall'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix (TEST)')
    plt.tight_layout()
    plt.savefig(OUT_CM, dpi=160)
    print(f"Saved figure: {OUT_CM}")

if __name__ == '__main__':
    main()
