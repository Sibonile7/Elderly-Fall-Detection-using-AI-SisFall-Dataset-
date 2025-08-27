# src/visualize_results.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
OUT  = ROOT

def must_load(p: Path, name: str):
    if not p.exists():
        print(f"[ERROR] Missing {name} at: {p}")
        print("Run:  python src/eval_test.py  first to generate it.")
        sys.exit(1)
    return np.load(p)

def save_confusions(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ADL','Fall'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix (Counts) – Test')
    plt.tight_layout(); plt.savefig(OUT / 'confusion_counts.png', dpi=160); plt.close()

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['ADL','Fall'])
    disp.plot(cmap=plt.cm.Blues, values_format='.2f')
    plt.title('Confusion Matrix (Normalized) – Test')
    plt.tight_layout(); plt.savefig(OUT / 'confusion_normalized.png', dpi=160); plt.close()

def save_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('ROC Curve – Test'); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / 'roc_curve.png', dpi=160); plt.close()

def save_pr(y_true, y_prob):
    prec, rec, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(); plt.plot(rec, prec, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision–Recall Curve – Test'); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / 'precision_recall_curve.png', dpi=160); plt.close()

def save_threshold_sweep(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 17)
    precs, recs, accs = [], [], []
    for th in thresholds:
        yhat = (y_prob >= th).astype(int)
        tp = np.sum((y_true==1) & (yhat==1))
        fp = np.sum((y_true==0) & (yhat==1))
        fn = np.sum((y_true==1) & (yhat==0))
        tn = np.sum((y_true==0) & (yhat==0))
        prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1); acc = (tp+tn) / len(y_true)
        precs.append(prec); recs.append(rec); accs.append(acc)
    import matplotlib.pyplot as plt
    plt.figure(); plt.plot(thresholds, precs, label='Precision')
    plt.plot(thresholds, recs, label='Recall'); plt.plot(thresholds, accs, label='Accuracy')
    plt.xlabel('Decision Threshold (Fall probability)'); plt.ylabel('Score')
    plt.title('Threshold vs Precision/Recall/Accuracy – Test'); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / 'threshold_sweep.png', dpi=160); plt.close()

def main():
    y_true = must_load(PROC / 'y_test.npy', 'y_test.npy')
    y_pred = must_load(PROC / 'y_pred.npy', 'y_pred.npy')
    y_prob = must_load(PROC / 'y_prob.npy', 'y_prob.npy')

    save_confusions(y_true, y_pred)
    save_roc(y_true, y_prob)
    save_pr(y_true, y_prob)
    save_threshold_sweep(y_true, y_prob)

    print("Saved: confusion_counts.png, confusion_normalized.png, roc_curve.png, precision_recall_curve.png, threshold_sweep.png")

if __name__ == '__main__':
    main()
