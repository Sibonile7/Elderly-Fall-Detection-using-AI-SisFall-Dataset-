import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

from dataset import SisFallWindows
from model import CNNGRUFall

X_PATH = 'data/processed/X_windows.npy'
Y_PATH = 'data/processed/y_labels.npy'
SAVE = 'models/best_model.pt'

EPOCHS = 15
BS = 256
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

def train_one(model, train_dl, val_dl):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    best_f1, best = 0.0, None
    train_losses, val_f1s = [], []

    for epoch in range(1, EPOCHS+1):
        # ---- Training ----
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)
        train_losses.append(avg_loss)

        # ---- Validation ----
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(DEVICE)
                logits = model(xb)
                yp.append(torch.argmax(logits, 1).cpu().numpy())
                yt.append(yb.numpy())
        yt = np.concatenate(yt); yp = np.concatenate(yp)
        from sklearn.metrics import f1_score
        f1 = f1_score(yt, yp)
        val_f1s.append(f1)

        print(f"epoch {epoch:02d} loss={avg_loss:.3f} f1={f1:.3f}")

        if f1 > best_f1:
            best_f1, best = f1, model.state_dict()

    return best, best_f1, train_losses, val_f1s

def main():
    ds_full = SisFallWindows(X_PATH, Y_PATH)
    y = np.load(Y_PATH)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_overall, best_score = None, -1
    history = []

    for fold, (tr, va) in enumerate(skf.split(np.zeros_like(y), y), 1):
        print(f"\n=== Fold {fold}/5 ===")
        tr_ds = SisFallWindows(X_PATH, Y_PATH, idx=tr)
        va_ds = SisFallWindows(X_PATH, Y_PATH, idx=va)
        tr_dl = DataLoader(tr_ds, batch_size=BS, shuffle=True, num_workers=2)
        va_dl = DataLoader(va_ds, batch_size=BS, shuffle=False, num_workers=2)

        model = CNNGRUFall(in_ch=9, n_classes=2)
        state, f1, train_losses, val_f1s = train_one(model, tr_dl, va_dl)
        history.append((train_losses, val_f1s))

        if f1 > best_score:
            best_score = f1
            best_overall = state

    os.makedirs('models', exist_ok=True)
    torch.save({'state_dict': best_overall}, SAVE)
    print('Saved best model to', SAVE)

    # ---- Final evaluation on last fold ----
    model = CNNGRUFall(in_ch=9, n_classes=2).to(DEVICE)
    model.load_state_dict(best_overall)
    va_loader = DataLoader(va_ds, batch_size=BS)
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in va_loader:
            xb = xb.to(DEVICE)
            yp.append(torch.argmax(model(xb), 1).cpu().numpy())
            yt.append(yb.numpy())
    yt = np.concatenate(yt); yp = np.concatenate(yp)

    print('\nValidation report (last fold):')
    print(classification_report(yt, yp, digits=3))

    acc = accuracy_score(yt, yp)
    print(f"Validation Accuracy: {acc:.3f}")
    cm = confusion_matrix(yt, yp)
    print("Confusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ADL", "Fall"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Validation)")
    plt.show()

    # ---- Plot training curves (last fold) ----
    train_losses, val_f1s = history[-1]
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_f1s, label="Val F1")
    plt.legend()
    plt.title("Training vs Validation")
    plt.xlabel("Epoch")
    plt.show()

if __name__ == '__main__':
    main()
