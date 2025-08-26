import os, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

from dataset import SisFallWindows
from model import CNNGRUFall

X_PATH = 'data/processed/X_windows.npy'
Y_PATH = 'data/processed/y_labels.npy'
IDX_TRAIN = 'data/processed/idx_train.npy'
IDX_VAL   = 'data/processed/idx_val.npy'
IDX_TEST  = 'data/processed/idx_test.npy'
SAVE = 'models/best_model.pt'

EPOCHS = 15
BS = 256
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

def train_one(model, train_dl, val_dl, crit):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    best_f1, best = 0.0, None
    train_losses, val_f1s = [], []
    patience, wait = 3, 0  # early stopping

    for epoch in range(1, EPOCHS+1):
        model.train()
        tot = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tot += loss.item()
        tr_loss = tot / max(1, len(train_dl))

        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(DEVICE)
                logits = model(xb)
                yp.append(torch.argmax(logits,1).cpu().numpy())
                yt.append(yb.numpy())
        yt = np.concatenate(yt); yp = np.concatenate(yp)
        f1 = f1_score(yt, yp)

        train_losses.append(tr_loss); val_f1s.append(f1)
        print(f"epoch {epoch:02d} loss={tr_loss:.3f} f1_val={f1:.3f}")

        if f1 > best_f1:
            best_f1, best = f1, model.state_dict(); wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    return best, best_f1, train_losses, val_f1s

def main():
    # Load splits
    tr_idx = np.load(IDX_TRAIN); va_idx = np.load(IDX_VAL); te_idx = np.load(IDX_TEST)

    # Datasets & loaders
    tr_ds = SisFallWindows(X_PATH, Y_PATH, idx=tr_idx)
    va_ds = SisFallWindows(X_PATH, Y_PATH, idx=va_idx)
    te_ds = SisFallWindows(X_PATH, Y_PATH, idx=te_idx)
    tr_dl = DataLoader(tr_ds, batch_size=BS, shuffle=True,  num_workers=2)
    va_dl = DataLoader(va_ds, batch_size=BS, shuffle=False, num_workers=2)
    te_dl = DataLoader(te_ds, batch_size=BS, shuffle=False, num_workers=2)

    # Optional: class weights if falls are minority
    y_all = np.load(Y_PATH)
    neg, pos = int((y_all==0).sum()), int((y_all==1).sum())
    w = torch.tensor([1.0, neg/max(pos,1)], dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)

    model = CNNGRUFall(in_ch=9, n_classes=2)
    state, best_f1, train_losses, val_f1s = train_one(model, tr_dl, va_dl, crit)

    os.makedirs('models', exist_ok=True)
    torch.save({'state_dict': state}, SAVE)
    print('Saved best model to', SAVE)

    # ---- Test evaluation ----
    model = CNNGRUFall(in_ch=9, n_classes=2).to(DEVICE)
    model.load_state_dict(state)
    yt, yp = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(DEVICE)
            logits = model(xb)
            yp.append(torch.argmax(logits,1).cpu().numpy())
            yt.append(yb.numpy())
    yt = np.concatenate(yt); yp = np.concatenate(yp)

    print('\nTEST report:')
    print(classification_report(yt, yp, digits=3))
    acc = accuracy_score(yt, yp)
    print(f"TEST Accuracy: {acc:.3f}")

    cm = confusion_matrix(yt, yp)
    print("TEST Confusion Matrix:\n", cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ADL','Fall'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (TEST)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_test.png', dpi=160); plt.close()

    # Curves (from last training)
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_f1s, label='Val F1')
    plt.legend(); plt.title('Training vs Validation')
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=160); plt.close()

if __name__ == '__main__':
    main()
