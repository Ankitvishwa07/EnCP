import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import holidays
from sklearn.preprocessing import StandardScaler
from config import TARGET, TRAIN_END, VAL_END

L = 48          # input window (hours)
H = 24          # output window (hours)
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(s=SEED):
    np.random.seed(s); torch.manual_seed(s)

def build_frame(hourly):
    """Per-hour table: scaled-later target + calendar covariates."""
    df = pd.DataFrame(index=hourly.index)
    df["y"] = hourly[TARGET]
    df["h_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["h_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["wd_sin"] = np.sin(2 * np.pi * df.index.weekday / 7)
    df["wd_cos"] = np.cos(2 * np.pi * df.index.weekday / 7)
    df["weekend"] = (df.index.weekday >= 5).astype(float)
    fr = holidays.France(years=range(2006, 2012))
    hd = pd.DatetimeIndex(pd.to_datetime(list(fr.keys())))
    df["holiday"] = df.index.normalize().isin(hd).astype(float)
    return df

def make_windows(df_scaled, y_raw, start_idx, end_idx, L=L, H=H):
    """
    Window i: inputs = rows [i-L, i), targets = y[i : i+H].
    Targets are restricted to [start_idx, end_idx) so each split only
    predicts its own period (inputs may look back into earlier data).
    """
    arr = df_scaled.values.astype(np.float32)
    X, Y, T = [], [], []
    for i in range(max(start_idx, L), end_idx - H + 1):
        X.append(arr[i - L:i])
        Y.append(y_raw[i:i + H])
        T.append(i)
    return np.stack(X), np.stack(Y).astype(np.float32), np.array(T)

def prepare(hourly, L=L, H=H):
    df = build_frame(hourly)
    n_train = int((df.index <= TRAIN_END).sum())
    n_val_end = int((df.index <= VAL_END).sum())
    n = len(df)

    scaler = StandardScaler().fit(df.iloc[:n_train][["y"]])         # target scaler (train only)
    df_s = df.copy()
    df_s["y"] = scaler.transform(df[["y"]])                         # other columns are already in [-1, 1]
    y_raw = df["y"].values

    tr = make_windows(df_s, y_raw, 0, n_train, L, H)
    va = make_windows(df_s, y_raw, n_train, n_val_end, L, H)
    te = make_windows(df_s, y_raw, n_val_end, n, L, H)
    return df, scaler, tr, va, te

class EarlyStopper:
    def __init__(self, patience=7):
        self.best, self.bad, self.patience, self.state = 1e9, 0, patience, None
    def step(self, loss, model):
        if loss < self.best - 1e-5:
            self.best, self.bad = loss, 0
            self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            self.bad += 1
        return self.bad >= self.patience

def train_model(model, tr, va, scaler, epochs=50, lr=1e-3, wd=0.0, bs=64, patience=7, verbose=True):
    """Trains on scaled targets; windows hold raw y, so scale them here."""
    mu, sd = float(scaler.mean_[0]), float(scaler.scale_[0])
    Xtr, Ytr = torch.tensor(tr[0]), torch.tensor((tr[1] - mu) / sd)
    Xva, Yva = torch.tensor(va[0]).to(DEVICE), torch.tensor((va[1] - mu) / sd).to(DEVICE)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.HuberLoss()
    stopper = EarlyStopper(patience)
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr[idx].to(DEVICE), Ytr[idx].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vloss = sum(loss_fn(model(Xva[j:j + 512]), Yva[j:j + 512]).item() * len(Xva[j:j + 512])
                        for j in range(0, len(Xva), 512)) / len(Xva)
        if verbose:
            print(f"epoch {ep+1:>2} | train {tot/n:.4f} | val {vloss:.4f}")
        if stopper.step(vloss, model):
            if verbose: print("early stop")
            break
    model.load_state_dict(stopper.state)
    return model, stopper.best

def predict(model, data, scaler):
    mu, sd = float(scaler.mean_[0]), float(scaler.scale_[0])
    model.eval()
    X = torch.tensor(data[0])
    out = []
    with torch.no_grad():
        for j in range(0, len(X), 512):
            out.append(model(X[j:j + 512].to(DEVICE)).cpu().numpy())
    return np.concatenate(out) * sd + mu            # back to kWh