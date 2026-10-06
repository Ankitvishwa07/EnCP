import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import itertools

from deep_common import *
from features import load_hourly
from evaluate import metrics

class PosEnc(nn.Module):
    def __init__(self, d, maxlen=500):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransNet(nn.Module):
    def __init__(self, n_in, d=64, layers=2, dropout=0.1, out=H):
        super().__init__()
        self.proj = nn.Linear(n_in, d)
        self.pos = PosEnc(d)
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d,
                                           dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d, out)
    def forward(self, x):
        return self.head(self.enc(self.pos(self.proj(x)))[:, -1])

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    set_seed()
    hourly = load_hourly()
    df, scaler, tr, va, te = prepare(hourly)
    n_in = tr[0].shape[2]

    # small grid, chosen on VALIDATION loss only (16 runs; reduce if slow)
    grid = list(itertools.product([32, 64], [1, 2], [0.1, 0.3], [1e-3, 5e-4]))
    best_loss, best_cfg, best_state = 1e9, None, None
    for d, nl, dr, lr in grid:
        set_seed()
        m = TransNet(n_in, d=d, layers=nl, dropout=dr)
        m, vloss = train_model(m, tr, va, scaler, epochs=30, lr=lr, wd=1e-4, patience=5, verbose=False)
        print(f"d={d} layers={nl} dropout={dr} lr={lr} -> val loss {vloss:.4f}")
        if vloss < best_loss:
            best_loss, best_cfg = vloss, (d, nl, dr, lr)
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
    print("\nbest config (d, layers, dropout, lr):", best_cfg, "| val loss", round(best_loss, 4))

    d, nl, dr, lr = best_cfg
    model = TransNet(n_in, d=d, layers=nl, dropout=dr)
    model.load_state_dict(best_state)
    model.to(DEVICE)
    torch.save(model.state_dict(), "models/transformer.pt")

    pred = predict(model, te, scaler)
    actual = te[1]
    res = {name: metrics(actual[:, k], pred[:, k]) for name, k in [("1h", 0), ("24h", 23)]}
    print("\nTransformer test metrics")
    print(pd.DataFrame(res).T.round(4))
    print("all 24 steps MAE:", round(float(np.mean(np.abs(actual - pred))), 4))

    pd.DataFrame(res).T.to_csv("results/metrics_transformer.csv")
    np.savez("results/preds_transformer.npz", pred=pred, actual=actual,
             start=df.index[te[2]].values.astype("int64"))