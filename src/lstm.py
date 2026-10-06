import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from deep_common import *
from features import load_hourly
from evaluate import metrics

class LSTMNet(nn.Module):
    def __init__(self, n_in, hid=64, out=H, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hid, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hid, out)
    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1])

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    set_seed()
    hourly = load_hourly()
    df, scaler, tr, va, te = prepare(hourly)
    print("windows:", len(tr[0]), len(va[0]), len(te[0]), "| device:", DEVICE)

    model = LSTMNet(n_in=tr[0].shape[2])
    model, best = train_model(model, tr, va, scaler, epochs=50, lr=1e-3, wd=0.0)
    torch.save(model.state_dict(), "models/lstm.pt")
    joblib.dump(scaler, "models/deep_scaler.pkl")

    pred = predict(model, te, scaler)            # shape (n_windows, 24), in kWh
    actual = te[1]

    # step 1 -> "1h" result, step 24 -> "24h" result
    res = {}
    for name, k in [("1h", 0), ("24h", 23)]:
        res[name] = metrics(actual[:, k], pred[:, k])
    print("\nLSTM test metrics")
    print(pd.DataFrame(res).T.round(4))
    print("all 24 steps MAE:", round(float(np.mean(np.abs(actual - pred))), 4))

    pd.DataFrame(res).T.to_csv("results/metrics_lstm.csv")
    start_times = df.index[te[2]]
    np.savez("results/preds_lstm.npz", pred=pred, actual=actual, start=start_times.values.astype("int64"))