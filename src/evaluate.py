import numpy as np
import pandas as pd

def metrics(y, p, eps=1e-3):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAPE": float(np.mean(np.abs((y - p) / np.maximum(np.abs(y), eps))) * 100),
    }

def rolling_origin(data, cols, make_model, n_folds=5, test_days=30):
    """
    Expanding-window CV. Splits by TARGET time so training never sees a
    target from the fold's test period. Returns mean metrics over folds.
    make_model: function returning a fresh, unfitted sklearn-style model.
    """
    t = data["target_time"]
    end = t.max()
    results = []
    for i in range(n_folds, 0, -1):
        te_end = end - pd.Timedelta(days=test_days * (i - 1))
        te_start = te_end - pd.Timedelta(days=test_days)
        tr = data[t <= te_start]
        te = data[(t > te_start) & (t <= te_end)]
        m = make_model()
        m.fit(tr[cols], tr["target"])
        results.append(metrics(te["target"], m.predict(te[cols])))
    return pd.DataFrame(results).mean().to_dict()

def rolling_origin_naive(data, col, n_folds=5, test_days=30):
    """Same folds for a naive baseline (no training needed)."""
    t = data["target_time"]
    end = t.max()
    results = []
    for i in range(n_folds, 0, -1):
        te_end = end - pd.Timedelta(days=test_days * (i - 1))
        te_start = te_end - pd.Timedelta(days=test_days)
        te = data[(t > te_start) & (t <= te_end)]
        results.append(metrics(te["target"], te[col]))
    return pd.DataFrame(results).mean().to_dict()