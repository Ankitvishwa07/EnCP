import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluate import metrics
import holidays

# ---------- 1. master table ----------
base = pd.read_csv("results/metrics_baselines_trees.csv")
rows = [{"model": r.model, "horizon": f"{r.horizon}h",
         "MAE": r.test_MAE, "RMSE": r.test_RMSE, "MAPE": r.test_MAPE} for r in base.itertuples()]
for name, f in [("LSTM", "results/metrics_lstm.csv"), ("Transformer", "results/metrics_transformer.csv")]:
    m = pd.read_csv(f, index_col=0)
    for hz, r in m.iterrows():
        rows.append({"model": name, "horizon": hz, "MAE": r.MAE, "RMSE": r.RMSE, "MAPE": r.MAPE})
master = pd.DataFrame(rows)
table = master.pivot(index="model", columns="horizon", values=["MAE", "RMSE", "MAPE"])
table = table.reindex(columns=pd.MultiIndex.from_product([["MAE", "RMSE", "MAPE"], ["1h", "24h", "168h"]]))
table.round(4).to_csv("results/comparison.csv")
print(table.round(4).to_string())

# ---------- 2. error analysis at 24h (fair for all models) ----------
t24 = pd.read_parquet("results/preds_trees_h24.parquet")
t24["target_time"] = pd.to_datetime(t24["target_time"])
t24 = t24.set_index("target_time")
t24.index = pd.DatetimeIndex(t24.index).as_unit("ns")

def deep_to_series(path):
    z = np.load(path)
    raw = z["start"]
    # saved as integers; the time unit depends on the pandas version, so detect it
    for unit in ["ns", "us", "ms", "s"]:
        start = pd.to_datetime(raw, unit=unit)
        if 2006 <= start.min().year <= 2011:
            break
    else:
        raise ValueError("could not detect time unit of saved timestamps")
    idx = pd.DatetimeIndex(start + pd.Timedelta(hours=23)).as_unit("ns")   # 24th step target time
    return pd.Series(z["pred"][:, 23], index=idx)

t24["lstm"] = deep_to_series("results/preds_lstm.npz")
t24["transformer"] = deep_to_series("results/preds_transformer.npz")
print("rows before dropna:", len(t24), "| missing LSTM:", t24["lstm"].isna().sum())
t24 = t24.dropna()
models = ["xgb", "rf", "lr", "lstm", "transformer"]

# by hour of day
err_hour = pd.DataFrame({m: (t24[m] - t24["actual"]).abs().groupby(t24.index.hour).mean() for m in models})
err_hour[["xgb", "lstm", "transformer"]].plot(kind="bar", figsize=(11, 4), width=0.85,
                                              title="24h MAE by target hour (test)")
plt.ylabel("MAE kWh"); plt.tight_layout(); plt.savefig("results/err_by_hour.png", dpi=150); plt.close()

# weekday / weekend / holiday
fr = holidays.France(years=range(2010, 2011))
hol = pd.DatetimeIndex(pd.to_datetime(list(fr.keys())))
day_type = np.where(t24.index.normalize().isin(hol), "holiday",
                    np.where(t24.index.weekday >= 5, "weekend", "weekday"))
by_type = pd.DataFrame({m: (t24[m] - t24["actual"]).abs().groupby(day_type).mean() for m in models})
print("\n24h MAE by day type:\n", by_type.round(4))
by_type.to_csv("results/err_by_daytype.csv")

# peaks (top 5% of actual) vs the rest
thr = t24["actual"].quantile(0.95)
peak = t24["actual"] >= thr
rows = []
for m in models:
    e = t24[m] - t24["actual"]
    rows.append({"model": m, "MAE_peak": e[peak].abs().mean(), "MAE_other": e[~peak].abs().mean(),
                 "bias_peak": e[peak].mean()})
pk = pd.DataFrame(rows).set_index("model")
print("\nPeak (top 5%) vs other hours, 24h:\n", pk.round(4))
pk.to_csv("results/err_peaks.csv")

# 10 worst days for the best model
best = "xgb"
daily_err = (t24[best] - t24["actual"]).abs().groupby(t24.index.date).mean().sort_values(ascending=False)
print("\n10 worst days:\n", daily_err.head(10).round(3))
fig, axes = plt.subplots(5, 2, figsize=(13, 14))
for ax, day in zip(axes.ravel(), daily_err.head(10).index):
    d = t24[t24.index.date == day]
    ax.plot(d.index.hour, d["actual"], label="actual"); ax.plot(d.index.hour, d[best], label=best)
    ax.set_title(str(day)); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("results/worst_days.png", dpi=130); plt.close()