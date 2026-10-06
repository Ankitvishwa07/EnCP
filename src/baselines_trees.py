import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import HORIZONS
from features import make_features, split, load_hourly
from evaluate import metrics, rolling_origin, rolling_origin_naive

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
hourly = load_hourly()

def make_lr():  return make_pipeline(StandardScaler(), LinearRegression())
def make_rf():  return RandomForestRegressor(n_estimators=200, max_depth=18, min_samples_leaf=3,
                                             n_jobs=-1, random_state=0)
def make_xgb(depth=6, lr=0.03, n=400):
    return XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, subsample=0.8,
                        colsample_bytree=0.8, n_jobs=-1, random_state=0)

rows = []
for h in HORIZONS:
    print(f"\n===== horizon {h}h =====")
    data, cols = make_features(h, hourly)
    tr, va, te = split(data)
    Xtr, ytr, Xva, yva, Xte, yte = tr[cols], tr["target"], va[cols], va["target"], te[cols], te["target"]

    # --- naive baselines
    for name, col in [("Naive yesterday", "naive_day"), ("Naive last week", "naive_week")]:
        r = {"model": name, "horizon": h, **{f"test_{k}": v for k, v in metrics(yte, te[col]).items()}}
        r.update({f"val_{k}": v for k, v in metrics(yva, va[col]).items()})
        r.update({f"cv_{k}": v for k, v in rolling_origin_naive(data, col).items()})
        rows.append(r)

    # --- linear regression
    lr_m = make_lr().fit(Xtr, ytr)
    r = {"model": "Linear Regression", "horizon": h,
         **{f"val_{k}": v for k, v in metrics(yva, lr_m.predict(Xva)).items()},
         **{f"test_{k}": v for k, v in metrics(yte, lr_m.predict(Xte)).items()},
         **{f"cv_{k}": v for k, v in rolling_origin(data, cols, make_lr).items()}}
    rows.append(r)

    # --- random forest
    rf_m = make_rf().fit(Xtr, ytr)
    r = {"model": "Random Forest", "horizon": h,
         **{f"val_{k}": v for k, v in metrics(yva, rf_m.predict(Xva)).items()},
         **{f"test_{k}": v for k, v in metrics(yte, rf_m.predict(Xte)).items()},
         **{f"cv_{k}": v for k, v in rolling_origin(data, cols, make_rf, n_folds=3).items()}}
    rows.append(r)

    # --- XGBoost: pick settings on VALIDATION only
    best, best_mae = None, 1e9
    for depth in (4, 6, 8):
        for lr in (0.03, 0.06):
            m = make_xgb(depth, lr).fit(Xtr, ytr)
            mae = metrics(yva, m.predict(Xva))["MAE"]
            if mae < best_mae:
                best, best_mae, best_cfg = m, mae, (depth, lr)
    print("best XGB (depth, lr):", best_cfg, "| val MAE", round(best_mae, 4))
    d, l = best_cfg
    r = {"model": "XGBoost", "horizon": h,
         **{f"val_{k}": v for k, v in metrics(yva, best.predict(Xva)).items()},
         **{f"test_{k}": v for k, v in metrics(yte, best.predict(Xte)).items()},
         **{f"cv_{k}": v for k, v in rolling_origin(data, cols, lambda: make_xgb(d, l), n_folds=3).items()}}
    rows.append(r)

    # --- save models, test predictions, feature importance
    joblib.dump(best, f"models/xgb_h{h}.pkl")
    joblib.dump(rf_m, f"models/rf_h{h}.pkl")
    joblib.dump(lr_m, f"models/lr_h{h}.pkl")
    pd.DataFrame({"target_time": te["target_time"], "actual": yte,
                  "xgb": best.predict(Xte), "rf": rf_m.predict(Xte),
                  "lr": lr_m.predict(Xte), "naive_day": te["naive_day"],
                  "naive_week": te["naive_week"]}).to_parquet(f"results/preds_trees_h{h}.parquet")
    imp = pd.Series(best.feature_importances_, index=cols).sort_values().tail(15)
    imp.plot(kind="barh", figsize=(7, 5), title=f"XGBoost feature importance (h={h})")
    plt.tight_layout(); plt.savefig(f"results/xgb_importance_h{h}.png", dpi=150); plt.close()

res = pd.DataFrame(rows)
res.to_csv("results/metrics_baselines_trees.csv", index=False)

print("\n=== TEST MAE (kWh/hour) ===")
print(res.pivot(index="model", columns="horizon", values="test_MAE").round(4))
print("\n=== TEST RMSE ===")
print(res.pivot(index="model", columns="horizon", values="test_RMSE").round(4))