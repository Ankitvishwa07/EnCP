import joblib
import numpy as np
import pandas as pd
from features import make_features, split, load_hourly

hourly = load_hourly()
data, cols = make_features(24, hourly)
tr, va, te = split(data)
m = joblib.load("models/xgb_h24.pkl")

res = va["target"] - m.predict(va[cols])
q10, q90 = res.quantile(0.10), res.quantile(0.90)
pred = m.predict(te[cols])

out = pd.DataFrame({"actual": te["target"].values, "pred": pred},
                   index=pd.DatetimeIndex(te["target_time"].values).as_unit("ns"))
out["lo"] = (out["pred"] + q10).clip(lower=0)
out["hi"] = out["pred"] + q90
out.to_parquet("results/forecast_test.parquet")
cover = ((out["actual"] >= out["lo"]) & (out["actual"] <= out["hi"])).mean()
print(f"band = pred + [{q10:.3f}, {q90:.3f}] | test coverage {cover:.1%} (target about 80%)")