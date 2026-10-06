import numpy as np
import pandas as pd
import holidays
from config import TARGET, TRAIN_END, VAL_END, HORIZONS

LAGS = [0, 1, 2, 3, 6, 12, 23, 24, 47, 48, 144, 167, 168]   # 144 and 167 are needed by the weekly naive baseline
ROLL_WINDOWS = [3, 24, 168]
SUB_COLS = ["sub1_kwh", "sub2_kwh", "sub3_kwh", "rem_kwh", "voltage"]

def load_hourly(path="data/processed/hourly.parquet"):
    return pd.read_parquet(path)

def make_features(h, hourly):
    """
    One row per forecast ORIGIN time t. Features use data at or before t only.
    Target = consumption at t+h. Calendar features describe the target time t+h
    (known in advance, so no leakage).
    """
    y = hourly[TARGET]
    X = pd.DataFrame(index=hourly.index)

    # lag features (relative to origin)
    for k in LAGS:
        X[f"lag_{k}"] = y.shift(k)

    # rolling statistics, ending at the origin (inclusive)
    for w in ROLL_WINDOWS:
        X[f"rmean_{w}"] = y.rolling(w).mean()
        X[f"rstd_{w}"] = y.rolling(w).std()

    # sub-meter and voltage values at origin
    for c in SUB_COLS:
        X[f"{c}_now"] = hourly[c]

    # calendar features of the target time
    tt = hourly.index + pd.Timedelta(hours=h)
    X["hour"] = tt.hour
    X["wd"] = tt.weekday
    X["month"] = tt.month
    X["weekend"] = (tt.weekday >= 5).astype(int)
    X["h_sin"] = np.sin(2 * np.pi * tt.hour / 24)
    X["h_cos"] = np.cos(2 * np.pi * tt.hour / 24)
    X["m_sin"] = np.sin(2 * np.pi * tt.month / 12)
    X["m_cos"] = np.cos(2 * np.pi * tt.month / 12)

    fr = holidays.France(years=range(2006, 2012))
    hol_days = pd.DatetimeIndex(pd.to_datetime(list(fr.keys())))
    X["holiday"] = tt.normalize().isin(hol_days).astype(int)

    feature_cols = list(X.columns)

    # naive baselines: latest same-hour value already known at origin t
    c_day = int(np.ceil(h / 24))
    c_week = int(np.ceil(h / 168))
    X["naive_day"] = y.shift(24 * c_day - h)
    X["naive_week"] = y.shift(168 * c_week - h)

    # target and bookkeeping columns (NOT features)
    X["target"] = y.shift(-h)
    X["target_time"] = tt

    data = X.dropna()
    return data, feature_cols

def split(data):
    """
    Split by TARGET time, so the training set never contains a target
    from the validation/test period. Features of a val/test row may come
    from earlier data, which is fine (that is how real forecasting works).
    """
    t = data["target_time"]
    train = data[t <= TRAIN_END]
    val = data[(t > TRAIN_END) & (t <= VAL_END)]
    test = data[t > VAL_END]
    return train, val, test

if __name__ == "__main__":
    hourly = load_hourly()
    for h in HORIZONS:
        data, cols = make_features(h, hourly)
        tr, va, te = split(data)
        print(f"h={h:>3} | features={len(cols)} | train={len(tr):,} val={len(va):,} test={len(te):,} "
              f"| NaNs={int(data.isna().sum().sum())}")
        data.to_parquet(f"data/processed/features_h{h}.parquet")
    print("\nFeature list:\n", cols)