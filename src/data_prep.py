import numpy as np
import pandas as pd

RAW = "data/raw/household_power_consumption.txt"
SHORT_GAP_MIN = 180          # gaps <= 3 hours are interpolated
WEEK = 7 * 24 * 60           # minutes in a week

# ---------- 1. Load and parse ----------
df = pd.read_csv(RAW, sep=";", na_values=["?", ""], low_memory=False)
df.index = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
df = df.drop(columns=["Date", "Time"]).astype(float)
df.columns = ["gap", "grp", "voltage", "intensity", "sub1", "sub2", "sub3"]
df = df[~df.index.duplicated(keep="first")].sort_index()

# ---------- 2. Expose missing minutes ----------
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="min")
df = df.reindex(full_idx)
print(f"Rows: {len(df):,} | Missing (gap col): {df['gap'].isna().sum():,} "
      f"({df['gap'].isna().mean()*100:.2f}%)")

# ---------- 3. Gap statistics (for your report) ----------
isna = df["gap"].isna()
run_id = (isna != isna.shift()).cumsum()
runs = isna.groupby(run_id).agg(["sum", "first"])
gap_lengths = runs.loc[runs["first"], "sum"]            # lengths of NaN runs only
print(f"Number of gaps: {len(gap_lengths)}")
print(f"  short (<= {SHORT_GAP_MIN} min): {(gap_lengths <= SHORT_GAP_MIN).sum()}")
print(f"  long  (>  {SHORT_GAP_MIN} min): {(gap_lengths > SHORT_GAP_MIN).sum()}")
print(f"  longest gap: {gap_lengths.max()} min ({gap_lengths.max()/1440:.1f} days)")
gap_lengths.describe().to_csv("results/gap_stats.csv")

# remember which minutes were imputed (used later for EDA plot)
df["imputed"] = isna.astype(float)

# ---------- 4. Imputation ----------
def impute(s: pd.Series) -> pd.Series:
    na = s.isna()
    rid = (na != na.shift()).cumsum()
    run_len = na.groupby(rid).transform("sum")
    short_mask = na & (run_len <= SHORT_GAP_MIN)

    out = s.copy()
    interp = s.interpolate(method="linear", limit_area="inside")
    out[short_mask] = interp[short_mask]                 # short gaps: interpolation

    for _ in range(4):                                   # long gaps: same time last week
        m = out.isna()
        if not m.any():
            break
        out[m] = out.shift(WEEK)[m]

    return out.ffill().bfill()                           # last-resort edge cases

value_cols = ["gap", "grp", "voltage", "intensity", "sub1", "sub2", "sub3"]
for c in value_cols:
    df[c] = impute(df[c])
assert df[value_cols].isna().sum().sum() == 0, "NaNs remain after imputation"

# ---------- 5. Derived features ----------
df["active_wh"] = df["gap"] * 1000 / 60                  # kW -> Wh per minute
df["sub_rem"] = (df["active_wh"] - df["sub1"] - df["sub2"] - df["sub3"]).clip(lower=0)

# ---------- 6. Resample ----------
sum_cols = ["active_wh", "sub1", "sub2", "sub3", "sub_rem"]
mean_cols = ["gap", "grp", "voltage", "intensity", "imputed"]

def resample(rule: str) -> pd.DataFrame:
    energy = df[sum_cols].resample(rule).sum() / 1000    # Wh -> kWh
    other = df[mean_cols].resample(rule).mean()
    out = pd.concat([energy, other], axis=1)
    return out.rename(columns={
        "active_wh": "active_kwh", "sub1": "sub1_kwh", "sub2": "sub2_kwh",
        "sub3": "sub3_kwh", "sub_rem": "rem_kwh", "imputed": "imputed_frac"})

hourly = resample("h")
daily = resample("D")

# ---------- 7. Save ----------
hourly.to_parquet("data/processed/hourly.parquet")
daily.to_parquet("data/processed/daily.parquet")

# ---------- 8. Sanity checks ----------
print("\nHourly shape:", hourly.shape, "| NaNs:", int(hourly.isna().sum().sum()))
print("Date range:", hourly.index.min(), "->", hourly.index.max())
print("Daily mean consumption: %.1f kWh/day" % daily["active_kwh"].mean())
print("Share imputed (hourly): %.2f%%" % (hourly["imputed_frac"].mean() * 100))
print(hourly.head())