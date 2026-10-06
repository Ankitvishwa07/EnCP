import matplotlib
matplotlib.use("Agg")                      # no window, just save files
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sns.set_theme(style="whitegrid")
hourly = pd.read_parquet("data/processed/hourly.parquet")
daily = pd.read_parquet("data/processed/daily.parquet")

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

h = hourly.copy()
h["hour"] = h.index.hour
h["wd"] = h.index.weekday
h["month"] = h.index.month
h["year"] = h.index.year
h["weekend"] = h["wd"] >= 5

def save(name):
    plt.tight_layout()
    plt.savefig(f"results/{name}.png", dpi=150)
    plt.close()
    print("saved", name)

# 1. Full time series (daily) ---------------------------------------------
fig, ax = plt.subplots(figsize=(14, 4))
daily["active_kwh"].plot(ax=ax, lw=0.8)
ax.set(title="Daily energy consumption", ylabel="kWh/day", xlabel="")
save("01_daily_timeseries")

# 2. Daily profile: mean by hour, weekday vs weekend -----------------------
fig, ax = plt.subplots(figsize=(8, 4))
for flag, label in [(False, "Weekday"), (True, "Weekend")]:
    h[h["weekend"] == flag].groupby("hour")["active_kwh"].mean().plot(ax=ax, marker="o", label=label)
ax.set(title="Average consumption by hour of day", xlabel="Hour", ylabel="kWh/hour")
ax.legend()
save("02_daily_profile")

# 3. Weekly profile --------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
wk = daily.groupby(daily.index.weekday)["active_kwh"].mean()
ax.bar(DAYS, wk.values)
ax.set(title="Average daily consumption by weekday", ylabel="kWh/day")
save("03_weekly_profile")

# 4. Yearly (seasonal) profile, one line per full year ---------------------
fig, ax = plt.subplots(figsize=(9, 4))
mon = daily.groupby([daily.index.year, daily.index.month])["active_kwh"].mean().unstack(0)
mon.index = [MONTHS[i - 1] for i in mon.index]
mon.plot(ax=ax, marker="o")
ax.set(title="Average daily consumption by month", ylabel="kWh/day", xlabel="")
ax.legend(title="Year")
save("04_yearly_profile")

# 5. Heatmap: hour of day vs weekday ---------------------------------------
pivot = h.pivot_table(values="active_kwh", index="hour", columns="wd", aggfunc="mean")
pivot.columns = DAYS
fig, ax = plt.subplots(figsize=(7, 7))
sns.heatmap(pivot, cmap="YlOrRd", annot=False, cbar_kws={"label": "kWh/hour"}, ax=ax)
ax.invert_yaxis()
ax.set(title="Mean consumption: hour of day vs weekday")
save("05_heatmap_hour_weekday")

# 6. ACF and PACF (justify lags 24 and 168) --------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 7))
plot_acf(hourly["active_kwh"], lags=200, ax=axes[0])
plot_pacf(hourly["active_kwh"], lags=200, method="ywm", ax=axes[1])
axes[0].set_title("Autocorrelation (spikes at 24, 48, ... and 168)")
axes[1].set_title("Partial autocorrelation")
for a in axes:
    for lag in (24, 168):
        a.axvline(lag, color="red", ls="--", alpha=0.4)
save("06_acf_pacf")

# 7. Sub-meter share -------------------------------------------------------
cols = ["sub1_kwh", "sub2_kwh", "sub3_kwh", "rem_kwh"]
labels = ["Kitchen (sub1)", "Laundry (sub2)", "Water heater/AC (sub3)", "Other (remainder)"]
totals = hourly[cols].sum()
print("\nSub-meter share (%):")
print((totals / totals.sum() * 100).round(1).to_string())

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(totals, labels=labels, autopct="%1.1f%%", startangle=90)
ax.set_title("Share of total consumption")
save("07_submeter_pie")

by_hour = h.groupby("hour")[cols].mean()
share = by_hour.div(by_hour.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(10, 4))
share.plot(kind="bar", stacked=True, ax=ax, width=0.9)
ax.set(title="Sub-meter share by hour of day", ylabel="% of consumption", xlabel="Hour")
ax.legend(labels, bbox_to_anchor=(1.01, 1), loc="upper left")
save("08_submeter_share_by_hour")

# 8. Imputed regions highlighted -------------------------------------------
fig, ax = plt.subplots(figsize=(14, 4))
daily["active_kwh"].plot(ax=ax, lw=0.8, color="steelblue")
mask = (daily["imputed_frac"] > 0.05).values
ax.fill_between(daily.index, 0, 1, where=mask, transform=ax.get_xaxis_transform(),
                color="red", alpha=0.3, label="Days with imputed data")
ax.set(title="Imputed regions (red)", ylabel="kWh/day", xlabel="")
ax.legend()
save("09_imputed_regions")

# 9. Summary numbers for the report ----------------------------------------
print("\nPeak hour (mean): ", int(h.groupby("hour")["active_kwh"].mean().idxmax()), ":00")
print("Weekend vs weekday daily mean (kWh):",
      daily[daily.index.weekday >= 5]["active_kwh"].mean().round(2), "vs",
      daily[daily.index.weekday < 5]["active_kwh"].mean().round(2))
print("Highest month:", MONTHS[daily.groupby(daily.index.month)["active_kwh"].mean().idxmax() - 1],
      "| Lowest month:", MONTHS[daily.groupby(daily.index.month)["active_kwh"].mean().idxmin() - 1])