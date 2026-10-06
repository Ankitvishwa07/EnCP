import numpy as np
import pandas as pd
from config import TARGET, TRAIN_END, OFFPEAK_HOURS, PRICE_OFFPEAK, PRICE_NORMAL
from features import load_hourly

DEFAULT_Q = 0.80                       # planner threshold (quantile of training consumption)
EVAL_Q = 0.90                          # peaks are always MEASURED against the train 90th percentile
LOADS = {"sub2_kwh": ("washing machine / laundry", range(8, 22)),
         "sub3_kwh": ("water heater", range(24))}
DEEP_PATH = "results/preds_transformer.npz"

def price(hour):
    return PRICE_OFFPEAK if hour in OFFPEAK_HOURS else PRICE_NORMAL

# ---------- perceive ----------
def load_deep(path):
    z = np.load(path)
    raw = z["start"]
    for unit in ["ns", "us", "ms", "s"]:            # saved as ints; unit depends on pandas version
        start = pd.to_datetime(raw, unit=unit)
        if 2006 <= start.min().year <= 2011:
            break
    else:
        raise ValueError("cannot detect time unit")
    return pd.DatetimeIndex(start).as_unit("ns"), z["pred"], z["actual"]

def day_windows(path=DEEP_PATH):
    """One (index, forecast, actual) tuple per test day, forecast issued at midnight."""
    start, pred, actual = load_deep(path)
    days = []
    for i in np.where(start.hour == 0)[0]:
        idx = pd.date_range(start[i], periods=24, freq="h")
        days.append((idx,
                     pd.Series(np.clip(pred[i], 0, None).astype(float), index=idx),
                     pd.Series(actual[i].astype(float), index=idx)))
    return days

def build_profiles(hourly):
    tr = hourly[:TRAIN_END]
    return {c: tr.groupby(tr.index.hour)[c].mean() for c in LOADS}     # typical kWh by hour

# ---------- plan ----------
def plan(curve, profiles, thr):
    f = curve.copy().astype(float)
    rows = []
    for col, (label, window) in LOADS.items():
        for hr in list(f.index[f > thr]):
            kwh = min(float(profiles[col][hr.hour]), f[hr] - thr)
            if kwh <= 1e-6:
                continue
            cands = [c for c in f.index if c.hour in window and c != hr and f[c] + kwh <= thr]
            if not cands:
                continue
            dst = min(cands, key=lambda c: (price(c.hour), f[c]))      # cheapest, then emptiest
            f[hr] -= kwh
            f[dst] += kwh
            rows.append(dict(load=col, label=label, src=hr, dst=dst, kwh=kwh,
                             saving=kwh * (price(hr.hour) - price(dst.hour))))
    cols = ["load", "label", "src", "dst", "kwh", "saving"]
    return pd.DataFrame(rows, columns=cols), f

# ---------- act + evaluate on one day ----------
def simulate_day(idx, plan_curve, act, sub, profiles, thr_plan, thr_eval):
    """Plan on plan_curve (forecast or actual=oracle), apply moves to the ACTUAL load."""
    sched, _ = plan(plan_curve, profiles, thr_plan)
    new = act.copy().astype(float)
    moved_total = 0.0
    for r in sched.itertuples():
        moved = min(r.kwh, float(sub[r.load].loc[r.src]))              # cannot move more than was used
        new[r.src] -= moved
        new[r.dst] += moved
        moved_total += moved
    prices = np.array([price(h) for h in idx.hour])
    before = float((act - thr_eval).clip(lower=0).sum())
    after = float((new - thr_eval).clip(lower=0).sum())
    res = dict(n_moves=len(sched), kwh_moved=moved_total,
               peak_before=before, peak_after=after,
               peak_reduction_pct=(before - after) / before * 100 if before > 0 else np.nan,
               max_before=float(act.max()), max_after=float(new.max()),
               cost_saving=float((act.values * prices).sum() - (new.values * prices).sum()))
    return res, sched, new

# ---------- explain ----------
def explain(sched):
    if sched.empty:
        return ["No shiftable peak load found for this day. No changes needed."]
    return [f"Run the {r.label} at {r.dst:%H:%M} instead of {r.src:%H:%M} to avoid the peak "
            f"(moves about {r.kwh:.2f} kWh, saves about EUR {r.saving:.3f})." for r in sched.itertuples()]

def explain_llm(sched, model="llama3.2:3b"):
    """Optional free local LLM via Ollama; falls back to the template."""
    try:
        import ollama
        txt = sched[["label", "src", "dst", "kwh"]].to_string()
        r = ollama.chat(model=model, messages=[{"role": "user",
            "content": f"Turn this appliance schedule into 3 short friendly tips for a homeowner:\n{txt}"}])
        return r["message"]["content"]
    except Exception:
        return "\n".join(explain(sched))

# ---------- run over all test days ----------
if __name__ == "__main__":
    hourly = load_hourly()
    prof = build_profiles(hourly)
    train = hourly[:TRAIN_END][TARGET]
    thr_eval = train.quantile(EVAL_Q)
    days = day_windows()
    print(f"{len(days)} test days | peak threshold used for measuring: {thr_eval:.2f} kWh/h")

    rows = []
    for q in (0.80, 0.90):
        thr_plan = train.quantile(q)
        for idx, fc, act in days:
            sub = hourly.loc[idx]
            for mode, curve in (("forecast", fc), ("oracle", act)):
                r, _, _ = simulate_day(idx, curve, act, sub, prof, thr_plan, thr_eval)
                rows.append({"day": idx[0].date(), "mode": mode, "q": q, **r})
    df = pd.DataFrame(rows)
    df["max_drop"] = df["max_before"] - df["max_after"]
    df.to_csv("results/agent_days.csv", index=False)

    g = df.groupby(["q", "mode"])
    summ = g.agg(days_with_peaks=("peak_before", lambda s: int((s > 0).sum())),
                 mean_peak_red_pct=("peak_reduction_pct", "mean"),
                 std_peak_red_pct=("peak_reduction_pct", "std"),
                 kwh_moved_per_day=("kwh_moved", "mean"),
                 eur_saved_per_day=("cost_saving", "mean"),
                 max_hour_drop_kwh=("max_drop", "mean"))
    sums = g[["peak_before", "peak_after"]].sum()
    summ["overall_peak_red_pct"] = (sums.peak_before - sums.peak_after) / sums.peak_before * 100
    summ.round(3).to_csv("results/agent_summary.csv")
    print(summ.round(3).to_string())

    # example schedule + advice
    idx, fc, act = max(days, key=lambda d: d[2].max())
    _, sched, _ = simulate_day(idx, fc, act, hourly.loc[idx], prof, train.quantile(DEFAULT_Q), thr_eval)
    print("\nExample day:", idx[0].date())
    print(sched[["label", "src", "dst", "kwh", "saving"]].round(3).to_string(index=False))
    print("\n".join(explain(sched)))