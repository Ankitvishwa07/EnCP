import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)                                   # relative paths work from anywhere
sys.path.insert(0, str(ROOT / "src"))
from agent import (day_windows, build_profiles, simulate_day, explain, explain_llm,
                   price, DEFAULT_Q, EVAL_Q)
from config import TARGET, TRAIN_END
from features import load_hourly

st.set_page_config(page_title="Energy Advisor", layout="wide")

@st.cache_data
def get_hourly(): return load_hourly()
@st.cache_data
def get_daily(): return pd.read_parquet("data/processed/daily.parquet")
@st.cache_data
def get_fc(): return pd.read_parquet("results/forecast_test.parquet")
@st.cache_data
def get_days(): return day_windows()
@st.cache_data
def get_master():
    base = pd.read_csv("results/metrics_baselines_trees.csv")
    rows = [dict(model=r.model, horizon=f"{r.horizon}h", MAE=r.test_MAE, RMSE=r.test_RMSE, MAPE=r.test_MAPE)
            for r in base.itertuples()]
    for name, f in [("LSTM", "results/metrics_lstm.csv"), ("Transformer", "results/metrics_transformer.csv")]:
        m = pd.read_csv(f, index_col=0)
        for hz, r in m.iterrows():
            rows.append(dict(model=name, horizon=hz, MAE=r.MAE, RMSE=r.RMSE, MAPE=r.MAPE))
    return pd.DataFrame(rows)

page = st.sidebar.radio("Page", ["Explorer", "Model comparison", "Live forecast", "Agent schedule"])

# ---------------- 1. Explorer ----------------
if page == "Explorer":
    st.title("Historical consumption explorer")
    res = st.selectbox("Resolution", ["hourly", "daily"])
    d = get_hourly() if res == "hourly" else get_daily()
    col = st.selectbox("Series", ["active_kwh", "sub1_kwh", "sub2_kwh", "sub3_kwh", "rem_kwh", "voltage"])
    rng = st.date_input("Date range", [d.index.min().date(), d.index.max().date()])
    if len(rng) == 2:
        sel = d.loc[str(rng[0]):str(rng[1]), [col]]
        st.plotly_chart(px.line(sel, y=col, labels={"index": "", col: col}), use_container_width=True)
        st.caption(f"Mean {sel[col].mean():.2f} | max {sel[col].max():.2f} | total {sel[col].sum():,.0f}")

# ---------------- 2. Model comparison ----------------
elif page == "Model comparison":
    st.title("Model comparison (test set, Apr-Nov 2010)")
    master = get_master()
    metric = st.selectbox("Metric", ["MAE", "RMSE", "MAPE"])
    st.dataframe(master.pivot(index="model", columns="horizon", values=metric)
                 .reindex(columns=["1h", "24h", "168h"]).round(4), use_container_width=True)
    fig = px.bar(master, x="model", y=metric, color="horizon", barmode="group",
                 category_orders={"horizon": ["1h", "24h", "168h"]})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("LSTM and Transformer are 24-step models, so 168h is n/a. MAPE is inflated by near-zero hours.")

# ---------------- 3. Live forecast ----------------
elif page == "Live forecast":
    st.title("24h-ahead forecast (XGBoost) with 80% band")
    fc = get_fc()
    days = sorted(set(fc.index.date))
    c1, c2 = st.columns(2)
    start = c1.selectbox("Start day", days, index=len(days) // 2)
    n = c2.slider("Days to show", 1, 7, 2)
    s = pd.Timestamp(start)
    d = fc.loc[s: s + pd.Timedelta(days=n) - pd.Timedelta(hours=1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d.index, y=d["lo"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(31,119,180,0.2)", name="80% band"))
    fig.add_trace(go.Scatter(x=d.index, y=d["pred"], name="Forecast", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=d.index, y=d["actual"], name="Actual", line=dict(color="black", width=1.5)))
    fig.update_layout(yaxis_title="kWh per hour", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.metric("MAE over shown period (kWh/h)", f"{(d['pred'] - d['actual']).abs().mean():.3f}")

# ---------------- 4. Agent schedule ----------------
else:
    st.title("Energy advisor: recommended schedule")
    hourly = get_hourly()
    prof = build_profiles(hourly)
    days = get_days()
    labels = [d[0][0].date() for d in days]
    c1, c2 = st.columns(2)
    sel = c1.selectbox("Day", labels, index=len(labels) // 2)
    q = c2.slider("Peak threshold (quantile of training consumption)", 0.70, 0.95, DEFAULT_Q, 0.05)
    idx, fc, act = days[labels.index(sel)]
    train = hourly[:TRAIN_END][TARGET]
    thr_plan, thr_eval = train.quantile(q), train.quantile(EVAL_Q)
    res, sched, new_act = simulate_day(idx, fc, act, hourly.loc[idx], prof, thr_plan, thr_eval)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Moves planned", res["n_moves"])
    m2.metric("Energy shifted (kWh)", f"{res['kwh_moved']:.2f}")
    m3.metric("Cost saved (EUR)", f"{res['cost_saving']:.3f}")
    red = res["peak_reduction_pct"]
    m4.metric("Peak load reduced", "no peak today" if np.isnan(red) else f"{red:.0f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=fc.values, name="Forecast"))
    fig.add_trace(go.Scatter(x=idx, y=act.values, name="Actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=idx, y=new_act.values, name="Actual after schedule",
                             line=dict(color="green", dash="dash")))
    fig.add_hline(y=thr_plan, line_dash="dot", line_color="red", annotation_text="planning threshold")
    fig.add_hline(y=thr_eval, line_dash="dot", line_color="orange", annotation_text="measured peak threshold")
    off = [h for h in range(24) if price(h) < price(12)]
    fig.update_layout(yaxis_title="kWh per hour", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Off-peak (cheaper) hours: {min(off)}..{max(off)} wrap-around 22:00-06:00.")

    st.subheader("Schedule")
    if sched.empty:
        st.info("No shiftable peak load found for this day.")
    else:
        show = sched.assign(src=sched.src.dt.strftime("%H:%M"), dst=sched.dst.dt.strftime("%H:%M"))
        st.dataframe(show[["label", "src", "dst", "kwh", "saving"]].round(3)
                     .rename(columns={"label": "load", "src": "from", "dst": "to",
                                      "kwh": "kWh moved", "saving": "EUR saved"}),
                     use_container_width=True)
    st.subheader("Advice")
    if st.checkbox("Use local LLM (needs Ollama running)", value=False):
        st.write(explain_llm(sched))
    else:
        for line in explain(sched):
            st.write("- " + line)