"""
Nurex V4.1 Intraday - Live Dashboard (new engine only)
========================================================
Standalone Streamlit app for the rebuilt, point-in-time V4.1 intraday model.

This is deliberately separate from dashboard_v4_1.py, which is the older
multi-generation "Institutional High-Alpha Command Center" comparison tool
(V3.1/V3.2/V4.0 plus the pre-rebuild V4.1 model). This file shows ONLY the
new engine: v4_1_intraday.dashboard_adapter + the locked paper-trading
journal (data/v41_journal.sqlite). Nothing here is recomputed with hindsight -
the journal table is the honest, gate-closure-time record.

Run:  streamlit run dashboard_v41_live.py --server.port 5006
"""
import os
import sys
from datetime import date

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath("."))

from v4_1_intraday import dashboard_adapter as v41id
from v4_1_intraday import journal as J

st.set_page_config(page_title="Nurex V4.1 Intraday (Live)", page_icon="\U0001f512", layout="wide")

cfg = v41id.config()
AREAS = cfg["areas"]
TZ = cfg["local_tz"]

st.title("\U0001f512 Nurex V4.1 Intraday - Live Paper Trading")
st.caption(
    "Point-in-time engine only (rebuilt 17 Sep 2026). Every row below is a decision the model made "
    "using only the data available before that quarter's gate closed - never recomputed with hindsight."
)

with st.sidebar:
    st.header("Controls")
    area_view = st.radio("Area", AREAS, horizontal=True)
    lookback_days = st.slider("Journal lookback (days)", 1, 60, 14)
    if st.button("\U0001f504 Refresh now"):
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------------- model status
st.subheader("Model status")
cols = st.columns(len(AREAS))
for c, area in zip(cols, AREAS):
    with c:
        info = v41id.model_info(area)
        if info is None:
            st.warning(f"{area}: no trained model found - run `python train_v4_1.py train`")
            continue
        dp = info.get("decision_params", {})
        st.markdown(f"**{area}**")
        st.caption(f"trained until {info.get('trained_until', '?')} | n_train={info.get('n_train', '?')}")
        buy = dp.get("buy")
        sell = dp.get("sell")
        st.write(f"BUY: {buy if buy else 'off'}")
        st.write(f"SELL: {sell if sell else 'off'}")
        rep = info.get("replay")
        if rep:
            st.metric(f"{area} backtest net €", f"{rep.get('net_eur', 0):,.0f}",
                       f"{rep.get('net_eur_per_mwh', 0):.2f} €/MWh")
            st.caption(f"Sharpe {rep.get('sharpe_daily_ann', float('nan')):.2f} | "
                       f"max DD {rep.get('max_drawdown_eur', 0):,.0f} € | "
                       f"({info.get('replay_file', '')})")

st.divider()

# ---------------------------------------------------------------- journal summary (real, live)
st.subheader(f"Live paper-trading journal - last {lookback_days} day(s)")
summ = J.summary(cfg, days=lookback_days)
if summ.empty:
    st.info("Journal is empty for this window. The `cycle` task locks decisions 20 minutes before "
            "each gate closes - give it a bit more time, or check `logs\\v41_cycle.log` on the PC.")
else:
    st.dataframe(
        summ.set_index("area")[
            ["locked", "open", "settled_quarters", "trades", "mwh", "net_eur",
             "eur_per_mwh", "win_rate_pct", "missed_gates", "first", "last"]
        ],
        use_container_width=True,
    )
    total_net = summ["net_eur"].sum()
    total_trades = int(summ["trades"].sum())
    total_missed = int(summ["missed_gates"].sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Combined net € (both areas)", f"{total_net:,.2f}")
    m2.metric("Total settled trades", total_trades)
    if total_missed:
        m3.metric("Missed gates (PC was off)", total_missed, delta_color="inverse")
    else:
        m3.metric("Missed gates", 0)

st.divider()

# ---------------------------------------------------------------- full locked ledger
st.subheader(f"Locked decisions - {area_view}")
start = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=lookback_days)
j = J.read(cfg, area=area_view, start=start)
if j.empty:
    st.info("No locked decisions yet for this area/window.")
else:
    j = j.sort_values("quarter_utc", ascending=False).copy()
    j["quarter_local"] = j["quarter_utc"].dt.tz_localize("UTC").dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %H:%M")
    j["locked"] = "\U0001f512"
    show = j[["quarter_local", "locked", "status", "action", "mwh", "edge", "exp_spread",
              "spread_actual", "pnl_eur", "hour_all_same_side", "reason"]]
    show = show.rename(columns={
        "quarter_local": "Quarter (local)", "locked": "", "status": "Status", "action": "Action",
        "mwh": "MWh", "edge": "Edge €", "exp_spread": "Exp. spread €", "spread_actual": "Actual spread €",
        "pnl_eur": "PnL €", "hour_all_same_side": "Hourly candidate", "reason": "Reason",
    })
    st.dataframe(show, use_container_width=True, height=520)

st.divider()

# ---------------------------------------------------------------- data sources
st.subheader("Data sources")
st.dataframe(v41id.sources_status(), hide_index=True, use_container_width=True)

st.caption(
    "This page reads only data/v41_journal.sqlite and the trained model bundles - it never recomputes "
    "a past decision with newer data. For the older multi-generation comparison tool (V3.1/V3.2/V4.0 and "
    "the pre-rebuild V4.1), see dashboard_v4_1.py on port 5005."
)
