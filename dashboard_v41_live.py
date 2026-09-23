"""
Nurex V4.1 Intraday Dashboard (V4.1 only)
=========================================
Shows only the rebuilt, point-in-time V4.1 intraday engine. Imports nothing from src/
(no V2/V3/V3.2/V4.0 code). Data: v4_1_intraday (models, locked journal) + the V4.2 store.

Default view = LOCKED: the paper-trading journal, i.e. the decision locked before each gate
closed with the data available then. "Recomputed" re-runs today's model over the day
(in-sample for past days) and is for analysis only. Sizes are MWh per 15-min quarter.

Run:  scripts_v41\\run_dashboard_v41_live.bat   ->  http://127.0.0.1:5006
"""
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("Nurex_V4_2"))

from v4_1_intraday import dashboard_adapter as v41id
from v4_1_intraday import journal as J
from v4_1_intraday import ledger_v41 as L

st.set_page_config(page_title="Nurex V4.1 Intraday", page_icon="⚡", layout="wide")

cfg = v41id.config()
AREAS = cfg["areas"]
TZ = cfg["local_tz"]
SRC_ICON = {"LOCKED": "\U0001f512 LOCKED", "MISSED": "⛔ MISSED", "FORECAST": "\U0001f52e FORECAST",
            "NO RECORD": "— NO RECORD", "RECOMPUTED": "♻ RECOMPUTED"}
ACT_ICON = {"BUY": "\U0001f7e2 BUY", "SELL": "\U0001f534 SELL", "HOLD": "⚪ HOLD"}


@st.cache_data(ttl=300, show_spinner=False)
def _ledger(area, day, view):
    return L.day_ledger(area, day, view)


@st.cache_data(ttl=300, show_spinner=False)
def _status(area):
    return L.status(area)


@st.cache_data(ttl=300, show_spinner=False)
def _daily(days):
    return L.journal_daily(days=days)


@st.cache_data(ttl=3600, show_spinner=False)
def _info(area):
    return v41id.model_info(area)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚡ Nurex V4.1")
    area = st.radio("Bidding zone", AREAS, horizontal=True)
    today_local = pd.Timestamp.now(tz=TZ).date()
    day = st.date_input("Delivery day", value=today_local, max_value=today_local + timedelta(days=1))
    view_lbl = st.radio("Decisions shown", ["\U0001f512 Locked (paper-trading record)", "♻ Recomputed (analysis)"],
                        help="Locked = decided before gate closure and never changed - the honest record. "
                             "Recomputed = today's model re-run over the day; in-sample for past days.")
    view = "locked" if view_lbl.startswith("\U0001f512") else "recomputed"
    show_borders = st.checkbox("Show flow per border", value=False)
    perf_days = st.slider("Performance window (days)", 3, 60, 14)
    if st.button("\U0001f504 Refresh now", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Page time {pd.Timestamp.now(tz=TZ):%Y-%m-%d %H:%M} ({TZ})")

day_str = pd.Timestamp(day).strftime("%Y-%m-%d")

st.title(f"⚡ Nurex V4.1 Intraday — {area}")
st.caption("Point-in-time V4.1 engine only. Decision time = delivery − 60 min; decisions are locked "
           "20 min before that. Sizes in MWh per 15-min quarter (MW = MWh × 4). Simulation / paper trading.")

# ------------------------------------------------------------------ status bar
info = _info(area) or {}
stt = _status(area)
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown("**Model**")
    st.caption(f"trained until {str(info.get('trained_until', '?'))[:16]} UTC")
    dp = info.get("decision_params", {}) or {}
    st.caption(f"BUY {dp.get('buy') or 'off'} · SELL {dp.get('sell') or 'off'}")
with c2:
    st.markdown("**Costs**")
    st.caption(f"{cfg.cost_per_mwh:.2f} EUR/MWh (all-in, per traded MWh)")
with c3:
    st.markdown("**Imbalance data**")
    age = stt.get("imb_age_min")
    if age is None:
        st.error("no imbalance data readable")
    else:
        ok = age <= stt.get("guard_min", 90)
        (st.success if ok else st.error)(f"last {stt['last_imbalance_utc']:%H:%M} UTC · {age:.0f} min old")
        if not ok:
            st.caption("Older than the stale-data guard: new decisions are forced to HOLD.")
with c4:
    st.markdown("**Journal**")
    ll = stt.get("last_lock_utc")
    if ll is None:
        st.warning("no locks yet")
    else:
        mins = (stt["now_utc"] - ll).total_seconds() / 60
        (st.success if mins <= 30 else st.warning)(f"last lock {mins:.0f} min ago")
        if mins > 30:
            st.caption("The 15-min cycle (task Nurex_V41_Cycle) looks stalled.")
with c5:
    st.markdown("**Risk state**")
    m = stt.get("risk_mult")
    if m is None:
        st.warning(str(stt.get("risk_reason")))
    elif m >= 1.0:
        st.success("full size")
    elif m > 0:
        st.warning(f"size ×{m:.1f} ({stt.get('risk_reason')})")
    else:
        st.error(f"trading stopped ({stt.get('risk_reason')})")

st.divider()

# ------------------------------------------------------------------ day ledger
with st.spinner(f"Building {area} {day_str} ({view})..."):
    try:
        led, linfo = _ledger(area, day_str, view)
    except Exception as e:
        led, linfo = pd.DataFrame(), {"error": str(e)}

if led.empty:
    st.warning(f"No day-ahead prices in the store for {area} {day_str} yet (published ~13:00 CET the day before). "
               + (f"Error: {linfo['error']}" if "error" in linfo else ""))
    st.stop()

pnl = pd.to_numeric(led["pnl_eur"], errors="coerce")
traded = led["action"].isin(["BUY", "SELL"])
settled_tr = traded & pnl.notna()
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Positions", int(traded.sum()), f"{int((led['action'] == 'BUY').sum())} BUY / {int((led['action'] == 'SELL').sum())} SELL",
          delta_color="off")
k2.metric("Volume", f"{led.loc[traded, 'mwh'].sum():.1f} MWh")
k3.metric("Net PnL (settled)", f"€{pnl[settled_tr].sum():,.2f}")
k4.metric("Win rate", f"{100 * (pnl[settled_tr] > 0).mean():.0f}%" if settled_tr.any() else "—")
k5.metric("Settled quarters", f"{int(led['actual_settled_imbalance_eur'].notna().sum())}/{len(led)}")
if view == "locked":
    k6.metric("Missed gates", int((led["source"] == "MISSED").sum()),
              help="Quarters whose gate passed before the cycle ran (PC off or cycle blocked) - locked as HOLD.")
else:
    k6.metric("View", "recomputed", help="In-sample for past days - not a performance record.")

if view == "recomputed":
    st.info("Recomputed view: today's model re-run over this day with the replay's risk overlay. "
            "For past days this is in-sample and is NOT the performance record - switch to Locked for that.")
notes = []
if linfo.get("de") not in ("ok",):
    notes.append(f"DE prices: {linfo.get('de')}")
if linfo.get("flows") not in ("ok",):
    notes.append(f"flows: {linfo.get('flows')}")
if linfo.get("recomputed") not in ("ok",):
    notes.append(f"forecasts: {linfo.get('recomputed')}")
if notes:
    st.caption("ℹ️ " + " · ".join(notes))

tbl = pd.DataFrame({
    "Quarter": led["time_dk"].str.slice(11, 16),
    "Source": led["source"].map(SRC_ICON).fillna(led["source"]),
    "Position": led["action"].map(ACT_ICON).fillna(led["action"]),
    "MWh": led["mwh"].round(2),
    "MW": led["mw"].round(1),
    "DK spot €": led["spot_price_eur"].round(2),
    "DE spot €": led["de_spot_eur"].round(2),
    "Net flow MW": led["net_flow_mw"].round(0),
    "Exp. spread €": pd.to_numeric(led["exp_spread"], errors="coerce").round(2),
    "q10": pd.to_numeric(led["q10"], errors="coerce").round(1),
    "q50": pd.to_numeric(led["q50"], errors="coerce").round(1),
    "q90": pd.to_numeric(led["q90"], errors="coerce").round(1),
    "P(up)": pd.to_numeric(led["p_up"], errors="coerce").round(2),
    "P(down)": pd.to_numeric(led["p_down"], errors="coerce").round(2),
    "Pred. imb €": led["pred_imb_eur"].round(2),
    "Settled imb €": led["actual_settled_imbalance_eur"].round(2),
    "Actual spread €": pd.to_numeric(led["spread_actual"], errors="coerce").round(2),
    "PnL €": pnl.round(2),
    "Reason": led["reason"].fillna(""),
})
if show_borders:
    for _, lbl in v41id.FLOW_BORDERS.get(area, []):
        c = f"flow_{lbl}"
        if c in led.columns:
            tbl.insert(tbl.columns.get_loc("Net flow MW") + 1, f"→{lbl} MW", led[c].round(0))

st.subheader(f"96-quarter ledger — {day_str}")
st.dataframe(
    tbl, hide_index=True, width="stretch", height=560,
    column_config={
        "MWh": st.column_config.NumberColumn(format="%.2f", help="Energy per 15-min quarter"),
        "MW": st.column_config.NumberColumn(format="%.1f", help="= MWh × 4"),
        "Net flow MW": st.column_config.NumberColumn(format="%+.0f", help="Scheduled exchange, + = export"),
        "PnL €": st.column_config.NumberColumn(format="%.2f"),
        "Exp. spread €": st.column_config.NumberColumn(help="Expected imbalance − DA spot"),
    })
st.caption("Source: \U0001f512 locked before gate · ⛔ gate missed (HOLD) · \U0001f52e live forecast, "
           "not locked yet · — no journal record (before the journal started). PnL = side × MWh × "
           "(imbalance − spot) − costs; only settled quarters.")

# price picture for the day
with st.expander("\U0001f4c8 Prices through the day", expanded=False):
    ch = pd.DataFrame({
        "DK spot": led["spot_price_eur"].values,
        "Predicted imbalance": led["pred_imb_eur"].values,
        "Settled imbalance": led["actual_settled_imbalance_eur"].values,
    }, index=led["time_dk"].str.slice(11, 16))
    st.line_chart(ch, height=300)

# upcoming gates (today only)
if day_str == today_local.strftime("%Y-%m-%d"):
    nowu = pd.Timestamp.now(tz="UTC").tz_localize(None)
    gate = led["quarter_utc"] - cfg.lead
    up = led[gate > nowu].head(8)
    if len(up):
        st.subheader("Next gates")
        g = gate[up.index]
        st.dataframe(pd.DataFrame({
            "Quarter": up["time_dk"].str.slice(11, 16),
            "Gate (local)": g.dt.tz_localize("UTC").dt.tz_convert(TZ).dt.strftime("%H:%M"),
            "Locks at (local)": (g - pd.Timedelta(minutes=int(cfg["intraday"].get("lock_ahead_minutes", 20))))
                .dt.tz_localize("UTC").dt.tz_convert(TZ).dt.strftime("%H:%M"),
            "Status": up["source"].map(SRC_ICON).fillna(up["source"]),
            "Current view": up["action"].map(ACT_ICON).fillna(up["action"]),
            "MWh": up["mwh"].round(2),
            "Exp. spread €": pd.to_numeric(up["exp_spread"], errors="coerce").round(2),
        }), hide_index=True, width="stretch")

st.divider()

# ------------------------------------------------------------------ performance (locked journal only)
st.subheader(f"Paper-trading performance — locked journal, last {perf_days} days")
dly = _daily(perf_days)
if dly.empty:
    st.info("No settled locked decisions in this window yet.")
else:
    piv = dly.pivot_table(index="day", columns="area", values="net_eur", aggfunc="sum").fillna(0.0)
    cum = piv.cumsum()
    p1, p2 = st.columns(2)
    with p1:
        st.caption("Net € per day")
        st.bar_chart(piv, height=260)
    with p2:
        st.caption("Cumulative net €")
        st.line_chart(cum, height=260)
    summ = J.summary(cfg, days=perf_days)
    if not summ.empty:
        st.dataframe(summ.rename(columns={
            "area": "Zone", "locked": "Locked", "open": "Unsettled", "settled_quarters": "Settled",
            "trades": "Trades", "mwh": "MWh", "net_eur": "Net €", "eur_per_mwh": "€/MWh",
            "win_rate_pct": "Win %", "missed_gates": "Missed gates", "first": "First", "last": "Last"}),
            hide_index=True, width="stretch")
rep = info.get("replay")
if rep:
    st.caption(f"Reference (backtest, not live): walk-forward replay {area} — {rep.get('trades', 0):,} trades, "
               f"{rep.get('net_eur', 0):,.0f} € ({rep.get('net_eur_per_mwh', 0):.2f} €/MWh), "
               f"max DD {rep.get('max_drawdown_eur', 0):,.0f} € · {info.get('replay_file', '')}")

st.divider()
with st.expander("\U0001f4e1 Data sources (coverage)", expanded=False):
    try:
        st.dataframe(v41id.sources_status(), hide_index=True, width="stretch")
    except Exception as e:
        st.warning(f"Source status unavailable: {e}")
st.caption(f"Store read from: {stt.get('store_source')}. This page imports nothing from V2/V3/V4.0. "
           "The multi-generation comparison tool remains at dashboard_v4_1.py (port 5005).")
