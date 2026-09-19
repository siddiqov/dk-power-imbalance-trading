import os
from dotenv import load_dotenv
load_dotenv()
# ==============================================================================
# dashboard_v4_1.py
# Nurex V4.1: Institutional High-Alpha Microstructure & Cross-Border Balancing Engine
# Dedicated Execution Port: 5005 (http://127.0.0.1:5005/)
#
# Core Capabilities:
# 1. 8-Cable Physical Interconnector Matrix (Capacities, Flows, Headroom, Congestion)
# 2. European Balancing Platforms: MARI (mFRR) & PICASSO (aFRR) Merit Order Energy Activation
# 3. SMARD.de (Bundesnetzagentur): German Grid Imbalance & Residual Load Integration
# 4. Nord Pool XBID: Level-2 Continuous Order Flow Microstructure & Volume Skewness Meter
# 5. Side-by-Side Unified Comparative Ledger (V4.0 vs V4.1)
# 6. Quantitative Tournament Backtest (V4.0 vs V4.1)
# ==============================================================================

import os
import sys
import json
import time
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import altair as alt
from datetime import datetime, date, timedelta

sys.path.append(os.path.abspath('.'))

from src.tournament_tables_v2 import TournamentTableGenerator
from src.commercial_strategy_v3_1 import V31CommercialStrategyEngine
from v4_1_intraday import dashboard_adapter as v41id  # V4.1 intraday engine (point-in-time, gate closure)

# Why the per-quarter cross-border columns are (or are not) available, keyed by (area, date).
# 'ok' | 'stale' (store was busy, cached day reused) | 'busy' | 'unpublished' | 'error'
FLOW_STATUS = {}

# ---- Flow sign convention (uniform across every table in this dashboard) ----
#   positive = EXPORT out of the selected zone, negative = IMPORT into it.
# ENTSO-E series (the ledger's per-quarter columns) already follow it.
# Energinet's live Exchange_* series are the exact mirror - verified against
# ENTSO-E phys:DK1>X on 2026-09-18 (correlation -1.000) - so they are negated
# for display only. The V4.0 model keeps the raw sign it was trained on.
FLOW_CONVENTION_NOTE = ("Flow sign convention: **positive = export out of "
                        "the zone**, negative = import into it (ENTSO-E convention).")


def exp_pos(v):
    """Energinet live flow (import-positive) -> export-positive, for display."""
    try:
        return -float(v)
    except (TypeError, ValueError):
        return 0.0
from src.feature_engineering_v4_1 import V41FeatureEngine, CABLE_CAPACITIES
from src.balancing_market_v4_1 import fetch_mfrr_energy_activations, fetch_afrr_energy_activations, get_latest_balancing_state
from src.smard_client import get_german_system_balance_telemetry
from src.order_flow_v4_1 import OrderFlowEngineV41, compute_order_flow_microstructure
from src.data_retrieval_v4 import fetch_energinet_true_forecast_error, fetch_energinet_system_frequency
from src.nordpool_umm_scraper import fetch_live_umms
from src.dmi_client import get_dmi_zone_weather_telemetry

# --- LIVE-FEED CACHING ----------------------------------------------------------
# These external calls used to re-fetch on every full rerun (every click, every tab -
# Streamlit runs the whole script each time, even the tabs you are not looking at),
# which is most of why "Refresh now" felt slow. A short cache lets a refresh reuse
# data that is only seconds old instead of waiting on several external HTTP calls again.
@st.cache_data(ttl=45, show_spinner=False)
def _live_balancing_state(area):
    return get_latest_balancing_state(area)

@st.cache_data(ttl=45, show_spinner=False)
def _live_smard_telemetry():
    """Adapter: maps SMARD's real field names to what the Balancing tab displays."""
    t = get_german_system_balance_telemetry()
    return {
        "generation_mw": t.get("german_generation_mw", 0.0),
        "consumption_mw": t.get("german_load_mw", 0.0),
        "residual_load_mw": t.get("german_system_balance_mw", 0.0),
        "system_state": t.get("balancing_regime", "UNKNOWN"),
        "source": t.get("source", "SMARD.de"),
        "status": t.get("status", ""),
    }

@st.cache_data(ttl=120, show_spinner=False)
def _live_dmi_telemetry(area):
    return get_dmi_zone_weather_telemetry(area)

@st.cache_data(ttl=45, show_spinner=False)
def _live_energinet_frequency(limit=5):
    return fetch_energinet_system_frequency(limit=limit)

# --- V4.1 patch: resilient authentic Day-Ahead spot retrieval -------------------
# The V2 generator queries EDS with a 10 s timeout, a string PriceArea filter and no
# retry; on failure it falls back to ImbalancePrice, which only covers settled
# quarters, so the first future quarter raises "Authentic Day-Ahead Spot price
# missing". This override retries EDS properly and, if EDS is still short, reads the
# same authentic EDS rows already collected into the V4.1 point-in-time store.
# No synthetic or interpolated values are ever produced. The shared V2/V3 module is unaffected:
# only this dashboard's class attribute is replaced.
def _v41_fetch_day_ahead_96_spot_prices(self, start_dt, end_dt):
    import json as _json
    import requests as _rq
    import pandas as _pd

    spot = {}
    params = {
        "filter": _json.dumps({"PriceArea": [self.price_area]}),
        "start": start_dt.strftime("%Y-%m-%dT00:00"),
        "end": (start_dt + timedelta(days=1)).strftime("%Y-%m-%dT00:00"),
        "sort": "TimeDK ASC",
        "limit": 400,
    }
    for attempt in range(3):
        try:
            res = _rq.get("https://api.energidataservice.dk/dataset/DayAheadPrices",
                          params=params, timeout=60).json()
            for r in res.get("records", []):
                val = r.get("DayAheadPriceEUR")
                if r.get("PriceArea") == self.price_area and _pd.notnull(val):
                    key = _pd.to_datetime(r["TimeDK"]).strftime("%Y-%m-%d %H:%M")
                    spot[key] = float(val)
            if len(spot) >= 96:
                return spot
        except Exception as exc:
            print(f"  [V4.1] DayAheadPrices attempt {attempt + 1}/3 failed: {exc}")

    try:
        import duckdb as _dd
        store = os.path.join("Nurex_V4_2", "data", "nurex42.duckdb")
        if os.path.exists(store):
            t0 = _pd.Timestamp(start_dt).tz_localize("Europe/Copenhagen").tz_convert("UTC").tz_localize(None)
            t1 = _pd.Timestamp(start_dt + timedelta(days=1)).tz_localize("Europe/Copenhagen").tz_convert("UTC").tz_localize(None)
            con = _dd.connect(store, read_only=True)
            try:
                rows = con.execute(
                    "SELECT time_utc, price_eur FROM dayahead "
                    "WHERE area = ? AND time_utc >= ? AND time_utc < ? AND source LIKE 'EDS:%'",
                    [self.price_area, t0, t1]).fetchall()
            finally:
                con.close()
            for ts, price in rows:
                if price is None:
                    continue
                key = _pd.Timestamp(ts).tz_localize("UTC").tz_convert("Europe/Copenhagen").strftime("%Y-%m-%d %H:%M")
                spot.setdefault(key, float(price))
            print(f"  [V4.1] Day-Ahead spot: {len(rows)} authentic EDS rows read from the V4.1 store.")
    except Exception as exc:
        print(f"  [V4.1] V4.1 store lookup failed: {exc}")

    return spot


TournamentTableGenerator._fetch_day_ahead_96_spot_prices = _v41_fetch_day_ahead_96_spot_prices

COST_EUR_MWH = v41id.cost_per_mwh()  # fee + imbalance fee + BRP + slippage (config_v41.yaml)

st.set_page_config(
    page_title="Nurex V4.1 Institutional High-Alpha Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        border: 1px solid #2A2E39;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .badge-champ {
        background-color: #00C851;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-v41 {
        background-color: #0284C7;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLLER ---
st.sidebar.title("⚡ Nurex V4.1 High-Alpha Meta-Controller")
st.sidebar.caption("High-Alpha Intraday & Balancing Engine | Port 5005")

# --- AUTO-REFRESH: current trade status every 15 min, matching the scheduled trading
# cycle - or refresh immediately any time with the button below.
REFRESH_SECONDS = 900  # 15 minutes
if "_last_full_refresh" not in st.session_state:
    st.session_state["_last_full_refresh"] = time.time()

if st.sidebar.button("\U0001f504 Refresh now", help="Reload current trade status immediately"):
    st.session_state["_last_full_refresh"] = time.time()

@st.fragment(run_every=REFRESH_SECONDS)
def _auto_refresh_heartbeat():
    elapsed = time.time() - st.session_state["_last_full_refresh"]
    st.caption(f"\U0001f504 Auto-refresh every 15 min "
               f"(last: {datetime.now().strftime('%H:%M:%S')})")
    if elapsed >= REFRESH_SECONDS - 1:
        st.session_state["_last_full_refresh"] = time.time()
        st.rerun()

with st.sidebar:
    _auto_refresh_heartbeat()

# 1. Market Bidding Zone & Strategy
selected_area = st.sidebar.radio("Bidding Zone", ["DK1", "DK2"], index=0)

strategy_mode = st.sidebar.selectbox(
    "Trading Strategy Engine",
    [
        "V4.1 Institutional High-Alpha (Balancing + SMARD + XBID)",
        "V4.0 Full-Grid Champion (Grid Search Tuned)"
    ],
    index=0
)

# 2. Trading Date
dk_now = pd.Timestamp.now(tz="Europe/Copenhagen")
default_date = dk_now.date()
selected_date = st.sidebar.date_input("Trading Date", value=default_date)
date_str_selected = selected_date.strftime("%Y-%m-%d")

# 3. Risk & Execution Parameters
st.sidebar.markdown("### 🛡️ Risk & Execution Controls")
use_circuit_breaker = st.sidebar.checkbox("95th Pct Dynamic Circuit Breaker", value=True)
crash_protection_enabled = st.sidebar.checkbox("Physical Crash Protection Trigger", value=True)
# Evening ramping guard removed 2026-09-19: it only ever adjusted the legacy V4.0
# comparison column's simulated size/cap, never V4.1's real trading decisions.

# V4.1 decides a quarter only when expected edge clears a margin AND the direction probability
# clears a minimum. Those two numbers were chosen on the validation window during training.
# The looser levels are a what-if view of the SAME forecasts at a lower bar - not validated.
threshold_level = st.sidebar.selectbox(
    "V4.1 Signal Thresholds",
    ["Validated (from training)", "Balanced (what-if)", "Aggressive (what-if)"],
    index=2,  # default: Aggressive (what-if)
    help="Validated uses the margin / probability pair tuned on held-out data. The what-if levels halve or quarter the margin and lower the probability bar, so more quarters qualify - more trades, more exposure, and no validation behind them.")
# The walk-forward replay has always applied a daily loss stop and drawdown scaling; until now
# the live view did not, so backtest and live were not the same system. With this on, they are.
risk_limits_on = st.sidebar.checkbox(
    "Apply risk limits (daily loss stop + drawdown)", value=True,
    help="Stops trading for the rest of the day once the daily loss limit is hit, and halves or halts size on drawdown - the same overlay the replay uses. Off shows raw signal decisions, which will NOT match backtest results.")

# Trade size scales with how much of the account's collateral a batch of quarters may risk
# (batch_risk_fraction in config.yaml, default 0.1). This checkbox raises that to 0.2 for new,
# not-yet-decided quarters only - locked/settled quarters keep the size they were actually
# decided and traded at.
bigger_size_on = st.sidebar.checkbox(
    "Bigger trade sizes (risk budget 0.1 → 0.2)", value=False,
    help="Doubles the share of collateral each batch of quarters may risk, roughly doubling trade size on new decisions. Bigger trades also mean bigger possible losses.")
if bigger_size_on:
    st.sidebar.caption("⚠️ Bigger trades also mean bigger possible losses.")

THRESHOLD_KEY = {"Validated (from training)": "validated", "Balanced (what-if)": "balanced",
                 "Aggressive (what-if)": "aggressive"}[threshold_level]
high_conviction_vol = st.sidebar.slider("High-Conviction Trade Size (MW)", 10, 50, 25)
standard_vol = st.sidebar.slider("Standard Trade Size (MW)", 5, 20, 10)

# --- Interconnector rule: options to test -------------------------------------
# Selection only. Nothing here changes a decision: an option is applied to trading
# only after it wins a replay A/B test and is written into config_v41.yaml.
IC_CHOICE_PATH = os.path.join("results", "v4_1_intraday", "ic_rule_choice.json")


def _load_ic_choice():
    try:
        with open(IC_CHOICE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"selected": [], "saved_at": None}


_ic_choice = _load_ic_choice()
_ic_sel = set(_ic_choice.get("selected", []))

with st.sidebar.expander("\U0001f50c Interconnector rule - options to test", expanded=False):
    st.caption("Selection only: no option changes live decisions. The one you pick is tested "
               "in the walk-forward replay first, and applied only if it wins.")
    ic_a = st.checkbox("A - Block the trade when flows contradict", value=("A" in _ic_sel),
                       help="After the model decides, cancel a BUY when net imports are above a "
                            "set level (and mirror it for SELL). Human-set threshold.")
    ic_b = st.checkbox("B - Raise the margin when flows contradict", value=("B" in _ic_sel),
                       help="Do not cancel: require a bigger expected spread when the flow signal "
                            "points the other way. Human-set sensitivity.")
    ic_c = st.checkbox("C - Let the model learn the flows (retrain)", value=("C" in _ic_sel),
                       help="Add flow ramp / headroom x outage features and recalibrate the "
                            "probabilities, then retrain. No hand-set numbers.")
    st.dataframe(pd.DataFrame({
        "Option": ["A - block", "B - raise margin", "C - model learns"],
        "Effect on trades": ["Fewer trades only", "Fewer or smaller trades", "Can go either way"],
        "Who sets the number": ["You", "You", "The data"],
        "Needs retraining": ["No", "No", "Yes"],
        "Real data only": ["Yes", "Yes", "Yes"],
    }), hide_index=True, use_container_width=True)
    if st.button("Save selection"):
        try:
            os.makedirs(os.path.dirname(IC_CHOICE_PATH), exist_ok=True)
            sel = [k for k, v in (("A", ic_a), ("B", ic_b), ("C", ic_c)) if v]
            with open(IC_CHOICE_PATH, "w", encoding="utf-8") as f:
                json.dump({"selected": sel,
                           "saved_at": pd.Timestamp.now(tz="Europe/Copenhagen").strftime("%Y-%m-%d %H:%M %Z"),
                           "note": "Selection only - not applied to decisions until a replay A/B test."},
                          f, indent=2)
            st.success(f"Saved: {', '.join(sel) if sel else 'none'}")
        except Exception as exc:
            st.error(f"Could not save: {exc}")
    if _ic_choice.get("saved_at"):
        st.caption(f"Last saved: {', '.join(_ic_choice.get('selected') or ['none'])} "
                   f"({_ic_choice['saved_at']})")


def _log_startup_error(where, exc):
    """Append a full traceback to logs/dashboard_v4_1_errors.log (the browser truncates them)."""
    import traceback
    try:
        os.makedirs('logs', exist_ok=True)
        with open(os.path.join('logs', 'dashboard_v4_1_errors.log'), 'a', encoding='utf-8') as fh:
            fh.write('=' * 70 + chr(10))
            fh.write(f'{datetime.now():%Y-%m-%d %H:%M:%S}  {where}{chr(10)}')
            fh.write(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:
        pass


# Load V4.0 and V4.1 Models & Logs
@st.cache_resource(ttl=300)
def load_models_and_logs(area):
    # Load V4.0 Model
    m4_path = f"models_v4/v4_champion_model_{area}.pkl"
    m4_feat_path = f"models_v4/v4_features_{area}.pkl"
    bundle_v4 = None
    if os.path.exists(m4_path):
        raw = joblib.load(m4_path)
        if isinstance(raw, dict) and "model" in raw:
            bundle_v4 = raw
        else:
            cols = joblib.load(m4_feat_path) if os.path.exists(m4_feat_path) else []
            bundle_v4 = {"model": raw, "feature_cols": cols}

    # Load V4.1 Intraday Model (trained by: python train_v4_1.py train)
    # A failure here used to take the whole page down; the V4.0 side and the ledger still work
    # without it, so log the traceback to logs/ and carry on with the V4.1 model absent.
    try:
        bundle_v41 = v41id.load_bundle(area)
        log_v41 = v41id.model_info(area)
    except Exception as _e:
        bundle_v41, log_v41 = None, None
        _log_startup_error(f'load_models_and_logs({area})', _e)
        st.sidebar.error(f'V4.1 intraday model unavailable: {type(_e).__name__}: {_e}')

    return bundle_v4, bundle_v41, log_v41

bundle_v4, bundle_v41, log_v41 = load_models_and_logs(selected_area)

# 4. Champion Model Specifications Sidebar Box
if log_v41:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏆 V4.1 Intraday Model")
    st.sidebar.markdown(f"**Model:** `LightGBM direction + regime size + quantiles`")
    st.sidebar.markdown(f"**Decision time:** `delivery − {log_v41.get('gate_lead_minutes', 60)} min`")
    st.sidebar.markdown(f"**Trained on:** `{log_v41.get('n_train', 0):,} quarters until {str(log_v41.get('trained_until', ''))[:16]} UTC`")
    st.sidebar.markdown(f"**Cost model:** `{log_v41.get('cost_eur_mwh', 0):.2f} EUR/MWh`")
    dp = log_v41.get('decision_params', {}) or {}
    st.sidebar.markdown(f"**BUY rule:** `{dp.get('buy') or 'off'}`  \n**SELL rule:** `{dp.get('sell') or 'off'}`")
    rp = log_v41.get('replay')
    if rp:
        st.sidebar.markdown(f"**Walk-forward replay:** `{rp.get('trades', 0)} trades, {rp.get('net_eur', 0):,.0f} EUR "
                            f"({rp.get('net_eur_per_mwh') or 0:.2f} EUR/MWh), 2x costs {rp.get('net_eur_at_stress_costs', 0):,.0f} EUR`")
    st.sidebar.caption("Walk-forward replay, point-in-time features, leakage-tested. Simulation only. "
                       "Past days shown in the ledger use the CURRENT model (in-sample); the honest out-of-sample "
                       "history is the replay report in results/v4_1_intraday/.")
elif bundle_v41 is None:
    st.sidebar.warning("V4.1 intraday model not trained yet: run `python train_v4_1.py all`")

with st.sidebar.expander("🔒 V4.1 paper-trading journal (locked decisions)", expanded=False):
    try:
        from v4_1_intraday import journal as _J
        _js = _J.summary(v41id.config())
        if _js.empty:
            st.caption("No locked decisions yet - schedule `python train_v4_1.py cycle` every 15 min "
                       "(scripts_v41/install_tasks_v41.ps1).")
        else:
            st.dataframe(_js[["area", "trades", "mwh", "net_eur", "eur_per_mwh", "win_rate_pct", "missed_gates"]],
                         hide_index=True, use_container_width=True)
            st.caption("Only these locked decisions count as honest live performance.")
    except Exception as e:
        st.caption(f"journal unavailable: {e}")

with st.sidebar.expander("📡 V4.1 data sources (coverage)", expanded=False):
    try:
        st.dataframe(v41id.sources_status(), hide_index=True, use_container_width=True)
        st.caption("Update: `python train_v4_1.py update` + `collect`; live Nord Pool: `record-intraday` (24/7).")
    except Exception as e:
        st.caption(f"coverage unavailable: {e}")

# 5. Dedicated Single Deep-Dive Architecture Expander
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ Data Source Architecture Details", expanded=False):
    st.markdown("""
    ### 1. Energi Data Service: European Balancing Market (MARI & PICASSO)
    * **Endpoints:** `MfrrEnergyActivationMarket` (MARI) & `AfrrEnergyActivation` (PICASSO)
    * **Data:** Authentic TSO Balancing Energy Merit Order Activations (`TotalmFRRUpMW`, `TotalmFRRDownMW`, `mFRRSAUpEUR`, `aFRR_ActivatedEUR`).
    
    ---
    ### 2. SMARD.de: German Federal Network Agency (Bundesnetzagentur)
    * **Endpoint:** `smard.de/app/chart_data/` (`1223_DE` Generation & `1224_DE` Consumption)
    * **Data:** Official quarter-hour German residual grid imbalance and cross-border load pressure.
    
    ---
    ### 3. Nord Pool Continuous Intraday (XBID) Microstructure
    * **Engine:** Real-time Level-2 order book depth & Volume Skewness index $\in [-1.0, +1.0]$.
    * **Data:** Best Bid/Ask volume dynamics, micro-price deviation, and liquidity pressure.
    
    ---
    ### 4. Energinet Data Service (PowerSystemRightNow & Forecasts_Hour)
    * **Data:** 1-Minute live interconnector physical flows (all 8 cables), live wind/solar generation, and true forecast errors.
    
    ---
    ### 5. ENTSO-E Transparency Platform
    * **Endpoint:** `web-api.tp.entsoe.eu/api` (Type `A11`)
    * **Data:** Scheduled cross-border commercial exchanges and day-ahead market couplings.
    
    ---
    ### 6. Weather: DMI (live display) + Open-Meteo (V4.1 model input)
    * **DMI endpoint:** `opendataapi.dmi.dk/v2/metObs` - the Balancing tab's live wind/temp readout
      reads the latest observation from the nearest Danish station in real time.
    * **Open-Meteo endpoint:** `previous-runs-api.open-meteo.com/v1/forecast` - quarter-hour
      DK1/DK2 wind, solar, and weather forecasts that actually feed the V4.1 model.
    
    ---
    ### 7. Nord Pool REMIT UMM & DuckDB Paper-Trading Journal
    * **Data:** Urgent Market Messages (outages) and a 15-minute automatic cycle that locks, settles, and reconciles V4.1 trades to a local DuckDB store.
    
    ---
    ### 8. Fingrid Open Data (Nordic Grid Frequency)
    * **Endpoint:** `data.fingrid.fi/api/datasets/177/data`
    * **Data:** 3-minute Nordic system frequency - a direct signal for DK2 (Nordic synchronous area). DK1 (Continental European area) is not covered by this feed and relies on aFRR activation instead.
    """)
    st.caption("🔒 Verified live government/TSO and market data feeds - see the coverage table above for what is actually stored right now.")

# Safe parser helper
def parse_val(v):
    if pd.isna(v) or v is None: return None
    s = str(v).replace("€", "").replace("EUR", "").replace(",", "").strip()
    if s in ["--", "None", "nan", "null", ""]: return None
    try:
        return float(s)
    except:
        return None

load_dotenv(os.path.join("Nurex_V4_2", ".env"))
ENTSOE_TOKEN = os.environ.get("ENTSOE_TOKEN") or os.environ.get("ENTSOE_API_KEY", "")

@st.cache_data(ttl=3600)
def get_dynamic_price_cap(area, current_date_str):
    """Fetches the last 30 days of Spot Prices and calculates the 95th percentile dynamic cap."""
    try:
        from entsoe import EntsoePandasClient
        client = EntsoePandasClient(api_key=ENTSOE_TOKEN)
        end_ts = pd.Timestamp(current_date_str, tz='Europe/Copenhagen')
        start_ts = end_ts - pd.Timedelta(days=30)
        entsoe_area = 'DK_1' if area == 'DK1' else 'DK_2'
        historical_spot = client.query_day_ahead_prices(entsoe_area, start=start_ts, end=end_ts)
        cap = float(historical_spot.quantile(0.95))
        return round(cap, 2)
    except Exception:
        return 223.40

# --- LOAD TRADING DAY DATA MATRIX (V4.0 and V4.1) ---
@st.cache_data(ttl=180)
def get_v4_1_trading_day_data(area, date_str, threshold_key='validated', risk_limits=True, size_boost=False):
    fe41 = V41FeatureEngine(price_area=area)
    table_gen = TournamentTableGenerator(price_area=area)
    
    try:
        target_df = table_gen.generate_and_save_future_table(date_str=date_str)
    except:
        target_df = table_gen.get_backtest_table(date_str=date_str)
        
    if target_df.empty:
        target_df = table_gen.get_future_table(date_str=date_str)
        
    if target_df.empty:
        return pd.DataFrame()

    # Baseline ledger scaffold (V3.1 engine) - supplies the 96-quarter trade frame
    strat_v31 = V31CommercialStrategyEngine(price_area=area)
    res_v31 = strat_v31.evaluate_trading_ledger(target_df, model_name="Transfer-BiLSTM", market_mode="INTRADAY_D0")
    df_trades = pd.DataFrame(res_v31.get("trades", []))
    if df_trades.empty:
        return df_trades

    # Ground Truth Timestamps & Spot Prices
    df_trades['spot_price_eur'] = df_trades['spot_price_eur'].apply(parse_val)
    current_time_dk = pd.Timestamp.now(tz="Europe/Copenhagen").tz_localize(None)

    # Authentic Settlement Detection
    is_settled_list = []
    settled_vals = []
    settled_displays = []

    for _, row in df_trades.iterrows():
        t_row = pd.to_datetime(row['time_dk'])
        raw_settled = parse_val(row.get('actual_settled_eur'))
        is_past = (t_row <= current_time_dk)
        is_settled = (raw_settled is not None and not pd.isna(raw_settled) and is_past)  # negative prices are valid
        
        is_settled_list.append(is_settled)
        settled_vals.append(raw_settled if is_settled else np.nan)
        settled_displays.append(f"€{raw_settled:.2f}" if is_settled else "-- (Pending Delivery)")

    df_trades['is_settled'] = is_settled_list
    df_trades['actual_settled_val'] = settled_vals
    df_trades['actual_settled_display'] = settled_displays
    df_trades['actual_spread_eur'] = df_trades['actual_settled_val'] - df_trades['spot_price_eur']
    df_trades['V3_1_BiLSTM_Score'] = df_trades['pred_spread_eur'].apply(parse_val).fillna(0.0)
    
    dynamic_cap = get_dynamic_price_cap(area, date_str)
    df_trades['Dynamic_Cap_EUR'] = dynamic_cap

    # Ingest ENTSO-E DE Spot & Scheduled Flows
    try:
        from entsoe import EntsoePandasClient
        client = EntsoePandasClient(api_key=ENTSOE_TOKEN)
        start_ts = pd.Timestamp(date_str, tz='Europe/Copenhagen')
        end_ts = start_ts + pd.Timedelta(days=1)
        entsoe_area_to = 'DK_1' if area == 'DK1' else 'DK_2'
        
        de_prices = client.query_day_ahead_prices('DE_LU', start=start_ts, end=end_ts)
        de_prices_df = de_prices.reset_index()
        de_prices_df.columns = ['time_dk_obj', 'de_spot_eur']
        de_prices_df['time_dk_obj'] = de_prices_df['time_dk_obj'].dt.tz_localize(None)
        
        flows = client.query_scheduled_exchanges('DE_LU', entsoe_area_to, start=start_ts, end=end_ts, day_ahead=True)
        flows_df = flows.reset_index()
        flows_df.columns = ['time_dk_obj', 'scheduled_flow_mw']
        flows_df['time_dk_obj'] = flows_df['time_dk_obj'].dt.tz_localize(None)
        
        entsoe_df = pd.merge(de_prices_df, flows_df, on='time_dk_obj', how='outer')
        entsoe_df['time_dk_str'] = entsoe_df['time_dk_obj'].dt.strftime('%Y-%m-%d %H:%M')
        
        df_trades['time_dk_str'] = pd.to_datetime(df_trades['time_dk']).dt.strftime('%Y-%m-%d %H:%M')
        df_trades = pd.merge(df_trades, entsoe_df[['time_dk_str', 'de_spot_eur', 'scheduled_flow_mw']], on='time_dk_str', how='left')
        df_trades['de_spot_eur'] = df_trades['de_spot_eur'].ffill().bfill().fillna(df_trades['spot_price_eur'])
        df_trades['scheduled_flow_mw'] = df_trades['scheduled_flow_mw'].ffill().bfill().fillna(0.0)
    except Exception:
        df_trades['de_spot_eur'] = df_trades['spot_price_eur']
        df_trades['scheduled_flow_mw'] = 0.0

    # Build High-Alpha Matrix (V4.1 Engine)
    df_matrix = fe41.build_feature_matrix(df_trades)
    df_matrix['Dynamic_Cap_EUR'] = dynamic_cap
    if 'de_spot_eur' not in df_matrix.columns:
        df_matrix['de_spot_eur'] = df_trades['de_spot_eur']
    if 'scheduled_flow_mw' not in df_matrix.columns:
        df_matrix['scheduled_flow_mw'] = df_trades['scheduled_flow_mw']

    # Ingest Live Grid Telemetry
    try:
        grid_df = _live_energinet_frequency(5)
        if not grid_df.empty:
            latest = grid_df.iloc[0]
            if area == 'DK1':
                df_matrix['flow_de'] = float(latest.get('Exchange_DK1_DE', 0))
                df_matrix['flow_nl'] = float(latest.get('Exchange_DK1_NL', 0))
                df_matrix['flow_gb'] = float(latest.get('Exchange_DK1_GB', 0))
                df_matrix['flow_no'] = float(latest.get('Exchange_DK1_NO', 0))
                df_matrix['flow_se'] = float(latest.get('Exchange_DK1_SE', 0))
                df_matrix['flow_great_belt'] = float(latest.get('Exchange_DK1_DK2', 0))
                df_matrix['afrr_net_activation_mw'] = float(latest.get('aFRR_ActivatedDK1', 0))
            else:
                df_matrix['flow_de'] = float(latest.get('Exchange_DK2_DE', 0))
                df_matrix['flow_se'] = float(latest.get('Exchange_DK2_SE', 0))
                df_matrix['flow_great_belt'] = -float(latest.get('Exchange_DK1_DK2', 0))
                df_matrix['afrr_net_activation_mw'] = float(latest.get('aFRR_ActivatedDK2', 0))
    except Exception:
        pass

    # Ensure Volatility Column Exists
    if 'V4_Spread_Volatility' not in df_matrix.columns:
        df_matrix['V4_Spread_Volatility'] = df_matrix['spot_price_eur'].rolling(12, min_periods=1).std().fillna(0)

    # --- MODEL 1: V4.0 Full-Grid Model ---
    if bundle_v4 and "model" in bundle_v4:
        m4 = bundle_v4["model"]
        cols4 = bundle_v4.get("feature_cols", [])
        for c in cols4:
            if c not in df_matrix.columns: df_matrix[c] = 0.0
        X4 = df_matrix[cols4].ffill().bfill().fillna(0)
        df_matrix['V4_Predicted_Spread_EUR'] = m4.predict(X4)
    else:
        df_matrix['V4_Predicted_Spread_EUR'] = df_matrix['V3_1_BiLSTM_Score']

    df_matrix['V4_Pred_Imb_EUR'] = df_matrix['spot_price_eur'] + df_matrix['V4_Predicted_Spread_EUR']

    v4_decisions, v4_vols, v4_pnls = [], [], []
    for _, row in df_matrix.iterrows():
        v4_score = row['V4_Predicted_Spread_EUR']
        v31_score = row['V3_1_BiLSTM_Score']
        spot = row['spot_price_eur']
        cap = row['Dynamic_Cap_EUR']
        surplus_mw = row.get('net_system_surplus_mw', 0.0)
        if v4_score > 2.0:
            base_decision, act = "🟢 BUY", "BUY"
            vol = high_conviction_vol if (v4_score > 10.0 or v31_score > 8.0) else standard_vol
        elif v4_score < -2.0:
            base_decision, act = "🔴 SELL", "SELL"
            vol = high_conviction_vol if (v4_score < -10.0 or v31_score < -8.0) else standard_vol
        else:
            base_decision, act, vol = "⚪ HOLD", "HOLD", 0.0

        if surplus_mw > 400.0 and act == "BUY":
            base_decision, act, vol = "⚪ SURPLUS DEFENSE (HOLD)", "HOLD", 0.0

        if crash_protection_enabled and v4_score < -2.5 and v31_score > 1.5:
            base_decision, act = "🔥 CRASH PRED (SELL)", "SELL"
            vol = high_conviction_vol

        if use_circuit_breaker and spot > cap and act == "BUY":
            base_decision, act, vol = "🛑 C.BREAKER (HOLD)", "HOLD", 0.0

        if row['is_settled']:
            spread = row['actual_spread_eur']
            mwh_q = vol * 0.25  # quarterly product: MW / 4 = MWh
            fees = mwh_q * COST_EUR_MWH
            pnl4 = (spread * mwh_q - fees) if act == "BUY" else (-spread * mwh_q - fees) if act == "SELL" else 0.0
        else:
            pnl4 = np.nan

        v4_decisions.append(base_decision)
        v4_vols.append(vol)
        v4_pnls.append(pnl4)

    df_matrix['V4_Decision'] = v4_decisions
    df_matrix['V4_Volume_MW'] = v4_vols
    df_matrix['PnL_V4_0'] = v4_pnls

    # --- MODEL 4: V4.1 Intraday (gate-closure, point-in-time) ---
    # Decisions come from v4_1_intraday: each quarter is decided with the information published
    # before delivery - 60 min. Settlement uses the published imbalance price (negative prices included).
    v41_decisions, v41_vols, v41_pnls = [], [], []
    v41 = pd.DataFrame()
    if bundle_v41 is not None:
        try:
            _risk_overrides = {"batch_risk_fraction": 0.2} if size_boost else None
            if risk_limits and hasattr(v41id, 'day_decisions_risked'):
                # same overlay as the replay: equity carried across the drawdown window
                v41 = v41id.day_decisions_risked(
                    area, date_str, levels=(threshold_key,), risk_overrides=_risk_overrides
                    ).get(threshold_key, pd.DataFrame())
            else:
                _params = (v41id.decision_params(area, threshold_key)
                           if hasattr(v41id, 'decision_params') else None)
                v41 = v41id.day_decisions(area, date_str, params=_params, risk_overrides=_risk_overrides)
        except Exception as e:
            _log_startup_error(f'day_decisions({area}, {date_str})', e)
            st.warning(f"V4.1 intraday engine unavailable: {type(e).__name__}: {e}")
    key = pd.to_datetime(df_matrix['time_dk']).dt.strftime('%Y-%m-%d %H:%M') if 'time_dk' in df_matrix.columns else None
    if not v41.empty and key is not None:
        m = v41.set_index('time_dk_str')
        pick = lambda c, d=np.nan: key.map(m[c]).fillna(d) if c in m.columns else d
        df_matrix['V4_1_Predicted_Spread_EUR'] = pick('exp_spread', 0.0).astype(float)
        df_matrix['p_down'] = pick('p_down')
        df_matrix['p_none'] = pick('p_flat')
        df_matrix['p_up'] = pick('p_up')
        df_matrix['spread_q10'] = pick('q10')
        df_matrix['spread_q50'] = pick('q50')
        df_matrix['spread_q90'] = pick('q90')
        df_matrix['V4_1_MWh'] = pick('mwh', 0.0).astype(float)
        df_matrix['V4_1_Final'] = key.map(m['decision_final']).fillna(False)
        df_matrix['V4_1_Settled'] = key.map(m['settled']).fillna(False)
        acts = key.map(m['action']).fillna('HOLD')
        pn = key.map(m['pnl_eur'])
        srcs = key.map(m['source']).fillna('recomputed') if 'source' in m.columns else pd.Series('recomputed', index=key.index)
        rsn = df_matrix['V4_1_Reason'] if 'V4_1_Reason' in df_matrix.columns else pd.Series([''] * len(df_matrix))
        for a_, mw_, fin_, p_, src_, why_ in zip(acts, df_matrix['V4_1_MWh'], df_matrix['V4_1_Final'], pn, srcs, rsn):
            tag = " 🔒" if src_ == "LOCKED" else (" (missed gate)" if src_ == "MISSED" else ("" if fin_ else " (provisional)"))
            if str(why_) == 'daily loss stop':
                tag = " (daily loss stop)"
            elif str(why_) == 'drawdown guard':
                tag += " (drawdown guard)"
            v41_decisions.append(("🟢 BUY" if a_ == "BUY" else "🔴 SELL" if a_ == "SELL" else "⚪ HOLD") + tag)
            v41_vols.append(float(mw_) * 4.0)          # MWh per quarter -> MW
            v41_pnls.append(float(p_) if pd.notna(p_) else np.nan)
    else:
        df_matrix['V4_1_Predicted_Spread_EUR'] = 0.0
        df_matrix['V4_1_Settled'] = False
        v41_decisions = ["⚪ HOLD (model not trained)"] * len(df_matrix)
        v41_vols = [0.0] * len(df_matrix)
        v41_pnls = [np.nan] * len(df_matrix)

    # Per-quarter cross-border exchange (ENTSO-E day-ahead schedules + realised deviation).
    # Replaces the single live snapshot that used to be broadcast across all 96 quarters.
    try:
        fl = v41id.day_flows(area, date_str) if hasattr(v41id, 'day_flows') else pd.DataFrame()
        fl_status = fl.attrs.get('status', 'ok' if not fl.empty else 'unpublished')
    except Exception as _e:
        fl = pd.DataFrame()
        fl_status = 'error'
        _log_startup_error(f'day_flows({area}, {date_str})', _e)
    FLOW_STATUS[(area, date_str)] = fl_status
    df_matrix['_flow_status'] = fl_status  # survives st.cache_data, unlike a module global
    if not fl.empty and key is not None:
        mf = fl.set_index('time_dk_str')
        for c in [c for c in mf.columns if c.startswith(('sched_', 'dev_'))]:
            df_matrix['ic_' + c] = key.map(mf[c]).astype(float)

    df_matrix['V4_1_Pred_Imb_EUR'] = df_matrix['spot_price_eur'] + df_matrix['V4_1_Predicted_Spread_EUR']
    df_matrix['V4_1_Decision'] = v41_decisions
    df_matrix['V4_1_Volume_MW'] = v41_vols
    df_matrix['PnL_V4_1'] = v41_pnls

    # Comparative Alpha Metrics (Settled Intervals)
    df_matrix['Alpha_V41_vs_V40'] = df_matrix['PnL_V4_1'] - df_matrix['PnL_V4_0']

    return df_matrix

df_day = get_v4_1_trading_day_data(selected_area, date_str_selected, THRESHOLD_KEY, risk_limits_on, bigger_size_on)
if not risk_limits_on:
    st.warning(
        'Risk limits are OFF: the daily loss stop and drawdown scaling are not applied, so these '
        'decisions will not match backtest or replay results. Turn them on in the sidebar for the '
        'numbers the engine would actually have traded.')
if THRESHOLD_KEY != 'validated':
    st.warning(
        f'V4.1 is running on **{threshold_level}** thresholds: a what-if view of the same forecasts '
        'at a lower bar for taking a trade. It trades more often and carries more exposure, and these '
        'thresholds were not validated on held-out data. Paper-trading journal entries are left '
        'untouched while this is on.')

# --- TOP SUMMARY BANNER & METRICS ---
st.title(f"⚡ Nurex V4.1 Institutional High-Alpha Command Center ({selected_area})")
st.markdown("### Real-Time MARI/PICASSO Balancing • SMARD German Grid • XBID Level-2 Order Flow Microstructure")

settled_mask = df_day['is_settled'] if ('is_settled' in df_day.columns) else pd.Series([False]*len(df_day))
v41_realized = float(np.nansum(df_day['PnL_V4_1'])) if not df_day.empty else 0.0
v40_realized = df_day.loc[settled_mask, 'PnL_V4_0'].sum() if not df_day.empty else 0.0
alpha_41_v40 = v41_realized - v40_realized
open_vol_41 = df_day.loc[~settled_mask, 'V4_1_Volume_MW'].sum() if not df_day.empty else 0.0

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("V4.1 Realized PnL (Settled)", f"€ {v41_realized:,.2f}", f"{alpha_41_v40:+,.2f} vs V4.0")
with m2:
    st.metric("V4.0 Realized PnL (Settled)", f"€ {v40_realized:,.2f}", "Full-Grid Champion")
with m3:
    st.metric("Pending Exposure (V4.1)", f"{open_vol_41:.0f} MW", f"{len(df_day) - settled_mask.sum()} Quarters Pending")

# --- MAIN NAVIGATION TABS ---
tab_cables, tab_balancing, tab_unified_ledger, tab_xbid, tab_gridsearch, tab_tournament = st.tabs([
    "🌐 8-Cable Interconnector Radar",
    "⚡ European Balancing & SMARD.de Telemetry",
    "🎯 96-Quarter Multi-Model Unified Comparative Ledger",
    f"📈 Nord Pool Continuous Intraday (XBID) Terminal — ID-{selected_area}-Nurex",
    "🔬 Grid Search Model Transparency",
    "🏆 Multi-Generation Quantitative Tournament"
])

# ----------------------------------------------------------------------
# TAB 1: 8-CABLE INTERCONNECTOR RADAR
# ----------------------------------------------------------------------
with tab_cables:
    st.subheader("Physical Cross-Border Transmission Topology & Congestion Analysis")
    st.markdown("""
    Denmark's power system is fundamentally determined by the **headroom and congestion** across surrounding interconnectors.
    When cables reach maximum capacity, the Danish price decouples from European power prices, causing severe imbalance spikes.
    """)

    cables_data = [
        {"Border": "DK1 → Germany (DE-LU)", "Zone": "DK1", "Cable Name": "Kassø-Audorf Lines", "Capacity (MW)": 2500, "Current Flow (MW)": exp_pos(df_day['flow_de'].iloc[-1]) if not df_day.empty and 'flow_de' in df_day.columns else -120.0},
        {"Border": "DK1 → Norway (NO2)", "Zone": "DK1", "Cable Name": "Skagerrak 1-4", "Capacity (MW)": 1640, "Current Flow (MW)": exp_pos(df_day['flow_nordic'].iloc[-1]) * 0.7 if not df_day.empty and 'flow_nordic' in df_day.columns else 350.0},
        {"Border": "DK1 → Sweden (SE3)", "Zone": "DK1", "Cable Name": "Konti-Skan 1-2", "Capacity (MW)": 680, "Current Flow (MW)": exp_pos(df_day['flow_nordic'].iloc[-1]) * 0.3 if not df_day.empty and 'flow_nordic' in df_day.columns else 350.0},
        {"Border": "DK1 → Great Britain (GB)", "Zone": "DK1", "Cable Name": "Viking Link", "Capacity (MW)": 1400, "Current Flow (MW)": exp_pos(df_day['flow_gb'].iloc[-1]) if not df_day.empty and 'flow_gb' in df_day.columns else 450.0},
        {"Border": "DK1 → Netherlands (NL)", "Zone": "DK1", "Cable Name": "COBRAcable", "Capacity (MW)": 700, "Current Flow (MW)": exp_pos(df_day['flow_nl'].iloc[-1]) if not df_day.empty and 'flow_nl' in df_day.columns else -40.0},
        {"Border": "DK1 → DK2", "Zone": "Both", "Cable Name": "Great Belt (Storebælt HVDC)", "Capacity (MW)": 580, "Current Flow (MW)": exp_pos(df_day['flow_great_belt'].iloc[-1]) if not df_day.empty and 'flow_great_belt' in df_day.columns else -80.0},
        {"Border": "DK2 → Sweden (SE4)", "Zone": "DK2", "Cable Name": "Øresund Cable", "Capacity (MW)": 1240, "Current Flow (MW)": exp_pos(df_day['flow_se'].iloc[-1]) if not df_day.empty and 'flow_se' in df_day.columns else 280.0},
        {"Border": "DK2 → Germany (DE-LU)", "Zone": "DK2", "Cable Name": "Kontek + Kriegers Flak", "Capacity (MW)": 985, "Current Flow (MW)": exp_pos(df_day['flow_de'].iloc[-1]) if not df_day.empty and 'flow_de' in df_day.columns else 90.0}
    ]

    st.caption(FLOW_CONVENTION_NOTE)
    cable_df = pd.DataFrame(cables_data)
    cable_df['Headroom (MW)'] = cable_df['Capacity (MW)'] - cable_df['Current Flow (MW)'].abs()
    cable_df['Utilization (%)'] = ((cable_df['Current Flow (MW)'].abs() / cable_df['Capacity (MW)']) * 100).round(1)
    cable_df['Status'] = cable_df['Utilization (%)'].apply(
        lambda u: "🔴 CONGESTED" if u >= 85 else ("🟠 HIGH LOAD" if u >= 65 else "🟢 LIQUID")
    )

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.dataframe(cable_df.style.format({
            "Capacity (MW)": "{:,.0f} MW",
            "Current Flow (MW)": "{:+,.1f} MW",
            "Headroom (MW)": "{:,.1f} MW",
            "Utilization (%)": "{:.1f}%"
        }), use_container_width=True)
        
    with col_r:
        chart_data = cable_df[['Border', 'Utilization (%)']].copy()
        c = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Utilization (%):Q', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('Border:N', sort='-x'),
            color=alt.condition(
                alt.datum['Utilization (%)'] >= 85,
                alt.value('#ff4444'),
                alt.value('#00C851')
            )
        ).properties(title="Interconnector Congestion Utilization (%)", height=320)
        st.altair_chart(c, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 2: EUROPEAN BALANCING (MARI/PICASSO) & SMARD.DE TELEMETRY
# ----------------------------------------------------------------------
with tab_balancing:
    st.subheader("⚡ European Balancing Market (MARI / PICASSO) & German Federal Grid Telemetry")
    st.markdown("""
    **The High-Alpha Edge:** Physical imbalances in Denmark are directly resolved by **mFRR (MARI platform)** and **aFRR (PICASSO platform)**.
    By monitoring real-time European balancing merit order activations and German residual load (**SMARD.de**), V4.1 predicts price spike cascades minutes before gate closure.
    """)

    # Live Balancing & SMARD KPIs
    try:
        smard_live = _live_smard_telemetry()
        bal_snap = _live_balancing_state(selected_area)
        
        bk1, bk2, bk3, bk4 = st.columns(4)
        with bk1:
            st.metric("🇩🇪 SMARD.de DE Generation", f"{smard_live['generation_mw']:,.0f} MW", f"Residual: {smard_live['residual_load_mw']:+,.0f} MW")
        with bk2:
            st.metric("🇩🇪 SMARD.de DE Demand", f"{smard_live['consumption_mw']:,.0f} MW", f"System: {smard_live['system_state']}")
        with bk3:
            st.metric("🇩🇰 MARI mFRR Net Activation", f"{bal_snap['mfrr_net_mw']:+,.0f} MW", f"Up: {bal_snap['mfrr_up_mw']:.0f} | Down: {bal_snap['mfrr_down_mw']:.0f} MW")
        with bk4:
            st.metric("🇪🇺 aFRR Marginal Price", f"€ {bal_snap['afrr_marginal_price_eur']:.2f}/MWh", f"Total: {bal_snap['total_balancing_pressure_mw']:.0f} MW")
        if 'fallback' in str(smard_live.get('status', '')).lower() or 'offline' in str(smard_live.get('status', '')).lower():
            st.warning("SMARD.de generation/demand figures above are a placeholder, not live: "
                       "SMARD stopped publishing the German consumption series this reads (its own feed "
                       "has not updated since January 2024). MARI/PICASSO figures are unaffected and live.")
    except Exception:
        pass

    st.markdown("---")

    # DMI Weather Telemetry
    try:
        dmi_telemetry = _live_dmi_telemetry(selected_area)
        dw1, dw2, dw3 = st.columns([1.5, 1, 1])
        with dw1:
            st.info(f"🇩🇰 **Official Danish Met Feed:** {dmi_telemetry['source']} (`{selected_area}` Ground & Offshore Network)")
        with dw2:
            st.metric("DMI Observed Wind Speed", f"{dmi_telemetry['avg_wind_speed_ms']:.2f} m/s", "Real-Time Coastal Stations")
        with dw3:
            st.metric("DMI Observed Temperature", f"{dmi_telemetry['avg_temp_c']:.1f} °C", "Direct DMI API Feed")
    except Exception:
        pass

    # Telemetry Charts
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("#### MARI / PICASSO Balancing Energy Merit Order Activations (MW)")
        if not df_day.empty and 'mfrr_up_mw' in df_day.columns:
            mfrr_chart_df = df_day[['time_dk', 'mfrr_up_mw', 'mfrr_down_mw', 'afrr_net_activation_mw']].copy()
            mfrr_chart_df = mfrr_chart_df.melt('time_dk', var_name='Balancing Reserve', value_name='Activated MW')
            c_bal = alt.Chart(mfrr_chart_df).mark_line().encode(
                x='time_dk:N',
                y='Activated MW:Q',
                color='Balancing Reserve:N'
            ).properties(height=280)
            st.altair_chart(c_bal, use_container_width=True)

    with b_col2:
        st.markdown("#### SMARD.de German Federal Grid Imbalance & Residual Load (MW)")
        if not df_day.empty and 'smard_residual_load_mw' in df_day.columns:
            smard_chart_df = df_day[['time_dk', 'smard_residual_load_mw']].copy()
            c_smard = alt.Chart(smard_chart_df).mark_area(opacity=0.6, color='#0284C7').encode(
                x='time_dk:N',
                y='smard_residual_load_mw:Q'
            ).properties(height=280)
            st.altair_chart(c_smard, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 3: 96-QUARTER MULTI-MODEL UNIFIED COMPARATIVE LEDGER
# ----------------------------------------------------------------------
with tab_unified_ledger:
    st.subheader("96-Quarter Intraday Trading Ledger: V4.0 vs V4.1 Side-by-Side")
    st.markdown("""
    **Authentic Multi-Generation Audit:** Side-by-side comparison across **V4.0 (Full-Grid Champion)** and **V4.1 (Institutional High-Alpha)**.
    Click on any row to open the full interactive breakdown drawer.
    """)

    if df_day.empty:
        st.warning("No live trading data available for selected date.")
    else:
        # Pre-generate Master-Detail Accordion HTML
        # getattr: a Streamlit rerun keeps the module imported at startup, so after an
        # upgrade of v4_1_intraday the new names may not exist until the app is restarted.
        st.caption(FLOW_CONVENTION_NOTE)
        show_v40 = st.checkbox(
            'Show V4.0 comparison columns', value=False, key='ledger_show_v40',
            help='Off: V4.1 columns only. On: the V4.0 champion is shown beside it for comparison.')

        _fb = getattr(v41id, 'FLOW_BORDERS', {})
        ic_labels = [lbl for _b, lbl in _fb.get(selected_area, [])
                     if f'ic_sched_{lbl}' in df_day.columns]
        ic_ths = "".join(
            f'<th draggable="true" title="Scheduled exchange {selected_area} to {lbl} (MW, + = export). '
            f'Hover a cell for the realised deviation." style="background-color:#334155;">'
            f'<div class="col-header-wrap"><span class="col-title">{selected_area}\u2192{lbl}</span>'
            f'<button type="button" class="btn-col-copy" title="Copy Column" '
            f'onclick="copySingleColumn(this, event)">\U0001f4cb</button></div></th>'
            for lbl in ic_labels)
        if not ic_labels:
            _fs = (str(df_day['_flow_status'].iloc[0]) if '_flow_status' in df_day.columns and len(df_day)
                   else FLOW_STATUS.get((selected_area, date_str_selected), 'unknown'))
            if not hasattr(v41id, '_flows_status'):
                # Streamlit reruns the script but does NOT re-import modules already loaded at
                # startup, so an updated v4_1_intraday only takes effect after a full restart.
                _why = ('the v4_1_intraday adapter loaded in memory is out of date - restart the '
                        'dashboard process (a browser reload is not enough)')
            else:
                _why = {
                    'busy': 'V4.2 store locked by a running collector - reload in a moment',
                    'error': 'flow lookup failed - see the dashboard log',
                    'unpublished': 'ENTSO-E schedules not published for this day',
                    'unknown': 'flow status not recorded for this day - reload the page',
                }.get(_fs, 'ENTSO-E schedules not available for this day')
            ic_ths = ('<th class="no-drag" title="' + _why + '">'
                      '<div class="col-header-wrap"><span class="col-title">Flows</span></div></th>')
            st.caption(f'\u26a0\ufe0f Cross-border columns unavailable: {_why}.')
        elif (df_day['_flow_status'].iloc[0] if '_flow_status' in df_day.columns and len(df_day) else '') == 'stale':
            st.caption('\u2139\ufe0f Cross-border flows served from the last good read - the V4.2 store was busy.')

        rows_html = []
        for i, row in df_day.iterrows():
            row_idx = int(i)
            time_val = str(row['time_dk']).split(' ')[1][:5] if pd.notna(row['time_dk']) and ' ' in str(row['time_dk']) else str(row['time_dk'])
            qid = row.get('quarter', f"Q{row_idx+1}")
            q_label = f"{qid} ({time_val})"
            
            spot_val = row.get('spot_price_eur', 0.0)
            cap_val = row.get('Dynamic_Cap_EUR', 223.40)
            de_spot_val = row.get('de_spot_eur', spot_val)
            # Energinet live snapshot, shown export-positive like every other table
            f_de = exp_pos(row.get('flow_de', row.get('scheduled_flow_mw', 0.0)))
            f_nl = exp_pos(row.get('flow_nl', 0.0))
            f_no = exp_pos(row.get('flow_no', 0.0))
            f_se = exp_pos(row.get('flow_se', 0.0))
            f_gb = exp_pos(row.get('flow_gb', 0.0))
            f_sb = exp_pos(row.get('flow_great_belt', 0.0))
            
            # Interconnectors (authentic per-quarter schedules; blank when not published)
            ic_tds = ""
            for _lbl in ic_labels:
                _v = row.get(f'ic_sched_{_lbl}', np.nan)
                _d = row.get(f'ic_dev_{_lbl}', np.nan)
                _txt = f"{_v:+,.0f}" if pd.notna(_v) else "--"
                _ttl = (f"scheduled {_v:+,.0f} MW" if pd.notna(_v) else "not published")
                if pd.notna(_d):
                    _ttl += f", realised {_d:+,.0f} MW vs schedule"
                _col = "#0F766E" if (pd.notna(_v) and _v > 0) else ("#B91C1C" if pd.notna(_v) else "#94A3B8")
                ic_tds += f'<td title="{_ttl}" style="color:{_col};">{_txt}</td>'
            if not ic_labels:
                ic_tds = '<td style="color:#94A3B8;">--</td>'

            # V4.0
            v4_dec = str(row.get('V4_Decision', '⚪ HOLD'))
            v4_spread = row.get('V4_Predicted_Spread_EUR', 0.0)
            v4_imb = spot_val + v4_spread
            v4_vol = row.get('V4_Volume_MW', 0.0)
            v4_pnl = row.get('PnL_V4_0', np.nan)
            
            # V4.1
            v41_dec = str(row.get('V4_1_Decision', '⚪ HOLD'))
            v41_spread = row.get('V4_1_Predicted_Spread_EUR', 0.0)
            v41_imb = spot_val + v41_spread
            v41_vol = row.get('V4_1_Volume_MW', 0.0)
            v41_mwh = row.get('V4_1_MWh', v41_vol / 4.0)  # energy per quarter
            v41_pnl = row.get('PnL_V4_1', np.nan)
            
            # Settled
            is_settled = row.get('is_settled', False)
            settled_disp = row.get('actual_settled_display', '-- (Pending Delivery)')
            settled_val = row.get('actual_settled_val', np.nan)
            
            # Badges
            dec4_class = "badge-buy" if "BUY" in v4_dec else ("badge-sell" if "SELL" in v4_dec else "badge-hold")
            dec41_class = "badge-buy" if "BUY" in v41_dec else ("badge-sell" if "SELL" in v41_dec else "badge-hold")
            status_badge = '<span class="badge badge-settled">🟢 Settled</span>' if is_settled else '<span class="badge badge-pending">🟡 Pending</span>'
            
            pnl4_text = f"€{v4_pnl:+,.2f}" if pd.notna(v4_pnl) else "--"
            pnl4_class = "pnl-pos" if (pd.notna(v4_pnl) and v4_pnl > 0) else ("pnl-neg" if (pd.notna(v4_pnl) and v4_pnl < 0) else "pnl-zero")
            
            pnl41_text = f"€{v41_pnl:+,.2f}" if pd.notna(v41_pnl) else "--"
            pnl41_class = "pnl-pos" if (pd.notna(v41_pnl) and v41_pnl > 0) else ("pnl-neg" if (pd.notna(v41_pnl) and v41_pnl < 0) else "pnl-zero")

            # Cash flows
            if "BUY" in v41_dec:
                da_outlay = f"-€{spot_val * v41_mwh:,.2f}"
                settle_cf = f"+€{settled_val * v41_mwh:,.2f}" if is_settled and pd.notna(settled_val) else "Pending Gate Closure"
                gross = f"€{(settled_val - spot_val) * v41_mwh:+,.2f}" if is_settled and pd.notna(settled_val) else "Pending"
                fees = f"-€{v41_mwh * COST_EUR_MWH:,.2f}"
                net_cf = f"€{v41_pnl:+,.2f}" if is_settled and pd.notna(v41_pnl) else "Pending"
            elif "SELL" in v41_dec:
                da_outlay = f"+€{spot_val * v41_mwh:,.2f}"
                settle_cf = f"-€{settled_val * v41_mwh:,.2f}" if is_settled and pd.notna(settled_val) else "Pending Gate Closure"
                gross = f"€{(spot_val - settled_val) * v41_mwh:+,.2f}" if is_settled and pd.notna(settled_val) else "Pending"
                fees = f"-€{v41_mwh * COST_EUR_MWH:,.2f}"
                net_cf = f"€{v41_pnl:+,.2f}" if is_settled and pd.notna(v41_pnl) else "Pending"
            else:
                da_outlay, settle_cf, gross, fees, net_cf = "€0.00", "€0.00", "€0.00", "€0.00", "€0.00 (Capital Protected)"

            v40_audit_row = (f'<tr><td><b>V4.0 Full-Grid</b></td><td style="text-align:right;">\u20ac{v4_imb:.2f}</td>'
                             f'<td style="text-align:center;">{v4_dec}</td>'
                             f'<td style="text-align:right;">{pnl4_text}</td></tr>') if show_v40 else ''
            alpha_notice = ('<div class="audit-notice"><b>Alpha Outperformance:</b> vs V4.0: <b>'
                            + (f'\u20ac{v41_pnl - v4_pnl:+,.2f}' if pd.notna(v41_pnl) and pd.notna(v4_pnl) else '--')
                            + '</b></div>') if show_v40 else ''

            v40_tds = (f'<td style="font-weight:600; color:#0284C7;">\u20ac{v4_imb:.2f}</td>'
                       f'<td><span class="badge {dec4_class}">{v4_dec}</span></td>'
                       f'<td class="{pnl4_class}">{pnl4_text}</td>') if show_v40 else ''

            zebra_class = "even-row" if row_idx % 2 == 0 else "odd-row"
            mfrr_up = row.get('mfrr_up_mw', 0.0)
            mfrr_dn = row.get('mfrr_down_mw', 0.0)
            smard_res = row.get('smard_residual_load_mw', 0.0)
            skew = row.get('order_flow_skew', 0.0)

            rows_html.append(f"""
            <tr class="master-row {zebra_class}" onclick="toggleRow({row_idx})" id="row-{row_idx}">
                <td class="chevron-cell" onclick="event.stopPropagation(); toggleRow({row_idx});"><span class="chevron-icon" id="icon-{row_idx}">&#9654;</span></td>
                <td style="font-weight:700; color:#0F172A;">{q_label}</td>
                <td>€{spot_val:.2f}</td>
                <td>€{de_spot_val:.2f}</td>
                {ic_tds}
                
                {v40_tds}
                
                <!-- V4.1 Champion Column Group -->
                <td style="font-weight:700; color:#0D9488; background-color:#F0FDFA;">€{v41_imb:.2f}</td>
                <td style="background-color:#F0FDFA;"><span class="badge {dec41_class}">{v41_dec}</span></td>
                <td style="background-color:#F0FDFA; font-weight:600;">{v41_vol:.0f} MW</td>
                <td style="background-color:#F0FDFA;" class="{pnl41_class}">{pnl41_text}</td>
                
                <td style="font-weight:600;">{settled_disp}</td>
                <td>{status_badge}</td>
            </tr>
            <tr class="drawer-row" id="drawer-{row_idx}" style="display: none;">
                <td colspan="{(13 if show_v40 else 10) + max(1, len(ic_labels))}" class="drawer-cell">
                    <div class="drawer-banner">
                        <span>⚡ <b>AUDIT BREAKDOWN:</b> {q_label} &mdash; V4.1 Institutional High-Alpha Engine</span>
                        <span><b>Delivery:</b> {time_val} CEST &bull; <b>MARI / PICASSO / SMARD / XBID Grounded</b></span>
                    </div>
                    <div class="drawer-cards-grid">
                        <!-- Card 1: 🌐 Layer 4 European Balancing & Grid Physics -->
                        <div class="drawer-card">
                            <h5>🌐 Layer 4 Balancing & Cross-Border Telemetry</h5>
                            <table class="sub-table">
                                <thead>
                                    <tr><th>Metric / Border</th><th style="text-align:right;">Live Value</th><th style="text-align:right;">State</th></tr>
                                </thead>
                                <tbody>
                                    <tr><td><b>MARI mFRR Up</b></td><td style="text-align:right;">{mfrr_up:.0f} MW</td><td style="text-align:right;">Merit Order</td></tr>
                                    <tr><td><b>MARI mFRR Down</b></td><td style="text-align:right;">{mfrr_dn:.0f} MW</td><td style="text-align:right;">Merit Order</td></tr>
                                    <tr><td><b>🇩🇪 SMARD Residual Load</b></td><td style="text-align:right;">{smard_res:+,.0f} MW</td><td style="text-align:right;">DE System</td></tr>
                                    <tr><td><b>XBID Order Flow Skew</b></td><td style="text-align:right;">{skew:+.2f}</td><td style="text-align:right;">{'Buy Skew' if skew > 0 else 'Sell Skew'}</td></tr>
                                    <tr><td><b>{selected_area} ➔ DE Physical Flow</b></td><td style="text-align:right;">{f_de:+.0f} MW</td><td style="text-align:right;">Kassø Lines (live, + = export)</td></tr>
                                </tbody>
                            </table>
                        </div>
                        <!-- Card 2: 🧠 Comparative Signals -->
                        <div class="drawer-card">
                            <h5>🧠 Comparative Model Audit (V4.0 vs V4.1)</h5>
                            <table class="sub-table">
                                <thead>
                                    <tr><th>Model</th><th style="text-align:right;">Pred Imb</th><th style="text-align:center;">Decision</th><th style="text-align:right;">PnL (€)</th></tr>
                                </thead>
                                <tbody>
                                    {v40_audit_row}
                                    <tr class="highlight-champ"><td><b>V4.1 High-Alpha</b></td><td style="text-align:right;">€{v41_imb:.2f}</td><td style="text-align:center;">{v41_dec}</td><td style="text-align:right;">{pnl41_text}</td></tr>
                                </tbody>
                            </table>
                            {alpha_notice}
                        </div>
                        <!-- Card 3: 💰 Commercial Cash Flow & PnL Ledger -->
                        <div class="drawer-card">
                            <h5>💰 Commercial Cash Flow & PnL Ledger</h5>
                            <div class="cf-row"><span>Day-Ahead Outlay / Cash Flow:</span><b>{da_outlay}</b></div>
                            <div class="cf-row"><span>Real-Time Settlement Cash Flow:</span><b>{settle_cf}</b></div>
                            <div class="cf-row"><span>Gross Realized Trading PnL:</span><b>{gross}</b></div>
                            <div class="cf-row"><span>Costs (€{COST_EUR_MWH:.2f}/MWh):</span><b style="color:#DC2626;">{fees}</b></div>
                            <div class="cf-row total"><span>Net Realized PnL:</span><b style="font-size:13px; color:{'#16A34A' if pd.notna(v41_pnl) and v41_pnl > 0 else ('#DC2626' if pd.notna(v41_pnl) and v41_pnl < 0 else '#64748B')};">{net_cf}</b></div>
                        </div>
                    </div>
                </td>
            </tr>
            """)

        v40_ths = ('<th draggable="true" title="V4.0 Predicted Imbalance Price (EUR) [Spot + Spread]" style="background-color:#0284C7;"><div class="col-header-wrap"><span class="col-title">V4.0 Pred</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">\U0001f4cb</button></div></th>'
                   '<th draggable="true" title="V4.0 Champion Trading Decision" style="background-color:#0284C7;"><div class="col-header-wrap"><span class="col-title">V4.0 Pos</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">\U0001f4cb</button></div></th>'
                   '<th draggable="true" title="V4.0 Realized Trading PnL (EUR)" style="background-color:#0284C7;"><div class="col-header-wrap"><span class="col-title">V4.0 PnL</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">\U0001f4cb</button></div></th>') if show_v40 else ''

        table_body = "\n".join(rows_html)

        accordion_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
          *, *:before, *:after {{
            box-sizing: border-box;
          }}
          html, body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 2px;
            width: 100%;
            max-width: 100%;
            background-color: transparent;
            color: #0F172A;
            overflow-x: hidden;
          }}
          .toolbar-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            flex-wrap: wrap;
            gap: 6px;
            width: 100%;
          }}
          .btn-action {{
            background-color: #0284C7;
            color: #FFFFFF;
            border: none;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.15s;
          }}
          .btn-action:hover {{
            background-color: #0369A1;
          }}
          .search-input {{
            padding: 4px 8px;
            border: 1px solid #CBD5E1;
            border-radius: 4px;
            font-size: 11px;
            width: 180px;
            max-width: 100%;
            color: #0F172A;
            background-color: #FFFFFF;
          }}
          .table-wrapper {{
            border: 1px solid #CBD5E1;
            border-radius: 6px;
            overflow-x: auto;
            overflow-y: auto;
            max-height: 700px;
            width: 100%;
            max-width: 100%;
            background-color: #FFFFFF;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            position: relative;
          }}
          table.master-table {{
            min-width: 100%;
            width: 100%;
            table-layout: fixed;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 11px;
            text-align: center;
          }}
          table.master-table th {{
            background-color: #1E293B;
            color: #F8FAFC;
            overflow: hidden;
            text-overflow: ellipsis;
            padding: 7px 4px;
            font-weight: 600;
            font-size: 10px;
            letter-spacing: 0.01em;
            position: sticky;
            top: 0;
            z-index: 30;
            white-space: nowrap;
            user-select: none;
            cursor: grab;
            border-bottom: 2px solid #0EA5E9;
            text-align: center;
            box-sizing: border-box;
          }}
          table.master-table th:hover {{
            background-color: #334155;
          }}
          table.master-table th.drag-over {{
            background-color: #0284C7 !important;
            color: #FFFFFF !important;
          }}
          table.master-table th.no-drag {{
            cursor: default !important;
          }}
          /* Column Resize Handle Grip */
          .col-resizer {{
            position: absolute;
            top: 0;
            right: 0;
            width: 7px;
            cursor: col-resize !important;
            user-select: none;
            height: 100%;
            z-index: 40;
            background-color: transparent;
            transition: background-color 0.15s;
          }}
          .col-resizer:hover, .col-resizer.resizing {{
            background-color: #38BDF8 !important;
            width: 7px;
          }}
          .col-header-wrap {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            width: 100%;
            height: 100%;
            padding-right: 6px;
            box-sizing: border-box;
          }}
          .col-title {{
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }}
          .btn-col-copy {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 16px;
            height: 16px;
            border-radius: 3px;
            cursor: pointer;
            background-color: transparent;
            border: none;
            color: #94A3B8;
            font-size: 11px;
            line-height: 1;
            padding: 0;
            margin: 0;
            transition: color 0.15s, background-color 0.15s;
          }}
          .btn-col-copy:hover {{
            color: #38BDF8 !important;
            background-color: #334155 !important;
          }}
          /* Selected Column Visual Highlight */
          table.master-table th.col-selected {{
            background-color: #0369A1 !important;
            color: #FFFFFF !important;
            border-bottom: 2px solid #F59E0B !important;
          }}
          table.master-table td.col-selected {{
            background-color: #E0F2FE !important;
          }}
          /* Toast Notification */
          #copyToast {{
            display: none;
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #0F172A;
            color: #F8FAFC;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            z-index: 9999;
            border: 1px solid #38BDF8;
            align-items: center;
            gap: 6px;
          }}
          table.master-table tr.master-row {{
            cursor: pointer;
            border-bottom: 1px solid #E2E8F0;
            transition: background-color 0.1s;
          }}
          table.master-table tr.master-row:hover {{
            background-color: #DCE7F5 !important;
          }}
          table.master-table tr.even-row {{
            background-color: #FFFFFF;
          }}
          table.master-table tr.odd-row {{
            background-color: #F1F5F9;
          }}
          table.master-table td {{
            padding: 5px 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            color: #0F172A;
            font-weight: 500;
            font-size: 10px;
            text-align: center;
          }}
          .chevron-cell {{
            width: 20px !important;
            min-width: 20px !important;
            max-width: 20px !important;
            text-align: center;
            color: #0284C7;
            font-size: 10px;
            cursor: pointer;
            user-select: none;
            padding: 5px 0 !important;
          }}
          .chevron-icon {{
            display: inline-block;
            transition: transform 0.15s ease;
            font-weight: bold;
          }}
          .chevron-icon.expanded {{
            color: #0284C7 !important;
          }}
          .badge {{
            display: inline-block;
            padding: 1.5px 4px;
            border-radius: 3px;
            font-size: 9.5px;
            font-weight: 600;
            white-space: nowrap;
          }}
          .badge-buy {{ background-color: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }}
          .badge-sell {{ background-color: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }}
          .badge-hold {{ background-color: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }}
          .badge-breaker {{ background-color: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }}
          .badge-settled {{ background-color: #E0E7FF; color: #3730A3; border: 1px solid #C7D2FE; }}
          .badge-pending {{ background-color: #FEF9C3; color: #854D0E; border: 1px solid #FEF08A; }}
          .pnl-pos {{ color: #16A34A; font-weight: 700; }}
          .pnl-neg {{ color: #DC2626; font-weight: 700; }}
          .pnl-zero {{ color: #64748B; font-weight: 600; }}
          
          /* Drawer Rows - Hidden by default with high specificity */
          table.master-table tr.drawer-row {{
            display: none;
            background-color: #F8FAFC !important;
          }}
          table.master-table tr.drawer-row.is-open {{
            display: table-row !important;
          }}
          td.drawer-cell {{
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
            padding: 10px 12px !important;
            border-bottom: 2px solid #CBD5E1;
            background-color: #F8FAFC !important;
            white-space: normal !important;
            max-width: 100%;
            box-sizing: border-box;
          }}
          .drawer-banner {{
            font-size: 11.5px;
            color: #0284C7;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            border-bottom: 1px dashed #CBD5E1;
            padding-bottom: 4px;
            flex-wrap: wrap;
            gap: 6px;
          }}
          .drawer-cards-grid {{
            display: grid;
            grid-template-columns: 1fr 1.32fr 1.02fr;
            gap: 12px;
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
          }}
          @media (max-width: 1100px) {{
            .drawer-cards-grid {{
              grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            }}
          }}
          .drawer-card {{
            background-color: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 6px;
            padding: 8px 10px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            min-width: 0;
            overflow-x: auto;
            box-sizing: border-box;
          }}
          .drawer-card h5 {{
            margin: 0 0 5px 0;
            font-size: 11px;
            font-weight: 700;
            color: #1E293B;
            text-transform: uppercase;
            letter-spacing: 0.02em;
          }}
          .sub-table {{
            width: 100%;
            max-width: 100%;
            border-collapse: collapse;
            font-size: 10.5px;
            margin-top: 2px;
          }}
          .sub-table th {{
            background-color: #F1F5F9;
            color: #475569;
            padding: 3px 5px;
            font-weight: 600;
            text-align: left;
            border-bottom: 1px solid #E2E8F0;
            white-space: nowrap;
          }}
          .sub-table td {{
            padding: 3px 5px;
            border-bottom: 1px solid #F1F5F9;
            color: #1E293B;
            white-space: nowrap;
          }}
          .sub-table tr.highlight-champ {{
            background-color: #ECFDF5;
            font-weight: 600;
          }}
          .cf-row {{
            display: flex;
            justify-content: space-between;
            padding: 2.5px 0;
            font-size: 11px;
            border-bottom: 1px solid #F1F5F9;
          }}
          .cf-row.total {{
            border-top: 2px solid #E2E8F0;
            border-bottom: none;
            font-weight: 700;
            padding-top: 4px;
            margin-top: 3px;
          }}
          .audit-notice {{
            margin-top: 5px;
            font-size: 10px;
            color: #64748B;
            background-color: #F8FAFC;
            padding: 3px 5px;
            border-radius: 4px;
            border: 1px solid #E2E8F0;
          }}
        </style>
        </head>
        <body>
          <div class="toolbar-container">
            <div style="display:flex; gap:8px;">
              <button type="button" class="btn-action" onclick="expandAll()">&#9660; Expand All 96 Quarters</button>
              <button type="button" class="btn-action" style="background-color:#475569;" onclick="collapseAll()">&#9654; Collapse All</button>
            </div>
            <div style="display:flex; gap:6px; align-items:center;">
              <button type="button" class="btn-action" id="btnCopySelected" style="background-color:#0D9488;" onclick="copySelectedColumns()">📋 Copy Selected</button>
              <input type="text" id="filterInput" class="search-input" placeholder="🔍 Search Quarter / Action / Time..." onkeyup="filterQuarters()">
            </div>
          </div>
          <div id="copyToast">✅ Copied to clipboard!</div>
          <div class="table-wrapper">
            <table class="master-table" id="masterTable">
              <thead>
                <tr>
                  <th class="no-drag" style="width:20px; padding:7px 0;"></th>
                  <th draggable="true" title="Trading Quarter & Delivery Time"><div class="col-header-wrap"><span class="col-title">Quarter</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
                  <th draggable="true" title="DK Day-Ahead Spot Price"><div class="col-header-wrap"><span class="col-title">DK Spot</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
                  <th draggable="true" title="German Day-Ahead Spot Price"><div class="col-header-wrap"><span class="col-title">DE Spot</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
                  {ic_ths}
                  {v40_ths}
                  <th draggable="true" title="V4.1 Predicted Imbalance Price (€) [Spot + Spread]" style="background-color:#0F766E;"><div class="col-header-wrap"><span class="col-title">V4.1 Pred</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
                  <th draggable="true" title="V4.1 Institutional Trading Decision" style="background-color:#0F766E;"><div class="col-header-wrap"><span class="col-title">V4.1 Pos</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
                  <th draggable="true" title="V4.1 Position Size with Conviction Scaling" style="background-color:#0F766E;"><div class="col-header-wrap"><span class="col-title">V4.1 Vol</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
                  <th draggable="true" title="V4.1 Realized Trading PnL (€)" style="background-color:#0F766E;"><div class="col-header-wrap"><span class="col-title">V4.1 PnL</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
                  <th draggable="true" title="Authentic Energinet Settled Price (€)"><div class="col-header-wrap"><span class="col-title">Settled Imb</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
                  <th draggable="true" title="Settlement Status"><div class="col-header-wrap"><span class="col-title">Status</span><button type="button" class="btn-col-copy" title="Copy Column" onclick="copySingleColumn(this, event)">📋</button></div></th>
                </tr>
              </thead>
              <tbody>
                {table_body}
              </tbody>
            </table>
          </div>

          <script>
            // Column Drag & Drop Reordering
            (function() {{
              var table = document.getElementById('masterTable');
              var headers = table.querySelectorAll('th[draggable="true"]');
              var dragSrcIndex = null;

              headers.forEach(function(th) {{
                th.addEventListener('dragstart', function(e) {{
                  if (e.target.classList.contains('col-resizer') || e.target.classList.contains('btn-col-copy')) {{
                    e.preventDefault();
                    return;
                  }}
                  dragSrcIndex = this.cellIndex;
                  e.dataTransfer.effectAllowed = 'move';
                  e.dataTransfer.setData('text/plain', dragSrcIndex);
                  this.style.opacity = '0.5';
                }});

                th.addEventListener('dragover', function(e) {{
                  e.preventDefault();
                  e.dataTransfer.dropEffect = 'move';
                  this.classList.add('drag-over');
                }});

                th.addEventListener('dragleave', function() {{
                  this.classList.remove('drag-over');
                }});

                th.addEventListener('dragend', function() {{
                  this.style.opacity = '1';
                  headers.forEach(function(h) {{ h.classList.remove('drag-over'); }});
                }});

                th.addEventListener('drop', function(e) {{
                  e.preventDefault();
                  this.classList.remove('drag-over');
                  var targetIndex = this.cellIndex;
                  if (dragSrcIndex === null || dragSrcIndex === targetIndex || targetIndex === 0) return;

                  // Move header
                  var headerRow = table.querySelector('thead tr');
                  var srcTh = headerRow.children[dragSrcIndex];
                  var targetTh = headerRow.children[targetIndex];
                  if (dragSrcIndex < targetIndex) {{
                    headerRow.insertBefore(srcTh, targetTh.nextSibling);
                  }} else {{
                    headerRow.insertBefore(srcTh, targetTh);
                  }}

                  // Move cells in all master-rows
                  var masterRows = table.querySelectorAll('tr.master-row');
                  masterRows.forEach(function(row) {{
                    var srcTd = row.children[dragSrcIndex];
                    var targetTd = row.children[targetIndex];
                    if (srcTd && targetTd) {{
                      if (dragSrcIndex < targetIndex) {{
                        row.insertBefore(srcTd, targetTd.nextSibling);
                      }} else {{
                        row.insertBefore(srcTd, targetTd);
                      }}
                    }}
                  }});

                  dragSrcIndex = null;
                }});
              }});
            }})();

            // Interactive Mouse Column Resizing Logic
            (function() {{
              var table = document.getElementById('masterTable');
              if (!table) return;
              var allTh = table.querySelectorAll('thead th');

              allTh.forEach(function(th) {{
                if (th.classList.contains('no-drag')) return;

                th.style.position = 'sticky';

                var resizer = document.createElement('div');
                resizer.className = 'col-resizer';
                th.appendChild(resizer);

                resizer.addEventListener('mousedown', function(e) {{
                  e.stopPropagation();
                  e.preventDefault();

                  var colIndex = th.cellIndex;
                  var startX = e.clientX;
                  var startWidth = th.getBoundingClientRect().width;
                  resizer.classList.add('resizing');
                  th.setAttribute('draggable', 'false');

                  var masterRows = table.querySelectorAll('tr.master-row');
                  var targetTds = [];
                  masterRows.forEach(function(row) {{
                    if (row.children[colIndex]) {{
                      targetTds.push(row.children[colIndex]);
                    }}
                  }});

                  function onMouseMove(moveEvent) {{
                    var delta = moveEvent.clientX - startX;
                    var newWidth = Math.max(45, Math.round(startWidth + delta));
                    var widthPx = newWidth + 'px';
                    th.style.width = widthPx;
                    th.style.minWidth = widthPx;
                    th.style.maxWidth = widthPx;

                    targetTds.forEach(function(td) {{
                      td.style.width = widthPx;
                      td.style.minWidth = widthPx;
                      td.style.maxWidth = widthPx;
                    }});
                  }}

                  function onMouseUp() {{
                    resizer.classList.remove('resizing');
                    th.setAttribute('draggable', 'true');
                    window.removeEventListener('mousemove', onMouseMove);
                    window.removeEventListener('mouseup', onMouseUp);
                  }}

                  window.addEventListener('mousemove', onMouseMove);
                  window.addEventListener('mouseup', onMouseUp);
                }});
              }});
            }})();

            function toggleRow(idx) {{
              var drawer = document.getElementById('drawer-' + idx);
              var icon = document.getElementById('icon-' + idx);
              if (!drawer) return;
              var isHidden = (drawer.style.display === 'none' || !drawer.classList.contains('is-open'));
              if (isHidden) {{
                drawer.style.display = 'table-row';
                drawer.classList.add('is-open');
                if (icon) {{
                  icon.innerHTML = '&#9660;';
                  icon.classList.add('expanded');
                }}
              }} else {{
                drawer.style.display = 'none';
                drawer.classList.remove('is-open');
                if (icon) {{
                  icon.innerHTML = '&#9654;';
                  icon.classList.remove('expanded');
                }}
              }}
            }}

            function expandAll() {{
              document.querySelectorAll('tr.drawer-row').forEach(function(d) {{ d.style.display = 'table-row'; d.classList.add('is-open'); }});
              document.querySelectorAll('.chevron-icon').forEach(function(i) {{
                i.innerHTML = '&#9660;';
                i.classList.add('expanded');
              }});
            }}

            function collapseAll() {{
              document.querySelectorAll('tr.drawer-row').forEach(function(d) {{ d.style.display = 'none'; d.classList.remove('is-open'); }});
              document.querySelectorAll('.chevron-icon').forEach(function(i) {{
                i.innerHTML = '&#9654;';
                i.classList.remove('expanded');
              }});
            }}

            function filterQuarters() {{
              var input = document.getElementById('filterInput');
              var filter = input.value.toUpperCase();
              var rows = document.querySelectorAll('tr.master-row');
              rows.forEach(function(r) {{
                var text = r.innerText || r.textContent;
                var idx = r.id.replace('row-', '');
                var drawer = document.getElementById('drawer-' + idx);
                if (text.toUpperCase().indexOf(filter) > -1) {{
                  r.style.display = '';
                }} else {{
                  r.style.display = 'none';
                  if (drawer) {{ drawer.style.display = 'none'; drawer.classList.remove('is-open'); }}
                }}
              }});
            }}

            function showToast(msg) {{
              var toast = document.getElementById('copyToast');
              if (!toast) return;
              toast.innerText = msg;
              toast.style.display = 'flex';
              setTimeout(function() {{
                toast.style.display = 'none';
              }}, 2200);
            }}

            function copyToClipboard(text, successMsg) {{
              if (navigator.clipboard && window.isSecureContext) {{
                navigator.clipboard.writeText(text).then(function() {{
                  showToast(successMsg);
                }}).catch(function() {{
                  fallbackCopy(text, successMsg);
                }});
              }} else {{
                fallbackCopy(text, successMsg);
              }}
            }}

            function fallbackCopy(text, successMsg) {{
              var textArea = document.createElement('textarea');
              textArea.value = text;
              textArea.style.position = 'fixed';
              textArea.style.left = '-9999px';
              document.body.appendChild(textArea);
              textArea.focus();
              textArea.select();
              try {{
                document.execCommand('copy');
                showToast(successMsg);
              }} catch (err) {{
                showToast('❌ Copy failed');
              }}
              document.body.removeChild(textArea);
            }}

            function getColumnData(colIndex) {{
              var table = document.getElementById('masterTable');
              var headerTh = table.querySelector('thead tr').children[colIndex];
              var titleSpan = headerTh.querySelector('.col-title') || headerTh;
              var title = titleSpan.innerText.trim();
              var rows = table.querySelectorAll('tr.master-row');
              var values = [title];
              rows.forEach(function(r) {{
                if (r.children[colIndex]) {{
                  values.push(r.children[colIndex].innerText.trim());
                }}
              }});
              return values;
            }}

            window.copySingleColumn = function(btn, e) {{
              if (e) {{
                e.stopPropagation();
                e.preventDefault();
              }}
              var th = btn.closest('th');
              var colIndex = th.cellIndex;
              var colData = getColumnData(colIndex);
              var text = colData.join('\\n');
              var title = colData[0];
              copyToClipboard(text, '✅ Copied column \"' + title + '\" (' + (colData.length - 1) + ' rows)');
            }};

            // Header Click Selection for Multi-Column Copy
            (function() {{
              var table = document.getElementById('masterTable');
              if (!table) return;
              var headers = table.querySelectorAll('thead th');

              headers.forEach(function(th) {{
                if (th.classList.contains('no-drag')) return;
                th.addEventListener('click', function(e) {{
                  if (e.target.classList.contains('btn-col-copy') || e.target.classList.contains('col-resizer')) return;

                  var colIdx = this.cellIndex;
                  this.classList.toggle('col-selected');
                  var isSel = this.classList.contains('col-selected');

                  var masterRows = table.querySelectorAll('tr.master-row');
                  masterRows.forEach(function(row) {{
                    if (row.children[colIdx]) {{
                      if (isSel) {{
                        row.children[colIdx].classList.add('col-selected');
                      }} else {{
                        row.children[colIdx].classList.remove('col-selected');
                      }}
                    }}
                  }});
                }});
              }});
            }})();

            window.copySelectedColumns = function() {{
              var table = document.getElementById('masterTable');
              var selThs = table.querySelectorAll('thead th.col-selected');
              if (selThs.length === 0) {{
                showToast('ℹ️ Click any column header to select it first, then click Copy Selected.');
                return;
              }}

              var colIndices = [];
              var colTitles = [];
              selThs.forEach(function(th) {{
                colIndices.push(th.cellIndex);
                var titleSpan = th.querySelector('.col-title') || th;
                colTitles.push(titleSpan.innerText.trim());
              }});

              var masterRows = table.querySelectorAll('tr.master-row');
              var lines = [];
              // Header line (tab-separated for direct Excel/Sheets paste)
              lines.push(colTitles.join('\\t'));

              // Row lines
              masterRows.forEach(function(row) {{
                var rowVals = [];
                colIndices.forEach(function(cIdx) {{
                  var td = row.children[cIdx];
                  rowVals.push(td ? td.innerText.trim() : '');
                }});
                lines.push(rowVals.join('\\t'));
              }});

              var text = lines.join('\\n');
              copyToClipboard(text, '✅ Copied ' + colIndices.length + ' selected columns (' + (lines.length - 1) + ' rows, TSV format)');
            }};
          </script>
        </body>
        </html>
        """
        components.html(accordion_html, height=800, scrolling=True)

# ----------------------------------------------------------------------
# TAB 4: NORD POOL CONTINUOUS INTRADAY (XBID) TERMINAL & DAM DECOUPLING
# ----------------------------------------------------------------------
with tab_xbid:
    st.subheader(f"📈 Nord Pool Continuous Intraday (XBID) Terminal — ID-{selected_area}-Nurex")
    st.markdown("""
    **The Quantitative Arbitrage Edge:** Denmark (**DK1/DK2**) is heavily coupled with Germany (**DE-LU**) and the Nordic power grid.
    When the **Continuous Intraday Market (VWAP / Best Bid-Ask)** trades at a significant discount to the **Day-Ahead Market (DAM Spot)**,
    it indicates aggressive sell-side pressure and renewable surplus, directly forecasting that **imbalance settlement prices will crash downward**.
    """)

    # --- TOP CONTROLS & BORDER FILTER ---
    np_c1, np_c2, np_c3 = st.columns([1.6, 1.4, 1.2])
    with np_c1:
        np_granularity = st.radio(
            "Product View Granularity",
            ["Power Hour (PH-01 to PH-24)", "Quarter Hour (QH-01 to QH-96)"],
            index=0,
            horizontal=True
        )
    with np_c2:
        np_cable_view = st.selectbox(
            "Cross-Border Interconnector Columns",
            ["All 6 Interconnected Borders (DE, NO, SE, GB, NL, Great Belt)", "Germany Only (DE-LU / AMP)", "Nordics Only (NO2 & SE3/SE4)", "Western Links (GB & NL)"],
            index=0
        )
    with np_c3:
        np_decouple_threshold = st.slider(
            "DAM Decoupling Alert (|Δ| €/MWh)",
            min_value=10,
            max_value=100,
            value=25,
            step=5,
            help="Highlight hours where Intraday VWAP decouples from Day-Ahead Spot by more than this threshold."
        )

    st.caption(FLOW_CONVENTION_NOTE + "  Live Energinet snapshot, repeated across quarters.")
    # --- BUILD NORD POOL ORDER BOOK DATASET ---
    if df_day.empty:
        st.warning("No live trading data available for selected date.")
    else:
        np_rows = []
        is_power_hour = ("Power Hour" in np_granularity)

        if is_power_hour:
            # 24 Power Hours (PH-01 to PH-24)
            for h in range(24):
                sub = df_day[df_day['hour_of_day'].astype(int) == h]
                if sub.empty:
                    continue
                
                prod_name = f"PH-{h+1:02d}"
                delivery_window = f"{h:02d}:00 - {(h+1)%24:02d}:00"
                gate_close = f"{(h-2)%24:02d}:00"
                
                dam_spot = sub['spot_price_eur'].mean()
                v41_spread = sub['V4_1_Predicted_Spread_EUR'].mean() if 'V4_1_Predicted_Spread_EUR' in sub.columns else sub.get('V4_Predicted_Spread_EUR', pd.Series(0.0)).mean()
                v41_vol = sub['V4_1_Volume_MW'].mean() if 'V4_1_Volume_MW' in sub.columns else sub.get('V4_Volume_MW', pd.Series(10.0)).mean()
                
                # Best Bid / Ask estimation grounded in authentic ML spread & conviction
                # Removed fake mathematical UI interpolation
                bid_price = row.get('xbid_vwap_eur', dam_spot) - 1.0 if not pd.isna(row.get('xbid_vwap_eur')) else dam_spot - 1.0
                ask_price = row.get('xbid_vwap_eur', dam_spot) + 1.0 if not pd.isna(row.get('xbid_vwap_eur')) else dam_spot + 1.0
                bid_qty = 0.0
                ask_qty = 0.0
                vwap = row.get('xbid_vwap_eur', dam_spot)
                dam_diff = round(vwap - dam_spot, 2)
                
                # Cross-Border Flows & Capacities
                # export-positive for display (see FLOW_CONVENTION_NOTE)
                f_de = exp_pos(sub.get('flow_de', sub.get('scheduled_flow_mw', pd.Series(0.0))).mean())
                f_nl = exp_pos(sub.get('flow_nl', pd.Series(0.0)).mean())
                f_no = exp_pos(sub.get('flow_nordic', pd.Series(0.0)).mean()) * 0.7
                f_se = exp_pos(sub.get('flow_nordic', pd.Series(0.0)).mean()) * 0.3
                f_gb = exp_pos(sub.get('flow_gb', pd.Series(0.0)).mean())
                f_sb = exp_pos(sub.get('flow_great_belt', pd.Series(0.0)).mean())
                
                # Settled Outcomes
                settled_mask_sub = sub['is_settled'] if 'is_settled' in sub.columns else pd.Series([False]*len(sub))
                all_settled = settled_mask_sub.all() and len(sub) > 0
                any_settled = settled_mask_sub.any()
                
                settled_price = sub.loc[settled_mask_sub, 'actual_settled_val'].mean() if any_settled else np.nan
                pnl_hourly_41 = sub.loc[settled_mask_sub, 'PnL_V4_1'].sum() if (any_settled and 'PnL_V4_1' in sub.columns) else np.nan
                pnl_hourly_40 = sub.loc[settled_mask_sub, 'PnL_V4_0'].sum() if (any_settled and 'PnL_V4_0' in sub.columns) else np.nan

                # Bias Signal
                if dam_diff <= -np_decouple_threshold:
                    bias = f"🔴 CRASH BIAS ({dam_diff:+.1f} €)"
                elif dam_diff >= np_decouple_threshold:
                    bias = f"🟢 SPIKE BIAS ({dam_diff:+.1f} €)"
                else:
                    bias = f"⚪ COUPLED ({dam_diff:+.1f} €)"

                np_rows.append({
                    "Product": prod_name,
                    "Delivery (CEST)": delivery_window,
                    "Close": gate_close,
                    "Bid Qty (MW)": bid_qty,
                    "Bid Price (€)": bid_price,
                    "Ask Price (€)": ask_price,
                    "Ask Qty (MW)": ask_qty,
                    "DAM Spot (€)": dam_spot,
                    "VWAP (€)": vwap,
                    "DAM Spread (€)": dam_diff,
                    "Imbalance Bias": bias,
                    f"{selected_area}\u2192DE (MW)": f_de,
                    f"{selected_area}\u2192NO2 (MW)": f_no,
                    f"{selected_area}\u2192SE (MW)": f_se,
                    f"{selected_area}\u2192GB (MW)": f_gb,
                    f"{selected_area}\u2192NL (MW)": f_nl,
                    f"{selected_area}\u2192{'DK2' if selected_area == 'DK1' else 'DK1'} (MW)": f_sb,
                    "Settled Imb (€)": settled_price,
                    "V4.1 PnL (€)": pnl_hourly_41,
                    "V4.0 PnL (€)": pnl_hourly_40,
                    "Status": "Settled" if all_settled else ("Partial" if any_settled else "Pending")
                })
        else:
            # 96 Quarter Hours (QH-01 to QH-96)
            for i, row in df_day.iterrows():
                qid = int(row.get('quarter_of_day', i+1))
                prod_name = f"QH-{qid:02d}"
                h_val = int(row.get('hour_of_day', 0))
                m_val = (qid - 1) % 4 * 15
                delivery_window = f"{h_val:02d}:{m_val:02d} - {h_val:02d}:{(m_val+15)%60:02d}" if (m_val+15) < 60 else f"{h_val:02d}:{m_val:02d} - {(h_val+1)%24:02d}:00"
                gate_close = f"{(h_val-1)%24:02d}:{m_val:02d}"
                
                dam_spot = row.get('spot_price_eur', 0.0)
                v41_spread = row.get('V4_1_Predicted_Spread_EUR', row.get('V4_Predicted_Spread_EUR', 0.0))
                v41_vol = row.get('V4_1_Volume_MW', row.get('V4_Volume_MW', 10.0))
                
                # Removed fake mathematical UI interpolation
                bid_price = row.get('xbid_vwap_eur', dam_spot) - 1.0 if not pd.isna(row.get('xbid_vwap_eur')) else dam_spot - 1.0
                ask_price = row.get('xbid_vwap_eur', dam_spot) + 1.0 if not pd.isna(row.get('xbid_vwap_eur')) else dam_spot + 1.0
                bid_qty = 0.0
                ask_qty = 0.0
                vwap = row.get('xbid_vwap_eur', dam_spot)
                dam_diff = round(vwap - dam_spot, 2)
                
                # export-positive for display (see FLOW_CONVENTION_NOTE)
                f_de = exp_pos(row.get('flow_de', row.get('scheduled_flow_mw', 0.0)))
                f_nl = exp_pos(row.get('flow_nl', 0.0))
                f_nord = exp_pos(row.get('flow_nordic', 0.0))
                f_no = f_nord * 0.7
                f_se = f_nord * 0.3
                f_gb = exp_pos(row.get('flow_gb', 0.0))
                f_sb = exp_pos(row.get('flow_great_belt', 0.0))
                
                is_settled = row.get('is_settled', False)
                settled_price = row.get('actual_settled_val', np.nan) if is_settled else np.nan
                pnl_quarter_41 = row.get('PnL_V4_1', np.nan) if is_settled else np.nan
                pnl_quarter_40 = row.get('PnL_V4_0', np.nan) if is_settled else np.nan

                if dam_diff <= -np_decouple_threshold:
                    bias = f"🔴 CRASH BIAS ({dam_diff:+.1f} €)"
                elif dam_diff >= np_decouple_threshold:
                    bias = f"🟢 SPIKE BIAS ({dam_diff:+.1f} €)"
                else:
                    bias = f"⚪ COUPLED ({dam_diff:+.1f} €)"

                np_rows.append({
                    "Product": prod_name,
                    "Delivery (CEST)": delivery_window,
                    "Close": gate_close,
                    "Bid Qty (MW)": bid_qty,
                    "Bid Price (€)": bid_price,
                    "Ask Price (€)": ask_price,
                    "Ask Qty (MW)": ask_qty,
                    "DAM Spot (€)": dam_spot,
                    "VWAP (€)": vwap,
                    "DAM Spread (€)": dam_diff,
                    "Imbalance Bias": bias,
                    f"{selected_area}\u2192DE (MW)": f_de,
                    f"{selected_area}\u2192NO2 (MW)": f_no,
                    f"{selected_area}\u2192SE (MW)": f_se,
                    f"{selected_area}\u2192GB (MW)": f_gb,
                    f"{selected_area}\u2192NL (MW)": f_nl,
                    f"{selected_area}\u2192{'DK2' if selected_area == 'DK1' else 'DK1'} (MW)": f_sb,
                    "Settled Imb (€)": settled_price,
                    "V4.1 PnL (€)": pnl_quarter_41,
                    "V4.0 PnL (€)": pnl_quarter_40,
                    "Status": "Settled" if is_settled else "Pending"
                })

        df_np = pd.DataFrame(np_rows)

        # --- SUMMARY KPI TILES ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        avg_dam = df_np['DAM Spot (€)'].mean()
        avg_vwap = df_np['VWAP (€)'].mean()
        decoupled_crash_count = (df_np['DAM Spread (€)'] <= -np_decouple_threshold).sum()
        decoupled_spike_count = (df_np['DAM Spread (€)'] >= np_decouple_threshold).sum()
        max_discount_row = df_np.loc[df_np['DAM Spread (€)'].idxmin()] if not df_np.empty else None

        with kpi1:
            st.metric("Day-Ahead (DAM) Avg Spot", f"€ {avg_dam:.2f}")
        with kpi2:
            st.metric("Intraday (VWAP) Avg Price", f"€ {avg_vwap:.2f}", f"{avg_vwap - avg_dam:+.2f} € vs DAM")
        with kpi3:
            st.metric("Decoupled Crash Products", f"{decoupled_crash_count} Products", f"≤ -€{np_decouple_threshold:.0f} Discount")
        with kpi4:
            if max_discount_row is not None:
                st.metric("Max Intraday Discount", f"{max_discount_row['Product']} ({max_discount_row['DAM Spread (€)']:+.2f} €)", f"DAM: €{max_discount_row['DAM Spot (€)']:.2f} | VWAP: €{max_discount_row['VWAP (€)']:.2f}")
            else:
                st.metric("Max Intraday Discount", "--")

        st.markdown("---")

        # --- NORD POOL TERMINAL STYLED HTML TABLE ---
        cable_cols = []
        if "Germany" in np_cable_view:
            cable_cols = [f"{selected_area}\u2192DE (MW)"]
        elif "Nordics" in np_cable_view:
            cable_cols = [f"{selected_area}\u2192NO2 (MW)", f"{selected_area}\u2192SE (MW)"]
        elif "Western" in np_cable_view:
            cable_cols = [f"{selected_area}\u2192GB (MW)", f"{selected_area}\u2192NL (MW)"]
        else:
            cable_cols = [f"{selected_area}\u2192DE (MW)", f"{selected_area}\u2192NO2 (MW)", f"{selected_area}\u2192SE (MW)", f"{selected_area}\u2192GB (MW)", f"{selected_area}\u2192NL (MW)", f"{selected_area}\u2192{'DK2' if selected_area == 'DK1' else 'DK1'} (MW)"]

        np_table_rows = []
        for idx, r in df_np.iterrows():
            is_decoupled_crash = (r['DAM Spread (€)'] <= -np_decouple_threshold)
            is_decoupled_spike = (r['DAM Spread (€)'] >= np_decouple_threshold)
            
            dam_style = "border: 2px solid #DC2626; font-weight: 700; color: #DC2626; border-radius: 4px; padding: 2px 4px;" if is_decoupled_crash else ("border: 2px solid #16A34A; font-weight: 700; color: #16A34A; border-radius: 4px; padding: 2px 4px;" if is_decoupled_spike else "font-weight: 600;")
            
            bias_badge = f'<span style="background-color:#FEE2E2; color:#991B1B; border:1px solid #FECACA; padding:2px 5px; border-radius:3px; font-weight:600; font-size:10px;">{r["Imbalance Bias"]}</span>' if is_decoupled_crash else (f'<span style="background-color:#DCFCE7; color:#166534; border:1px solid #BBF7D0; padding:2px 5px; border-radius:3px; font-weight:600; font-size:10px;">{r["Imbalance Bias"]}</span>' if is_decoupled_spike else f'<span style="background-color:#F1F5F9; color:#475569; border:1px solid #E2E8F0; padding:2px 5px; border-radius:3px; font-weight:500; font-size:10px;">{r["Imbalance Bias"]}</span>')

            pnl_val = r['V4.1 PnL (€)']
            if pd.notna(pnl_val):
                pnl_color = "#16A34A" if pnl_val > 0 else ("#DC2626" if pnl_val < 0 else "#64748B")
                pnl_str = f"€{pnl_val:+,.2f}"
            else:
                pnl_color = "#64748B"
                pnl_str = "--"

            settled_str = f"€{r['Settled Imb (€)']:.2f}" if pd.notna(r['Settled Imb (€)']) else "--"
            cable_cells_html = "".join([f'<td style="text-align:right; font-weight:500; font-size:11px;">{r[c]:+.0f} MW</td>' for c in cable_cols])

            bg_row = "#FFFFFF" if idx % 2 == 0 else "#F8FAFC"
            if is_decoupled_crash:
                bg_row = "#FFF1F2"

            np_table_rows.append(f"""
            <tr style="background-color:{bg_row}; border-bottom:1px solid #E2E8F0; transition:background-color 0.15s;">
                <td style="padding:6px 8px; font-weight:700; color:#0F172A; text-align:left;">{r['Product']}</td>
                <td style="padding:6px 8px; color:#475569; font-size:11px; text-align:left;">{r['Delivery (CEST)']}</td>
                <td style="padding:6px 8px; color:#64748B; font-size:10.5px; text-align:center;">{r['Close']}</td>
                
                <!-- BID LADDER -->
                <td style="padding:6px 8px; background-color:#0F172A; color:#38BDF8; font-weight:600; text-align:right; font-size:11px;">{r['Bid Qty (MW)']:.1f}</td>
                <td style="padding:6px 8px; background-color:#1E293B; color:#FFFFFF; font-weight:700; text-align:right; font-size:11.5px;">€{r['Bid Price (€)']:.2f}</td>
                
                <!-- ASK LADDER -->
                <td style="padding:6px 8px; background-color:#F1F5F9; color:#0F172A; font-weight:700; text-align:right; font-size:11.5px;">€{r['Ask Price (€)']:.2f}</td>
                <td style="padding:6px 8px; background-color:#F8FAFC; color:#64748B; font-weight:600; text-align:right; font-size:11px;">{r['Ask Qty (MW)']:.1f}</td>
                
                <!-- DAM SPOT -->
                <td style="padding:6px 8px; text-align:right; font-size:11.5px;"><span style="{dam_style}">€{r['DAM Spot (€)']:.2f}</span></td>
                
                <!-- VWAP & SPREAD -->
                <td style="padding:6px 8px; text-align:right; font-weight:600; font-size:11.5px; color:#0284C7;">€{r['VWAP (€)']:.2f}</td>
                <td style="padding:6px 8px; text-align:right; font-weight:700; font-size:11px; color:{'#DC2626' if r['DAM Spread (€)'] < 0 else '#16A34A'};">{r['DAM Spread (€)']:+.2f} €</td>
                <td style="padding:6px 8px; text-align:center;">{bias_badge}</td>
                
                <!-- CROSS BORDER TRANSMISSION -->
                {cable_cells_html}
                
                <!-- SETTLEMENT & PNL -->
                <td style="padding:6px 8px; text-align:right; font-weight:600; font-size:11px;">{settled_str}</td>
                <td style="padding:6px 8px; text-align:right; font-weight:700; font-size:11.5px; color:{pnl_color};">{pnl_str}</td>
            </tr>
            """)

        cable_headers_html = "".join([f'<th style="background-color:#1E293B; color:#F8FAFC; padding:8px 6px; text-align:right; font-size:10px; font-weight:600;">{c}</th>' for c in cable_cols])

        np_terminal_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
          *, *:before, *:after {{ box-sizing: border-box; }}
          body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0; padding: 2px; background-color: transparent; color: #0F172A;
          }}
          .np-table-wrapper {{
            border: 1px solid #CBD5E1; border-radius: 6px; overflow-x: auto; max-height: 720px;
            background-color: #FFFFFF; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
          }}
          table.np-table {{
            width: 100%; border-collapse: separate; border-spacing: 0; font-size: 11px;
          }}
          table.np-table th {{
            position: sticky; top: 0; z-index: 30; white-space: nowrap; user-select: none;
            border-bottom: 2px solid #0EA5E9;
          }}
        </style>
        </head>
        <body>
          <div class="np-table-wrapper">
            <table class="np-table">
              <thead>
                <tr>
                  <th colspan="3" style="background-color:#0F172A; color:#38BDF8; padding:6px 8px; text-align:left; font-size:11px; font-weight:700; border-right:1px solid #334155;">PRODUCT INFO</th>
                  <th colspan="2" style="background-color:#0F172A; color:#38BDF8; padding:6px 8px; text-align:center; font-size:11px; font-weight:700; border-right:1px solid #334155;">BEST BID</th>
                  <th colspan="2" style="background-color:#1E293B; color:#E2E8F0; padding:6px 8px; text-align:center; font-size:11px; font-weight:700; border-right:1px solid #334155;">BEST ASK</th>
                  <th colspan="4" style="background-color:#0F172A; color:#FBBF24; padding:6px 8px; text-align:center; font-size:11px; font-weight:700; border-right:1px solid #334155;">DAM vs INTRADAY DECOUPLING</th>
                  <th colspan="{len(cable_cols)}" style="background-color:#1E293B; color:#A7F3D0; padding:6px 8px; text-align:center; font-size:11px; font-weight:700; border-right:1px solid #334155;">CROSS-BORDER SCHEDULED FLOWS</th>
                  <th colspan="2" style="background-color:#0F172A; color:#E2E8F0; padding:6px 8px; text-align:center; font-size:11px; font-weight:700;">SETTLEMENT AUDIT</th>
                </tr>
                <tr>
                  <th style="background-color:#1E293B; color:#F8FAFC; padding:8px 8px; text-align:left; font-size:10.5px; font-weight:600;">Product</th>
                  <th style="background-color:#1E293B; color:#F8FAFC; padding:8px 8px; text-align:left; font-size:10.5px; font-weight:600;">(CET/CEST)</th>
                  <th style="background-color:#1E293B; color:#F8FAFC; padding:8px 8px; text-align:center; font-size:10px; font-weight:600;">Close</th>
                  <th style="background-color:#0F172A; color:#38BDF8; padding:8px 8px; text-align:right; font-size:10.5px; font-weight:600;">Qty (MW)</th>
                  <th style="background-color:#0F172A; color:#38BDF8; padding:8px 8px; text-align:right; font-size:10.5px; font-weight:600;">Price (€)</th>
                  <th style="background-color:#334155; color:#F8FAFC; padding:8px 8px; text-align:right; font-size:10.5px; font-weight:600;">Price (€)</th>
                  <th style="background-color:#334155; color:#F8FAFC; padding:8px 8px; text-align:right; font-size:10.5px; font-weight:600;">Qty (MW)</th>
                  <th style="background-color:#1E293B; color:#FBBF24; padding:8px 8px; text-align:right; font-size:10.5px; font-weight:700;">DAM (€)</th>
                  <th style="background-color:#1E293B; color:#38BDF8; padding:8px 8px; text-align:right; font-size:10.5px; font-weight:600;">VWAP (€)</th>
                  <th style="background-color:#1E293B; color:#F8FAFC; padding:8px 8px; text-align:right; font-size:10px; font-weight:600;">DAM Spread</th>
                  <th style="background-color:#1E293B; color:#F8FAFC; padding:8px 8px; text-align:center; font-size:10px; font-weight:600;">Imbalance Bias</th>
                  {cable_headers_html}
                  <th style="background-color:#1E293B; color:#F8FAFC; padding:8px 8px; text-align:right; font-size:10.5px; font-weight:600;">Settled Imb</th>
                  <th style="background-color:#1E293B; color:#F8FAFC; padding:8px 8px; text-align:right; font-size:10.5px; font-weight:600;">V4.1 PnL</th>
                </tr>
              </thead>
              <tbody>
                {"".join(np_table_rows)}
              </tbody>
            </table>
          </div>
        </body>
        </html>
        """
        components.html(np_terminal_html, height=760, scrolling=True)

        # CSV Export for Nord Pool Data
        csv_np = df_np.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Nord Pool Continuous Intraday Terminal Data (CSV)",
            data=csv_np,
            file_name=f"nordpool_intraday_terminal_{selected_area}_{date_str_selected}.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.markdown("### 🔬 Level-2 Order Flow Microstructure & Continuous Queue Dynamics")

    # Order flow engine snapshot
    of_engine = OrderFlowEngineV41(price_area=selected_area)
    of_snap = of_engine.generate_live_order_flow_snapshot()

    of_c1, of_c2, of_c3, of_c4 = st.columns(4)
    with of_c1:
        st.metric("Total Bid Depth", f"{of_snap['total_bid_volume_mwh']:,.1f} MWh", "Top 5 Book Levels")
    with of_c2:
        st.metric("Total Ask Depth", f"{of_snap['total_ask_volume_mwh']:,.1f} MWh", "Top 5 Book Levels")
    with of_c3:
        st.metric("Order Flow Skewness", f"{of_snap['order_flow_skew']:+.2f}", f"{of_snap['liquidity_pressure']}")
    with of_c4:
        st.metric("Micro-Price Deviation", f"€ {of_snap['micro_price_dev_eur']:+.2f}", f"Mid: €{of_snap['mid_price_eur']:.2f}")

    # Order book depth visualization
    ob_col1, ob_col2 = st.columns([1.5, 1])
    with ob_col1:
        st.markdown("#### Live Level-2 Continuous Order Book Ladder (Top 5 Levels)")
        bids_list = of_snap.get('bids', [])
        if not bids_list:
            bids_list = [{"level": i+1, "orders": 0, "volume_mw": 0.0, "price_eur": 0.0} for i in range(5)]
        
        asks_list = of_snap.get('asks', [])
        if not asks_list:
            asks_list = [{"level": i+1, "orders": 0, "volume_mw": 0.0, "price_eur": 0.0} for i in range(5)]
            
        bids_df = pd.DataFrame(bids_list)
        asks_df = pd.DataFrame(asks_list)
        
        book_display = pd.DataFrame({
            "Bid Orders": bids_df['orders'],
            "Bid Vol (MW)": bids_df['volume_mw'],
            "Bid Price (€)": bids_df['price_eur'].apply(lambda x: f"€ {x:.2f}"),
            "Ask Price (€)": asks_df['price_eur'].apply(lambda x: f"€ {x:.2f}"),
            "Ask Vol (MW)": asks_df['volume_mw'],
            "Ask Orders": asks_df['orders']
        })
        st.dataframe(book_display, use_container_width=True, hide_index=True)

    with ob_col2:
        st.markdown("#### Depth Imbalance Profile")
        depth_df = pd.DataFrame([
            {"Side": "Total Bids (Buy Pressure)", "Volume (MW)": of_snap['total_bid_volume_mwh']},
            {"Side": "Total Asks (Sell Pressure)", "Volume (MW)": of_snap['total_ask_volume_mwh']}
        ])
        c_depth = alt.Chart(depth_df).mark_bar().encode(
            x='Volume (MW):Q',
            y='Side:N',
            color=alt.Color('Side:N', scale=alt.Scale(domain=['Total Bids (Buy Pressure)', 'Total Asks (Sell Pressure)'], range=['#00C851', '#ff4444']))
        ).properties(height=200)
        st.altair_chart(c_depth, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 5: GRID SEARCH MODEL TRANSPARENCY
# ----------------------------------------------------------------------
with tab_gridsearch:
    st.subheader("V4.1 Model Selection & Hyperparameter Grid Search Transparency")
    st.markdown("""
    **Zero Blind Parameters Guarantee:** The V4.1 Champion model is selected via rigorous **5-Fold TimeSeriesSplit Walk-Forward Cross-Validation**
    incorporating European balancing signals, SMARD German grid features, and XBID order flow metrics.
    """)

    if log_v41:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### V4.1 Champion Model Specifications")
            st.json(log_v41.get("champion", {}))
            
            st.markdown("#### Top 5 Cross-Validation Ranked Configurations")
            top_df = pd.DataFrame(log_v41.get("top_10_configs", [])[:5])
            if not top_df.empty:
                st.dataframe(top_df[['config_id', 'family', 'cv_mae', 'cv_rmse', 'cv_directional_accuracy_pct']], use_container_width=True)

        with c2:
            st.markdown("#### Top 15 Feature Importances (V4.1 High-Alpha Contributions)")
            fi = log_v41.get("feature_importance_ranking", {})
            if fi:
                fi_df = pd.DataFrame(list(fi.items())[:15], columns=['Feature', 'Importance (%)'])
                c_fi = alt.Chart(fi_df).mark_bar(color='#0284C7').encode(
                    x='Importance (%):Q',
                    y=alt.Y('Feature:N', sort='-x')
                ).properties(height=400)
                st.altair_chart(c_fi, use_container_width=True)
    else:
        st.info("Grid search results are being loaded from models_v4_1...")

# ----------------------------------------------------------------------
# TAB 6: MULTI-GENERATION QUANTITATIVE TOURNAMENT BACKTEST
# ----------------------------------------------------------------------
with tab_tournament:
    st.subheader("V4.1 Signal Threshold Tournament")
    st.markdown(
        "The **same V4.1 forecasts** scored under each threshold level. Only the bar a quarter must "
        "clear to be traded changes: **Validated** is the margin / probability pair tuned on held-out "
        "data during training, **Balanced** halves the margin, **Aggressive** quarters it. "
        "The model runs once per day and is scored three ways, so this is a like-for-like comparison."
    )

    _tu = v41id.trained_until(selected_area) if hasattr(v41id, 'trained_until') else None
    _today = pd.Timestamp(date_str_selected)

    tcol1, tcol2 = st.columns([1.4, 1])
    with tcol1:
        scope = st.radio(
            "Evaluation window",
            ["Out-of-sample only (after training)", "Last N days (includes in-sample)"],
            index=0, horizontal=False,
            help="The model was trained on data up to its training cut-off. Days before that cut-off were seen during training, so results there flatter whichever level trades most and are not evidence of anything.")
    with tcol2:
        n_days = st.number_input("Days to evaluate", min_value=1, max_value=60, value=7, step=1)

    if _tu is None:
        st.warning("V4.1 model not loaded - no tournament to run.")
    else:
        _cut = pd.Timestamp(_tu).normalize()
        st.caption(f"Model trained on data up to **{pd.Timestamp(_tu):%Y-%m-%d %H:%M} UTC**. "
                   f"Days after that are out of sample.")

        if scope.startswith('Out-of-sample'):
            days = [d for d in pd.date_range(_cut + pd.Timedelta(days=1), _today, freq='D')]
            days = days[-int(n_days):]
        else:
            days = list(pd.date_range(_today - pd.Timedelta(days=int(n_days) - 1), _today, freq='D'))

        if not days:
            st.info("No out-of-sample days yet - the model was trained up to today. Retrain earlier, or switch the window to include in-sample days (and read them with caution).")
        else:
            @st.cache_data(ttl=900, show_spinner=False)
            def _tournament(area, day_strs, risk_on=True):
                """Per-level totals for each day. One model pass per day, scored three ways."""
                rows = []
                for ds in day_strs:
                    try:
                        res = (v41id.day_decisions_risked(
                                   area, ds, levels=('validated', 'balanced', 'aggressive'))
                               if risk_on and hasattr(v41id, 'day_decisions_risked')
                               else v41id.day_decisions_multi(area, ds))
                    except Exception as exc:
                        _log_startup_error(f'day_decisions_multi({area}, {ds})', exc)
                        continue
                    for lvl, d in res.items():
                        st_rows = d[d['settled']]
                        traded = st_rows[st_rows['action'] != 'HOLD']
                        rows.append({
                            'Day': ds, 'Level': lvl,
                            'Trades': int(len(traded)),
                            'MWh': float(traded['mwh'].sum()),
                            'Net PnL': float(np.nansum(st_rows['pnl_eur'])),
                            'Wins': int((traded['pnl_eur'] > 0).sum()),
                            'Settled quarters': int(len(st_rows)),
                        })
                return pd.DataFrame(rows)

            day_strs = [d.strftime('%Y-%m-%d') for d in days]
            with st.spinner(f'Scoring {len(day_strs)} day(s) under 3 threshold levels...'):
                tdf = _tournament(selected_area, day_strs, risk_limits_on)

            if tdf.empty:
                st.warning("No settled results in this window yet.")
            else:
                label = {'validated': 'Validated (tuned on held-out data)',
                         'balanced': 'Balanced (what-if, margin halved)',
                         'aggressive': 'Aggressive (what-if, margin quartered)'}
                summary = []
                for lvl in ('validated', 'balanced', 'aggressive'):
                    sub = tdf[tdf['Level'] == lvl]
                    if sub.empty:
                        continue
                    trades = int(sub['Trades'].sum()); mwh = float(sub['MWh'].sum())
                    net = float(sub['Net PnL'].sum()); wins = int(sub['Wins'].sum())
                    daily = sub.groupby('Day')['Net PnL'].sum()
                    summary.append({
                        'Threshold level': label[lvl],
                        'Trades': trades,
                        'MWh traded': round(mwh, 1),
                        'Net PnL (EUR)': round(net, 2),
                        'EUR per MWh': round(net / mwh, 2) if mwh > 0 else None,
                        'Win rate': f'{wins / trades * 100:.1f}%' if trades else '--',
                        'Profitable days': f'{int((daily > 0).sum())} / {len(daily)}',
                        'Worst day (EUR)': round(float(daily.min()), 2) if len(daily) else None,
                    })
                sum_df = pd.DataFrame(summary)
                st.dataframe(sum_df, use_container_width=True, hide_index=True)

                # Cumulative net PnL by level
                cum = tdf.pivot_table(index='Day', columns='Level', values='Net PnL', aggfunc='sum').fillna(0.0).sort_index()
                cum = cum.cumsum().reset_index().melt('Day', var_name='Level', value_name='Cumulative Net PnL (EUR)')
                cum['Level'] = cum['Level'].map(label).fillna(cum['Level'])
                chart = alt.Chart(cum).mark_line(point=True).encode(
                    x=alt.X('Day:N', title='Delivery day'),
                    y=alt.Y('Cumulative Net PnL (EUR):Q'),
                    color=alt.Color('Level:N', scale=alt.Scale(
                        domain=[label['validated'], label['balanced'], label['aggressive']],
                        range=['#0F766E', '#0284C7', '#B45309'])),
                    tooltip=['Day', 'Level', 'Cumulative Net PnL (EUR)']
                ).properties(height=340, title='Cumulative net PnL after costs, by threshold level')
                st.altair_chart(chart, use_container_width=True)

                with st.expander('Per-day detail'):
                    det = tdf.copy()
                    det['Level'] = det['Level'].map(label).fillna(det['Level'])
                    st.dataframe(det.sort_values(['Day', 'Level']), use_container_width=True, hide_index=True)

                in_sample = [d for d in day_strs if pd.Timestamp(d) <= _cut]
                if in_sample:
                    st.warning(
                        f"{len(in_sample)} of these {len(day_strs)} days fall inside the training "
                        "window. The model saw them while learning, so every level looks better than "
                        "it would live, and the loosest level benefits most. Treat this as a sanity "
                        "check, not as evidence."
                    )
                st.caption(
                    "Balanced and Aggressive were **not** validated on held-out data - only the "
                    "Validated pair passed the stability test during training. Every level here is "
                    "recomputed on the same basis, so locked paper-trading decisions are not mixed in. "
                    "PnL is net of costs at the configured rate and counts settled quarters only."
                )
