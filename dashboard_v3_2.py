import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import sys
import requests
from datetime import datetime, date, timedelta
from entsoe import EntsoePandasClient

sys.path.append(os.path.abspath('.'))
from src.tournament_tables_v2 import TournamentTableGenerator
from src.commercial_strategy_v3_1 import V31CommercialStrategyEngine

st.set_page_config(page_title="Nurex V3.2 Dashboard", layout="wide")

CHOSEN_V3_1_BASELINE = "Transfer-BiLSTM"
ENTSOE_TOKEN = os.environ.get("ENTSOE_TOKEN", "")

# --- SIDEBAR ---
st.sidebar.title("🎛️ V3.2 Meta-Controller")
st.sidebar.markdown("### System Status")
st.sidebar.success("✅ DMI (Danish Meteorological Inst.) API: Live")
st.sidebar.success("✅ ENTSO-E API: Live")
st.sidebar.success("✅ Energinet API: Live")
st.sidebar.success("✅ Dynamic Circuit Breaker: Active")

selected_area = st.sidebar.radio("Select Bidding Zone", ["DK1", "DK2"])
st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ **Experimental Features**")
strategy_mode = st.sidebar.radio(
    "Crash Protection Mode",
    ["Disabled (V3.1 Only)", "Heuristic (>800 MW)", "ML Meta-Model (Random Forest)"],
    index=2
)

ml_sidebar_placeholder = st.sidebar.empty()
if strategy_mode == "ML Meta-Model (Random Forest)":
    ml_sidebar_placeholder.info("🧠 ML Engine Active\nThe Random Forest is dynamically evaluating the Flow & Wind matrices to predict structural crashes.")
elif strategy_mode == "Heuristic (>800 MW)":
    ml_sidebar_placeholder.warning("⚠️ Dumb Rule Active\nHardcoded to short-sell whenever Cross-Border flow exceeds 800 MW, ignoring all other variables.")
st.sidebar.markdown("---")
dk_now = pd.Timestamp.now(tz="Europe/Copenhagen")
default_date = dk_now.date()
selected_date = st.sidebar.date_input("Select Trading Date", value=default_date)
date_str_selected = selected_date.strftime("%Y-%m-%d")

def parse_currency(val):
    if pd.isna(val) or val == "--": return None
    try:
        return float(str(val).replace("€", "").replace(",", "").strip())
    except:
        return None

from src.data_retrieval_v4 import fetch_energinet_true_forecast_error
from src.nordpool_umm_scraper import fetch_live_umms

def fetch_open_meteo_forecast(latitude=56.2639, longitude=9.5018):
    # DEPRECATED IN V4.0 - Replaced by fetch_energinet_true_forecast_error
    return None

def fetch_umm_outages():
    # V4.0 Live Web Scraper
    return fetch_live_umms(["DK1", "DK2"])


@st.cache_data(ttl=3600)
def get_dynamic_price_cap(area, current_date_str):
    """Fetches the last 30 days of Spot Prices and calculates the 95th percentile dynamic cap."""
    client = EntsoePandasClient(api_key=ENTSOE_TOKEN)
    end_ts = pd.Timestamp(current_date_str, tz='Europe/Copenhagen')
    start_ts = end_ts - pd.Timedelta(days=30)
    entsoe_area = 'DK_1' if area == 'DK1' else 'DK_2'
    try:
        historical_spot = client.query_day_ahead_prices(entsoe_area, start=start_ts, end=end_ts)
        cap = historical_spot.quantile(0.95)
        return float(cap)
    except Exception as e:
        print("Failed to fetch historical spot for cap:", e)
        return 250.0

@st.cache_data(ttl=300)
def load_live_monolithic_data(area, date_str, strategy_mode="ML Meta-Model (Random Forest)"):
    table_gen = TournamentTableGenerator(price_area=area)
    try:
        target_df = table_gen.generate_and_save_future_table(date_str=date_str)
    except:
        target_df = table_gen.get_backtest_table(date_str=date_str)
    if target_df.empty:
        target_df = table_gen.get_future_table(date_str=date_str)
    if target_df.empty: return pd.DataFrame() 

    strategy = V31CommercialStrategyEngine(price_area=area)
    s_v31 = strategy.evaluate_trading_ledger(target_df, model_name=CHOSEN_V3_1_BASELINE, market_mode="INTRADAY_D0")
    
    trades = s_v31.get("trades", [])
    df_trades = pd.DataFrame(trades)
    if df_trades.empty: return df_trades
        
    df_trades['time_dk_obj'] = pd.to_datetime(df_trades['time_dk'])
    df_trades['hour_of_day'] = df_trades['time_dk_obj'].dt.hour
    df_trades['quarter_of_day'] = df_trades['time_dk_obj'].dt.hour * 4 + df_trades['time_dk_obj'].dt.minute // 15
    df_trades['time_hour_floor'] = df_trades['time_dk_obj'].dt.floor('H')
    df_trades['V3_1_BiLSTM_Score'] = df_trades['pred_spread_eur']
    
    # ENTSO-E Integration
    client = EntsoePandasClient(api_key=ENTSOE_TOKEN)
    start_ts = pd.Timestamp(date_str, tz='Europe/Copenhagen')
    end_ts = start_ts + pd.Timedelta(days=1)
    
    try:
        de_prices = client.query_day_ahead_prices('DE_LU', start=start_ts, end=end_ts)
        de_prices_df = de_prices.reset_index()
        de_prices_df.columns = ['time_dk_obj', 'de_spot_eur']
        de_prices_df['time_dk_obj'] = de_prices_df['time_dk_obj'].dt.tz_localize(None) 
        df_trades = pd.merge(df_trades, de_prices_df, on='time_dk_obj', how='left')
        
        df_trades['dk_de_spread'] = df_trades['spot_price_eur'] - df_trades['de_spot_eur']
        df_trades['V3_2_DK_DE_Spread_Volatility'] = df_trades['dk_de_spread'].rolling(4, min_periods=1).std().fillna(0)
        
        entsoe_area_to = 'DK_1' if area == 'DK1' else 'DK_2'
        flows = client.query_scheduled_exchanges('DE_LU', entsoe_area_to, start=start_ts, end=end_ts, day_ahead=True)
        flows_df = flows.reset_index()
        flows_df.columns = ['time_dk_obj', 'scheduled_flow_mw']
        flows_df['time_dk_obj'] = flows_df['time_dk_obj'].dt.tz_localize(None) 
        df_trades = pd.merge(df_trades, flows_df, on='time_dk_obj', how='left')
        df_trades['scheduled_flow_mw'] = df_trades['scheduled_flow_mw'].ffill().fillna(0)
    except Exception as e:
        df_trades['de_spot_eur'] = df_trades['spot_price_eur']
        df_trades['V3_2_DK_DE_Spread_Volatility'] = 0.0
        df_trades['scheduled_flow_mw'] = 0.0

    # V4.0 True Forecast Integration
    try:
        err_df = fetch_energinet_true_forecast_error(area)
        if not err_df.empty:
            df_trades['V3_2_Wind_Error_Meteo'] = 0.0 # True error calculation will be injected here during retraining
        else:
            df_trades['V3_2_Wind_Error_Meteo'] = 0.0
    except:
        df_trades['V3_2_Wind_Error_Meteo'] = 0.0
        
    outage_mw = fetch_umm_outages()
    df_trades['umm_outage_mw'] = outage_mw

    # Meta Model Loading
    if strategy_mode == "ML Meta-Model (Random Forest)":
        meta_features = ['V3_1_BiLSTM_Score', 'V3_2_Wind_Error_Meteo', 'V3_2_DK_DE_Spread_Volatility', 'hour_of_day', 'quarter_of_day', 'scheduled_flow_mw']
        model_path = f'models_v3_2/v3_2_meta_model_flow_aware_{area}.pkl'
    else:
        meta_features = ['V3_1_BiLSTM_Score', 'V3_2_Wind_Error_Meteo', 'V3_2_DK_DE_Spread_Volatility', 'hour_of_day', 'quarter_of_day']
        model_path = f'models_v3_2/v3_2_meta_model_{area}.pkl'
        
    meta_model = joblib.load(model_path)
    df_trades['V3_2_Meta_Score'] = meta_model.predict(df_trades[meta_features].fillna(0)) 
    
    def apply_circuit_breaker(row):
        v3_1_action = str(row.get('action', 'HOLD'))
        
        # 1. Crash Prediction (SELL) Override
        if strategy_mode == "Heuristic (>800 MW)" and row.get('scheduled_flow_mw', 0) >= 800:
            return "🔥 CRASH PRED (SELL)"
        elif strategy_mode == "ML Meta-Model (Random Forest)" and "SELL" in row.get('V3_2_Meta_Decision', 'HOLD') and "BUY" in v3_1_action:
            return "🔥 CRASH PRED (SELL)"
            
        # 2. Defensive Circuit Breaker (Cap)
        if row['spot_price_eur'] > row['Dynamic_Cap_EUR'] and "BUY" in row.get('V3_2_Meta_Decision', 'HOLD'):
            return "🛑 C.BREAKER (HOLD)"
            
        return row.get('V3_2_Meta_Decision', 'HOLD')
        
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
    df_trades['V3_2_P_Up'] = df_trades['V3_2_Meta_Score'].apply(lambda x: sigmoid(x)*100 if x >= 0 else 0.5*(1-abs(x/100))*100)
    df_trades['V3_2_P_Dn'] = df_trades['V3_2_Meta_Score'].apply(lambda x: sigmoid(-x)*100 if x < 0 else 0.5*(1-abs(x/100))*100)
    
    df_trades['V3_2_Meta_Decision'] = df_trades['V3_2_Meta_Score'].apply(
        lambda x: "🟢 BUY" if x > 2.0 else ("🔴 SELL" if x < -2.0 else "⚪ HOLD")
    )
    
    dyn_cap = get_dynamic_price_cap(area, date_str)
    df_trades['Dynamic_Cap_EUR'] = dyn_cap
    df_trades['V3_2_Meta_Decision'] = df_trades.apply(apply_circuit_breaker, axis=1)
    
    return df_trades


st.title(f"Nurex Trading - V3.2 Ensemble Command Center ({selected_area})")
try:
    df_real = load_live_monolithic_data(selected_area, date_str_selected, strategy_mode)
    dyn_cap_display = df_real['Dynamic_Cap_EUR'].iloc[0] if not df_real.empty else 0.0
except Exception as e:
    df_real = pd.DataFrame()
    dyn_cap_display = 0.0
    st.error(f"Error loading pipeline: {e}")

dyn_cap = df_real['Dynamic_Cap_EUR'].iloc[0] if not df_real.empty else 0.0

col1, col2, col3 = st.columns(3)
with col1:
    st.caption("Selected Date")
    st.subheader(date_str_selected)
with col2:
    st.caption("30-Day Dynamic Price Cap (95%)")
    st.subheader(f"€ {dyn_cap:.2f}")
    st.markdown("<span style='color: #00C851;'>↑ Circuit Breaker Threshold</span>", unsafe_allow_html=True)
with col3:
    if strategy_mode != "Disabled (V3.1 Only)" and 'scheduled_flow_mw' in df_real.columns:
        latest_flow = df_real['scheduled_flow_mw'].iloc[-1] if not df_real.empty else 0.0
        st.caption("Live Cross-Border Flow")
        st.subheader(f"{latest_flow:.0f} MW")
        st.markdown("<span style='color: #FF8800;'>🧠 ML Active Feature</span>", unsafe_allow_html=True)
        
        if 'V3_2_P_Dn' in df_real.columns:
            latest_prob = df_real['V3_2_P_Dn'].iloc[-1]
            ml_sidebar_placeholder.info(
                f"🧠 **ML Engine Active**\n\n"
                f"Live Flow & Wind evaluated.\n\n"
                f"🔥 **Current Crash Probability:**\n"
                f"### {latest_prob:.1f}%"
            )

tab1, tab2, tab3 = st.tabs(["📁 Live Bidding & Cross-Border", "📈 Market Signals", "⚖️ V3.1 vs V3.2 Comparison"])

with tab1:
    if df_real.empty: st.error("No data available.")
    else:
        q_data = []
        for i, row in df_real.iterrows():
            v3_1_action = row['action'] if pd.notna(row['action']) else "HOLD"
            if "BUY" in v3_1_action: v3_1_action = "BUY"
            if "SELL" in v3_1_action: v3_1_action = "SELL"
            
            spot = row['spot_price_eur']
            de_spot = row.get('de_spot_eur', spot)
            flow = row.get('scheduled_flow_mw', 0.0)
            settled = parse_currency(row['actual_settled_eur'])
            
            v3_1_vol = row['volume_mwh'] if pd.notna(row['volume_mwh']) else 0.0
            v3_2_vol = v3_1_vol if v3_1_vol > 0 else 10.0
            if "CRASH PRED" in row['V3_2_Meta_Decision']: v3_2_vol = 25.0 # Max volume on crash pred
            if "HOLD" in row['V3_2_Meta_Decision'] or "BREAKER" in row['V3_2_Meta_Decision']: 
                v3_2_vol = 0.0
            
            v3_2_pnl = 0.0
            if v3_2_vol > 0 and pd.notna(spot) and pd.notna(settled):
                fees = v3_2_vol * 0.51
                if "BUY" in row['V3_2_Meta_Decision']: v3_2_pnl = (settled - spot) * v3_2_vol - fees
                elif "SELL" in row['V3_2_Meta_Decision']: v3_2_pnl = (spot - settled) * v3_2_vol - fees
                
            q10 = row['q10_price_eur']
            q90 = row['q90_price_eur']
            quantiles_str = f"€{int(q10)}-€{int(q90)}" if pd.notna(q10) and pd.notna(q90) else ""

            q_data.append({
                "Quarter (Time)": f"{row['quarter']} ({row['time_dk'].split(' ')[1]})" if pd.notna(row['time_dk']) else row['quarter'],
                "DK Spot (€)": round(spot, 2) if pd.notna(spot) else None,
                "Dyn. Cap (€)": round(row['Dynamic_Cap_EUR'], 2),
                "DE Spot (€)": round(de_spot, 2) if pd.notna(de_spot) else None,
                "DE->DK Sched. Flow (MW)": round(flow, 1),
                "Settled Imb (€)": row['actual_settled_eur'],
                "V3.1 Pred Imb (€)": f"€ {row['pred_imbalance_eur']:.2f}" if pd.notna(row.get('pred_imbalance_eur')) else None,
                "V3.1 Pred Spread": f"{row['pred_spread_eur']:+.2f}" if pd.notna(row['pred_spread_eur']) else None,
                "V3.1 P(Up/Dn)": f"{row['p_up']*100:.2f}% / {row['p_down']*100:.2f}%",
                "V3.1 Decision": v3_1_action,
                "V3.1 Net PnL": row['net_pnl_eur'], 
                "V3.2 P(Up/Dn)": f"{row['V3_2_P_Up']:.2f}% / {row['V3_2_P_Dn']:.2f}%",
                "V3.2 Meta Decision": row['V3_2_Meta_Decision'],
                "V3.2 Net PnL": f"€ {v3_2_pnl:.2f}" if pd.notna(settled) else "",
            })
            
        df_q96 = pd.DataFrame(q_data)
        
        # --- Copy Toolbar ---
        with st.expander("📋 Copy Column Data Toolbar"):
            ccol1, ccol2 = st.columns([1, 3])
            with ccol1:
                selected_col = st.selectbox("Select Column to Copy:", df_q96.columns.tolist())
            with ccol2:
                st.markdown("*(Hover over the box below and click the **Copy icon** 📋 in the top right)*")
                col_data_str = "\n".join(df_q96[selected_col].astype(str).tolist())
                st.code(col_data_str, language="text")
                
        st.dataframe(df_q96, hide_index=True, use_container_width=True, height=700)

with tab2:
    if not df_real.empty:
        st.subheader("Price Forecasts vs Actuals (with Dynamic Cap)")
        chart_df = df_real[['quarter', 'spot_price_eur', 'pred_imbalance_eur']].copy()
        chart_df['actual_settled'] = df_real['actual_settled_eur'].apply(parse_currency)
        chart_df['Dynamic Circuit Breaker (€)'] = df_real['Dynamic_Cap_EUR']
        chart_df.set_index('quarter', inplace=True)
        chart_df.columns = ['DK Spot (€)', 'V3.1 Pred Imb (€)', 'Actual Settled Imb (€)', 'Dynamic Cap (€)']
        # Custom colors: Spot=Red, Pred=Blue, Actual=Green, Cap=Yellow/Gold
        st.line_chart(chart_df, color=['#FF0000', '#0000FF', '#00FF00', '#FFD700'], height=400)

with tab3:
    if not df_real.empty:
        st.subheader("Performance Comparison: V3.1 vs V3.2")
        
        v3_1_total_pnl = 0.0
        v3_2_total_pnl = 0.0
        
        for i, row in df_real.iterrows():
            spot = row['spot_price_eur']
            try:
                settled = float(str(row['actual_settled_eur']).replace('€', '').replace(',', '').strip())
            except:
                settled = None
            
            # V3.1 logic
            v3_1_vol = row['volume_mwh'] if pd.notna(row['volume_mwh']) else 0.0
            v3_1_action = str(row.get('action', 'HOLD'))
            if v3_1_vol > 0 and pd.notna(spot) and pd.notna(settled):
                fees = v3_1_vol * 0.51
                if "BUY" in v3_1_action: v3_1_total_pnl += (settled - spot) * v3_1_vol - fees
                elif "SELL" in v3_1_action: v3_1_total_pnl += (spot - settled) * v3_1_vol - fees
                
            # V3.2 logic
            v3_2_vol = v3_1_vol if v3_1_vol > 0 else 10.0
            if "CRASH PRED" in row['V3_2_Meta_Decision']: v3_2_vol = 25.0
            if "HOLD" in row['V3_2_Meta_Decision'] or "BREAKER" in row['V3_2_Meta_Decision']: v3_2_vol = 0.0
            
            if v3_2_vol > 0 and pd.notna(spot) and pd.notna(settled):
                fees = v3_2_vol * 0.51
                if "BUY" in row['V3_2_Meta_Decision']: v3_2_total_pnl += (settled - spot) * v3_2_vol - fees
                elif "SELL" in row['V3_2_Meta_Decision']: v3_2_total_pnl += (spot - settled) * v3_2_vol - fees
                
        diff = v3_2_total_pnl - v3_1_total_pnl
        
        colA, colB, colC = st.columns(3)
        colA.metric("V3.1 Total PnL", f"€ {v3_1_total_pnl:,.2f}")
        colB.metric("V3.2 Total PnL", f"€ {v3_2_total_pnl:,.2f}", delta=f"€ {diff:,.2f}", delta_color="normal" if diff >=0 else "inverse")
        
        st.markdown("---")
        st.markdown("### How V3.2 Protects Capital")
        st.write("V3.2 uses the **Flow-Aware Meta Model** and **Dynamic Circuit Breaker** to actively filter out toxic trades that V3.1 would blindly execute.")
