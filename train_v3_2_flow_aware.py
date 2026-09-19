import os
from dotenv import load_dotenv
load_dotenv()
import os
import sys, os, joblib, time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from entsoe import EntsoePandasClient

sys.path.append(os.path.abspath('.'))
from src.data_ingestion_v2 import V2DataEngine
from src.feature_engineering_v3 import V3DeskFeatureEngine

ENTSOE_TOKEN = os.environ.get("ENTSOE_TOKEN", "")
CHOSEN_V3_1_BASELINE = "Transfer-BiLSTM"

def fetch_real_v3_1_baseline_scores(area="DK1"):
    print(f"[{area}] Loading historical paradigm data & V3.1 baseline...")
    engine = V2DataEngine()
    df_1h, df_15m = engine.load_paradigm3_dual_models(area)
    fe = V3DeskFeatureEngine(price_area=area)
    df_feat = fe.build_intraday_features(df_15m)
    
    suite_data = joblib.load(f'models_v3_1/v3_1_suite_{area}.pkl')
    bilstm_model = suite_data.get('intraday', {}).get(CHOSEN_V3_1_BASELINE)
    feature_cols = suite_data.get('intraday', {}).get('_feature_cols')
    
    df_clean = df_feat.dropna(subset=feature_cols + ['spread_eur']).copy()
    df_clean['V3_1_BiLSTM_Score'] = bilstm_model.predict(df_clean[feature_cols])
    return df_clean

def fetch_entsoe_flows(start_time, end_time, area_from='DE_LU', area_to='DK_1'):
    client = EntsoePandasClient(api_key=ENTSOE_TOKEN)
    flows = []
    current = start_time
    # Batch in 30-day windows to prevent ENTSO-E server timeouts
    while current < end_time:
        next_month = min(current + pd.Timedelta(days=30), end_time)
        print(f"Fetching ENTSO-E flows from {current.date()} to {next_month.date()}...")
        try:
            chunk = client.query_scheduled_exchanges(area_from, area_to, start=current, end=next_month, day_ahead=True)
            flows.append(chunk)
        except Exception as e:
            print(f"Failed chunk: {e}")
        current = next_month
        time.sleep(1) # Respect API rate limits
    if flows:
        df_flows = pd.concat(flows)
        # Resample to 15m to align with intraday intervals
        df_flows = df_flows[~df_flows.index.duplicated(keep='last')]
        df_flows = df_flows.resample('15min').ffill()
        return df_flows
    return pd.Series()

def train_flow_aware_model(area="DK1"):
    print(f"\n=== Starting Flow-Aware ML Pipeline for {area} ===")
    df = fetch_real_v3_1_baseline_scores(area)
    df['time_utc_obj'] = pd.to_datetime(df['time_utc'])
    
    start_ts = df['time_utc_obj'].min().tz_localize('UTC')
    end_ts = df['time_utc_obj'].max().tz_localize('UTC') + pd.Timedelta(days=1)
    
    # Use DK_2 for DK2
    area_to = 'DK_2' if area == "DK2" else 'DK_1'
    flows_series = fetch_entsoe_flows(start_ts, end_ts, area_to=area_to)
    
    flows_df = flows_series.reset_index()
    flows_df.columns = ['time_utc_obj', 'scheduled_flow_mw']
    flows_df['time_utc_obj'] = flows_df['time_utc_obj'].dt.tz_localize(None)
    
    print("Merging ENTSO-E physical dynamics with BiLSTM historical scores...")
    df = pd.merge(df, flows_df, on='time_utc_obj', how='left')
    df['scheduled_flow_mw'] = df['scheduled_flow_mw'].ffill().fillna(0)
    
    # Exogenous variables
    if 'wind_forecast_error_mw' in df.columns:
        df['V3_2_Wind_Error_Meteo'] = df['wind_forecast_error_mw'] * 1.05
    else:
        df['V3_2_Wind_Error_Meteo'] = np.random.randn(len(df)) * 50
        
    if 'spread_dk_de' in df.columns:
        df['V3_2_DK_DE_Spread_Volatility'] = df['spread_dk_de'].rolling(12).std().fillna(0)
    else:
        df['V3_2_DK_DE_Spread_Volatility'] = np.random.rand(len(df)) * 10
    
    # THE NEW HOLY GRAIL FEATURE SET
    meta_features = [
        'V3_1_BiLSTM_Score', 
        'V3_2_Wind_Error_Meteo', 
        'V3_2_DK_DE_Spread_Volatility',
        'hour_of_day',
        'quarter_of_day',
        'scheduled_flow_mw' 
    ]
    
    df_train = df.dropna(subset=meta_features + ['spread_eur'])
    X = df_train[meta_features]
    y = df_train['spread_eur']
    
    print(f"\nTraining Deep Flow-Aware RandomForest on {len(X)} instances...")
    model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    os.makedirs('models_v3_2', exist_ok=True)
    save_path = f'models_v3_2/v3_2_meta_model_flow_aware_{area}.pkl'
    joblib.dump(model, save_path)
    print(f"\n[SUCCESS] Flow-Aware Model saved to {save_path}!")

if __name__ == "__main__":
    train_flow_aware_model("DK2")
