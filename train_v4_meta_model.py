import os
import joblib
import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def train_v4_meta_model(area='DK1'):
    print(f"[{area}] Training V4.0 Flow & True Forecast Aware Meta-Model...")
    
    conn = duckdb.connect('energy_data.db')
    
    # We join historical predictions (v2_hourly_imbalance or similar) with electricity_balance and forecasts_hour
    # For simplicity of V4 scaffolding, we simulate loading the existing df_clean and merging actuals
    # In a fully connected state, we pull from the ML features table.
    
    # Instead of writing complex SQL here, let's load the exact same base data used by V3.2
    # and augment it with duckdb queries.
    import sys
    sys.path.append(os.path.abspath('.'))
    from train_v3_2_flow_aware import fetch_real_v3_1_baseline_scores
    
    try:
        df_clean = fetch_real_v3_1_baseline_scores(area)
        df_clean['time_utc_obj'] = pd.to_datetime(df_clean['time_utc']).dt.tz_localize(None)
    except Exception as e:
        print(f"Error loading base paradigm data: {e}")
        return
        
    print(f"Base data loaded: {len(df_clean)} records. Merging V4 true features from DuckDB...")
    
    # Query true forecast errors and true exchanges from DB
    query = f"""
    SELECT 
        eb.time_utc,
        eb.exchange_continent as flow_continent,
        eb.exchange_nordic as flow_nordic,
        eb.exchange_gb as flow_uk,
        (eb.total_wind - fh.forecast_day_ahead) as true_wind_error
    FROM electricity_balance eb
    LEFT JOIN forecasts_hour fh 
        ON eb.time_utc = fh.time_utc AND eb.price_area = fh.price_area 
        AND fh.forecast_type = 'Wind'
    WHERE eb.price_area = '{area}'
    """
    
    v4_features_df = conn.execute(query).fetchdf()
    # Normalize time_utc to match df_clean
    v4_features_df['time_utc_obj'] = pd.to_datetime(v4_features_df['time_utc']).dt.tz_localize(None)
    
    df = pd.merge(df_clean, v4_features_df, on='time_utc_obj', how='left')
    
    # Fill missing with rolling medians to simulate continuous flow
    for col in ['flow_continent', 'flow_nordic', 'flow_uk', 'true_wind_error']:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0)
    
    df['V4_Spread_Volatility'] = df['spread_eur'].rolling(12).std().fillna(0)
    
    # V4.0 Meta Features
    meta_features = [
        'V3_1_BiLSTM_Score', 
        'true_wind_error', 
        'V4_Spread_Volatility',
        'hour_of_day',
        'quarter_of_day',
        'flow_continent',
        'flow_nordic',
        'flow_uk'
    ]
    
    df_train = df.dropna(subset=meta_features + ['spread_eur'])
    X = df_train[meta_features]
    y = df_train['spread_eur']
    
    print(f"\nTraining Deep V4 RandomForest on {len(X)} instances...")
    model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    os.makedirs('models_v4', exist_ok=True)
    save_path = f'models_v4/v4_meta_model_{area}.pkl'
    joblib.dump(model, save_path)
    
    # Save the feature columns so the dashboard knows what to inject
    joblib.dump(meta_features, f'models_v4/v4_features_{area}.pkl')
    print(f"\n[SUCCESS] V4.0 Model saved to {save_path}!")

if __name__ == '__main__':
    train_v4_meta_model("DK1")
    train_v4_meta_model("DK2")
