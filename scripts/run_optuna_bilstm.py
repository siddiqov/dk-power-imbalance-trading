import os
import json
import numpy as np
import pandas as pd
from src.hyperparameter_optimizer_v3_1 import V31HyperparameterOptimizer
from src.feature_engineering_v4_1 import V41GridFeatureEngine

def run_bilstm_tuning(area="DK1", trials=15):
    print(f"\n[{area}] Starting BiLSTM Optuna Tuning...")
    fe = V41GridFeatureEngine(price_area=area)
    df_raw = fe.load_raw_dataset()
    df_feat = fe.build_feature_matrix(df_raw)
    
    df_clean = df_feat.dropna(subset=['target_spread_eur']).copy()
    y = df_clean['target_spread_eur'].values
    
    feature_cols = [c for c in df_clean.columns if c not in [
        'time_dk', 'time_utc', 'delivery_start', 'delivery_end', 
        'imbalance_price_eur', 'spread_eur', 'direction', 'target_spread_eur',
        'action', 'volume_mwh', 'pnl_eur'
    ] and not isinstance(df_clean[c].iloc[0], str)]
    
    for c in feature_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')
        
    X = df_clean[feature_cols].fillna(0.0).values
    
    opt = V31HyperparameterOptimizer(price_area=area, output_dir="models_v3_1")
    best_bilstm = opt.optimize_bilstm(X, y, n_trials=trials)
    
    param_file = opt.param_file
    if os.path.exists(param_file):
        with open(param_file, 'r', encoding='utf-8') as f:
            all_params = json.load(f)
    else:
        all_params = {}
        
    all_params['Transfer-BiLSTM'] = best_bilstm
    
    with open(param_file, 'w', encoding='utf-8') as f:
        json.dump(all_params, f, indent=4)
        
    print(f"[{area}] BiLSTM parameters updated: {best_bilstm}")

if __name__ == "__main__":
    for a in ["DK1", "DK2"]:
        run_bilstm_tuning(a, trials=15)
