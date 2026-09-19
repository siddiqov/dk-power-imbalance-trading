import os
import json
import numpy as np
import pandas as pd
import torch
import sqlite3
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from src.deep_models_v3_1 import PyTorchBiLSTMRegressor
from src.feature_engineering_v4_1 import V41GridFeatureEngine

def generate_oof_bilstm(area="DK1"):
    print(f"[{area}] Generating Honest Out-Of-Fold BiLSTM Predictions...")
    fe = V41GridFeatureEngine(price_area=area)
    df_raw = fe.load_raw_dataset()
    df_feat = fe.build_feature_matrix(df_raw)
    
    df_clean = df_feat.dropna(subset=['target_spread_eur']).copy()
    if 'time_dk' not in df_clean.columns:
        df_clean = df_clean.reset_index()
    if 'time_dk' in df_clean.columns:
        df_clean = df_clean.sort_values('time_dk').reset_index(drop=True)
    else:
        # Fallback if the index wasn't named time_dk
        df_clean['time_dk'] = df_clean.index
        df_clean = df_clean.sort_values('time_dk').reset_index(drop=True)
    
    y = df_clean['target_spread_eur'].values
    times = df_clean['time_dk'].values
    
    feature_cols = [c for c in df_clean.columns if c not in [
        'time_dk', 'time_utc', 'delivery_start', 'delivery_end', 
        'imbalance_price_eur', 'spread_eur', 'direction', 'target_spread_eur',
        'action', 'volume_mwh', 'pnl_eur'
    ] and not isinstance(df_clean[c].iloc[0], str)]
    
    for c in feature_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')
        
    X = df_clean[feature_cols].fillna(0.0).values
    
    # 5-Fold TimeSeriesSplit for OOF
    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros(len(y))
    
    # Load hyperparams if available
    param_file = "models_v3_1/v3_1_hyperparameters.json"
    hyperparams = {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.001}
    if os.path.exists(param_file):
        with open(param_file, 'r', encoding='utf-8') as f:
            all_params = json.load(f)
            if area in all_params and 'Transfer-BiLSTM' in all_params[area]:
                hyperparams.update(all_params[area]['Transfer-BiLSTM'])
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"  [{area}] Fold {fold+1}/5...")
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        
        model = PyTorchBiLSTMRegressor(
            hidden_dim=hyperparams.get('hidden_dim', 128),
            num_layers=hyperparams.get('num_layers', 2),
            lr=hyperparams.get('lr', 0.001),
            epochs=15,
            batch_size=64
        )
        
        model.fit(X_tr_sc, y_tr)
        preds = model.predict(X_va_sc)
        oof_preds[val_idx] = preds
        
    # Save OOF to database or file
    df_oof = pd.DataFrame({
        'time_dk': times,
        'oof_bilstm_score': oof_preds
    })
    
    df_oof.to_parquet(f"data/oof_bilstm_{area}.parquet", index=False)
    print(f"[{area}] OOF Predictions Saved. Length: {len(df_oof)}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for a in ["DK1", "DK2"]:
        generate_oof_bilstm(a)
