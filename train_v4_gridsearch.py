# ==============================================================================
# train_v4_gridsearch.py
# Nurex V4.0 Systematic Hyperparameter Grid Search & Champion Model Selector
# 
# ZERO Blind/Random Parameters:
# - Strict TimeSeriesSplit(n_splits=5) Walk-Forward Cross-Validation
# - Multi-Algorithm Tournament: LightGBM vs CatBoost vs Random Forest
# - Grid Optimization over Estimators, Depths, Learning Rates, Regularization
# - Serialization of Champion Models & Complete Metric Logs to models_v4/
# ==============================================================================

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

sys.path.append(os.path.abspath('.'))
from src.feature_engineering_v4 import V4GridFeatureEngine
from train_v3_2_flow_aware import fetch_real_v3_1_baseline_scores

def run_grid_search_for_area(area="DK1"):
    print(f"\n=======================================================", flush=True)
    print(f"  STARTING INSTITUTIONAL V4.0 GRID SEARCH FOR {area}", flush=True)
    print(f"=======================================================", flush=True)
    start_time_all = time.time()

    # 1. Ingest 4-Layer Feature Matrix
    fe = V4GridFeatureEngine(price_area=area)
    df_raw = fe.load_raw_dataset()
    df_matrix = fe.build_feature_matrix(df_raw)
    
    # 2. Attach V3.1 BiLSTM Baseline Momentum Scores
    print(f"[{area}] Fetching & attaching V3.1 BiLSTM Neural Momentum...", flush=True)
    try:
        df_v31 = fetch_real_v3_1_baseline_scores(area)
        df_v31['time_utc'] = pd.to_datetime(df_v31['time_utc'])
        score_df = df_v31[['time_utc', 'V3_1_BiLSTM_Score']].drop_duplicates(subset=['time_utc'])
        df_matrix['time_utc'] = pd.to_datetime(df_matrix['time_utc'])
        df_matrix = pd.merge(df_matrix, score_df, on='time_utc', how='left')
        df_matrix['V3_1_BiLSTM_Score'] = df_matrix['V3_1_BiLSTM_Score'].ffill().bfill().fillna(0.0)
    except Exception as e:
        print(f"[{area}] Warning loading V3.1 scores: {e}. Using zero baseline.", flush=True)
        df_matrix['V3_1_BiLSTM_Score'] = 0.0

    # 3. Assemble Full Feature Set (35 features)
    feature_cols = ['V3_1_BiLSTM_Score'] + fe.get_feature_column_names()
    for col in feature_cols:
        if col not in df_matrix.columns:
            df_matrix[col] = 0.0

    target_col = 'target_spread_eur'
    clean_df = df_matrix.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    X = clean_df[feature_cols]
    y = clean_df[target_col]
    print(f"[{area}] Clean Dataset: {len(X)} rows | {len(feature_cols)} features.", flush=True)

    # 4. Strict TimeSeriesSplit Cross-Validation (Walk-Forward)
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # 5. Define Systematic Hyperparameter Grid (NO blind or random parameters)
    candidate_configs = []

    # A. LightGBM Candidates
    for n_est in [100, 200]:
        for depth in [4, 6]:
            for lr in [0.03, 0.08]:
                for subsample in [0.8, 1.0]:
                    for reg_l in [1.0, 5.0]:
                        candidate_configs.append({
                            "family": "LightGBM",
                            "params": {
                                "n_estimators": n_est,
                                "max_depth": depth,
                                "learning_rate": lr,
                                "subsample": subsample,
                                "subsample_freq": 1 if subsample < 1.0 else 0,
                                "reg_lambda": reg_l,
                                "random_state": 42,
                                "n_jobs": -1,
                                "verbose": -1
                            }
                        })

    # B. CatBoost Candidates
    for n_est in [100, 150]:
        for depth in [4, 6]:
            for lr in [0.04, 0.08]:
                for l2 in [3.0, 7.0]:
                    candidate_configs.append({
                        "family": "CatBoost",
                        "params": {
                            "iterations": n_est,
                            "depth": depth,
                            "learning_rate": lr,
                            "l2_leaf_reg": l2,
                            "random_seed": 42,
                            "verbose": 0,
                            "thread_count": -1
                        }
                    })

    # C. Random Forest Candidates
    for n_est in [100, 150]:
        for depth in [6, 10]:
            for min_split in [2, 5]:
                candidate_configs.append({
                    "family": "RandomForest",
                    "params": {
                        "n_estimators": n_est,
                        "max_depth": depth,
                        "min_samples_split": min_split,
                        "random_state": 42,
                        "n_jobs": -1
                    }
                })

    print(f"[{area}] Total Configurations in Grid: {len(candidate_configs)}", flush=True)
    print(f"[{area}] Running 5-Fold Walk-Forward Cross-Validation...", flush=True)

    results = []

    for idx, config in enumerate(candidate_configs, 1):
        family = config["family"]
        params = config["params"]

        fold_maes, fold_rmses, fold_r2s, fold_das = [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if family == "LightGBM":
                model = LGBMRegressor(**params)
            elif family == "CatBoost":
                model = CatBoostRegressor(**params)
            else:
                model = RandomForestRegressor(**params)

            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            mae = mean_absolute_error(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            r2 = r2_score(y_val, preds)
            da = np.mean((np.sign(preds) == np.sign(y_val))) * 100.0

            fold_maes.append(mae)
            fold_rmses.append(rmse)
            fold_r2s.append(r2)
            fold_das.append(da)

        mean_mae = float(np.mean(fold_maes))
        mean_rmse = float(np.mean(fold_rmses))
        mean_r2 = float(np.mean(fold_r2s))
        mean_da = float(np.mean(fold_das))

        results.append({
            "config_id": idx,
            "family": family,
            "params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in params.items() if k not in ['verbose', 'n_jobs', 'thread_count']},
            "cv_mae": round(mean_mae, 4),
            "cv_rmse": round(mean_rmse, 4),
            "cv_r2": round(mean_r2, 4),
            "cv_directional_accuracy_pct": round(mean_da, 2)
        })

        if idx % 5 == 0 or idx == len(candidate_configs):
            print(f"  [{area}] Config {idx:2d}/{len(candidate_configs)} | {family:12s} | MAE: {mean_mae:.3f} EUR | DA: {mean_da:.1f}%", flush=True)

    # 6. Rank Results & Select Champion Model
    results_sorted = sorted(results, key=lambda r: (r["cv_mae"], -r["cv_directional_accuracy_pct"]))
    champion_info = results_sorted[0]

    print(f"\n[{area}] =======================================================", flush=True)
    print(f"[{area}] GRID SEARCH COMPLETED IN {time.time() - start_time_all:.1f}s", flush=True)
    print(f"[{area}] CHAMPION MODEL: {champion_info['family']} (Config #{champion_info['config_id']})", flush=True)
    print(f"[{area}] CV MAE: {champion_info['cv_mae']} EUR/MWh", flush=True)
    print(f"[{area}] CV RMSE: {champion_info['cv_rmse']} EUR/MWh", flush=True)
    print(f"[{area}] CV Directional Accuracy: {champion_info['cv_directional_accuracy_pct']}%", flush=True)
    print(f"[{area}] Best Hyperparameters: {json.dumps(champion_info['params'], indent=2)}", flush=True)
    print(f"[{area}] =======================================================", flush=True)

    # 7. Retrain Champion on 100% of Clean Historical Data
    champ_family = champion_info["family"]
    champ_params = champion_info["params"]

    if champ_family == "LightGBM":
        full_params = dict(champ_params)
        full_params.update({"random_state": 42, "n_jobs": -1, "verbose": -1})
        champion_model = LGBMRegressor(**full_params)
    elif champ_family == "CatBoost":
        full_params = dict(champ_params)
        full_params.update({"random_seed": 42, "verbose": 0, "thread_count": -1})
        champion_model = CatBoostRegressor(**full_params)
    else:
        full_params = dict(champ_params)
        full_params.update({"random_state": 42, "n_jobs": -1})
        champion_model = RandomForestRegressor(**full_params)

    champion_model.fit(X, y)

    # Extract Feature Importances
    if hasattr(champion_model, "feature_importances_"):
        raw_imp = champion_model.feature_importances_
        norm_imp = (raw_imp / np.sum(raw_imp)) * 100.0
        feature_importance_dict = {f: round(float(imp), 2) for f, imp in zip(feature_cols, norm_imp)}
        feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
    else:
        feature_importance_dict = {}

    # 8. Save Artifacts
    os.makedirs('models_v4', exist_ok=True)
    
    log_path = f'models_v4/grid_search_results_{area}.json'
    with open(log_path, 'w') as f:
        json.dump({
            "area": area,
            "timestamp": datetime.now().isoformat(),
            "total_configs_evaluated": len(candidate_configs),
            "champion": champion_info,
            "feature_importance_ranking": feature_importance_dict,
            "top_10_configs": results_sorted[:10],
            "all_results": results_sorted
        }, f, indent=2)
    print(f"[{area}] Saved full grid search log to {log_path}", flush=True)

    bundle_path = f'models_v4/v4_champion_model_{area}.pkl'
    bundle = {
        "model": champion_model,
        "family": champ_family,
        "params": champ_params,
        "feature_cols": feature_cols,
        "champion_metrics": {
            "cv_mae": champion_info["cv_mae"],
            "cv_rmse": champion_info["cv_rmse"],
            "cv_r2": champion_info["cv_r2"],
            "cv_da_pct": champion_info["cv_directional_accuracy_pct"]
        },
        "feature_importance": feature_importance_dict,
        "trained_date": datetime.now().isoformat(),
        "total_training_rows": len(X)
    }
    joblib.dump(bundle, bundle_path)
    print(f"[{area}] Successfully serialized V4.0 Champion Model to {bundle_path}", flush=True)

    return bundle

if __name__ == "__main__":
    t_start = time.time()
    bundle_dk1 = run_grid_search_for_area("DK1")
    bundle_dk2 = run_grid_search_for_area("DK2")
    print(f"\n=======================================================", flush=True)
    print(f"  V4.0 TOTAL PIPELINE FINISHED IN {time.time() - t_start:.1f} SECONDS", flush=True)
    print(f"=======================================================", flush=True)
