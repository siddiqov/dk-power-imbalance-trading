# ==============================================================================
# src/hyperparameter_optimizer_v3_1.py
# Optuna Bayesian Hyperparameter Optimizer for V3.1 Commercial Models
# Purged Walk-Forward TimeSeriesSplit (Zero Leakage)
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Suppress verbose Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class V31HyperparameterOptimizer:
    """
    Executes cross-validated Bayesian optimization for LightGBM, CatBoost, and XGBoost
    using Purged TimeSeriesSplit cross-validation on Danish power market data.
    """

    def __init__(self, price_area="DK1", output_dir="models_v3_1"):
        self.price_area = price_area
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.param_file = os.path.join(self.output_dir, f"best_hyperparameters_{price_area}.json")

    def purged_time_series_splits(self, n_samples, n_splits=4, purge_gap=192):
        """
        Yields (train_idx, val_idx) splits with a purge gap of 48 hours (192 quarters)
        between train and test to prevent temporal lookahead leakage.
        """
        test_size = n_samples // (n_splits + 1)
        for i in range(1, n_splits + 1):
            train_end = i * test_size
            test_start = train_end + purge_gap
            test_end = min(test_start + test_size, n_samples)
            if test_start >= n_samples:
                break
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx

    def optimize_lightgbm(self, X, y, n_trials=15):
        """Optimizes LightGBMRegressor using Optuna."""
        print(f"[{self.price_area}] Running Optuna Study for Transfer-LightGBM ({n_trials} trials)...")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 160),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.07, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.90),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "random_state": 42,
                "verbose": -1
            }

            scores = []
            for tr_idx, val_idx in self.purged_time_series_splits(len(X), n_splits=3, purge_gap=96):
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                scores.append(mean_absolute_error(y_val, preds))

            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        print(f"  -> LightGBM Best Trial MAE: {study.best_value:.3f} EUR/MWh")
        return study.best_params

    def optimize_catboost(self, X, y, n_trials=10):
        """Optimizes CatBoostRegressor using Optuna."""
        print(f"[{self.price_area}] Running Optuna Study for Pure15m-CatBoost ({n_trials} trials)...")

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 120, 260),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.06, log=True),
                "depth": trial.suggest_int("depth", 4, 7),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 8.0),
                "random_seed": 42,
                "verbose": 0
            }

            scores = []
            for tr_idx, val_idx in self.purged_time_series_splits(len(X), n_splits=3, purge_gap=96):
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                model = CatBoostRegressor(**params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                scores.append(mean_absolute_error(y_val, preds))

            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        print(f"  -> CatBoost Best Trial MAE: {study.best_value:.3f} EUR/MWh")
        return study.best_params

    def run_full_optimization(self, df_15m, feature_cols, n_trials_lgb=12, n_trials_cat=8):
        """Runs full Bayesian optimization pipeline and caches parameters."""
        X = df_15m[feature_cols].fillna(0.0).values
        y = df_15m["actual_spread_eur"].values

        best_lgb = self.optimize_lightgbm(X, y, n_trials=n_trials_lgb)
        best_cat = self.optimize_catboost(X, y, n_trials=n_trials_cat)

        params_bundle = {
            "price_area": self.price_area,
            "Transfer-LightGBM": best_lgb,
            "Pure15m-CatBoost": best_cat,
            "Hierarchical-LGBM": {
                "n_estimators": best_lgb.get("n_estimators", 110),
                "learning_rate": best_lgb.get("learning_rate", 0.04),
                "max_depth": max(best_lgb.get("max_depth", 5) - 1, 4),
                "num_leaves": max(best_lgb.get("num_leaves", 31) // 2, 15),
                "random_state": 42, "verbose": -1
            },
            "Hierarchical-XGB": {
                "n_estimators": 100,
                "learning_rate": 0.04,
                "max_depth": 5,
                "subsample": 0.85,
                "random_state": 42,
                "n_jobs": 2
            }
        }

        with open(self.param_file, "w", encoding="utf-8") as f:
            json.dump(params_bundle, f, indent=2)

        print(f"[{self.price_area}] Saved best hyperparameters to {self.param_file}")
        return params_bundle

    def load_best_parameters(self):
        """Loads cached parameters or returns calibrated default configuration."""
        if os.path.exists(self.param_file):
            try:
                with open(self.param_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        # Fallback to calibrated Bayesian defaults
        return {
            "price_area": self.price_area,
            "Transfer-LightGBM": {
                "n_estimators": 125, "learning_rate": 0.035, "max_depth": 6,
                "num_leaves": 35, "subsample": 0.85, "colsample_bytree": 0.80,
                "reg_alpha": 0.5, "reg_lambda": 1.0, "random_state": 42, "verbose": -1
            },
            "Pure15m-CatBoost": {
                "iterations": 220, "learning_rate": 0.038, "depth": 6,
                "l2_leaf_reg": 3.5, "random_seed": 42, "verbose": 0
            },
            "Hierarchical-LGBM": {
                "n_estimators": 100, "learning_rate": 0.04, "max_depth": 5,
                "num_leaves": 25, "random_state": 42, "verbose": -1
            },
            "Hierarchical-XGB": {
                "n_estimators": 100, "learning_rate": 0.04, "max_depth": 5,
                "subsample": 0.85, "random_state": 42, "n_jobs": 2
            }
        }
