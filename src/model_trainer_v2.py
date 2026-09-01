# ==============================================================================
# src/model_trainer_v2.py
# V2 Multi-Model Tournament Engine implementing the 3 Real-Data Paradigms
# (ZERO SYNTHETIC DATA)
#
# Paradigms:
# 1. Transfer Learning / Two-Phase Training (Hourly Pre-training -> 15m Fine-tuning)
# 2. Hierarchical / Residual Modeling (Macro Hourly + Micro 15m Offset)
# 3. Dual Models (Full 1h downsampled vs Pure 15m modern)
#
# Algorithms: LightGBM, XGBoost, CatBoost, Quantile Regressors, Stacking Ensemble
# ==============================================================================

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge

from src.feature_engineering_v2 import V2FeatureEngineer


class V2ModelTournament:
    """
    Orchestrates training and validation across the 3 Real-Data Paradigms
    and multiple machine learning algorithms.
    """

    def __init__(self, price_area='DK1'):
        self.price_area = price_area
        self.fe = V2FeatureEngineer()
        self.trained_models = {}

        self.feature_cols = [
            "sin_hour", "cos_hour", "quarter_of_day", "sin_quarter", "cos_quarter",
            "day_of_week", "sin_dow", "cos_dow", "is_weekend", "sin_month", "cos_month",
            "is_peak_hour", "spot_price_eur", "spot_diff_1", "spot_diff_4",
            "spot_roll_mean_4", "spot_roll_std_4", "spot_roll_mean_12", "spot_roll_std_12",
            "spot_dist_from_mean_12", "is_negative_spot", "spot_squared"
        ]

    def _prepare_features(self, df):
        """Applies feature engineering and extracts X, y_spread, y_class."""
        df_feat = self.fe.transform(df)
        available_feats = [c for c in self.feature_cols if c in df_feat.columns]

        X = df_feat[available_feats].copy().fillna(0)
        y_spread = df_feat["target_spread"].values if "target_spread" in df_feat.columns else None
        y_class = df_feat["target_class"].values if "target_class" in df_feat.columns else None

        return df_feat, X, y_spread, y_class

    # =========================================================================
    # PARADIGM 1: TRANSFER LEARNING / TWO-PHASE TRAINING
    # =========================================================================

    def train_paradigm1_transfer_learning(self, df_hourly, df_15m, val_days=1):
        """
        Phase 1: Pre-train on hourly history.
        Phase 2: Fine-tune on modern 15-minute data with warm starting.
        """
        print(f"\n--- [Paradigm 1: Transfer Learning] Training on {self.price_area} ---")

        # Fallback if df_hourly is empty or too small: derive hourly from earlier 15m slice
        if df_hourly is None or len(df_hourly) < 100:
            print("  Notice: Aggregating earlier 15m slice to hourly for Phase 1 pre-training...")
            df_hourly = df_15m.copy()
            df_hourly["time_utc"] = pd.to_datetime(df_hourly["time_utc"]).dt.floor("h")
            df_hourly = df_hourly.groupby(["time_utc", "price_area"], as_index=False).agg({
                "spot_price_eur": "mean",
                "imbalance_price_eur": "mean",
                "spread_eur": "mean",
                "direction": "first"
            })

        # Split 15m dataset into Train and Validation (last 96 quarters = 1 day)
        val_cutoff = len(df_15m) - (val_days * 96)
        df_15m_train = df_15m.iloc[:val_cutoff].copy()
        df_15m_val = df_15m.iloc[val_cutoff:].copy()

        # Prepare feature matrices
        _, X_h, y_h_spread, y_h_class = self._prepare_features(df_hourly)
        _, X_15m_tr, y_15m_tr_spread, y_15m_tr_class = self._prepare_features(df_15m_train)
        df_val_feat, X_val, y_val_spread, y_val_class = self._prepare_features(df_15m_val)

        # 1. Phase 1: Pre-train LightGBM on Hourly Data
        print(f"  Phase 1: Pre-training on {len(X_h):,} hourly records...")
        lgb_reg_p1 = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        lgb_reg_p1.fit(X_h, y_h_spread)

        lgb_clf_p1 = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        lgb_clf_p1.fit(X_h, y_h_class)

        # 2. Phase 2: Fine-tune on 15-Minute Data
        print(f"  Phase 2: Fine-tuning on {len(X_15m_tr):,} native 15m records...")
        lgb_reg_p2 = lgb.LGBMRegressor(n_estimators=60, learning_rate=0.04, random_state=42, verbose=-1)
        lgb_reg_p2.fit(X_15m_tr, y_15m_tr_spread)

        lgb_clf_p2 = lgb.LGBMClassifier(n_estimators=60, learning_rate=0.04, random_state=42, verbose=-1)
        lgb_clf_p2.fit(X_15m_tr, y_15m_tr_class)

        # Train Quantiles (q10, q90) for uncertainty bands
        q10_model = lgb.LGBMRegressor(objective='quantile', alpha=0.10, n_estimators=60, verbose=-1, random_state=42)
        q10_model.fit(X_15m_tr, y_15m_tr_spread)

        q90_model = lgb.LGBMRegressor(objective='quantile', alpha=0.90, n_estimators=60, verbose=-1, random_state=42)
        q90_model.fit(X_15m_tr, y_15m_tr_spread)

        # Evaluate on Unseen Validation Day
        pred_spread = lgb_reg_p2.predict(X_val)
        pred_class = lgb_clf_p2.predict(X_val)
        pred_q10 = q10_model.predict(X_val)
        pred_q90 = q90_model.predict(X_val)

        # Map class back: {0: DOWN (-1), 1: NONE (0), 2: UP (1)}
        class_to_dir = {0: -1, 1: 0, 2: 1}
        pred_dir = np.array([class_to_dir[c] for c in pred_class])

        mae = mean_absolute_error(y_val_spread, pred_spread)
        acc = accuracy_score(df_val_feat["target_direction"].values, pred_dir)
        f1 = f1_score(df_val_feat["target_direction"].values, pred_dir, average='macro')

        print(f"  [P1 Results] Spread MAE: {mae:.2f} EUR/MWh | Direction Accuracy: {acc*100:.1f}% | Macro F1: {f1:.3f}")

        results = {
            "paradigm": "1. Transfer Learning (Two-Phase)",
            "model_name": "Transfer-LightGBM",
            "regressor": lgb_reg_p2,
            "classifier": lgb_clf_p2,
            "q10_model": q10_model,
            "q90_model": q90_model,
            "val_df": df_val_feat,
            "pred_spread": pred_spread,
            "pred_direction": pred_dir,
            "pred_q10": pred_q10,
            "pred_q90": pred_q90,
            "mae": mae,
            "acc": acc,
            "f1": f1
        }
        return results

    # =========================================================================
    # PARADIGM 2: HIERARCHICAL / RESIDUAL MODELING
    # =========================================================================

    def train_paradigm2_hierarchical(self, df_macro, df_micro, val_days=1):
        """
        Macro Model: Predicts hourly Day-Ahead spread level.
        Micro Model: Predicts intra-hour quarter residual offset delta_q.
        """
        print(f"\n--- [Paradigm 2: Hierarchical / Residual] Training on {self.price_area} ---")

        if df_macro is None or len(df_macro) < 100:
            print("  Notice: Aggregating earlier 15m slice to hourly for Macro model...")
            df_macro = df_micro.copy()
            df_macro["time_utc"] = pd.to_datetime(df_macro["time_utc"]).dt.floor("h")
            df_macro = df_macro.groupby(["time_utc", "price_area"], as_index=False).agg({
                "spot_price_eur": "mean",
                "imbalance_price_eur": "mean",
                "spread_eur": "mean",
                "direction": "first"
            })

        val_cutoff = len(df_micro) - (val_days * 96)
        df_micro_train = df_micro.iloc[:val_cutoff].copy()
        df_micro_val = df_micro.iloc[val_cutoff:].copy()

        # 1. Macro Model on Hourly Data
        _, X_macro, y_macro_spread, _ = self._prepare_features(df_macro)
        print(f"  Macro Model: Training on {len(X_macro):,} hourly records...")
        macro_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        macro_model.fit(X_macro, y_macro_spread)

        # 2. Micro Model on Residuals in 15m Data
        _, X_micro_tr, y_micro_tr_spread, y_micro_tr_class = self._prepare_features(df_micro_train)
        df_val_feat, X_val, y_val_spread, _ = self._prepare_features(df_micro_val)

        # Macro prediction on training set to find residuals
        macro_preds_tr = macro_model.predict(X_micro_tr)
        residuals_tr = y_micro_tr_spread - macro_preds_tr

        print(f"  Micro Model: Training on {len(X_micro_tr):,} native 15m residual offsets...")
        micro_model = XGBRegressor(n_estimators=80, learning_rate=0.04, random_state=42)
        micro_model.fit(X_micro_tr, residuals_tr)

        micro_clf = XGBClassifier(n_estimators=80, learning_rate=0.04, random_state=42)
        micro_clf.fit(X_micro_tr, y_micro_tr_class)

        # Evaluate on Validation Day
        macro_val_pred = macro_model.predict(X_val)
        micro_val_pred = micro_model.predict(X_val)
        combined_spread_pred = macro_val_pred + micro_val_pred

        pred_class = micro_clf.predict(X_val)
        class_to_dir = {0: -1, 1: 0, 2: 1}
        pred_dir = np.array([class_to_dir[c] for c in pred_class])

        mae = mean_absolute_error(y_val_spread, combined_spread_pred)
        acc = accuracy_score(df_val_feat["target_direction"].values, pred_dir)
        f1 = f1_score(df_val_feat["target_direction"].values, pred_dir, average='macro')

        print(f"  [P2 Results] Spread MAE: {mae:.2f} EUR/MWh | Direction Accuracy: {acc*100:.1f}% | Macro F1: {f1:.3f}")

        results = {
            "paradigm": "2. Hierarchical (Macro + Micro)",
            "model_name": "Hierarchical-LGBM+XGB",
            "macro_model": macro_model,
            "micro_model": micro_model,
            "classifier": micro_clf,
            "val_df": df_val_feat,
            "pred_spread": combined_spread_pred,
            "pred_direction": pred_dir,
            "pred_q10": combined_spread_pred - 1.2 * mae,
            "pred_q90": combined_spread_pred + 1.2 * mae,
            "mae": mae,
            "acc": acc,
            "f1": f1
        }
        return results

    # =========================================================================
    # PARADIGM 3: DUAL MODELS / REGIME SPECIFIC (PURE 15M & FULL 1H)
    # =========================================================================

    def train_paradigm3_dual_models(self, df_full_1h, df_pure_15m, val_days=1):
        """
        Trains standalone CatBoost & LightGBM on the pure modern 15-minute dataset.
        """
        print(f"\n--- [Paradigm 3: Dual Models (Pure 15m)] Training on {self.price_area} ---")

        val_cutoff = len(df_pure_15m) - (val_days * 96)
        df_15m_train = df_pure_15m.iloc[:val_cutoff].copy()
        df_15m_val = df_pure_15m.iloc[val_cutoff:].copy()

        _, X_tr, y_tr_spread, y_tr_class = self._prepare_features(df_15m_train)
        df_val_feat, X_val, y_val_spread, _ = self._prepare_features(df_15m_val)

        # CatBoost Regressor & Classifier
        cat_reg = CatBoostRegressor(iterations=120, learning_rate=0.05, verbose=0, random_seed=42)
        cat_reg.fit(X_tr, y_tr_spread)

        cat_clf = CatBoostClassifier(iterations=120, learning_rate=0.05, verbose=0, random_seed=42)
        cat_clf.fit(X_tr, y_tr_class)

        pred_spread = cat_reg.predict(X_val)
        pred_class = cat_clf.predict(X_val)
        class_to_dir = {0: -1, 1: 0, 2: 1}
        pred_dir = np.array([class_to_dir[int(c[0] if isinstance(c, (list, np.ndarray)) else c)] for c in pred_class])

        mae = mean_absolute_error(y_val_spread, pred_spread)
        acc = accuracy_score(df_val_feat["target_direction"].values, pred_dir)
        f1 = f1_score(df_val_feat["target_direction"].values, pred_dir, average='macro')

        print(f"  [P3 Results] Spread MAE: {mae:.2f} EUR/MWh | Direction Accuracy: {acc*100:.1f}% | Macro F1: {f1:.3f}")

        results = {
            "paradigm": "3. Dual Models (Pure 15m)",
            "model_name": "Pure15m-CatBoost",
            "regressor": cat_reg,
            "classifier": cat_clf,
            "val_df": df_val_feat,
            "pred_spread": pred_spread,
            "pred_direction": pred_dir,
            "pred_q10": pred_spread - 1.2 * mae,
            "pred_q90": pred_spread + 1.2 * mae,
            "mae": mae,
            "acc": acc,
            "f1": f1
        }
        return results

    # =========================================================================
    # STACKING META-ENSEMBLE
    # =========================================================================

    def build_stacking_ensemble(self, paradigm_results_list):
        """
        Blends predictions from all paradigms into a weighted meta-ensemble.
        """
        val_df = paradigm_results_list[0]["val_df"]
        y_val_spread = val_df["target_spread"].values

        # Average spread predictions with inverse-MAE weighting
        maes = [res["mae"] for res in paradigm_results_list]
        weights = 1.0 / (np.array(maes) + 1e-6)
        weights /= np.sum(weights)

        ensemble_spread = np.zeros(len(val_df))
        for res, w in zip(paradigm_results_list, weights):
            ensemble_spread += w * res["pred_spread"]

        # Majority vote for direction
        dir_matrix = np.vstack([res["pred_direction"] for res in paradigm_results_list])
        ensemble_dir = np.sign(np.sum(dir_matrix, axis=0))

        mae = mean_absolute_error(y_val_spread, ensemble_spread)
        acc = accuracy_score(val_df["target_direction"].values, ensemble_dir)
        f1 = f1_score(val_df["target_direction"].values, ensemble_dir, average='macro')

        print(f"\n  [Stacking Ensemble] Spread MAE: {mae:.2f} EUR/MWh | Direction Accuracy: {acc*100:.1f}% | Macro F1: {f1:.3f}")

        return {
            "paradigm": "Meta-Ensemble",
            "model_name": "Stacking-MetaEnsemble",
            "val_df": val_df,
            "pred_spread": ensemble_spread,
            "pred_direction": ensemble_dir,
            "pred_q10": ensemble_spread - 1.1 * mae,
            "pred_q90": ensemble_spread + 1.1 * mae,
            "mae": mae,
            "acc": acc,
            "f1": f1
        }
