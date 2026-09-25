# ==============================================================================
# src/model_trainer_v3_1.py
# V3.1 Advanced Multi-Paradigm Trainer with Optuna Tuning & Native PyTorch Deep Networks
# Complete 4-Paradigm Suite:
#   P1: Transfer-LightGBM (Optuna-Tuned) & Transfer-BiLSTM (PyTorch 2-Layer BiLSTM)
#   P2: Hierarchical-LGBM+XGB (Macro 1h + Micro 15m Residuals)
#   P3: Pure15m-CatBoost (Optuna-Tuned) & Transformer-TFT (PyTorch Multi-Head Self-Attention)
#   P4: Stacking-MetaEnsemble (Meta-Blending of Trees + Neural Networks)
#   Optimeering Quantiles (q10, q50, q90 Pinball Loss) & Direction Classifier
# ZERO SYNTHETIC DATA / 100% Genuine Danish Market Records
# ==============================================================================

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.linear_model import Ridge
import lightgbm as lgb
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor

from src.feature_engineering_v3 import V3DeskFeatureEngine
from src.data_ingestion_v2 import V2DataEngine
from src.deep_models_v3_1 import PyTorchBiLSTMRegressor, PyTorchTemporalAttentionRegressor
from src.hyperparameter_optimizer_v3_1 import V31HyperparameterOptimizer


class HierarchicalModelWrapper:
    def __init__(self, macro, micro):
        self.macro = macro
        self.micro = micro

    def predict(self, X):
        return self.macro.predict(X) + self.micro.predict(X)


class MetaEnsembleWrapper:
    def __init__(self, models_dict, meta_reg):
        self.models_dict = models_dict
        self.meta_reg = meta_reg

    def predict(self, X):
        preds = []
        for name, m in self.models_dict.items():
            try:
                preds.append(m.predict(X))
            except Exception:
                preds.append(np.zeros(len(X)))
        stacked = np.column_stack(preds)
        return self.meta_reg.predict(stacked)


class V31QuantileModelSuite:
    """
    V3.1 Commercial Model Suite integrating:
    - Optuna Bayesian tuned tree models (LightGBM, CatBoost, XGBoost)
    - Native PyTorch sequence models (Transfer-BiLSTM, Transformer-TFT)
    - Stacking Meta-Ensemble blending trees and deep sequence networks
    - Optimeering multi-quantile corridor (q10, q50, q90) and Tri-State Classifier
    """

    def __init__(self, price_area='DK1', model_dir='models_v3_1'):
        self.price_area = price_area
        self.model_dir = model_dir
        self.feature_engine = V3DeskFeatureEngine(price_area=price_area)
        self.opt_engine = V31HyperparameterOptimizer(price_area=price_area, output_dir=model_dir)
        self.models_day_ahead = {}
        self.models_intraday = {}
        self.models = {}
        os.makedirs(self.model_dir, exist_ok=True)

    def _train_suite_for_dataset(self, df_15m, df_hourly, feature_cols, suite_name="Day-Ahead"):
        print(f"\n--- [{self.price_area}] Training V3.1 {suite_name} Suite ({len(feature_cols)} features) ---")
        X_15m = df_15m[feature_cols].fillna(0.0).values
        _spread_col_15m = next((c for c in ["actual_spread_eur", "error_spread_eur", "actual_settled_imbalance_eur"] if c in df_15m.columns), None)
        if _spread_col_15m is None:
            raise KeyError(f"No spread column found in 15m data. Available: {list(df_15m.columns)}")
        y_15m = df_15m[_spread_col_15m].values
        y_dir = np.where(y_15m > 1.2, 2, np.where(y_15m < -1.2, 0, 1))

        X_1h = df_hourly[feature_cols].fillna(0.0).values
        _spread_col_1h = next((c for c in ["actual_spread_eur", "error_spread_eur", "actual_settled_imbalance_eur"] if c in df_hourly.columns), None)
        if _spread_col_1h is None:
            raise KeyError(f"No spread column found in 1h data. Available: {list(df_hourly.columns)}")
        y_1h = df_hourly[_spread_col_1h].values

        # Load Optuna-tuned parameters
        params = self.opt_engine.load_best_parameters()
        lgb_params = params.get("Transfer-LightGBM", {})
        cat_params = params.get("Pure15m-CatBoost", {})
        h_lgb_params = params.get("Hierarchical-LGBM", {})
        h_xgb_params = params.get("Hierarchical-XGB", {})

        models = {}

        # ---------------------------------------------------------------------
        # 1. PARADIGM 1: TRANSFER LEARNING
        # ---------------------------------------------------------------------
        print(f"  [P1.1] Transfer-LightGBM (Optuna Tuned: depth={lgb_params.get('max_depth', 6)}, lr={lgb_params.get('learning_rate', 0.035)})...")
        p1_pre = lgb.LGBMRegressor(**lgb_params)
        p1_pre.fit(X_1h, y_1h)

        p1_fine = lgb.LGBMRegressor(**lgb_params)
        p1_fine.fit(X_15m, y_15m, init_model=p1_pre)
        models["Transfer-LightGBM"] = p1_fine

        print(f"  [P1.2] Transfer-BiLSTM (Native PyTorch 2-Layer Bidirectional Recurrent Network)...")
        bilstm_params = self.opt_engine.load_best_parameters().get("Transfer-BiLSTM", {"hidden_dim": 64, "num_layers": 2, "lr": 0.004, "epochs": 6, "batch_size": 256})
        p1_bilstm = PyTorchBiLSTMRegressor(**bilstm_params)
        p1_bilstm.fit(X_15m, y_15m)
        models["Transfer-BiLSTM"] = p1_bilstm
        models["Deep-BiLSTM"] = p1_bilstm  # Backwards compatibility alias

        # ---------------------------------------------------------------------
        # 2. PARADIGM 2: HIERARCHICAL RESIDUAL MODELING
        # ---------------------------------------------------------------------
        print(f"  [P2] Hierarchical LGBM (Macro Trend) + XGBoost (Micro 15m Residuals)...")
        m_macro = lgb.LGBMRegressor(**h_lgb_params)
        m_macro.fit(X_1h, y_1h)

        macro_preds_15m = m_macro.predict(X_15m)
        residuals_15m = y_15m - macro_preds_15m

        m_micro = XGBRegressor(**h_xgb_params)
        m_micro.fit(X_15m, residuals_15m)
        models["Hierarchical-LGBM+XGB"] = HierarchicalModelWrapper(m_macro, m_micro)

        # ---------------------------------------------------------------------
        # 3. PARADIGM 3: DUAL HIGH-RESOLUTION MODELS (PURE 15M)
        # ---------------------------------------------------------------------
        print(f"  [P3.1] Pure15m-CatBoost (Optuna Tuned: depth={cat_params.get('depth', 6)}, l2={cat_params.get('l2_leaf_reg', 3.5)})...")
        m_cat = CatBoostRegressor(**cat_params)
        m_cat.fit(X_15m, y_15m)
        models["Pure15m-CatBoost"] = m_cat

        print(f"  [P3.2] Transformer-TFT (Native PyTorch Temporal Multi-Head Attention)...")
        m_tft = PyTorchTemporalAttentionRegressor(d_model=64, n_heads=4, num_layers=2, lr=0.003, epochs=6, batch_size=256)
        m_tft.fit(X_15m, y_15m)
        models["Transformer-TFT"] = m_tft

        # ---------------------------------------------------------------------
        # 4. PARADIGM 4: STACKING META-ENSEMBLE (Trees + PyTorch Deep Sequence)
        # ---------------------------------------------------------------------
        print(f"  [P4] Stacking Meta-Ensemble (Blending 3 Trees + 2 PyTorch Networks)...")
        ensemble_candidates = {
            "p1_lgb": p1_fine,
            "p1_lstm": p1_bilstm,
            "p2_hier": models["Hierarchical-LGBM+XGB"],
            "p3_cat": m_cat,
            "p3_tft": m_tft
        }
        cand_preds = np.column_stack([m.predict(X_15m) for m in ensemble_candidates.values()])
        meta_reg = Ridge(alpha=1.0)
        meta_reg.fit(cand_preds, y_15m)
        models["Stacking-MetaEnsemble"] = MetaEnsembleWrapper(ensemble_candidates, meta_reg)

        # ---------------------------------------------------------------------
        # 5. OPTIMEERING QUANTILES (q10, q50, q90 via Pinball Loss)
        # ---------------------------------------------------------------------
        print(f"  [Optimeering] Training Pinball Loss Quantiles (q10, q50, q90)...")
        for alpha in [0.10, 0.50, 0.90]:
            q_name = f"q{int(alpha*100)}"
            q_model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=alpha,
                n_estimators=150,
                learning_rate=0.035,
                max_depth=6,
                num_leaves=35,
                subsample=0.85,
                random_state=42,
                verbose=-1
            )
            q_model.fit(X_15m, y_15m)
            models[f"Quantile_{q_name}"] = q_model

        # ---------------------------------------------------------------------
        # 6. TRI-STATE DIRECTION CLASSIFIER
        # ---------------------------------------------------------------------
        print(f"  [Optimeering] Training Direction Classifier (Tri-State Softmax)...")
        # BUY-FIX [Change 1]: balanced sample weights so p_up is not systematically
        # underestimated and tune() can discover profitable BUY thresholds.
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_dir)

        m_clf = XGBClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=5,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            n_jobs=2
        )
        m_clf.fit(X_15m, y_dir, sample_weight=sample_weights)
        models["Direction_Classifier"] = m_clf
        models["_feature_cols"] = feature_cols

        return models

    def train_all_paradigms(self, run_optuna=False):
        print(f"\n{'='*80}")
        print(f"  [V3.1 RETRAINING] Training Dual Suites (Trees + PyTorch Networks) for {self.price_area}")
        print(f"{'='*80}")

        engine = V2DataEngine()
        df_hourly_raw, df_15m_raw = engine.load_paradigm1_transfer_learning(self.price_area)

        if df_hourly_raw is None or len(df_hourly_raw) < 100:
            print(f"  [Info] Creating 1h Macro historical aggregation from {len(df_15m_raw):,} native 15m records...")
            df_hourly_raw = df_15m_raw.copy()
            df_hourly_raw["time_utc"] = pd.to_datetime(df_hourly_raw["time_utc"]).dt.floor("h")
            df_hourly_raw = df_hourly_raw.groupby(["time_utc", "price_area"], as_index=False).agg({
                "spot_price_eur": "mean",
                "imbalance_price_eur": "mean",
                "spread_eur": "mean"
            })

        print(f"  -> Historical Hourly (Macro): {len(df_hourly_raw):,} records")
        print(f"  -> Modern 15-minute (Micro): {len(df_15m_raw):,} records")

        # 1. Day-Ahead Features
        df_da_15m = self.feature_engine.build_day_ahead_features(df_15m_raw)
        df_da_1h = self.feature_engine.build_day_ahead_features(df_hourly_raw)
        da_cols = [c for c in self.feature_engine.get_day_ahead_feature_names() if c in df_da_15m.columns]

        if run_optuna:
            self.opt_engine.run_full_optimization(df_da_15m, da_cols, n_trials_lgb=10, n_trials_cat=6)

        # 2. Train Day-Ahead Suite
        self.models_day_ahead = self._train_suite_for_dataset(df_da_15m, df_da_1h, da_cols, suite_name="Day-Ahead (D-1)")

        # 3. Intraday Features & Suite
        df_id_15m = self.feature_engine.build_intraday_features(df_15m_raw)
        df_id_1h = self.feature_engine.build_intraday_features(df_hourly_raw)
        id_cols = [c for c in self.feature_engine.get_intraday_feature_names() if c in df_id_15m.columns]

        self.models_intraday = self._train_suite_for_dataset(df_id_15m, df_id_1h, id_cols, suite_name="Intraday (D-0)")

        self.models = {
            "day_ahead": self.models_day_ahead,
            "intraday": self.models_intraday,
            "_default": self.models_day_ahead
        }

        save_path = os.path.join(self.model_dir, f"v3_1_suite_{self.price_area}.pkl")
        joblib.dump(self.models, save_path)
        print(f"\n[{self.price_area}] [SUCCESS] Saved V3.1 Dual Suites to {save_path}\n")

    def load_models(self):
        save_path = os.path.join(self.model_dir, f"v3_1_suite_{self.price_area}.pkl")
        print(f"[load_models] Attempting: {save_path} (exists={os.path.exists(save_path)})")
        if os.path.exists(save_path):
            try:
                data = joblib.load(save_path)
                if isinstance(data, dict) and "day_ahead" in data and "intraday" in data:
                    self.models = data
                    self.models_day_ahead = data["day_ahead"]
                    self.models_intraday = data["intraday"]
                else:
                    self.models = {"day_ahead": data, "intraday": data, "_default": data}
                    self.models_day_ahead = data
                    self.models_intraday = data
                print(f"[load_models] SUCCESS: {save_path}")
                return True
            except Exception as _e:
                print(f"[load_models] FAILED: {_e}")
                return False
        return False

    def predict_day_ahead_quantiles(self, df_day_d, market_mode="DAY_AHEAD_D1", approach="A", df_prev_day=None):
        if not self.models:
            loaded = self.load_models()
            if not loaded:
                self.train_all_paradigms()

        suite = self.models_day_ahead if market_mode == "DAY_AHEAD_D1" else self.models_intraday
        if not suite:
            suite = self.models.get("day_ahead", {})

        feat_cols = suite.get("_feature_cols", self.feature_engine.get_day_ahead_feature_names())
        if market_mode == "DAY_AHEAD_D1":
            df_feat = self.feature_engine.build_day_ahead_features(df_day_d)
        else:
            df_feat = self.feature_engine.build_intraday_features(df_day_d)

        # Check if actual settlement is present or if we are forecasting D+1 / live D-0 quarters
        has_actual_settlement = ("reg_persistence_quarters" in df_feat.columns and df_feat["reg_persistence_quarters"].abs().sum() > 0)
        
        if market_mode != "DAY_AHEAD_D1" and not has_actual_settlement:
            # Autoregressive forward persistence for D+1 Intraday
            available_cols_base = [c for c in feat_cols if c in df_feat.columns]
            X_base = df_feat[available_cols_base].fillna(0.0).values
            
            # Use preliminary spread predictions to determine trajectory
            if "Transfer-LightGBM" in suite:
                s_prelim = suite["Transfer-LightGBM"].predict(X_base)
            elif "Quantile_q50" in suite:
                s_prelim = suite["Quantile_q50"].predict(X_base)
            else:
                s_prelim = np.zeros(len(df_feat))
                
            dir_proj = np.where(s_prelim > 1.2, 1, np.where(s_prelim < -1.2, -1, 0))
            
            tail_streak = 0
            tail_dir = 0
            if approach == "A":
                # Approach A: Continuous Midnight Boundary Bridge
                # Check df_prev_day for actual settlements or latest available telemetry
                if df_prev_day is None:
                    try:
                        from datetime import timedelta
                        t_col = "time_dk" if "time_dk" in df_day_d.columns else "time_utc"
                        first_dt = pd.to_datetime(df_day_d.iloc[0][t_col])
                        prev_date_str = (first_dt - timedelta(days=1)).strftime("%Y-%m-%d")
                        p_file = f"results/96Q_backtest_table_{self.price_area}_{prev_date_str}.csv"
                        if os.path.exists(p_file):
                            df_prev_day = pd.read_csv(p_file)
                        else:
                            live_file = f"results/96Q_future_table_{self.price_area}.csv"
                            if os.path.exists(live_file):
                                df_prev_day = pd.read_csv(live_file)
                    except Exception:
                        df_prev_day = None
                        
                if df_prev_day is not None and not df_prev_day.empty:
                    p_sp = df_prev_day["spot_price_eur"].values if "spot_price_eur" in df_prev_day.columns else np.zeros(len(df_prev_day))
                    p_act_num = None
                    for c in ["actual_settled_imbalance_eur", "imbalance_price_eur", "ImbalancePriceEUR"]:
                        if c in df_prev_day.columns:
                            cleaned = df_prev_day[c].astype(str).str.replace("€", "").str.replace(",", "").str.strip()
                            parsed = pd.to_numeric(cleaned.replace("--", np.nan), errors="coerce")
                            if parsed.notnull().any():
                                p_act_num = parsed
                                break

                    p_pred_spread = None
                    for c in ["pred_spread_eur", "transfer_lgb_eur", "hierarchical_eur", "pure15m_catboost_eur"]:
                        if c in df_prev_day.columns:
                            if "pred_spread" in c:
                                p_pred_spread = pd.to_numeric(df_prev_day[c].astype(str).str.replace("€", "").str.replace("+", "").str.strip(), errors="coerce").fillna(0.0).values
                            else:
                                p_pred_spread = (pd.to_numeric(df_prev_day[c], errors="coerce").values - p_sp)
                            break
                    if p_pred_spread is None:
                        p_pred_spread = np.zeros(len(df_prev_day))

                    if p_act_num is not None:
                        act_spread = (p_act_num - p_sp).values
                        spread_blended = np.where(pd.notnull(act_spread), act_spread, p_pred_spread)
                    else:
                        spread_blended = p_pred_spread

                    dir_d = np.where(spread_blended > 1.2, 1, np.where(spread_blended < -1.2, -1, 0))
                    cs, cd = 0, 0
                    for d in dir_d:
                        if d != 0 and d == cd:
                            cs += 1
                        else:
                            cd = d
                            cs = 1 if d != 0 else 0
                    tail_streak, tail_dir = cs, cd
                                
            # Sequential streak propagation across the delivery horizon
            persistence = np.zeros(len(df_feat))
            cur_s = tail_streak if approach == "A" else 0
            cur_d = tail_dir if approach == "A" else 0
            for i in range(len(df_feat)):
                d = dir_proj[i]
                if d != 0 and d == cur_d:
                    cur_s += 1
                else:
                    cur_d = d
                    cur_s = 1 if d != 0 else 0
                persistence[i] = cur_s * cur_d
                
            df_feat["reg_persistence_quarters"] = persistence

        available_cols = [c for c in feat_cols if c in df_feat.columns]
        X = df_feat[available_cols].fillna(0.0).values

        q10_spread = suite["Quantile_q10"].predict(X)
        q50_spread = suite["Quantile_q50"].predict(X)
        q90_spread = suite["Quantile_q90"].predict(X)

        q10_spread = np.minimum(q10_spread, q50_spread)
        q90_spread = np.maximum(q90_spread, q50_spread)

        spot_arr = df_day_d["spot_price_eur"].values if "spot_price_eur" in df_day_d.columns else np.zeros(len(df_feat))

        clf = suite["Direction_Classifier"]
        probs_raw = clf.predict_proba(X)
        p_down = probs_raw[:, 0]
        p_bal = probs_raw[:, 1]
        p_up = probs_raw[:, 2]

        p_up_spike = np.clip((q90_spread - 20.0) / 60.0, 0.0, 1.0) * p_up

        predictions = {
            "quantiles": {
                "q10_spread": q10_spread,
                "q50_spread": q50_spread,
                "q90_spread": q90_spread,
                "q10_price": spot_arr + q10_spread,
                "q50_price": spot_arr + q50_spread,
                "q90_price": spot_arr + q90_spread
            },
            "probabilities": {
                "p_down": p_down,
                "p_balanced": p_bal,
                "p_up": p_up,
                "p_up_spike": p_up_spike
            }
        }

        for model_name in [
            "Transfer-LightGBM", "Transfer-BiLSTM", "Deep-BiLSTM",
            "Hierarchical-LGBM+XGB", "Pure15m-CatBoost", "Transformer-TFT",
            "Stacking-MetaEnsemble"
        ]:
            if model_name in suite:
                try:
                    s_pred = suite[model_name].predict(X)
                except Exception:
                    s_pred = q50_spread
                predictions[model_name] = {
                    "pred_spread": s_pred,
                    "pred_imbalance": df_day_d.get("spot_price_eur", 0.0) + s_pred
                }

        return predictions
