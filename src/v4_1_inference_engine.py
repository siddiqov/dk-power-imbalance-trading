# ==============================================================================
# src/v4_1_inference_engine.py
# Nurex V4.1 Institutional High-Alpha Inference Engine
#
# Generates 96-quarter intraday trading schedules for V4.1 incorporating:
# 1. 8-Cable Physical Interconnector Matrix (Capacities, Flows, Headroom)
# 2. European MARI (mFRR) & PICASSO (aFRR) Balancing Activation Pressures
# 3. German Federal Grid (SMARD.de) Residual Load & Surplus Spillover
# 4. Nord Pool XBID Level-2 Order Flow Microstructure Skewness
# 5. Multi-Horizon Velocity (d/dt) & Acceleration (d^2/dt^2) Momentum Signals
# 6. Tuned Stacking Super-Ensemble (LightGBM + CatBoost + Random Forest + Huber)
# 7. Dual Physical Circuit Breakers:
#    - Dynamic 95th-Percentile Spot Price Cap
#    - German System Surplus / Negative Residual Load Long Blocker
# ==============================================================================

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Ensure StackingSuperEnsembleV41 can be deserialized across modules
import __main__
try:
    from train_v4_1 import StackingSuperEnsembleV41
    setattr(__main__, "StackingSuperEnsembleV41", StackingSuperEnsembleV41)
    if "__main__" in sys.modules:
        sys.modules["__main__"].StackingSuperEnsembleV41 = StackingSuperEnsembleV41
except Exception:
    StackingSuperEnsembleV41 = None

from src.feature_engineering_v4_1 import V41FeatureEngine
from src.commercial_strategy_v3_1 import V31CommercialStrategyEngine

logger = logging.getLogger("V41InferenceEngine")

ENTSOE_TOKEN = os.getenv("ENTSOE_TOKEN", "01bb4846-6f4c-4e0f-8333-6c709b316594")


class V41InferenceEngine:
    def __init__(self, price_area: str = "DK1", base_volume_mwh: float = 10.0):
        self.price_area = price_area.upper()
        self.base_volume_mwh = base_volume_mwh
        self.model = None
        self.feature_cols = []
        self._load_model()

    def _load_model(self):
        paths_to_check = [
            os.path.join("models_v4_1", f"v4_1_champion_model_{self.price_area}.pkl"),
            os.path.join("models", "models_v4_1", f"v4_1_champion_model_{self.price_area}.pkl"),
        ]
        for path in paths_to_check:
            if os.path.exists(path):
                try:
                    raw = joblib.load(path)
                    if isinstance(raw, dict):
                        self.model = raw.get("model", raw)
                        self.feature_cols = raw.get("feature_cols", [])
                    else:
                        self.model = raw
                        feat_path = path.replace("champion_model", "features")
                        if os.path.exists(feat_path):
                            self.feature_cols = joblib.load(feat_path)
                    logger.info(f"[{self.price_area}] Loaded V4.1 Champion Model from {path} with {len(self.feature_cols)} features.")
                    return
                except Exception as e:
                    logger.warning(f"[{self.price_area}] Error loading V4.1 model from {path}: {e}")
        logger.warning(f"[{self.price_area}] V4.1 Champion Model not found in {paths_to_check}")

    def get_dynamic_price_cap(self, date_str: str) -> float:
        """Calculates 95th percentile spot price over the last 30 days via ENTSO-E or fallback."""
        try:
            from entsoe import EntsoePandasClient
            client = EntsoePandasClient(api_key=ENTSOE_TOKEN)
            end_ts = pd.Timestamp(date_str, tz="Europe/Copenhagen")
            start_ts = end_ts - pd.Timedelta(days=30)
            entsoe_area = "DK_1" if self.price_area == "DK1" else "DK_2"
            spot_series = client.query_day_ahead_prices(entsoe_area, start=start_ts, end=end_ts)
            cap = float(spot_series.quantile(0.95))
            return max(150.0, cap)
        except Exception:
            return 220.0

    def evaluate_96q_schedule(self, target_df: pd.DataFrame, date_str: str) -> list:
        """
        Generates 96 quarters of trading decisions using V4.1 High-Alpha Model.
        
        Args:
            target_df: 96-quarter dataframe from TournamentTableGenerator
            date_str: Delivery date 'YYYY-MM-DD'
            
        Returns:
            List of trade dictionaries compatible with SupabasePublisher.upsert_predictions
        """
        if target_df is None or target_df.empty:
            return []

        # 1. Base V3.1 BiLSTM evaluation to seed historical momentum
        strategy_v31 = V31CommercialStrategyEngine(price_area=self.price_area)
        try:
            s_v31 = strategy_v31.evaluate_trading_ledger(
                target_df, model_name="Transfer-BiLSTM", market_mode="INTRADAY_D0"
            )
            v31_trades = s_v31.get("trades", [])
        except Exception as e:
            logger.warning(f"[{self.price_area}] V3.1 baseline seeding failed: {e}")
            v31_trades = []

        df_trades = pd.DataFrame(v31_trades) if v31_trades else target_df.copy()
        if "pred_spread_eur" in df_trades.columns:
            df_trades["V3_1_BiLSTM_Score"] = pd.to_numeric(df_trades["pred_spread_eur"], errors="coerce").fillna(0.0)
        else:
            df_trades["V3_1_BiLSTM_Score"] = 0.0

        if "spot_price_eur" not in df_trades.columns:
            df_trades["spot_price_eur"] = pd.to_numeric(df_trades.get("spot_price", 0.0), errors="coerce").fillna(0.0)
        else:
            df_trades["spot_price_eur"] = pd.to_numeric(df_trades["spot_price_eur"], errors="coerce").fillna(0.0)

        # 2. Build V4.1 Feature Matrix (8-Cable matrix, MARI/PICASSO balancing, SMARD.de, Order Flow Skew)
        fe41 = V41FeatureEngine(price_area=self.price_area)
        try:
            df_matrix = fe41.build_feature_matrix(df_trades)
        except Exception as e:
            logger.warning(f"[{self.price_area}] V4.1 feature extraction error: {e}")
            df_matrix = df_trades.copy()

        # 3. Model Inference
        if self.model is not None and self.feature_cols:
            for c in self.feature_cols:
                if c not in df_matrix.columns:
                    df_matrix[c] = 0.0
            X = df_matrix[self.feature_cols].ffill().bfill().fillna(0.0)
            try:
                preds_v41 = self.model.predict(X)
            except Exception as e:
                logger.error(f"[{self.price_area}] V4.1 prediction error: {e}")
                preds_v41 = df_matrix["V3_1_BiLSTM_Score"].values
        else:
            # Fallback to V3.1 score + order flow skew proxy
            skew = df_matrix.get("order_flow_skew", 0.0)
            preds_v41 = df_matrix["V3_1_BiLSTM_Score"].values + (skew * 1.5)

        df_matrix["V4_1_Predicted_Spread_EUR"] = preds_v41

        # 4. Dynamic Safeguards & Physical Circuit Breakers
        dyn_cap = self.get_dynamic_price_cap(date_str)

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-float(x)))

        final_trades = []
        for i, row in df_matrix.iterrows():
            score = float(preds_v41[i])
            spot = float(row.get("spot_price_eur", 0.0))
            q_str = str(row.get("quarter", f"Q{i+1}"))
            t_dk_str = str(row.get("time_dk", ""))

            surplus_mw = float(row.get("net_system_surplus_mw", 0.0))
            smard_res = float(row.get("german_system_balance_mw", 0.0))
            skew = float(row.get("order_flow_skew", 0.0))
            h = float(row.get("hour_of_day", (i // 4)))
            is_late_evening = (h >= 21.5)

            # Directional Probabilities
            if score >= 0:
                p_up = float(round(sigmoid(score) * 100.0, 1))
                p_down = float(round(0.5 * (1.0 - min(1.0, abs(score) / 100.0)) * 100.0, 1))
            else:
                p_up = float(round(0.5 * (1.0 - min(1.0, abs(score) / 100.0)) * 100.0, 1))
                p_down = float(round(sigmoid(-score) * 100.0, 1))

            # Base signal & sizing
            if score > 2.0:
                candidate_action = "BUY Spot (Long)"
                vol = 10.0 if is_late_evening else (25.0 if (score > 8.0 or skew > 0.4) else 10.0)
            elif score < -2.0:
                candidate_action = "SELL Spot (Short)"
                vol = 10.0 if is_late_evening else (25.0 if (score < -8.0 or skew < -0.4) else 10.0)
            else:
                candidate_action = "HOLD"
                vol = 0.0

            # Safeguards & Physical Circuit Breakers:
            # 1. Heavy German Surplus / Spillover Long Blocker
            effective_cap = (dyn_cap * 0.90) if is_late_evening else dyn_cap
            if "BUY" in candidate_action and (surplus_mw > 400.0 or smard_res < -2000.0):
                action = "HOLD (Circuit Breaker)"
                direction = 0
                volume = 0.0
            # 2. Dynamic 95th-Percentile Spot Price Cap
            elif "BUY" in candidate_action and spot > effective_cap:
                action = "HOLD (Circuit Breaker)"
                direction = 0
                volume = 0.0
            elif "BUY" in candidate_action:
                action = "BUY Spot (Long)"
                direction = 1
                volume = vol
            elif "SELL" in candidate_action:
                action = "SELL Spot (Short)"
                direction = -1
                volume = vol
            else:
                action = "HOLD"
                direction = 0
                volume = 0.0

            pred_imb = round(spot + score, 2)
            pred_spread = round(score, 2)

            final_trades.append({
                "quarter": q_str,
                "time_dk": t_dk_str,
                "spot_price_eur": spot,
                "pred_imbalance_eur": pred_imb,
                "pred_spread_eur": pred_spread,
                "p_up": p_up,
                "p_down": p_down,
                "p_up_spike": 0.0,
                "action": action,
                "direction": direction,
                "volume_mwh": volume,
                "q10_price_eur": round(spot + score - 12.0, 2),
                "q50_price_eur": pred_imb,
                "q90_price_eur": round(spot + score + 12.0, 2),
                "status": "LOCKED_PENDING",
            })

        return final_trades
