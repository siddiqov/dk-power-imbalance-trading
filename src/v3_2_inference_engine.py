# ==============================================================================
# src/v3_2_inference_engine.py
# V3.2 Flow-Aware Meta-Model Inference Engine
#
# Generates 96-quarter intraday trading schedules for V3.2:
# 1. Base BiLSTM price spread expectation
# 2. Real-time Open-Meteo wind speed & forecast error
# 3. ENTSO-E DE-DK price spread volatility
# 4. Monolithic Random Forest Meta-Model evaluation
# 5. Dynamic 95th-percentile spot price circuit breaker
# ==============================================================================

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

logger = logging.getLogger("V32InferenceEngine")

CHOSEN_V3_1_BASELINE = "Transfer-BiLSTM"
ENTSOE_TOKEN = os.getenv("ENTSOE_TOKEN", os.environ.get("ENTSOE_TOKEN", ""))


class V32InferenceEngine:
    def __init__(self, price_area: str = "DK1", base_volume_mwh: float = 2.0):
        self.price_area = price_area.upper()
        self.base_volume_mwh = base_volume_mwh
        self.meta_model = None
        self._load_meta_model()

    def _load_meta_model(self):
        model_path = os.path.join("models_v3_2", f"v3_2_meta_model_flow_aware_{self.price_area}.pkl")
        if os.path.exists(model_path):
            try:
                self.meta_model = joblib.load(model_path)
                logger.info(f"[{self.price_area}] Loaded V3.2 Flow-Aware Meta-Model from {model_path}")
            except Exception as e:
                logger.warning(f"[{self.price_area}] Failed to load meta-model from {model_path}: {e}")
        else:
            logger.warning(f"[{self.price_area}] Meta-model file not found at {model_path}")

    def fetch_open_meteo_wind(self) -> pd.DataFrame:
        """Fetches Open-Meteo wind speed forecast for Denmark."""
        lat, lon = (56.2639, 9.5018) if self.price_area == "DK1" else (55.6761, 12.5683)
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": ["wind_speed_10m"],
                "timezone": "auto",
            }
            res = requests.get(url, params=params, timeout=5)
            if res.status_code == 200:
                data = res.json()
                df_w = pd.DataFrame({
                    "time": pd.to_datetime(data["hourly"]["time"]),
                    "wind_speed_10m": data["hourly"]["wind_speed_10m"],
                })
                df_w["time_hour_floor"] = df_w["time"].dt.tz_localize(None)
                return df_w
        except Exception as e:
            logger.warning(f"[{self.price_area}] Open-Meteo wind fetch failed: {e}")
        return pd.DataFrame()

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
            logger.info(f"[{self.price_area}] Calculated 95th percentile dynamic cap: €{cap:.2f}/MWh")
            return cap
        except Exception as e:
            logger.warning(f"[{self.price_area}] ENTSO-E dynamic cap query fallback: {e}")
            return 250.0

    def evaluate_96q_schedule(self, target_df: pd.DataFrame, date_str: str) -> list:
        """
        Generates 96 quarters of trading decisions using V3.2 Flow-Aware Meta-Model.
        
        Args:
            target_df: 96-quarter dataframe from TournamentTableGenerator
            date_str: Delivery date 'YYYY-MM-DD'
            
        Returns:
            List of trade dictionaries compatible with SupabasePublisher.upsert_predictions
        """
        if target_df is None or target_df.empty:
            return []

        from src.commercial_strategy_v3_1 import V31CommercialStrategyEngine
        strategy_v31 = V31CommercialStrategyEngine(price_area=self.price_area)
        
        # 1. Base V3.1 BiLSTM evaluation
        try:
            s_v31 = strategy_v31.evaluate_trading_ledger(
                target_df, model_name=CHOSEN_V3_1_BASELINE, market_mode="INTRADAY_D0"
            )
            v31_trades = s_v31.get("trades", [])
        except Exception as e:
            logger.warning(f"[{self.price_area}] V3.1 baseline evaluation failed: {e}")
            v31_trades = []

        df_trades = pd.DataFrame(v31_trades)
        if df_trades.empty:
            return []

        df_trades["time_dk_obj"] = pd.to_datetime(df_trades["time_dk"])
        df_trades["hour_of_day"] = df_trades["time_dk_obj"].dt.hour
        df_trades["quarter_of_day"] = df_trades["time_dk_obj"].dt.hour * 4 + df_trades["time_dk_obj"].dt.minute // 15
        df_trades["time_hour_floor"] = df_trades["time_dk_obj"].dt.floor("h")
        df_trades["V3_1_BiLSTM_Score"] = df_trades.get("pred_spread_eur", 0.0)

        # 2. Weather Integration
        df_wind = self.fetch_open_meteo_wind()
        if not df_wind.empty:
            df_trades = pd.merge(df_trades, df_wind[["time_hour_floor", "wind_speed_10m"]], on="time_hour_floor", how="left")
            df_trades["wind_speed_10m"] = df_trades["wind_speed_10m"].ffill().fillna(6.0)
            df_trades["V3_2_Wind_Error_Meteo"] = (df_trades["wind_speed_10m"] - 6.0) * 200.0
        else:
            df_trades["V3_2_Wind_Error_Meteo"] = 0.0

        # 3. ENTSO-E DE/DK Spread Volatility & Scheduled Flow
        df_trades["V3_2_DK_DE_Spread_Volatility"] = 0.0
        try:
            from entsoe import EntsoePandasClient
            client = EntsoePandasClient(api_key=ENTSOE_TOKEN)
            start_ts = pd.Timestamp(date_str, tz="Europe/Copenhagen")
            end_ts = start_ts + pd.Timedelta(days=1)
            de_prices = client.query_day_ahead_prices("DE_LU", start=start_ts, end=end_ts)
            de_prices_df = de_prices.reset_index()
            de_prices_df.columns = ["time_dk_obj", "de_spot_eur"]
            de_prices_df["time_dk_obj"] = de_prices_df["time_dk_obj"].dt.tz_localize(None)
            df_trades = pd.merge(df_trades, de_prices_df, on="time_dk_obj", how="left")
            df_trades["dk_de_spread"] = df_trades["spot_price_eur"] - df_trades["de_spot_eur"].ffill().fillna(df_trades["spot_price_eur"])
            df_trades["V3_2_DK_DE_Spread_Volatility"] = df_trades["dk_de_spread"].rolling(4, min_periods=1).std().fillna(0.0)
            
            # Fetch scheduled flow
            entsoe_area_to = 'DK_1' if self.price_area == 'DK1' else 'DK_2'
            flows = client.query_scheduled_exchanges('DE_LU', entsoe_area_to, start=start_ts, end=end_ts, day_ahead=True)
            flows_df = flows.reset_index()
            flows_df.columns = ['time_dk_obj', 'scheduled_flow_mw']
            flows_df['time_dk_obj'] = flows_df['time_dk_obj'].dt.tz_localize(None) 
            df_trades = pd.merge(df_trades, flows_df, on='time_dk_obj', how='left')
            df_trades['scheduled_flow_mw'] = df_trades['scheduled_flow_mw'].ffill().fillna(0.0)
        except Exception as e:
            logger.warning(f"[{self.price_area}] ENTSO-E queries failed: {e}")
            df_trades["V3_2_DK_DE_Spread_Volatility"] = 0.0
            df_trades["scheduled_flow_mw"] = 0.0

        # 4. Meta-Model Inference
        meta_features = [
            "V3_1_BiLSTM_Score",
            "V3_2_Wind_Error_Meteo",
            "V3_2_DK_DE_Spread_Volatility",
            "hour_of_day",
            "quarter_of_day",
            "scheduled_flow_mw"
        ]

        if self.meta_model is not None:
            X_meta = df_trades[meta_features].fillna(0.0)
            df_trades["V3_2_Meta_Score"] = self.meta_model.predict(X_meta)
        else:
            # Fallback to V3.1 BiLSTM score if model file missing
            df_trades["V3_2_Meta_Score"] = df_trades["V3_1_BiLSTM_Score"]

        # 5. Dynamic Circuit Breaker
        dyn_cap = self.get_dynamic_price_cap(date_str)

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-float(x)))

        final_trades = []
        for _, row in df_trades.iterrows():
            score = float(row.get("V3_2_Meta_Score", 0.0))
            spot = float(row.get("spot_price_eur", 0.0))
            q_str = str(row.get("quarter", "Q1"))
            t_dk_str = str(row.get("time_dk", ""))

            # Calculate Directional Probabilities
            if score >= 0:
                p_up = float(round(sigmoid(score) * 100.0, 1))
                p_down = float(round(0.5 * (1.0 - min(1.0, abs(score) / 100.0)) * 100.0, 1))
            else:
                p_up = float(round(0.5 * (1.0 - min(1.0, abs(score) / 100.0)) * 100.0, 1))
                p_down = float(round(sigmoid(-score) * 100.0, 1))

            # Signal Thresholds
            if score > 2.0:
                candidate_action = "BUY Spot (Long)"
            elif score < -2.0:
                candidate_action = "SELL Spot (Short)"
            else:
                candidate_action = "HOLD"

            v3_1_action = str(row.get("action", "HOLD"))

            # Dynamic Circuit Breaker & Crash Override
            if "SELL" in candidate_action and "BUY" in v3_1_action:
                action = "SELL (Crash Override)"
                direction = -1
                volume = 25.0
            elif "BUY" in candidate_action and spot > dyn_cap:
                action = "HOLD (Circuit Breaker)"
                direction = 0
                volume = 0.0
            elif "BUY" in candidate_action:
                action = "BUY Spot (Long)"
                direction = 1
                volume = self.base_volume_mwh
            elif "SELL" in candidate_action:
                action = "SELL Spot (Short)"
                direction = -1
                volume = self.base_volume_mwh
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
                "q10_price_eur": round(spot + score - 10.0, 2),
                "q50_price_eur": pred_imb,
                "q90_price_eur": round(spot + score + 10.0, 2),
                "status": "LOCKED_PENDING",
            })

        return final_trades
