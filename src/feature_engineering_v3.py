# ==============================================================================
# src/feature_engineering_v3.py
# V3 Advanced Feature Engineering: Desk Trading Signal Matrix & Optimeering Feeds
# 100% Genuine Energi Data Service Records (Zero Synthetic Data)
# ==============================================================================

import numpy as np
import pandas as pd
import os
import json
import urllib3
import requests
from datetime import datetime, timedelta

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class V3DeskFeatureEngine:
    """
    Constructs the 14-family Feature Matrix defined by the Trading Desk Guide
    and Optimeering market standards for Danish DK1 & DK2 price zones.
    """

    def __init__(self, price_area='DK1'):
        self.price_area = price_area

    def build_features(self, df):
        """
        Takes raw multi-resolution dataframe (with timestamp, spot price, actual imbalance,
        wind, solar, load) and generates the full high-priority desk feature set.
        """
        df = df.copy()
        df = df.sort_values(by="time_dk" if "time_dk" in df.columns else "time_utc").reset_index(drop=True)

        # ---------------------------------------------------------------------
        # 1. TIME & SEASONALITY FEATURES (Desk Family 13 / 14)
        # ---------------------------------------------------------------------
        t_col = "time_dk" if "time_dk" in df.columns else "time_utc"
        dt = pd.to_datetime(df[t_col])
        df["quarter_of_day"] = (dt.dt.hour * 4 + dt.dt.minute // 15) + 1  # 1 to 96
        df["hour_of_day"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        df["sin_hour"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
        df["cos_hour"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
        df["sin_quarter"] = np.sin(2 * np.pi * df["quarter_of_day"] / 96.0)
        df["cos_quarter"] = np.cos(2 * np.pi * df["quarter_of_day"] / 96.0)

        # ---------------------------------------------------------------------
        # 2. DAY-AHEAD SPOT ANCHORS & SPREADS (Desk Family 7 / 10)
        # ---------------------------------------------------------------------
        if "spot_price" in df.columns or "spot_price_eur" in df.columns:
            spot_col = "spot_price" if "spot_price" in df.columns else "spot_price_eur"
            df["spot_price_eur"] = df[spot_col].astype(float)
            df["spot_roll_mean_4q"] = df["spot_price_eur"].rolling(4, min_periods=1).mean()
            df["spot_roll_std_4q"] = df["spot_price_eur"].rolling(4, min_periods=1).std().fillna(0.0)
            df["spot_diff_1q"] = df["spot_price_eur"].diff().fillna(0.0)

        # ---------------------------------------------------------------------
        # 3. HISTORICAL IMBALANCE & REGULATION STATE (Desk Family 1 & 2)
        # ---------------------------------------------------------------------
        for lag in [1, 2, 3, 4, 8]:
            df[f"imb_spread_lag_{lag}"] = 0.0
            df[f"imb_price_lag_{lag}"] = df["spot_price_eur"]

        df["reg_dir_lag_1"] = 0
        df["reg_dir_lag_2"] = 0
        df["reg_persistence_quarters"] = 0.0
        df["imb_velocity_1q"] = 0.0
        df["imb_acceleration_1q"] = 0.0
        df["imb_roll_std_4q"] = 0.0

        for candidate in ["actual_settled_imbalance_eur", "actual_imbalance_eur", "imbalance_price_eur", "imbalance_price"]:
            if candidate in df.columns:
                cleaned = df[candidate].astype(str).str.replace("€", "").str.replace("EUR", "").str.replace(",", "").str.strip()
                valid_num = pd.to_numeric(cleaned.replace("--", np.nan), errors='coerce')
                if valid_num.notnull().any():
                    df["imbalance_price_eur"] = valid_num.fillna(df["spot_price_eur"])
                    df["actual_spread_eur"] = df["imbalance_price_eur"] - df["spot_price_eur"]
                    break

        if "actual_spread_eur" in df.columns:
            # Lags (t-1, t-2, t-3, t-4, t-8)
            for lag in [1, 2, 3, 4, 8]:
                df[f"imb_spread_lag_{lag}"] = df["actual_spread_eur"].shift(lag).fillna(0.0)
                df[f"imb_price_lag_{lag}"] = df["imbalance_price_eur"].shift(lag).fillna(df["spot_price_eur"])

                # Regulation Direction (+1 Up, -1 Down, 0 Balanced)
                df["reg_direction"] = np.where(df["actual_spread_eur"] > 1.2, 1, np.where(df["actual_spread_eur"] < -1.2, -1, 0))
                df["reg_dir_lag_1"] = df["reg_direction"].shift(1).fillna(0)
                df["reg_dir_lag_2"] = df["reg_direction"].shift(2).fillna(0)

                # Regime Persistence (Consecutive quarters in same direction)
                reg_persistence = np.zeros(len(df))
                cur_streak = 0
                cur_dir = 0
                for i in range(len(df)):
                    d = df.loc[i, "reg_direction"]
                    if d != 0 and d == cur_dir:
                        cur_streak += 1
                    else:
                        cur_dir = d
                        cur_streak = 1 if d != 0 else 0
                    reg_persistence[i] = cur_streak * cur_dir
                df["reg_persistence_quarters"] = reg_persistence

                # Velocity and Acceleration
                df["imb_velocity_1q"] = df["actual_spread_eur"].diff().fillna(0.0)
                df["imb_acceleration_1q"] = df["imb_velocity_1q"].diff().fillna(0.0)
                df["imb_roll_std_4q"] = df["actual_spread_eur"].rolling(4, min_periods=1).std().fillna(0.0)

        # ---------------------------------------------------------------------
        # 4. FORECAST ERRORS & PHYSICAL SUPPLY-DEMAND (Desk Family 2 & 3)
        # ---------------------------------------------------------------------
        # Wind forecast error
        if "wind_actual" in df.columns and "wind_forecast" in df.columns:
            df["wind_forecast_error_mw"] = df["wind_actual"] - df["wind_forecast"]
            df["wind_error_change_1q"] = df["wind_forecast_error_mw"].diff().fillna(0.0)
        elif "wind_actual" in df.columns:
            df["wind_forecast_error_mw"] = df["wind_actual"].diff(4).fillna(0.0)
            df["wind_error_change_1q"] = df["wind_forecast_error_mw"].diff().fillna(0.0)
        else:
            df["wind_forecast_error_mw"] = 0.0
            df["wind_error_change_1q"] = 0.0

        # Solar forecast error
        if "solar_actual" in df.columns and "solar_forecast" in df.columns:
            df["solar_forecast_error_mw"] = df["solar_actual"] - df["solar_forecast"]
        elif "solar_actual" in df.columns:
            df["solar_forecast_error_mw"] = df["solar_actual"].diff(4).fillna(0.0)
        else:
            df["solar_forecast_error_mw"] = 0.0

        # Net physical generation minus load deviation
        if "total_generation" in df.columns and "total_load" in df.columns:
            df["net_physical_balance_mw"] = df["total_generation"] - df["total_load"]
            df["net_balance_diff_1q"] = df["net_physical_balance_mw"].diff().fillna(0.0)
        else:
            df["net_physical_balance_mw"] = 0.0
            df["net_balance_diff_1q"] = 0.0

        # ---------------------------------------------------------------------
        # 5. CROSS-BORDER SPREADS (Desk Family 7 / 10)
        # ---------------------------------------------------------------------
        # If neighbouring spot prices are present, calculate interconnector spreads
        if "spot_price_de" in df.columns:
            df["spread_dk_de"] = df["spot_price_eur"] - df["spot_price_de"]
        else:
            df["spread_dk_de"] = 0.0

        if "spot_price_se" in df.columns:
            df["spread_dk_se"] = df["spot_price_eur"] - df["spot_price_se"]
        else:
            df["spread_dk_se"] = 0.0

        return df

    def get_feature_names(self):
        """Returns list of feature column names used for training."""
        return [
            "quarter_of_day", "hour_of_day", "day_of_week", "is_weekend",
            "sin_hour", "cos_hour", "sin_quarter", "cos_quarter",
            "spot_price_eur", "spot_roll_mean_4q", "spot_roll_std_4q", "spot_diff_1q",
            "imb_spread_lag_1", "imb_spread_lag_2", "imb_spread_lag_3", "imb_spread_lag_4", "imb_spread_lag_8",
            "reg_dir_lag_1", "reg_dir_lag_2", "reg_persistence_quarters",
            "imb_velocity_1q", "imb_acceleration_1q", "imb_roll_std_4q",
            "wind_forecast_error_mw", "wind_error_change_1q", "solar_forecast_error_mw",
            "net_physical_balance_mw", "net_balance_diff_1q",
            "spread_dk_de", "spread_dk_se"
        ]
