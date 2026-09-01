# ==============================================================================
# src/feature_engineering_v2.py
# V2 Feature Engineering & Day-Ahead Preprocessing Pipeline
#
# Generates rich tabular features for both 1-hour and 15-minute resolution
# without data leakage and strictly respecting Day-Ahead information availability.
# ==============================================================================

import pandas as pd
import numpy as np


class V2FeatureEngineer:
    """
    Constructs Day-Ahead predictive features for DK1 & DK2 imbalance forecasting.
    Works seamlessly on both hourly (Phase 1 / Macro) and 15-minute (Phase 2 / Micro) frames.
    """

    def __init__(self):
        pass

    def add_cyclical_time_features(self, df, time_col="time_utc"):
        """Adds sine and cosine encodings for time cycles (day, week, year)."""
        df = df.copy()
        dt = pd.to_datetime(df[time_col])

        # Hour of day (0-23)
        hour = dt.dt.hour + dt.dt.minute / 60.0
        df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
        df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)

        # Quarter of day (1 to 96)
        quarter = (dt.dt.hour * 4 + dt.dt.minute // 15) + 1
        df["quarter_of_day"] = quarter
        df["sin_quarter"] = np.sin(2 * np.pi * quarter / 96.0)
        df["cos_quarter"] = np.cos(2 * np.pi * quarter / 96.0)

        # Day of week (0=Mon, 6=Sun)
        dow = dt.dt.dayofweek
        df["day_of_week"] = dow
        df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
        df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)
        df["is_weekend"] = (dow >= 5).astype(int)

        # Month of year (1-12)
        month = dt.dt.month
        df["sin_month"] = np.sin(2 * np.pi * (month - 1) / 12.0)
        df["cos_month"] = np.cos(2 * np.pi * (month - 1) / 12.0)

        # Peak hours flag (08:00 - 20:00 on weekdays)
        df["is_peak_hour"] = ((hour >= 8) & (hour <= 20) & (df["is_weekend"] == 0)).astype(int)

        return df

    def add_spot_market_features(self, df, spot_col="spot_price_eur"):
        """Constructs momentum, rolling stats, and ramp signals from Day-Ahead Spot price."""
        df = df.copy()

        if spot_col not in df.columns:
            return df

        # Spot price differences (ramp rates)
        df["spot_diff_1"] = df[spot_col].diff().fillna(0)
        df["spot_diff_4"] = df[spot_col].diff(4).fillna(0)  # 1-hour lag in 15m data

        # Rolling statistics
        for window in [4, 12, 24, 96]:
            if len(df) > window:
                df[f"spot_roll_mean_{window}"] = df[spot_col].rolling(window, min_periods=1).mean()
                df[f"spot_roll_std_{window}"] = df[spot_col].rolling(window, min_periods=1).std().fillna(0)
                df[f"spot_dist_from_mean_{window}"] = df[spot_col] - df[f"spot_roll_mean_{window}"]

        # Negative price regime flag (common in Danish wind surplus)
        df["is_negative_spot"] = (df[spot_col] < 0).astype(int)
        df["spot_squared"] = np.sign(df[spot_col]) * (df[spot_col] ** 2)

        return df

    def add_target_labels(self, df, imb_col="imbalance_price_eur", spot_col="spot_price_eur", threshold=0.05):
        """Constructs both regression spread target and 3-class classification direction target."""
        df = df.copy()

        if imb_col in df.columns and spot_col in df.columns:
            # Spread target (Delta = P_imb - P_spot)
            df["target_spread"] = df[imb_col] - df[spot_col]

            # Direction target (+1: UP, -1: DOWN, 0: NONE)
            conditions = [
                df["target_spread"] > threshold,
                df["target_spread"] < -threshold
            ]
            choices = [1, -1]  # 1: UP, -1: DOWN
            df["target_direction"] = np.select(conditions, choices, default=0)

            # Map to classification class index for ML (0: DOWN, 1: NONE, 2: UP)
            direction_map = {-1: 0, 0: 1, 1: 2}
            df["target_class"] = df["target_direction"].map(direction_map)

        return df

    def transform(self, df):
        """Full transformation pipeline."""
        df = self.add_cyclical_time_features(df)
        df = self.add_spot_market_features(df)
        df = self.add_target_labels(df)
        df.dropna(subset=["spot_price_eur"], inplace=True)
        return df


if __name__ == '__main__':
    fe = V2FeatureEngineer()
    print("V2FeatureEngineer initialized successfully.")
