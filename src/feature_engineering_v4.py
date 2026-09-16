# ==============================================================================
# src/feature_engineering_v4.py
# Nurex V4.0 Advanced 4-Layer Feature Matrix & Multi-Cable Physical Grid Engine
# 
# Layer 1: Physical System State (Load, Wind, Solar, Net Flows)
# Layer 2: Expectation Deviations (Actual - DA Forecast, Intraday Revisions)
# Layer 3: Market Reaction & Interconnector Congestion (ATC - Flow Headroom)
# Layer 4: Real-Time TSO Balancing Response (aFRR / mFRR Sequence & Persistence)
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
import duckdb

# Cable Capacities (MW) defined by Energinet / Nord Pool
CABLE_CAPACITIES = {
    "DK1": {
        "continent": 2500.0,   # DK1 <-> Germany (DE-LU)
        "nordic": 2320.0,      # DK1 <-> NO2 (1640 MW) + DK1 <-> SE3 (680 MW)
        "gb": 1400.0,          # DK1 <-> Great Britain (Viking Link)
        "great_belt": 580.0    # DK1 <-> DK2 (Storebælt HVDC)
    },
    "DK2": {
        "continent": 985.0,    # DK2 <-> Germany (Kontek + Kriegers Flak)
        "nordic": 1240.0,      # DK2 <-> Sweden (SE4 Øresund)
        "gb": 0.0,             # DK2 has no direct cable to GB
        "great_belt": 580.0    # DK2 <-> DK1 (Storebælt HVDC)
    }
}

class V4GridFeatureEngine:
    """
    Constructs the comprehensive 4-Layer Feature Matrix for V4.0
    combining physics, deviations, market dynamics, and balancing activations.
    """
    def __init__(self, price_area='DK1', db_path="energy_data.db"):
        self.price_area = price_area.upper()
        self.db_path = db_path
        self.capacities = CABLE_CAPACITIES.get(self.price_area, CABLE_CAPACITIES["DK1"])

    def load_raw_dataset(self):
        """Loads and aligns native 15m imbalance data with physical electricity balance and forecasts."""
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            # Query base 15m imbalance data
            df_imb = con.execute(f"""
                SELECT 
                    time_utc,
                    price_area,
                    imbalance_price_eur,
                    spot_price_eur,
                    spread_eur,
                    direction,
                    satisfied_demand,
                    dominating_direction,
                    COALESCE(afrr_up_mw, 0) as afrr_up_mw,
                    COALESCE(afrr_down_mw, 0) as afrr_down_mw,
                    afrr_vwa_up_eur,
                    afrr_vwa_down_eur,
                    COALESCE(mfrr_marginal_up_eur, spot_price_eur) as mfrr_marginal_up_eur,
                    COALESCE(mfrr_marginal_down_eur, spot_price_eur) as mfrr_marginal_down_eur
                FROM v2_15min_imbalance
                WHERE price_area = '{self.price_area}'
                ORDER BY time_utc ASC
            """).fetchdf()

            # Query 15m physical balance data
            df_bal = con.execute(f"""
                SELECT 
                    time_utc,
                    total_load,
                    total_wind,
                    wind_offshore,
                    wind_onshore,
                    solar,
                    exchange_continent,
                    exchange_great_belt,
                    exchange_nordic,
                    exchange_gb,
                    net_exchange
                FROM electricity_balance
                WHERE price_area = '{self.price_area}'
                ORDER BY time_utc ASC
            """).fetchdf()

            # Query hourly forecast data (Downsampled / forward filled to 15m)
            df_fcs = con.execute(f"""
                SELECT 
                    time_utc,
                    forecast_day_ahead as wind_forecast_da,
                    forecast_intraday as wind_forecast_id
                FROM forecasts_hour
                WHERE price_area = '{self.price_area}' AND forecast_type = 'Offshore Wind'
                ORDER BY time_utc ASC
            """).fetchdf()

        finally:
            con.close()

        # Merge imbalance with physical balance
        df = pd.merge(df_imb, df_bal, on='time_utc', how='inner')
        
        # Merge with forecasts
        if not df_fcs.empty:
            df_fcs = df_fcs.drop_duplicates(subset=['time_utc'])
            df = pd.merge(df, df_fcs, on='time_utc', how='left')
            df['wind_forecast_da'] = df['wind_forecast_da'].ffill().bfill()
            df['wind_forecast_id'] = df['wind_forecast_id'].ffill().bfill()
        else:
            df['wind_forecast_da'] = df['total_wind'].rolling(4, min_periods=1).mean()
            df['wind_forecast_id'] = df['total_wind']

        df['time_utc'] = pd.to_datetime(df['time_utc'])
        df = df.sort_values('time_utc').reset_index(drop=True)
        return df

    def _fetch_live_prodex(self, date_str: str) -> pd.DataFrame:
        """Fetches authentic real-time 5-min production, wind, and 8-cable exchanges from Energinet."""
        import requests, json
        url = "https://api.energidataservice.dk/dataset/ElectricityProdex5MinRealtime"
        params = {
            "filter": json.dumps({"PriceArea": self.price_area}),
            "start": f"{date_str}T00:00",
            "end": f"{date_str}T23:59",
            "limit": 500
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                return pd.DataFrame()
            records = r.json().get("records", [])
            if not records:
                return pd.DataFrame()
            df_p = pd.DataFrame(records)
            df_p['time_dk_dt'] = pd.to_datetime(df_p['Minutes5DK'])
            df_15 = df_p.set_index('time_dk_dt').resample('15min').mean(numeric_only=True).reset_index()
            
            df_out = pd.DataFrame()
            df_out['time_dk_str'] = df_15['time_dk_dt'].dt.strftime('%Y-%m-%d %H:%M')
            df_out['wind_offshore'] = df_15.get('OffshoreWindPower', 0.0)
            df_out['wind_onshore'] = df_15.get('OnshoreWindPower', 0.0)
            df_out['total_wind'] = df_out['wind_offshore'] + df_out['wind_onshore']
            df_out['flow_de'] = df_15.get('ExchangeGermany', 0.0)
            df_out['flow_nl'] = df_15.get('ExchangeNetherlands', 0.0)
            df_out['flow_great_belt'] = df_15.get('ExchangeGreatBelt', 0.0)
            df_out['flow_no'] = df_15.get('ExchangeNorway', 0.0)
            df_out['flow_se'] = df_15.get('ExchangeSweden', 0.0)
            df_out['flow_gb'] = df_15.get('ExchangeGreatBritain', 0.0)
            df_out['solar'] = df_15.get('SolarPower', 0.0)
            
            df_out['exchange_continent'] = df_out['flow_de'] + df_out['flow_nl']
            df_out['exchange_great_belt'] = df_out['flow_great_belt']
            df_out['exchange_nordic'] = df_out['flow_no'] + df_out['flow_se']
            df_out['exchange_gb'] = df_out['flow_gb']
            df_out['net_exchange'] = (df_out['exchange_continent'] + df_out['exchange_great_belt'] + 
                                      df_out['exchange_nordic'] + df_out['exchange_gb'])
            prod_tot = (df_15.get('ProductionLt100MW', 0.0) + df_15.get('ProductionGe100MW', 0.0) + 
                        df_out['total_wind'] + df_out['solar'])
            df_out['total_load'] = (prod_tot - df_out['net_exchange']).clip(lower=800.0)
            return df_out
        except Exception:
            return pd.DataFrame()

    def build_feature_matrix(self, df_raw):
        """Constructs all 4 layers of institutional features."""
        df = df_raw.copy()

        # Extract date from df if possible
        d_str = None
        if 'time_dk' in df.columns and len(df) > 0:
            d_str = str(df['time_dk'].iloc[0]).split(' ')[0]
        elif 'time_utc' in df.columns and len(df) > 0:
            d_str = str(df['time_utc'].iloc[0]).split(' ')[0]

        # 1. Attempt to merge with authentic live Prodex telemetry
        merged_live = False
        if d_str:
            df_live = self._fetch_live_prodex(d_str)
            if not df_live.empty and 'time_dk' in df.columns:
                df['time_dk_str'] = pd.to_datetime(df['time_dk']).dt.strftime('%Y-%m-%d %H:%M')
                df = pd.merge(df, df_live, on='time_dk_str', how='left')
                if 'total_wind' in df.columns and df['total_wind'].notna().sum() > 10:
                    merged_live = True

        # 2. If live prodex not available, fallback to DuckDB electricity_balance
        if not merged_live and 'total_load' not in df.columns:
            try:
                con = duckdb.connect(self.db_path, read_only=True)
                df_bal = con.execute(f"""
                    SELECT 
                        time_utc,
                        total_load,
                        total_wind,
                        wind_offshore,
                        wind_onshore,
                        solar,
                        exchange_continent,
                        exchange_great_belt,
                        exchange_nordic,
                        exchange_gb,
                        net_exchange
                    FROM electricity_balance
                    WHERE price_area = '{self.price_area}'
                    ORDER BY time_utc ASC
                """).fetchdf()
                con.close()

                if 'time_utc' in df.columns:
                    df['time_utc'] = pd.to_datetime(df['time_utc'])
                    df_bal['time_utc'] = pd.to_datetime(df_bal['time_utc'])
                    df = pd.merge(df, df_bal, on='time_utc', how='left')
                elif 'time_dk' in df.columns:
                    df['time_dk_obj'] = pd.to_datetime(df['time_dk'])
                    df_bal['time_utc'] = pd.to_datetime(df_bal['time_utc'])
                    df = pd.merge_asof(
                        df.sort_values('time_dk_obj'),
                        df_bal.sort_values('time_utc'),
                        left_on='time_dk_obj',
                        right_on='time_utc',
                        direction='nearest'
                    )
            except Exception:
                pass

        # Ensure all physical columns exist with realistic defaults if null
        for col, def_val in [
            ('total_load', 3200.0),
            ('total_wind', 1100.0),
            ('solar', 50.0),
            ('wind_offshore', 600.0),
            ('wind_onshore', 500.0),
            ('exchange_continent', 0.0),
            ('exchange_great_belt', 0.0),
            ('exchange_nordic', 0.0),
            ('exchange_gb', 0.0),
            ('net_exchange', 0.0),
            ('wind_forecast_da', 1100.0),
            ('wind_forecast_id', 1100.0),
            ('afrr_up_mw', 0.0),
            ('afrr_down_mw', 0.0),
            ('mfrr_marginal_up_eur', 100.0),
            ('mfrr_marginal_down_eur', 100.0)
        ]:
            if col not in df.columns:
                df[col] = def_val
            else:
                df[col] = df[col].ffill().bfill().fillna(def_val)

        # -------------------------------------------------------------
        # LAYER 1: Physical Grid State
        # -------------------------------------------------------------
        # Net Generation & System Surplus
        df['physical_gen_mw'] = df['total_wind'] + df['solar']
        df['net_load_balance_mw'] = df['physical_gen_mw'] - df['total_load']
        df['net_system_surplus_mw'] = df['net_load_balance_mw'] + df['net_exchange']
        df['renewable_penetration'] = df['physical_gen_mw'] / (df['total_load'] + 1e-4)

        # Flow Components (Dedicated 8-Cable Telemetry)
        for flow_col, fallback_col in [
            ('flow_de', 'exchange_continent'),
            ('flow_nl', None),
            ('flow_no', 'exchange_nordic'),
            ('flow_se', None),
            ('flow_gb', 'exchange_gb'),
            ('flow_great_belt', 'exchange_great_belt')
        ]:
            if flow_col in df.columns:
                df[flow_col] = pd.Series(df[flow_col]).ffill().fillna(0.0)
            elif fallback_col and fallback_col in df.columns:
                df[flow_col] = pd.Series(df[fallback_col]).ffill().fillna(0.0)
            else:
                df[flow_col] = 0.0

        df['flow_continent'] = df['exchange_continent'].ffill().fillna(0.0) if 'exchange_continent' in df.columns else 0.0
        df['flow_nordic'] = df['exchange_nordic'].ffill().fillna(0.0) if 'exchange_nordic' in df.columns else 0.0

        # -------------------------------------------------------------
        # LAYER 2: Deviations from Expectation (Forecast Errors & Revisions)
        # -------------------------------------------------------------
        # Forecast Error: Actual - Day-Ahead Forecast
        df['wind_forecast_error_mw'] = df['total_wind'] - df['wind_forecast_da']
        # Forecast Revision: Intraday expectation change
        df['wind_forecast_revision_mw'] = df['wind_forecast_id'] - df['wind_forecast_da']
        # Load surprise proxy (deviation from 24h rolling expectation)
        df['load_surprise_mw'] = df['total_load'] - df['total_load'].rolling(96, min_periods=4).mean()

        # -------------------------------------------------------------
        # LAYER 3: Interconnector Headroom & Congestion Dynamics
        # -------------------------------------------------------------
        cap_cont = self.capacities["continent"]
        cap_nordic = self.capacities["nordic"]
        cap_gb = self.capacities["gb"]
        cap_gbelt = self.capacities["great_belt"]

        # Available Transfer Capacity Headroom: ATC - |Flow|
        df['headroom_continent_mw'] = np.maximum(0, cap_cont - df['flow_continent'].abs())
        df['headroom_nordic_mw'] = np.maximum(0, cap_nordic - df['flow_nordic'].abs())
        df['headroom_great_belt_mw'] = np.maximum(0, cap_gbelt - df['flow_great_belt'].abs())
        if cap_gb > 0:
            df['headroom_gb_mw'] = np.maximum(0, cap_gb - df['flow_gb'].abs())
        else:
            df['headroom_gb_mw'] = 0.0

        # Congestion Ratios [0 to 1]
        df['congestion_ratio_continent'] = (df['flow_continent'].abs() / cap_cont).clip(0, 1.0)
        df['congestion_ratio_nordic'] = (df['flow_nordic'].abs() / cap_nordic).clip(0, 1.0)
        df['congestion_ratio_great_belt'] = (df['flow_great_belt'].abs() / cap_gbelt).clip(0, 1.0)
        df['congestion_ratio_gb'] = (df['flow_gb'].abs() / (cap_gb if cap_gb > 0 else 1.0)).clip(0, 1.0)

        # Bottleneck Indicator (Is any critical interconnector > 85% congested?)
        df['is_severely_congested'] = (
            (df['congestion_ratio_continent'] > 0.85) | 
            (df['congestion_ratio_nordic'] > 0.85)
        ).astype(int)

        # -------------------------------------------------------------
        # LAYER 4: TSO Balancing & Reserve Dispatch Dynamics
        # -------------------------------------------------------------
        df['afrr_net_activation_mw'] = df['afrr_up_mw'] - df['afrr_down_mw']
        df['afrr_gross_volume_mw'] = df['afrr_up_mw'] + df['afrr_down_mw']
        df['afrr_trend_1h'] = df['afrr_net_activation_mw'] - df['afrr_net_activation_mw'].shift(4).fillna(0)
        
        # mFRR balancing spread
        df['mfrr_balancing_spread'] = df['mfrr_marginal_up_eur'] - df['mfrr_marginal_down_eur']
        
        # Ensure spread_eur exists
        if 'spread_eur' not in df.columns:
            if 'actual_settled_eur' in df.columns and 'spot_price_eur' in df.columns:
                s_act = pd.to_numeric(
                    df['actual_settled_eur'].astype(str).str.replace('€', '').str.replace('EUR', '').str.replace(',', '').str.strip(), 
                    errors='coerce'
                )
                s_spot = pd.to_numeric(
                    df['spot_price_eur'].astype(str).str.replace('€', '').str.replace('EUR', '').str.replace(',', '').str.strip(), 
                    errors='coerce'
                )
                df['spread_eur'] = (s_act - s_spot).fillna(0.0)
            elif 'pred_spread_eur' in df.columns:
                df['spread_eur'] = pd.to_numeric(df['pred_spread_eur'], errors='coerce').fillna(0.0)
            elif 'target_spread' in df.columns:
                df['spread_eur'] = pd.to_numeric(df['target_spread'], errors='coerce').fillna(0.0)
            else:
                df['spread_eur'] = 0.0

        # Regulation Regime Persistence
        df['spread_lag1'] = df['spread_eur'].shift(1).fillna(0)
        df['spread_lag2'] = df['spread_eur'].shift(2).fillna(0)
        df['spread_rolling_mean4'] = df['spread_eur'].shift(1).rolling(4, min_periods=1).mean().fillna(0)
        df['spread_rolling_std4'] = df['spread_eur'].shift(1).rolling(4, min_periods=1).std().fillna(0)

        # -------------------------------------------------------------
        # Time & Diurnal Harmonics
        # -------------------------------------------------------------
        if 'time_utc' not in df.columns:
            if 'time_dk' in df.columns:
                df['time_utc'] = pd.to_datetime(df['time_dk'])
            else:
                df['time_utc'] = pd.date_range(start="2026-01-01", periods=len(df), freq="15min")
        else:
            df['time_utc'] = pd.to_datetime(df['time_utc'])

        dt = df['time_utc']
        df['hour_of_day'] = dt.dt.hour + dt.dt.minute / 60.0
        df['quarter_of_day'] = dt.dt.hour * 4 + dt.dt.minute // 15 + 1
        df['day_of_week'] = dt.dt.dayofweek
        df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        df['is_peak_hour'] = ((df['hour_of_day'] >= 8) & (df['hour_of_day'] <= 20) & (df['is_weekend'] == 0)).astype(int)
        df['sin_hour'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)

        # Target Spread (EUR/MWh)
        df['target_spread_eur'] = df['spread_eur']

        # Clean NaNs
        df = df.dropna(subset=['target_spread_eur']).reset_index(drop=True)
        return df

    def get_feature_column_names(self):
        """Returns the complete list of institutional features."""
        features = [
            # Layer 1: Physical Grid State
            'total_load', 'total_wind', 'solar', 'net_exchange',
            'net_load_balance_mw', 'net_system_surplus_mw', 'renewable_penetration',
            'flow_continent', 'flow_nordic', 'flow_great_belt', 'flow_gb',
            # Layer 2: Deviations from Expectation
            'wind_forecast_error_mw', 'wind_forecast_revision_mw', 'load_surprise_mw',
            # Layer 3: Headroom & Congestion
            'headroom_continent_mw', 'headroom_nordic_mw', 'headroom_great_belt_mw',
            'congestion_ratio_continent', 'congestion_ratio_nordic', 'is_severely_congested',
            # Layer 4: TSO Balancing Dynamics
            'afrr_net_activation_mw', 'afrr_gross_volume_mw', 'afrr_trend_1h', 'mfrr_balancing_spread',
            'spread_lag1', 'spread_lag2', 'spread_rolling_mean4', 'spread_rolling_std4',
            # Temporal Harmonics
            'hour_of_day', 'quarter_of_day', 'day_of_week', 'is_peak_hour', 'sin_hour', 'cos_hour'
        ]
        return features
