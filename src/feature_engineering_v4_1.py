# ==============================================================================
# src/feature_engineering_v4_1.py
# Nurex V4.1 Advanced High-Alpha Feature Matrix & Microstructure Grid Engine
# 
# Layer 1: Physical System State (Load, Wind, Solar, 8-Cable Physical Net Flows)
# Layer 2: Expectation Deviations (True TSO Wind/Solar Error, Intraday Revisions)
# Layer 3: Market Reaction & Interconnector Congestion (ATC - Flow Headroom)
# Layer 4: Real-Time European Balancing & Microstructure:
#   - Danish MARI (mFRR) & PICASSO (aFRR) Activated Power & Marginal Prices
#   - German Bundesnetzagentur (SMARD.de) Grid System Balance & Spillover
#   - Continuous Intraday (XBID) Level-2 Order Flow Skewness & Micro-Price
#   - Velocity (d/dt), Acceleration (d^2/dt^2) & Multi-Horizon Momentum Signals
# ==============================================================================

import os
import sys
sys.path.append(os.path.abspath('.'))
import numpy as np
import pandas as pd

from src.feature_engineering_v4 import V4GridFeatureEngine, CABLE_CAPACITIES
from src.balancing_market_v4_1 import get_latest_balancing_state
from src.smard_client import get_german_system_balance_telemetry
from src.order_flow_v4_1 import compute_order_flow_microstructure

class V41GridFeatureEngine(V4GridFeatureEngine):
    """
    Constructs the V4.1 feature matrix integrating European balancing markets,
    German federal grid state, continuous XBID order flow microstructure, and
    multi-horizon velocity / acceleration momentum metrics.
    """
    def __init__(self, price_area='DK1', db_path="energy_data.db"):
        super().__init__(price_area=price_area, db_path=db_path)

    def build_feature_matrix(self, df_input=None):
        """
        Extends V4.0 feature matrix with Layer 4 MARI/PICASSO balancing,
        SMARD German balance, XBID order flow depth, and dynamic velocity terms.
        """
        df_base = super().build_feature_matrix(df_input)
        if df_base.empty:
            return df_base

        df = df_base.copy()

        # 1. Ingest European MARI / PICASSO Balancing Metrics
        try:
            bal_state = get_latest_balancing_state(self.price_area)
            df['mfrr_activated_down_mw'] = df.get('mfrr_down_mw', float(bal_state.get('mfrr_down_mw', 0.0)))
            df['mfrr_activated_up_mw'] = df.get('mfrr_up_mw', float(bal_state.get('mfrr_up_mw', 0.0)))
            df['mfrr_net_activation_mw'] = df['mfrr_activated_up_mw'] - df['mfrr_activated_down_mw']
            df['afrr_marginal_price_eur'] = float(bal_state.get('afrr_marginal_price_eur', 0.0))
            df['total_balancing_pressure_mw'] = float(bal_state.get('total_balancing_pressure_mw', 0.0))
        except Exception:
            df['mfrr_activated_down_mw'] = 0.0
            df['mfrr_activated_up_mw'] = 0.0
            df['mfrr_net_activation_mw'] = 0.0
            df['afrr_marginal_price_eur'] = 0.0
            df['total_balancing_pressure_mw'] = 0.0

        # 2. Ingest German Federal Grid (SMARD.de) System Balance
        try:
            smard_state = get_german_system_balance_telemetry()
            df['german_system_balance_mw'] = df.get('smard_residual_load_mw', float(smard_state.get('german_system_balance_mw', 0.0)))
            df['german_generation_mw'] = float(smard_state.get('german_generation_mw', 50000.0))
            df['german_load_mw'] = float(smard_state.get('german_load_mw', 50000.0))
        except Exception:
            df['german_system_balance_mw'] = 0.0
            df['german_generation_mw'] = 50000.0
            df['german_load_mw'] = 50000.0

        # 3. Ingest XBID Continuous Intraday Order Flow Microstructure
        try:
            df = compute_order_flow_microstructure(df)
        except Exception:
            df['xbid_order_flow_skew'] = 0.0
            df['xbid_micro_price_eur'] = df['spot_price_eur']
            df['xbid_vwap_eur'] = df['spot_price_eur']

        # Ensure order_flow_skew alias exists
        if 'order_flow_skew' not in df.columns:
            df['order_flow_skew'] = df.get('xbid_order_flow_skew', 0.0)

        # 4. Multi-Horizon Velocity (d/dt) & Acceleration (d^2/dt^2) Features
        # SMARD Residual Load Velocity (15-min and 1-hour change)
        df['smard_res_delta_15m'] = df['german_system_balance_mw'].diff(1).fillna(0.0)
        df['smard_res_delta_1h'] = df['german_system_balance_mw'].diff(4).fillna(0.0)

        # MARI / PICASSO Balancing Acceleration & Velocity
        df['mfrr_up_velocity'] = df['mfrr_activated_up_mw'].diff(1).fillna(0.0)
        df['mfrr_down_velocity'] = df['mfrr_activated_down_mw'].diff(1).fillna(0.0)
        df['mfrr_net_acceleration'] = df['mfrr_net_activation_mw'].diff(1).diff(1).fillna(0.0)

        # Order Flow Momentum & Exponential Moving Average
        df['order_flow_skew_ema4'] = df['order_flow_skew'].ewm(span=4, min_periods=1).mean().fillna(0.0)
        df['order_flow_momentum'] = df['order_flow_skew'].diff(2).fillna(0.0)

        # 5. Composite Cross-Border Price & Headroom Gradients
        df['balancing_spread_delta_eur'] = df['afrr_marginal_price_eur'] - df['spot_price_eur']
        headroom_tot = df.get('headroom_continent_mw', 500.0) + 1.0
        df['german_spillover_pressure_ratio'] = np.clip(df['german_system_balance_mw'] / headroom_tot, -5.0, 5.0)

        # DK-DE Spot Price Gradient
        de_spot = df.get('de_spot_eur', df['spot_price_eur'])
        df['dk_de_price_spread'] = df['spot_price_eur'] - de_spot
        df['dk_de_spread_velocity'] = df['dk_de_price_spread'].diff(1).fillna(0.0)

        # System Imbalance Momentum
        df['system_surplus_velocity'] = df.get('net_system_surplus_mw', pd.Series(0.0, index=df.index)).diff(1).fillna(0.0)

        return df

    def build_feature_matrix_v4_1(self, df_input=None):
        """Convenience alias matching V4.1 naming."""
        return self.build_feature_matrix(df_input)

# Alias for backwards compatibility
V41FeatureEngine = V41GridFeatureEngine

if __name__ == '__main__':
    print("Testing V4.1 Feature Matrix Builder with Velocity Features...")
    fe41 = V41GridFeatureEngine('DK1')
    try:
        sample_trades = pd.DataFrame({
            'time_dk': pd.date_range('2026-09-15 00:00', periods=8, freq='15min'),
            'spot_price_eur': [160.0, 165.0, 170.0, 175.0, 180.0, 175.0, 170.0, 165.0],
            'actual_settled_eur': [162.0, 160.0, 172.0, 180.0, 185.0, 170.0, 165.0, 160.0],
            'pred_spread_eur': [2.0, -5.0, 2.0, 5.0, 5.0, -5.0, -5.0, -5.0],
            'action': ['BUY', 'SELL', 'BUY', 'BUY', 'BUY', 'SELL', 'SELL', 'SELL'],
            'volume_mwh': [10.0, 25.0, 10.0, 10.0, 10.0, 25.0, 25.0, 25.0]
        })
        df_feat = fe41.build_feature_matrix(sample_trades)
        print("V4.1 Feature Matrix columns:", len(df_feat.columns))
        print("Velocity & Acceleration terms generated successfully:")
        print(df_feat[['smard_res_delta_15m', 'mfrr_up_velocity', 'order_flow_skew_ema4', 'dk_de_price_spread']].head(4))
    except Exception as e:
        print(f"Error: {e}")
