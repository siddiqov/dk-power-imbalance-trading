import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from src.nordpool_intraday_client import NordPoolIntradayClient

logger = logging.getLogger(__name__)

def compute_order_flow_microstructure(df_day, area='DK1'):
    if df_day.empty:
        return df_day

    df = df_day.copy()
    client = NordPoolIntradayClient()
    
    # We will fetch REAL historical contract statistics instead of mathematical generation
    date_str = None
    if 'time_dk' in df.columns and len(df) > 0:
        date_str = pd.to_datetime(df['time_dk'].iloc[0]).strftime('%Y-%m-%d')
    elif 'time_utc' in df.columns and len(df) > 0:
        date_str = pd.to_datetime(df['time_utc'].iloc[0]).strftime('%Y-%m-%d')
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
        
    df_contracts = client.get_contract_statistics(area, date_str)
    
    if df_contracts.empty:
        # No mock data allowed! Fill with zeros if API fails
        df['xbid_bid_vol_mw'] = 0.0
        df['xbid_ask_vol_mw'] = 0.0
        df['xbid_order_flow_skew'] = 0.0
        df['xbid_micro_price_eur'] = df.get('spot_price_eur', 0.0)
        df['xbid_vwap_eur'] = df.get('spot_price_eur', 0.0)
        return df
        
    # Merge real data on time
    if 'time_utc' in df.columns:
        merge_col = 'time_utc'
        df_contracts['time_utc'] = pd.to_datetime(df_contracts['time_utc'])
    elif 'time_dk' in df.columns:
        merge_col = 'time_dk'
        # Assuming delivery_start is UTC, we can convert to DK time
        df_contracts['time_dk'] = pd.to_datetime(df_contracts['delivery_start']).dt.tz_convert('Europe/Copenhagen').dt.tz_localize(None)
    else:
        # Fallback if time index is unknown
        df['xbid_bid_vol_mw'] = 0.0
        df['xbid_ask_vol_mw'] = 0.0
        df['xbid_order_flow_skew'] = 0.0
        df['xbid_micro_price_eur'] = df.get('spot_price_eur', 0.0)
        df['xbid_vwap_eur'] = df.get('spot_price_eur', 0.0)
        return df
        
    df = pd.merge(df, df_contracts[['time_utc' if merge_col == 'time_utc' else 'time_dk', 'buy_volume_mw', 'sell_volume_mw', 'vwap_eur']], on=merge_col, how='left')
    
    # Fill missing with 0 for volumes, and spot for vwap
    df['xbid_bid_vol_mw'] = df['buy_volume_mw'].fillna(0.0)
    df['xbid_ask_vol_mw'] = df['sell_volume_mw'].fillna(0.0)
    
    b_vol = df['xbid_bid_vol_mw']
    a_vol = df['xbid_ask_vol_mw']
    
    # Real mathematical skew from actual volumes, no synthetic generation
    skew = np.where((b_vol + a_vol) > 0, (b_vol - a_vol) / (b_vol + a_vol), 0.0)
    df['xbid_order_flow_skew'] = skew
    
    spot = df.get('spot_price_eur', 0.0)
    df['xbid_vwap_eur'] = df['vwap_eur'].fillna(spot)
    df['xbid_micro_price_eur'] = df['xbid_vwap_eur'] # Fallback to VWAP as micro-price if level-2 depth isn't available
    
    # Drop intermediate merge columns
    df = df.drop(columns=['buy_volume_mw', 'sell_volume_mw', 'vwap_eur'])
    
    return df

class OrderFlowEngineV41:
    def __init__(self, price_area='DK1'):
        self.price_area = price_area.upper()
        self.client = NordPoolIntradayClient()

    @staticmethod
    def compute(df, area='DK1'):
        return compute_order_flow_microstructure(df, area)

    def generate_live_order_flow_snapshot(self, reference_spot=85.0):
        # Using LIVE telemetry instead of synthetic ladders
        snapshot = self.client.get_live_order_book(self.price_area)
        snapshot['price_area'] = self.price_area
        return snapshot

if __name__ == "__main__":
    print("Testing Live XBID Order Flow Engine (V4.1)...")
    engine = OrderFlowEngineV41()
    snap = engine.generate_live_order_flow_snapshot()
    print("Live Skew:", snap.get('order_flow_skew'))
