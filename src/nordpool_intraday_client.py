import os
from dotenv import load_dotenv
load_dotenv()
# ==============================================================================
# src/nordpool_intraday_client.py
# Nord Pool Continuous Intraday (SIDC/XBID) Market Data & Execution Client
# Reads live 15-minute contract VWAP, volume, and order statistics for DK1 & DK2
# ==============================================================================

import os
import sys
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger("NordPoolIntraday")

class NordPoolIntradayClient:
    """
    Client for Nord Pool Intraday Market Data (API v2).
    Accesses 15-minute product VWAP, volume, and order flow metrics.
    """

    BASE_URL = "https://data-api.nordpoolgroup.com"
    # 2026-11-02: the old sts.nordpoolgroup.com login is rejected (401). Use the market-data login that the
    # models and the history downloader already use: NORDPOOL_MD_* from Nurex_V4_2/.env + sso.nordpoolgroup.com.
    SSO_URL = "https://sso.nordpoolgroup.com/connect/token"

    def __init__(self, cache_dir: str = "data/cache_nordpool", **_ignored):
        self.username, self.password = self._md_credentials()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.access_token = None
        self.token_expiry = None

    @staticmethod
    def _md_credentials():
        vals = {}
        try:
            from dotenv import dotenv_values
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            vals = dotenv_values(os.path.join(root, "Nurex_V4_2", ".env"))
        except Exception:
            pass
        u = (vals.get("NORDPOOL_MD_USERNAME") or os.environ.get("NORDPOOL_MD_USERNAME") or "").strip()
        p = (vals.get("NORDPOOL_MD_PASSWORD") or os.environ.get("NORDPOOL_MD_PASSWORD") or "").strip()
        return u, p

    def authenticate(self) -> bool:
        """OAuth2 password login against Nord Pool SSO (market-data client)."""
        if not self.username or not self.password:
            logger.warning("NORDPOOL_MD_USERNAME / NORDPOOL_MD_PASSWORD not found")
            return False
        import base64
        basic = base64.b64encode(b"client_marketdata_api:client_marketdata_api").decode()
        try:
            res = requests.post(self.SSO_URL, data={"grant_type": "password", "scope": "marketdata_api",
                                                    "username": self.username, "password": self.password},
                                headers={"Authorization": f"Basic {basic}",
                                         "Content-Type": "application/x-www-form-urlencoded"}, timeout=15)
            if res.status_code == 200:
                data = res.json()
                self.access_token = data.get("access_token")
                self.token_expiry = datetime.now() + timedelta(seconds=int(data.get("expires_in", 3600)) - 60)
                logger.info("Authenticated with Nord Pool SSO (market data).")
                return True
            logger.warning(f"Nord Pool login returned HTTP {res.status_code}")
            if res.status_code in (400, 401, 403):
                from src import api_breaker as _brk
                _brk.trip("nordpool_data_api", f"login HTTP {res.status_code}")
        except Exception as e:
            logger.debug(f"Auth attempt failed: {e}")
        return False

    def get_contract_statistics(self, area: str = "DK1", date_str: str = None) -> pd.DataFrame:
        """
        Fetches 15-minute contract statistics for the given area and delivery date.
        Returns DataFrame containing deliveryStart, averagePrice (VWAP), volume, highPrice, lowPrice.
        """
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        cache_file = os.path.join(self.cache_dir, f"contracts_{area}_{date_str}.json")

        # Check local cache first (valid for 10 minutes if today, permanent if past date)
        if os.path.exists(cache_file):
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
                is_today = (date_str == datetime.now().strftime("%Y-%m-%d"))
                if not is_today or (datetime.now() - mtime).total_seconds() < 600:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return self._parse_contracts_json(data, area)
            except Exception as e:
                logger.warning(f"Failed reading contract cache: {e}")

        # 2026-11-02: skip the API for a few hours after it rejected our login (dashboard speed)
        from src import api_breaker as _brk
        if _brk.blocked("nordpool_data_api"):
            return self._get_fallback_intraday_data(area, date_str)

        # Attempt API fetch
        headers = {"Accept": "application/json"}
        if self.access_token and (not self.token_expiry or datetime.now() < self.token_expiry):
            headers["Authorization"] = f"Bearer {self.access_token}"
        else:
            if self.authenticate():
                headers["Authorization"] = f"Bearer {self.access_token}"

        url = f"{self.BASE_URL}/api/v2/Intraday/ContractStatistics/ByAreas"
        params = {"areas": area, "date": date_str}

        try:
            res = requests.get(url, params=params, headers=headers, timeout=12)
            if res.status_code == 200:
                data = res.json()
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                return self._parse_contracts_json(data, area)
            else:
                logger.warning(f"Nord Pool API returned status {res.status_code}: {res.text[:120]}")
                if res.status_code in (401, 403):
                    _brk.trip("nordpool_data_api", f"HTTP {res.status_code}")
        except Exception as e:
            logger.warning(f"Failed fetching Nord Pool intraday contracts: {e}")

        # Fallback: return empty DataFrame with expected schema
        return self._get_fallback_intraday_data(area, date_str)

    def _parse_contracts_json(self, data: list, area: str) -> pd.DataFrame:
        rows = []
        for item in data:
            if item.get("area") == area or not item.get("area"):
                for c in item.get("contracts", []):
                    rows.append({
                        "delivery_start": c.get("deliveryStart"),
                        "delivery_end": c.get("deliveryEnd"),
                        "contract_id": c.get("contractId"),
                        "vwap_eur": c.get("averagePrice"),
                        "volume_mw": c.get("volume"),
                        "buy_volume_mw": c.get("buyVolume"),
                        "sell_volume_mw": c.get("sellVolume"),
                        "high_eur": c.get("highPrice"),
                        "low_eur": c.get("lowPrice"),
                        "last_eur": c.get("closePrice")
                    })
        df = pd.DataFrame(rows)
        if not df.empty:
            df['time_utc'] = pd.to_datetime(df['delivery_start']).dt.tz_localize(None)
        return df

    def _get_fallback_intraday_data(self, area: str, date_str: str) -> pd.DataFrame:
        cols = ["delivery_start", "delivery_end", "contract_id", "vwap_eur", "volume_mw", 
                "buy_volume_mw", "sell_volume_mw", "high_eur", "low_eur", "last_eur", "time_utc"]
        return pd.DataFrame(columns=cols)

    def get_live_order_book(self, area: str = "DK1") -> dict:
        """
        Fetches the live Level-2 order book for the nearest active contract.
        """
        df_contracts = self.get_contract_statistics(area)
        if df_contracts.empty:
            return self._get_fallback_order_book()
        from src import api_breaker as _brk
        if _brk.blocked("nordpool_data_api"):
            return self._get_fallback_order_book(float(df_contracts.iloc[-1]['vwap_eur']))
        
        now_utc = datetime.utcnow()
        future_contracts = df_contracts[df_contracts['time_utc'] >= now_utc].sort_values('time_utc')
        
        if future_contracts.empty:
            contract_id = df_contracts.iloc[-1]['contract_id']
            mid_price = df_contracts.iloc[-1]['vwap_eur']
        else:
            contract_id = future_contracts.iloc[0]['contract_id']
            mid_price = future_contracts.iloc[0]['vwap_eur']
            
        headers = {"Accept": "application/json"}
        if self.access_token and (not self.token_expiry or datetime.now() < self.token_expiry):
            headers["Authorization"] = f"Bearer {self.access_token}"
        else:
            if self.authenticate():
                headers["Authorization"] = f"Bearer {self.access_token}"
                
        url = f"{self.BASE_URL}/api/v2/Intraday/OrderBook/ByContractId"
        params = {"contractId": contract_id}
        
        try:
            res = requests.get(url, params=params, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                return self._parse_order_book_json(data, mid_price)
            logger.warning(f"Nord Pool order book returned HTTP {res.status_code}")
        except Exception as e:
            logger.warning(f"Failed fetching live order book: {e}")
            
        return self._get_fallback_order_book(mid_price)
        
    def _parse_order_book_json(self, data: dict, reference_price: float) -> dict:
        bids_raw = data.get('bids', [])
        asks_raw = data.get('asks', [])
        
        bids = []
        for i, b in enumerate(sorted(bids_raw, key=lambda x: x.get('price', 0), reverse=True)[:5]):
            bids.append({
                "level": i + 1,
                "orders": b.get('orders', 1),
                "volume_mw": b.get('volume', 0.0),
                "price_eur": b.get('price', 0.0)
            })
            
        asks = []
        for i, a in enumerate(sorted(asks_raw, key=lambda x: x.get('price', 0))[:5]):
            asks.append({
                "level": i + 1,
                "orders": a.get('orders', 1),
                "volume_mw": a.get('volume', 0.0),
                "price_eur": a.get('price', 0.0)
            })
            
        return self._compute_order_book_metrics(bids, asks, reference_price)
        
    def _get_fallback_order_book(self, reference_price: float = 85.0) -> dict:
        return self._compute_order_book_metrics([], [], reference_price)
        
    def _compute_order_book_metrics(self, bids: list, asks: list, reference_price: float) -> dict:
        total_bid_vol = sum(b['volume_mw'] for b in bids)
        total_ask_vol = sum(a['volume_mw'] for a in asks)
        
        if total_bid_vol + total_ask_vol == 0:
            skew = 0.0
            micro_price = reference_price
        else:
            skew = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
            if len(bids) > 0 and len(asks) > 0:
                best_bid = bids[0]['price_eur']
                best_ask = asks[0]['price_eur']
                b_vol1 = bids[0]['volume_mw']
                a_vol1 = asks[0]['volume_mw']
                if b_vol1 + a_vol1 > 0:
                    micro_price = (best_bid * a_vol1 + best_ask * b_vol1) / (b_vol1 + a_vol1)
                else:
                    micro_price = reference_price
            else:
                micro_price = reference_price
                
        micro_dev = micro_price - reference_price
        
        liquidity_status = (
            "Strong Buy Liquidity (Upward Tilt)" if skew > 0.10
            else ("Strong Sell Liquidity (Downward Drag)" if skew < -0.10
            else "Balanced Order Book")
        )
        if total_bid_vol + total_ask_vol == 0:
            liquidity_status = "No Market Data (API Disconnected)"
            
        return {
            "mid_price_eur": reference_price,
            "micro_price_eur": round(micro_price, 2),
            "micro_price_dev_eur": round(micro_dev, 2),
            "total_bid_volume_mwh": round(total_bid_vol, 1),
            "total_ask_volume_mwh": round(total_ask_vol, 1),
            "order_flow_skew": round(skew, 2),
            "liquidity_pressure": liquidity_status,
            "bids": bids,
            "asks": asks
        }

