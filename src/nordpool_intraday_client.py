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
    SSO_URL = "https://sts.nordpoolgroup.com/connect/token"

    def __init__(
        self,
        username: str = "api@nurex.energy",
        password: str = "sDr2T6?5h_2xV4SU?f",
        client_id: str = "IDAPI_NUREX_COPPER_READ",
        cache_dir: str = "data/cache_nordpool"
    ):
        self.username = os.getenv("NORDPOOL_USERNAME", username)
        self.password = os.getenv("NORDPOOL_PASSWORD", password)
        self.client_id = os.getenv("NORDPOOL_CLIENT_ID", client_id)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.access_token = None
        self.token_expiry = None

    def authenticate(self) -> bool:
        """
        Attempts OAuth2 authentication against Nord Pool STS.
        """
        payloads = [
            {
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
                "client_id": self.client_id,
                "scope": "marketdata_api"
            },
            {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.password,
                "scope": "marketdata_api"
            }
        ]

        for p in payloads:
            try:
                res = requests.post(self.SSO_URL, data=p, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    self.access_token = data.get("access_token")
                    expires_in = data.get("expires_in", 3600)
                    self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
                    logger.info("Successfully authenticated with Nord Pool STS.")
                    return True
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

        # Attempt API fetch
        headers = {}
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
