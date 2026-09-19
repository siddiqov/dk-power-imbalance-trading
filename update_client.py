import re
with open('src/nordpool_intraday_client.py', 'r') as f:
    content = f.read()

new_method = '''
    def get_live_order_book(self, area: str = "DK1") -> dict:
        \"\"\"
        Fetches the live Level-2 order book for the nearest active contract.
        \"\"\"
        df_contracts = self.get_contract_statistics(area)
        if df_contracts.empty:
            return self._get_fallback_order_book()
        
        now_utc = datetime.utcnow()
        future_contracts = df_contracts[df_contracts['time_utc'] >= now_utc].sort_values('time_utc')
        
        if future_contracts.empty:
            contract_id = df_contracts.iloc[-1]['contract_id']
            mid_price = df_contracts.iloc[-1]['vwap_eur']
        else:
            contract_id = future_contracts.iloc[0]['contract_id']
            mid_price = future_contracts.iloc[0]['vwap_eur']
            
        headers = {}
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
'''

content = content.replace('        return pd.DataFrame(columns=cols)', '        return pd.DataFrame(columns=cols)\n' + new_method)
with open('src/nordpool_intraday_client.py', 'w') as f:
    f.write(content)
