# ==============================================================================
# src/supabase_publisher.py
# Publishes Multi-Model predictions & trade audit ledger to Supabase for the Bidder Portal.
# Upserts all 96 quarters per price area; reconciles live settlements every 15 minutes.
# Supports both official supabase-py and zero-dependency direct PostgREST REST API.
# ==============================================================================

import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
import zoneinfo
import urllib3
import requests
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logger = logging.getLogger(__name__)

ALL_MODELS = [
    "StormHunter",
    "StormFusion",
    "NeuralFlash AI",
    "V4.1-HighAlpha",
    "V3.2-FlowAware",
    "Transformer-TFT",
    "Stacking-MetaEnsemble",
    "Hierarchical-LGBM+XGB",
    "Transfer-LightGBM",
    "Pure15m-CatBoost",
    "Deep-BiLSTM",
]


def _clean_numeric(val) -> Optional[float]:
    """Safely converts currency/status strings (e.g. '+€ 107.25', '-€ 91.17', '--') to numeric float."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if not pd.isna(val) else None
    s = str(val).replace("€", "").replace("EUR", "").replace(",", "").replace("+", "").strip()
    if s in ["--", "None", "nan", "null", ""]:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


clean_numeric = _clean_numeric


class SupabasePublisher:
    """
    Pushes multi-model predictions and trade audit records
    to Supabase `quarter_predictions` table via upsert.
    """

    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            logger.info("[SupabasePublisher] SUPABASE_URL / SUPABASE_SERVICE_KEY not configured. Running in Local Ledger mode (local SQLite & TSV dispatches updated).")
            self.client = None
            self.table = None
            self.log_table = None
            self.logs_table = None
            self.use_rest = False
            return

        self.url = url.rstrip("/")
        self.key = key
        self.use_rest = False

        # Attempt to import supabase client; fallback gracefully to direct REST API via requests
        try:
            from supabase import create_client
            self.client = create_client(url, key)
            self.table = self.client.table("quarter_predictions")
            self.log_table = self.client.table("prediction_updates_log")
            self.logs_table = self.log_table
        except (ImportError, Exception):
            logger.info("[SupabasePublisher] 'supabase' library not present or failed to load. Using direct PostgREST REST API.")
            self.client = "rest"
            self.use_rest = True
            self.table = None
            self.log_table = None
            self.logs_table = None

    def upsert_predictions(
        self,
        trades: list,
        price_area: str,
        delivery_date: str,
        model_name: str,
        market_mode: str = "INTRADAY_D0",
    ) -> int:
        """
        Upserts a 96-quarter commercial trading schedule to Supabase.
        
        Args:
            trades: List of trade dicts from V3CommercialStrategyEngine or DayAheadAuctionEngine
            price_area: 'DK1' or 'DK2'
            delivery_date: 'YYYY-MM-DD'
            model_name: Model name from ALL_MODELS
            market_mode: 'INTRADAY_D0' or 'DAY_AHEAD_D1'
        """
        if not trades:
            logger.warning(f"No trades to publish for {price_area} {delivery_date} [{market_mode}] model={model_name}")
            return 0

        if not self.client:
            logger.info(
                f"  [Local Ledger] Processed {len(trades)} quarters for {price_area} "
                f"({delivery_date}) [{market_mode}] model={model_name}"
            )
            return len(trades)

        cph_tz = zoneinfo.ZoneInfo("Europe/Copenhagen")
        d_parts = [int(p) for p in delivery_date.split("-")]

        rows = []
        for trade in trades:
            q_raw = trade.get("quarter", "Q1")
            q_str = q_raw.split()[0].replace("Q", "").strip()
            try:
                q_idx = int(q_str)
            except ValueError:
                q_idx = 1
            hour = (q_idx - 1) // 4
            minute = ((q_idx - 1) % 4) * 15
            dt = datetime(d_parts[0], d_parts[1], d_parts[2], hour, minute, tzinfo=cph_tz)
            time_dk_iso = dt.isoformat()

            p_up_val = _clean_numeric(trade.get("p_up"))
            if p_up_val is not None and p_up_val <= 1.0:
                p_up_val = round(p_up_val * 100.0, 1)

            p_down_val = _clean_numeric(trade.get("p_down"))
            if p_down_val is not None and p_down_val <= 1.0:
                p_down_val = round(p_down_val * 100.0, 1)

            p_up_spike_val = _clean_numeric(trade.get("p_up_spike"))
            if p_up_spike_val is not None and p_up_spike_val <= 1.0:
                p_up_spike_val = round(p_up_spike_val * 100.0, 1)

            row = {
                "market_mode": market_mode,
                "price_area": price_area,
                "delivery_date": delivery_date,
                "quarter_index": q_idx,
                "quarter_label": f"Q{q_idx}",
                "time_dk": time_dk_iso,
                "spot_price_eur": _clean_numeric(trade.get("spot_price_eur")),
                "pred_imbalance_eur": _clean_numeric(trade.get("pred_imbalance_eur")),
                "pred_spread_eur": _clean_numeric(trade.get("pred_spread_eur")),
                "p_up": p_up_val,
                "p_down": p_down_val,
                "p_up_spike": p_up_spike_val,
                "decision": trade.get("action"),
                "direction": trade.get("direction"),
                "volume_mwh": _clean_numeric(trade.get("volume_mwh")),
                "q10_price_eur": _clean_numeric(trade.get("q10_price_eur")),
                "q50_price_eur": _clean_numeric(trade.get("q50_price_eur")),
                "q90_price_eur": _clean_numeric(trade.get("q90_price_eur")),
                "model_name": model_name,
                "status": trade.get("status", "LOCKED_PENDING"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            if trade.get("actual_settled_eur") is not None:
                val = _clean_numeric(trade.get("actual_settled_eur"))
                if val is not None:
                    row["actual_settled_eur"] = val
            if trade.get("net_pnl_eur") is not None:
                val = _clean_numeric(trade.get("net_pnl_eur"))
                if val is not None:
                    row["net_pnl_eur"] = val
            if trade.get("fees_eur") is not None:
                val = _clean_numeric(trade.get("fees_eur"))
                if val is not None:
                    row["fees_eur"] = val
            if trade.get("gross_pnl_eur") is not None:
                val = _clean_numeric(trade.get("gross_pnl_eur"))
                if val is not None:
                    row["gross_pnl_eur"] = val

            rows.append(row)

        if not rows:
            logger.warning(f"No trades to upsert for {price_area} {delivery_date}")
            return 0

        batch_size = 50
        total_upserted = 0

        # Direct REST API fallback
        if self.use_rest:
            headers = {
                "apikey": self.key,
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates",
            }
            endpoint = f"{self.url}/rest/v1/quarter_predictions?on_conflict=market_mode,price_area,delivery_date,quarter_index,model_name"
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                try:
                    res = requests.post(endpoint, headers=headers, json=batch, timeout=20)
                    if res.status_code in (200, 201):
                        total_upserted += len(batch)
                    else:
                        logger.warning(f"  [Supabase REST] Upsert status {res.status_code}: {res.text[:200]}")
                except Exception as e:
                    logger.warning(f"  [Supabase REST] Error: {e}")

            self._log_update(price_area, delivery_date, total_upserted, len(trades))
            logger.info(
                f"  [Supabase REST] Upserted {total_upserted} quarters for {price_area} "
                f"({delivery_date}) [{market_mode}] model={model_name}"
            )
            return total_upserted

        # Official supabase-py client
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            try:
                self.table.upsert(
                    batch,
                    on_conflict="market_mode,price_area,delivery_date,quarter_index,model_name",
                ).execute()
            except Exception as e:
                err_str = str(e)
                if "23505" in err_str or "quarter_predictions_upsert_key" in err_str or "market_mode" in err_str or "duplicate key" in err_str:
                    logger.warning(
                        f"  [Supabase] 4-column unique constraint detected in DB. "
                        f"Preserving primary Intraday records for model={model_name}."
                    )
                    continue
                else:
                    raise e
            total_upserted += len(batch)

        self._log_update(price_area, delivery_date, total_upserted, len(trades))

        logger.info(
            f"  [Supabase] Upserted {total_upserted} quarters for {price_area} "
            f"({delivery_date}) [{market_mode}] model={model_name}"
        )
        return total_upserted

    def reconcile_settled_quarters(self, price_area: str, delivery_date: str) -> int:
        """
        Fetches authentic settled imbalance prices from Energi Data Service (ImbalancePrice)
        and reconciles all past quarters for all models in Supabase.
        """
        import json
        url_imb = "https://api.energidataservice.dk/dataset/ImbalancePrice"
        params = {
            "filter": json.dumps({"PriceArea": price_area}),
            "start": f"{delivery_date}T00:00",
            "end": f"{delivery_date}T23:59",
            "sort": "TimeDK ASC",
            "limit": 200,
        }

        settled_dict = {}
        try:
            res = requests.get(url_imb, params=params, verify=False, timeout=10).json()
            for r in res.get("records", []):
                if r.get("PriceArea") == price_area and pd.notnull(r.get("ImbalancePriceEUR")):
                    t_str = pd.to_datetime(r["TimeDK"]).strftime("%Y-%m-%d %H:%M")
                    settled_dict[t_str] = float(r["ImbalancePriceEUR"])
        except Exception as e:
            logger.warning(f"  [Reconciliation] Failed to query ImbalancePrice: {e}")
            return 0

        if not settled_dict or not self.client:
            return 0

        # Query pending records from Supabase
        pending_rows = []
        try:
            if self.use_rest:
                headers = {"apikey": self.key, "Authorization": f"Bearer {self.key}"}
                r = requests.get(
                    f"{self.url}/rest/v1/quarter_predictions?price_area=eq.{price_area}&delivery_date=eq.{delivery_date}&status=neq.SETTLED_AUDITED",
                    headers=headers,
                    timeout=15,
                )
                if r.status_code == 200:
                    pending_rows = r.json()
            else:
                query = (
                    self.table.select("*")
                    .eq("price_area", price_area)
                    .eq("delivery_date", delivery_date)
                    .neq("status", "SETTLED_AUDITED")
                )
                result = query.execute()
                pending_rows = result.data or []
        except Exception as e:
            logger.warning(f"  [Reconciliation] Failed to fetch pending rows: {e}")
            return 0

        if not pending_rows:
            return 0

        updated_count = 0
        fee_per_mwh = 0.51  # €0.06 Nord Pool + €0.20 TSO + €0.25 Slippage
        tax_rate = 0.22     # Danish 22% corporate tax

        updates_batch = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for row in pending_rows:
            q_idx = row["quarter_index"]
            hour = (q_idx - 1) // 4
            minute = ((q_idx - 1) % 4) * 15
            t_dk_str = f"{delivery_date} {hour:02d}:{minute:02d}"

            if t_dk_str in settled_dict:
                actual_price = settled_dict[t_dk_str]
                spot_price = float(row.get("spot_price_eur") or 0.0)
                volume = float(row.get("volume_mwh") or 0.0)
                action = str(row.get("decision") or "HOLD")

                direction = 1 if "BUY" in action else (-1 if "SELL" in action else 0)

                da_cash = -(direction * volume * spot_price)
                settle_cash = direction * volume * actual_price
                spread_cap = direction * (actual_price - spot_price)
                gross_pnl = da_cash + settle_cash
                fees = volume * fee_per_mwh if direction != 0 else 0.0
                tax = (gross_pnl - fees) * tax_rate if (gross_pnl - fees) > 0 else 0.0
                net_pnl = gross_pnl - fees - tax

                row_update = dict(row)
                row_update.pop("id", None)
                row_update["actual_settled_eur"] = round(actual_price, 2)
                row_update["spread_captured_eur"] = round(spread_cap, 2)
                row_update["da_cash_flow_eur"] = round(da_cash, 2)
                row_update["settle_cash_flow_eur"] = round(settle_cash, 2)
                row_update["gross_pnl_eur"] = round(gross_pnl, 2)
                row_update["fees_eur"] = round(fees, 2)
                row_update["tax_eur"] = round(tax, 2)
                row_update["net_pnl_eur"] = round(net_pnl, 2)
                row_update["status"] = "SETTLED_AUDITED"
                row_update["settled_at"] = now_iso
                row_update["updated_at"] = now_iso
                updates_batch.append(row_update)

        if updates_batch:
            on_conflict_cols = "market_mode,price_area,delivery_date,quarter_index,model_name"
            if self.use_rest:
                headers = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                    "Prefer": "resolution=merge-duplicates",
                }
                endpoint = f"{self.url}/rest/v1/quarter_predictions?on_conflict={on_conflict_cols}"
                for i in range(0, len(updates_batch), 50):
                    chunk = updates_batch[i : i + 50]
                    try:
                        res = requests.post(endpoint, headers=headers, json=chunk, timeout=20)
                        if res.status_code in (200, 201):
                            updated_count += len(chunk)
                    except Exception as e:
                        logger.warning(f"  [Reconciliation REST] Error: {e}")
            else:
                for i in range(0, len(updates_batch), 50):
                    chunk = updates_batch[i : i + 50]
                    try:
                        self.table.upsert(chunk, on_conflict=on_conflict_cols).execute()
                        updated_count += len(chunk)
                    except Exception as e:
                        logger.warning(f"  [Reconciliation] Upsert error: {e}")

            logger.info(
                f"  [Reconciliation] ✅ Audited and settled {updated_count} quarters for {price_area} ({delivery_date})"
            )

        return updated_count

    def _log_update(
        self,
        price_area: str,
        delivery_date: str,
        quarters_updated: int,
        total_quarters: int,
    ):
        """Append an entry to the prediction_updates_log table."""
        if self.use_rest:
            try:
                headers = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                }
                requests.post(
                    f"{self.url}/rest/v1/prediction_updates_log",
                    headers=headers,
                    json={
                        "price_area": price_area,
                        "delivery_date": delivery_date,
                        "quarters_updated": quarters_updated,
                        "total_quarters": total_quarters,
                    },
                    timeout=10,
                )
            except Exception as e:
                logger.warning(f"Failed to write update log via REST: {e}")
            return

        try:
            self.log_table.insert(
                {
                    "price_area": price_area,
                    "delivery_date": delivery_date,
                    "quarters_updated": quarters_updated,
                    "total_quarters": total_quarters,
                }
            ).execute()
        except Exception as e:
            logger.warning(f"Failed to write update log: {e}")

    def get_latest_update_time(self, price_area: str) -> Optional[str]:
        """Get the timestamp of the most recent prediction push for a zone."""
        if self.use_rest:
            try:
                headers = {"apikey": self.key, "Authorization": f"Bearer {self.key}"}
                r = requests.get(
                    f"{self.url}/rest/v1/prediction_updates_log?price_area=eq.{price_area}&order=triggered_at.desc&limit=1",
                    headers=headers,
                    timeout=10,
                )
                if r.status_code == 200 and r.json():
                    return r.json()[0]["triggered_at"]
            except Exception as e:
                logger.warning(f"Failed to query update log via REST: {e}")
            return None

        try:
            result = (
                self.log_table.select("triggered_at")
                .eq("price_area", price_area)
                .order("triggered_at", desc=True)
                .limit(1)
                .execute()
            )
            if result.data:
                return result.data[0]["triggered_at"]
        except Exception as e:
            logger.warning(f"Failed to query update log: {e}")
        return None
