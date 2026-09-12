# ==============================================================================
# src/supabase_publisher.py
# Publishes Multi-Model predictions & trade audit ledger to Supabase for the Bidder Portal.
# Upserts all 96 quarters per price area; reconciles live settlements every 15 minutes.
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
    "Transformer-TFT",
    "Stacking-MetaEnsemble",
    "Hierarchical-LGBM+XGB",
    "Transfer-LightGBM",
    "Pure15m-CatBoost",
    "Deep-BiLSTM",
]


class SupabasePublisher:
    """
    Pushes multi-model predictions and trade audit records
    to Supabase `quarter_predictions` table via upsert.
    """

    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise ValueError(
                "Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in environment. "
                "Copy .env.example to .env and fill in your Supabase credentials."
            )
        # Lazy import to avoid requiring supabase for non-publisher usage
        from supabase import create_client
        self.client = create_client(url, key)
        self.table = self.client.table("quarter_predictions")
        self.log_table = self.client.table("prediction_updates_log")

    def upsert_predictions(
        self,
        trades: List[Dict],
        price_area: str,
        delivery_date: str,
        model_name: str = "Transformer-TFT",
        market_mode: str = "INTRADAY_D0",
    ) -> int:
        """
        Upsert a list of trade dicts (from V3CommercialStrategyEngine.evaluate_trading_ledger)
        into Supabase. Returns the number of rows upserted.

        Args:
            trades: List of trade dicts from summary["trades"]
            price_area: 'DK1' or 'DK2'
            delivery_date: 'YYYY-MM-DD' string
            model_name: Model identifier, default 'Transformer-TFT'
            market_mode: 'INTRADAY_D0' or 'DAY_AHEAD_D1'
        """
        cph_tz = zoneinfo.ZoneInfo("Europe/Copenhagen")

        rows = []
        d_parts = [int(p) for p in delivery_date.split("-")]
        for trade in trades:
            q_raw = str(trade.get("quarter", "Q1")).split(" ")[0].replace("Q", "")
            try:
                q_idx = int(q_raw)
            except ValueError:
                q_idx = 1
            hour = (q_idx - 1) // 4
            minute = ((q_idx - 1) % 4) * 15
            dt = datetime(d_parts[0], d_parts[1], d_parts[2], hour, minute, tzinfo=cph_tz)
            time_dk_iso = dt.isoformat()

            row = {
                "market_mode": market_mode,
                "price_area": price_area,
                "delivery_date": delivery_date,
                "quarter_index": q_idx,
                "quarter_label": f"Q{q_idx}",
                "time_dk": time_dk_iso,
                "spot_price_eur": trade.get("spot_price_eur"),
                "pred_imbalance_eur": trade.get("pred_imbalance_eur"),
                "pred_spread_eur": trade.get("pred_spread_eur"),
                "p_up": trade.get("p_up"),
                "p_down": trade.get("p_down"),
                "p_up_spike": trade.get("p_up_spike"),
                "decision": trade.get("action"),
                "direction": trade.get("direction"),
                "volume_mwh": trade.get("volume_mwh"),
                "q10_price_eur": trade.get("q10_price_eur"),
                "q50_price_eur": trade.get("q50_price_eur"),
                "q90_price_eur": trade.get("q90_price_eur"),
                "model_name": model_name,
                "status": trade.get("status", "LOCKED_PENDING"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Optional settlement fields if trade was already audited
            if trade.get("actual_settled_eur") is not None:
                row["actual_settled_eur"] = trade.get("actual_settled_eur")
            if trade.get("net_pnl_eur") is not None:
                row["net_pnl_eur"] = trade.get("net_pnl_eur")
            if trade.get("fees_eur") is not None:
                row["fees_eur"] = trade.get("fees_eur")
            if trade.get("gross_pnl_eur") is not None:
                row["gross_pnl_eur"] = trade.get("gross_pnl_eur")

            rows.append(row)

        if not rows:
            logger.warning(f"No trades to upsert for {price_area} {delivery_date}")
            return 0

        # Upsert in batches (try with new composite constraint, fallback to legacy if not migrated yet)
        batch_size = 50
        total_upserted = 0

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            try:
                self.table.upsert(
                    batch,
                    on_conflict="market_mode,price_area,delivery_date,quarter_index,model_name",
                ).execute()
            except Exception as e:
                # If migration_v2 has not been applied yet, fallback to original 4-column constraint
                if "quarter_predictions_upsert_key" in str(e) or "market_mode" in str(e):
                    # Strip market_mode if column doesn't exist yet
                    for b in batch:
                        b.pop("market_mode", None)
                    self.table.upsert(
                        batch,
                        on_conflict="price_area,delivery_date,quarter_index,model_name",
                    ).execute()
                else:
                    raise e
            total_upserted += len(batch)

        # Log this update
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
        Calculates exact gross PnL, Nord Pool/TSO fees (€0.51/MWh), and net realized profit.
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

        if not settled_dict:
            logger.info(f"  [Reconciliation] No settled records yet for {price_area} on {delivery_date}")
            return 0

        # Query pending records from Supabase
        try:
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

        cph_tz = zoneinfo.ZoneInfo("Europe/Copenhagen")
        updated_count = 0
        fee_per_mwh = 0.51  # €0.06 Nord Pool + €0.20 TSO + €0.25 Slippage
        tax_rate = 0.22     # Danish 22% corporate tax

        updates_batch = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for row in pending_rows:
            q_idx = row["quarter_index"]
            hour = (q_idx - 1) // 4
            minute = ((q_idx - 1) % 4) * 15
            d_parts = [int(p) for p in delivery_date.split("-")]
            t_dk_str = f"{delivery_date} {hour:02d}:{minute:02d}"

            if t_dk_str in settled_dict:
                actual_price = settled_dict[t_dk_str]
                spot_price = float(row.get("spot_price_eur") or 0.0)
                volume = float(row.get("volume_mwh") or 0.0)
                action = str(row.get("decision") or "HOLD")

                direction = 0
                if "BUY" in action:
                    direction = 1
                elif "SELL" in action:
                    direction = -1

                da_cash = -(direction * volume * spot_price)
                settle_cash = direction * volume * actual_price
                spread_cap = direction * (actual_price - spot_price)
                gross_pnl = da_cash + settle_cash
                fees = volume * fee_per_mwh if direction != 0 else 0.0
                tax = (gross_pnl - fees) * tax_rate if (gross_pnl - fees) > 0 else 0.0
                net_pnl = gross_pnl - fees - tax

                row_update = {
                    "id": row["id"],
                    "actual_settled_eur": round(actual_price, 2),
                    "spread_captured_eur": round(spread_cap, 2),
                    "da_cash_flow_eur": round(da_cash, 2),
                    "settle_cash_flow_eur": round(settle_cash, 2),
                    "gross_pnl_eur": round(gross_pnl, 2),
                    "fees_eur": round(fees, 2),
                    "tax_eur": round(tax, 2),
                    "net_pnl_eur": round(net_pnl, 2),
                    "status": "SETTLED_AUDITED",
                    "settled_at": now_iso,
                    "updated_at": now_iso,
                }
                updates_batch.append(row_update)

        if updates_batch:
            # Batch update in chunks of 50
            for i in range(0, len(updates_batch), 50):
                chunk = updates_batch[i : i + 50]
                try:
                    self.table.upsert(chunk, on_conflict="id").execute()
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
