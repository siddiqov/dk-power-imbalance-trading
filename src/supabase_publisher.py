# ==============================================================================
# src/supabase_publisher.py
# Publishes TFT model quarter predictions to Supabase for the Bidder Portal.
# Upserts all 96 quarters per price area; only updates records that changed.
# ==============================================================================

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class SupabasePublisher:
    """
    Pushes quarter predictions from V3CommercialStrategyEngine
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
    ) -> int:
        """
        Upsert a list of trade dicts (from V3CommercialStrategyEngine.evaluate_trading_ledger)
        into Supabase. Returns the number of rows upserted.

        Args:
            trades: List of trade dicts from summary["trades"]
            price_area: 'DK1' or 'DK2'
            delivery_date: 'YYYY-MM-DD' string
            model_name: Model identifier, default 'Transformer-TFT'
        """
        import zoneinfo
        cph_tz = zoneinfo.ZoneInfo("Europe/Copenhagen")

        rows = []
        d_parts = [int(p) for p in delivery_date.split("-")]
        for trade in trades:
            q_idx = int(trade["quarter"].replace("Q", ""))
            hour = (q_idx - 1) // 4
            minute = ((q_idx - 1) % 4) * 15
            dt = datetime(d_parts[0], d_parts[1], d_parts[2], hour, minute, tzinfo=cph_tz)
            time_dk_iso = dt.isoformat()

            row = {
                "price_area": price_area,
                "delivery_date": delivery_date,
                "quarter_index": int(trade["quarter"].replace("Q", "")),
                "quarter_label": trade["quarter"],
                "time_dk": time_dk_iso,
                "spot_price_eur": trade["spot_price_eur"],
                "pred_imbalance_eur": trade["pred_imbalance_eur"],
                "pred_spread_eur": trade["pred_spread_eur"],
                "p_up": trade["p_up"],
                "p_down": trade["p_down"],
                "decision": trade["action"],
                "direction": trade["direction"],
                "volume_mwh": trade["volume_mwh"],
                "q10_price_eur": trade["q10_price_eur"],
                "q50_price_eur": trade["q50_price_eur"],
                "q90_price_eur": trade["q90_price_eur"],
                "model_name": model_name,
                "updated_at": datetime.utcnow().isoformat(),
            }
            rows.append(row)

        if not rows:
            logger.warning(f"No trades to upsert for {price_area} {delivery_date}")
            return 0

        # Upsert in batches (Supabase handles conflict on the UNIQUE constraint)
        batch_size = 50
        total_upserted = 0

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            result = self.table.upsert(
                batch,
                on_conflict="price_area,delivery_date,quarter_index,model_name",
            ).execute()
            total_upserted += len(batch)

        # Log this update
        self._log_update(price_area, delivery_date, total_upserted, len(trades))

        logger.info(
            f"✅ Upserted {total_upserted} quarters for {price_area} "
            f"({delivery_date}) model={model_name}"
        )
        return total_upserted

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
