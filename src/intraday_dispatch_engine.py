# ==============================================================================
# src/intraday_dispatch_engine.py
# 2-Hour Forward Intraday Dispatch Engine (12 Discrete Batches of 8 Quarters)
# Strict T-2h Gate Closure Protocol (Cutoff at T-2h15m: 21:45, 23:45, 01:45...)
# 100% Real Energi Data Service / Zero Leakage
# ==============================================================================

import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

from src.tournament_tables_v2 import TournamentTableGenerator
from src.commercial_strategy_v3 import V3CommercialStrategyEngine
try:
    from src.commercial_strategy_v3_1 import V31CommercialStrategyEngine
except ImportError:
    V31CommercialStrategyEngine = None


def get_danish_now():
    try:
        return datetime.now(ZoneInfo("Europe/Copenhagen"))
    except Exception:
        from datetime import timezone
        return datetime.now(timezone.utc) + timedelta(hours=2)


def get_2h_batch_metadata(batch_num: int, target_date: str = None, danish_dt=None):
    """
    Returns timing metadata for a given 2-hour batch (1 to 12).
    Each batch covers 8 consecutive 15-minute quarters (2.0 hours).
    """
    if danish_dt is None:
        danish_dt = get_danish_now()
        
    start_q = (batch_num - 1) * 8 + 1
    end_q = batch_num * 8
    start_hour = (batch_num - 1) * 2
    end_hour = batch_num * 2
    
    window_str = f"{start_hour:02d}:00 - {end_hour:02d}:00 CET"
    
    if batch_num == 1:
        cutoff_str = "D-1 21:45 CET (Deadline 22:00)"
    elif batch_num == 2:
        cutoff_str = "D-1 23:45 CET (Deadline 00:00)"
    else:
        cutoff_hour = start_hour - 3
        if cutoff_hour < 0:
            cutoff_hour += 24
        cutoff_str = f"Day D {cutoff_hour:02d}:45 CET (Deadline {start_hour-2:02d}:00)"
        
    return {
        "batch_num": batch_num,
        "start_quarter": start_q,
        "end_quarter": end_q,
        "quarter_range_str": f"Q{start_q}–Q{end_q}",
        "delivery_window": window_str,
        "cutoff_str": cutoff_str
    }


def get_upcoming_batch_info(danish_dt=None):
    """
    Determines the next active 2-hour upcoming delivery batch based on current Danish clock.
    """
    if danish_dt is None:
        danish_dt = get_danish_now()
        
    h = danish_dt.hour
    raw_batch = (h // 2) + 3
    if raw_batch > 12:
        batch_num = raw_batch - 12
        target_date = (danish_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        is_tomorrow = True
    else:
        batch_num = raw_batch
        target_date = danish_dt.strftime("%Y-%m-%d")
        is_tomorrow = False
        
    meta = get_2h_batch_metadata(batch_num, target_date=target_date, danish_dt=danish_dt)
    meta["target_date"] = target_date
    meta["is_tomorrow"] = is_tomorrow
    return meta


class Intraday2HourDispatchEngine:
    """
    Generates operational 2-hour (8-quarter) forward intraday dispatch schedules
    for physical trading execution, formatted for 1-click clipboard paste to Google Docs/Sheets.
    """

    def __init__(self, price_area="DK1", capital=20000.0, base_volume_mwh=2.0, profile="tier2_standard", version="v3_1"):
        self.price_area = price_area
        self.capital = float(capital)
        self.base_volume_mwh = float(base_volume_mwh)
        self.profile = profile or "tier2_standard"
        self.version = str(version).lower() if version else "v3_1"
        self.table_gen = TournamentTableGenerator(price_area=price_area)
        
        if ("v3_1" in self.version or "3.1" in self.version) and V31CommercialStrategyEngine is not None:
            self.strategy = V31CommercialStrategyEngine(
                price_area=price_area,
                capital=self.capital,
                base_volume_mwh=self.base_volume_mwh,
                profile=self.profile
            )
        else:
            self.strategy = V3CommercialStrategyEngine(
                price_area=price_area, 
                capital=self.capital, 
                base_volume_mwh=self.base_volume_mwh,
                profile=self.profile
            )

    def generate_batch(self, batch_num: int = None, date_str: str = None, model_name: str = "Transformer-TFT"):
        """
        Generates dispatch orders for a specific 2-hour batch (1-12) or auto-detects upcoming batch.
        """
        dk_now = get_danish_now()
        
        # Auto-resolve batch and date if not specified
        if batch_num is None or batch_num == "auto" or str(batch_num).lower() == "auto":
            auto_info = get_upcoming_batch_info(dk_now)
            batch_num = auto_info["batch_num"]
            if not date_str:
                date_str = auto_info["target_date"]
        else:
            batch_num = int(batch_num)
            if not date_str:
                # If batch 1 or 2, default to tomorrow if in evening, else today
                if batch_num in [1, 2] and dk_now.hour >= 20:
                    date_str = (dk_now + timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    date_str = dk_now.strftime("%Y-%m-%d")

        meta = get_2h_batch_metadata(batch_num, target_date=date_str, danish_dt=dk_now)
        start_q = meta["start_quarter"]
        end_q = meta["end_quarter"]

        # Fetch / generate full 96Q table for date
        try:
            target_df = self.table_gen.generate_and_save_future_table(date_str=date_str)
        except Exception as e:
            print(f"[Intraday2HourDispatch] Future table fallback: {e}")
            target_df = self.table_gen.get_backtest_table(date_str=date_str)

        if target_df.empty:
            target_df = self.table_gen.get_future_table()

        # Evaluate V3 Commercial Strategy
        summary = self.strategy.evaluate_trading_ledger(
            target_df, 
            model_name=model_name, 
            market_mode="INTRADAY_D0",
            profile=self.profile
        )

        all_trades = summary.get("trades", [])
        
        # Slice exact 8 quarters for this batch (start_q to end_q, 1-indexed)
        batch_trades_raw = all_trades[start_q - 1:end_q]
        
        # Format for output
        headers = [
            "Quarter", "Delivery Time (CET)", "Spot Price (€/MWh)", "Pred Imbalance (€/MWh)", 
            "Pred Spread (€/MWh)", "Quantiles (q10-q90)", "P(Up) / P(Dn)", "Recommended Action", 
            "Volume (MW)", "Status"
        ]
        tsv_lines = ["	".join(headers)]
        csv_lines = [",".join([f'"{h}"' for h in headers])]
        
        formatted_trades = []
        buy_count = 0
        sell_count = 0
        hold_count = 0
        total_vol = 0.0

        for t in batch_trades_raw:
            action_clean = t["action"].replace(" (Long)", "").replace(" (Short)", "")
            q_spread = f"€{t.get('q10_price_eur', 0):.1f} - €{t.get('q90_price_eur', 0):.1f}"
            
            p_up_val = float(t.get('p_up', 50.0))
            if p_up_val <= 1.0:
                p_up_val *= 100.0
            p_down_val = float(t.get('p_down', 50.0))
            if p_down_val <= 1.0:
                p_down_val *= 100.0
            p_ratio = f"{p_up_val:.0f}% Up / {p_down_val:.0f}% Dn"
            
            vol = float(t.get("volume_mwh", 0.0))
            if "BUY" in t["action"]:
                buy_count += 1
                total_vol += vol
            elif "SELL" in t["action"]:
                sell_count += 1
                total_vol += vol
            else:
                hold_count += 1

            row_tsv = [
                t["quarter"],
                t["time_dk"],
                f"€{t['spot_price_eur']:.2f}",
                f"€{t['pred_imbalance_eur']:.2f}",
                f"{t['pred_spread_eur']:+.2f} €/MWh",
                q_spread,
                p_ratio,
                action_clean,
                f"{vol:.1f} MW" if vol > 0 else "0.0 MW",
                f"Batch {batch_num} Dispatched"
            ]
            tsv_lines.append("\t".join(row_tsv))
            csv_lines.append(",".join([f'"{c}"' for c in row_tsv]))

            formatted_trades.append({
                "quarter": t["quarter"],
                "time_dk": t["time_dk"],
                "spot_price_eur": round(t["spot_price_eur"], 2),
                "pred_imbalance_eur": round(t["pred_imbalance_eur"], 2),
                "pred_spread_eur": round(t["pred_spread_eur"], 2),
                "q10_price_eur": round(t.get("q10_price_eur", 0), 2),
                "q90_price_eur": round(t.get("q90_price_eur", 0), 2),
                "p_up": round(p_up_val, 1),
                "p_down": round(p_down_val, 1),
                "action": action_clean,
                "volume_mwh": vol,
                "status": f"Batch {batch_num} ({meta['delivery_window']})"
            })

        tsv_payload = "\n".join(tsv_lines)
        csv_payload = "\n".join(csv_lines)
        actual_date = formatted_trades[0]["time_dk"][:10] if formatted_trades else date_str

        result = {
            "success": True,
            "price_area": self.price_area,
            "model_name": model_name,
            "profile": self.profile,
            "batch_num": batch_num,
            "date": actual_date,
            "quarter_range": meta["quarter_range_str"],
            "delivery_window": meta["delivery_window"],
            "gate_closure_cutoff": meta["cutoff_str"],
            "num_quarters": len(formatted_trades),
            "summary_metrics": {
                "total_dispatched_mw": round(total_vol, 1),
                "buy_count": buy_count,
                "sell_count": sell_count,
                "hold_count": hold_count
            },
            "trades": formatted_trades,
            "tsv_payload": tsv_payload,
            "csv_payload": csv_payload
        }

        return result

    def export_and_save_batch(self, batch_num: int = None, date_str: str = None, output_dir: str = "results/dispatch_batches"):
        """
        Saves TSV and CSV snapshots to the results/dispatch_batches folder.
        """
        os.makedirs(output_dir, exist_ok=True)
        res = self.generate_batch(batch_num=batch_num, date_str=date_str)
        b = res["batch_num"]
        d = res["date"]
        area = self.price_area
        
        # Save timestamped batch file
        tsv_file = os.path.join(output_dir, f"dispatch_{d}_Batch_{b}_{area}.tsv")
        with open(tsv_file, "w", encoding="utf-8") as f:
            f.write(res["tsv_payload"])
            
        # Save latest pointer file
        latest_file = os.path.join(output_dir, f"latest_dispatch_{area}.tsv")
        with open(latest_file, "w", encoding="utf-8") as f:
            f.write(res["tsv_payload"])
            
        res["saved_tsv_file"] = tsv_file
        res["latest_pointer_file"] = latest_file
        return res

    def generate_full_day_96q(self, date_str: str = None, model_name: str = "Transformer-TFT"):
        """
        Generates physical intraday dispatch schedules for all 96 quarters of Day D (e.g. Next Day).
        Supports any choice of model, DK1/DK2, and maps each quarter to its 2-hour batch and gate closure cutoff.
        """
        dk_now = get_danish_now()
        # Default to next day (D+1) if Danish time is >= 13:00 CET (post Day-Ahead spot auction clearing), else today (D)
        if not date_str:
            if dk_now.hour >= 13:
                date_str = (dk_now + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                date_str = dk_now.strftime("%Y-%m-%d")

        # Fetch/generate full 96Q table for date
        try:
            target_df = self.table_gen.generate_and_save_future_table(date_str=date_str)
        except Exception as e:
            print(f"[Intraday96QDispatch] Future table fallback: {e}")
            target_df = self.table_gen.get_backtest_table(date_str=date_str)

        if target_df.empty:
            target_df = self.table_gen.get_future_table()

        # Evaluate Commercial Strategy across all 96 quarters (handles Dynamic Conviction Tiers & Fixed Profiles)
        summary = self.strategy.evaluate_trading_ledger(
            target_df,
            model_name=model_name,
            market_mode="INTRADAY_D0",
            profile=self.profile
        )
        all_trades = summary.get("trades", [])

        headers = [
            "Quarter", "Batch", "Delivery Time (CET)", "Delivery Time (UTC)", "Spot Price (€/MWh)", 
            "Pred Imbalance (€/MWh)", "Pred Spread (€/MWh)", "Quantiles (q10-q90)", "P(Up) / P(Dn)", 
            "Recommended Action", "Volume (MW)", "DA Cash Flow (€)", "Gate Closure Cutoff", "Status"
        ]
        tsv_lines = ["\t".join(headers)]
        csv_lines = [",".join([f'"{h}"' for h in headers])]

        formatted_trades = []
        buy_count = 0
        sell_count = 0
        hold_count = 0
        total_vol = 0.0
        total_da_cash = 0.0

        for i, t in enumerate(all_trades):
            q_num = i + 1
            batch_num = ((q_num - 1) // 8) + 1
            meta = get_2h_batch_metadata(batch_num, target_date=date_str, danish_dt=dk_now)

            action_clean = t["action"].replace(" (Long)", "").replace(" (Short)", "")
            q_spread = f"€{t.get('q10_price_eur', 0):.1f} - €{t.get('q90_price_eur', 0):.1f}"
            
            p_up_val = float(t.get('p_up', 50.0))
            if p_up_val <= 1.0:
                p_up_val *= 100.0
            p_down_val = float(t.get('p_down', 50.0))
            if p_down_val <= 1.0:
                p_down_val *= 100.0
            p_ratio = f"{p_up_val:.0f}% Up / {p_down_val:.0f}% Dn"

            vol = float(t.get("volume_mwh", 0.0))
            spot = float(t.get("spot_price_eur", 0.0))
            da_flow = 0.0
            if "BUY" in t["action"]:
                buy_count += 1
                total_vol += vol
                da_flow = -(vol * spot)
            elif "SELL" in t["action"]:
                sell_count += 1
                total_vol += vol
                da_flow = (vol * spot)
            else:
                hold_count += 1

            total_da_cash += da_flow
            da_flow_str = f"{'+€' if da_flow >= 0 else '-€'} {abs(da_flow):.2f}" if action_clean != "HOLD" else "€ 0.00"

            time_dk = t.get("time_dk", "")
            time_utc = t.get("time_utc", "")
            if not time_utc and time_dk:
                try:
                    dt_dk = datetime.strptime(time_dk, "%Y-%m-%d %H:%M")
                    try:
                        from zoneinfo import ZoneInfo
                        dt_dk_zoned = dt_dk.replace(tzinfo=ZoneInfo("Europe/Copenhagen"))
                        time_utc = dt_dk_zoned.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        time_utc = (dt_dk - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    time_utc = time_dk

            row_tsv = [
                f"Q{q_num}", f"Batch {batch_num}", time_dk, time_utc,
                f"{spot:.2f}", f"{t['pred_imbalance_eur']:.2f}", f"{t['pred_spread_eur']:+.2f}",
                q_spread, p_ratio, action_clean, f"{vol:.1f}", da_flow_str, meta["cutoff_str"],
                "🔒 Order Locked" if action_clean != "HOLD" else "HOLD"
            ]
            tsv_lines.append("\t".join(row_tsv))
            csv_lines.append(",".join([f'"{col}"' for col in row_tsv]))

            formatted_trades.append({
                "quarter": f"Q{q_num}",
                "quarter_num": q_num,
                "batch_num": batch_num,
                "batch_label": f"Batch {batch_num} (Q{(batch_num-1)*8+1}–Q{batch_num*8})",
                "time_dk": time_dk,
                "time_utc": time_utc,
                "spot_price_eur": round(spot, 2),
                "pred_imbalance_eur": round(t["pred_imbalance_eur"], 2),
                "pred_spread_eur": round(t["pred_spread_eur"], 2),
                "q10_price_eur": round(t.get("q10_price_eur", 0), 2),
                "q90_price_eur": round(t.get("q90_price_eur", 0), 2),
                "p_up": round(p_up_val, 1),
                "p_down": round(p_down_val, 1),
                "action": action_clean,
                "volume_mwh": vol,
                "da_cash_flow": da_flow_str,
                "gate_closure_cutoff": meta["cutoff_str"],
                "delivery_window": meta["delivery_window"],
                "status": "🔒 Order Locked (Pending Delivery)" if action_clean != "HOLD" else "⚪ In Cash (HOLD)"
            })

        actual_date = formatted_trades[0]["time_dk"][:10] if formatted_trades else date_str
        return {
            "success": True,
            "price_area": self.price_area,
            "model_name": model_name,
            "profile": self.profile,
            "date": actual_date,
            "num_quarters": len(formatted_trades),
            "summary_metrics": {
                "total_dispatched_mw": round(total_vol, 1),
                "total_da_cash_eur": round(total_da_cash, 2),
                "buy_count": buy_count,
                "sell_count": sell_count,
                "hold_count": hold_count,
                "active_trades_pct": round(((buy_count + sell_count) / len(formatted_trades) * 100), 1) if formatted_trades else 0
            },
            "trades": formatted_trades,
            "tsv_payload": "\n".join(tsv_lines),
            "csv_payload": "\n".join(csv_lines)
        }
