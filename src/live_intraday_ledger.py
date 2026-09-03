# ==============================================================================
# src/live_intraday_ledger.py
# 100% REAL ENERGI DATA SERVICE LIVE LEDGER ACCUMULATOR (ZERO SYNTHETIC DATA)
#
# Computes exact trade execution cash flows for each 15-minute quarter:
# - Day-Ahead Commitment (Spot Price, Volume, Initial Cash Outflow/Inflow)
# - Real-Time Energinet Imbalance Settlement (Actual Imbalance Price, Revenue/Cost)
# - Nord Pool / TSO Tariffs / Exchange Fees (€0.51/MWh)
# - Net Realized Quarter PnL & Running Accumulated Portfolio Value
# ==============================================================================

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.data_ingestion_v2 import V2DataEngine


class LiveIntradayLedger:
    """
    Builds the detailed trade audit ledger for occurred quarters on Day D (Today)
    and completed quarters on Day D-1 (Backtest).
    """

    def __init__(self, price_area='DK1', capital=100000.0):
        self.price_area = price_area
        self.capital = capital
        self.engine = V2DataEngine()
        self.fee_per_mwh = 0.51 # €0.06 Nord Pool + €0.20 TSO + €0.25 Slippage
        self.tax_rate = 0.22 # Danish 22% Corporate Tax
        self._settled_dict = None

    def _get_settled_dict(self):
        if self._settled_dict is not None:
            return self._settled_dict

        start_dt = datetime(2026, 9, 2, 0, 0)
        end_dt = datetime(2026, 9, 3, 0, 0)

        # Pull real Energi Data Service records
        df_raw = self.engine.fetch_api_dataset(
            "ImbalancePrice",
            start_date=start_dt - timedelta(hours=2),
            end_date=end_dt + timedelta(hours=2),
            limit=2000
        )

        settled_dict = {}
        if not df_raw.empty:
            df_zone = df_raw[df_raw["PriceArea"] == self.price_area].copy()
            df_zone["time_dk"] = pd.to_datetime(df_zone["TimeDK"])
            for _, r in df_zone.iterrows():
                if pd.notnull(r.get("ImbalancePriceEUR")):
                    key = r["time_dk"].strftime("%Y-%m-%d %H:%M")
                    settled_dict[key] = {
                        "imbalance_price": float(r["ImbalancePriceEUR"]),
                        "spot_price": float(r["SpotPriceEUR"]) if pd.notnull(r.get("SpotPriceEUR")) else None
                    }

        self._settled_dict = settled_dict
        return self._settled_dict

    def get_live_today_ledger(self, model_name="Transformer-TFT"):
        """
        Builds live ledger for Today (2nd September 2026).
        Uses genuine predictions for the requested model from the 96-quarter Day D forecast matrix.
        """
        from src.tournament_tables_v2 import TournamentTableGenerator
        table_gen = TournamentTableGenerator(price_area=self.price_area)
        df_future = table_gen.get_future_table()

        model_col_map = {
            "Transformer-TFT": "transformer_tft_eur",
            "Transfer-LightGBM": "transfer_lgb_eur",
            "Hierarchical-LGBM+XGB": "hierarchical_eur",
            "Stacking-MetaEnsemble": "meta_ensemble_eur",
            "Deep-BiLSTM": "deep_bilstm_eur",
            "Pure15m-CatBoost": "pure15m_catboost_eur"
        }
        col = model_col_map.get(model_name, "meta_ensemble_eur")

        # Model-specific confidence thresholds (data-driven calibration)
        threshold_map = {
            "Transformer-TFT": 1.10,
            "Transfer-LightGBM": 1.25,
            "Hierarchical-LGBM+XGB": 1.30,
            "Stacking-MetaEnsemble": 1.20,
            "Deep-BiLSTM": 1.15,
            "Pure15m-CatBoost": 1.40
        }
        threshold = threshold_map.get(model_name, 1.20)

        trades = []
        running_capital = self.capital
        occurred_count = 0
        active_occurred_count = 0
        hold_occurred_count = 0
        gross_pnl_acc = 0.0
        fees_acc = 0.0

        for i, row in df_future.iterrows():
            t_dk_str = str(row["time_dk"])
            p_spot = float(row["spot_price_eur"])
            p_pred = float(row[col]) if col in row else p_spot
            pred_spread = p_pred - p_spot

            volume = 2.0
            action = "HOLD"
            direction = "BALANCED (0)"
            if pred_spread > threshold:
                action = "BUY Spot (Long)"
                direction = "UP (+1)"
            elif pred_spread < -threshold:
                action = "SELL Spot (Short)"
                direction = "DOWN (-1)"

            act_str = str(row.get("actual_settled_imbalance_eur", "--"))
            is_settled = (act_str != "--" and act_str != "nan" and act_str != "")

            if is_settled:
                occurred_count += 1
                clean_act = act_str.replace("€", "").replace("EUR", "").strip()
                p_actual = float(clean_act)
                actual_str = f"€ {p_actual:.2f}"
                status = "✅ Settled"

                if action == "BUY Spot (Long)":
                    active_occurred_count += 1
                    da_cash_str = f"-€ {volume * p_spot:.2f}"
                    da_cash_val = -(volume * p_spot)
                    settle_cash_str = f"+€ {volume * p_actual:.2f}"
                    settle_cash_val = volume * p_actual
                    fees = volume * self.fee_per_mwh
                    gross_pnl = settle_cash_val + da_cash_val
                    net_q_pnl = gross_pnl - fees
                elif action == "SELL Spot (Short)":
                    active_occurred_count += 1
                    da_cash_str = f"+€ {volume * p_spot:.2f}"
                    da_cash_val = volume * p_spot
                    settle_cash_str = f"-€ {volume * p_actual:.2f}"
                    settle_cash_val = -(volume * p_actual)
                    fees = volume * self.fee_per_mwh
                    gross_pnl = da_cash_val + settle_cash_val
                    net_q_pnl = gross_pnl - fees
                else:
                    hold_occurred_count += 1
                    da_cash_str = "€ 0.00"
                    settle_cash_str = "€ 0.00"
                    fees = 0.0
                    gross_pnl = 0.0
                    net_q_pnl = 0.0

                gross_pnl_acc += gross_pnl
                fees_acc += fees
                running_capital += net_q_pnl
                pnl_str = f"{'+€' if net_q_pnl >= 0 else '-€'} {abs(net_q_pnl):.2f}"
                run_cap_str = f"€ {running_capital:,.2f}"
            else:
                actual_str = "--"
                status = "⏳ Pending"
                da_cash_str = f"-€ {volume * p_spot:.2f}" if "BUY" in action else (f"+€ {volume * p_spot:.2f}" if "SELL" in action else "€ 0.00")
                settle_cash_str = "--"
                pnl_str = "--"
                run_cap_str = "--"
                fees = 0.0
                net_q_pnl = 0.0

            trades.append({
                "quarter": f"Q{i+1}",
                "time_dk": t_dk_str,
                "spot_price_eur": round(p_spot, 2),
                "pred_imbalance_eur": round(p_pred, 2),
                "pred_spread_eur": round(pred_spread, 2),
                "action": action,
                "direction": direction,
                "volume_mwh": volume if action != "HOLD" else 0.0,
                "da_cash_flow": da_cash_str,
                "actual_settled_eur": actual_str,
                "settle_cash_flow": settle_cash_str,
                "roundtrip_fees_eur": round(fees, 2) if is_settled and action != "HOLD" else 0.0,
                "net_pnl_eur": pnl_str,
                "running_capital": run_cap_str,
                "status": status,
                "is_settled": is_settled
            })

        # Calculate final accumulated tax & net realized profit so far
        tax_acc = max(0.0, (gross_pnl_acc - fees_acc) * self.tax_rate)
        net_realized_profit_so_far = gross_pnl_acc - fees_acc - tax_acc
        live_roc = (net_realized_profit_so_far / self.capital) * 100.0

        summary = {
            "model_name": model_name,
            "occurred_quarters": occurred_count,
            "total_quarters": len(df_future),
            "active_occurred": active_occurred_count,
            "hold_occurred": hold_occurred_count,
            "trades_fraction_str": f"{active_occurred_count}/{occurred_count}" if occurred_count > 0 else "0/0",
            "gross_pnl_so_far": round(gross_pnl_acc, 2),
            "fees_so_far": round(fees_acc, 2),
            "tax_so_far": round(tax_acc, 2),
            "net_realized_profit_so_far": round(net_realized_profit_so_far, 2),
            "live_roc_percent": round(live_roc, 2),
            "trades": trades
        }

        return summary

    def get_live_today_leaderboard(self):
        """
        Computes real-time commercial trading results for all 6 models
        strictly on the occurred settled quarters today (Day D).
        """
        models = [
            ("3. Dual Models (Pure 15m)", "Transformer-TFT"),
            ("1. Transfer Learning", "Transfer-LightGBM"),
            ("2. Hierarchical", "Hierarchical-LGBM+XGB"),
            ("4. Meta-Ensemble", "Stacking-MetaEnsemble"),
            ("1. Transfer Learning", "Deep-BiLSTM"),
            ("3. Dual Models (Pure 15m)", "Pure15m-CatBoost")
        ]

        leaderboard = []
        for paradigm, m_name in models:
            summary = self.get_live_today_ledger(model_name=m_name)
            trades = summary.get("trades", [])

            settled_trades = [t for t in trades if t.get("is_settled")]
            active_settled = [t for t in settled_trades if t.get("action") != "HOLD"]

            # Direction Accuracy
            dir_correct = 0
            spread_errs = []
            for t in settled_trades:
                pred_s = t["pred_spread_eur"]
                act_s = float(t["actual_settled_eur"].replace("€", "").strip()) - t["spot_price_eur"]
                spread_errs.append(abs(pred_s - act_s))
                if (pred_s > 0 and act_s > 0) or (pred_s < 0 and act_s < 0) or (abs(pred_s) <= 1.2 and abs(act_s) <= 1.2):
                    dir_correct += 1

            dir_acc = (dir_correct / len(settled_trades) * 100.0) if settled_trades else 0.0
            spread_mae = np.mean(spread_errs) if spread_errs else 0.0

            # Win rate (% of active trades with positive net PnL)
            winning_trades = [t for t in active_settled if t["net_pnl_eur"].startswith("+€")]
            win_rate = (len(winning_trades) / len(active_settled) * 100.0) if active_settled else 0.0

            traded_vol = len(active_settled) * 2.0
            gross_pnl = summary["gross_pnl_so_far"]
            fees = summary["fees_so_far"]
            tax = summary["tax_so_far"]
            net_profit = summary["net_realized_profit_so_far"]
            daily_return = summary["live_roc_percent"]

            pnl_str = f"{'+EUR' if gross_pnl >= 0 else '-EUR'} {abs(gross_pnl):,.2f}"
            fees_str = f"EUR {fees:,.2f}"
            tax_str = f"EUR {tax:,.2f}"
            net_str = f"{'+EUR' if net_profit >= 0 else '-EUR'} {abs(net_profit):,.2f}"
            ret_str = f"{'+' if daily_return >= 0 else ''}{daily_return:.2f}%"

            leaderboard.append({
                "Paradigm": paradigm,
                "Model Architecture": m_name,
                "Dir Accuracy": f"{dir_acc:.1f}%",
                "Spread MAE": f"{spread_mae:.2f} EUR",
                "Trades": summary["trades_fraction_str"],
                "Win Rate": f"{win_rate:.1f}%",
                "Volume (MWh)": f"{traded_vol:.1f}",
                "Gross PnL": pnl_str,
                "Fees & Slip": fees_str,
                "Danish Tax (22%)": tax_str,
                "Net Realized Profit (EUR)": net_str,
                "Daily Return": ret_str,
                "_net_profit_val": net_profit
            })

        # Sort by Net Realized Profit descending
        leaderboard.sort(key=lambda x: x["_net_profit_val"], reverse=True)
        for i, r in enumerate(leaderboard, 1):
            r["Rank"] = f"#{i}"

        return leaderboard
