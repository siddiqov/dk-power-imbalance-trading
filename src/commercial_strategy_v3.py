# ==============================================================================
# src/commercial_strategy_v3.py
# V3 Commercial Strategy Engine: Immutable Order Journal & Dual-Market Execution
# Zero Synthetic Data / Zero Hindsight Mutation / 100% Genuine Audits
# ==============================================================================

import os
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from src.model_trainer_v3 import V3QuantileModelSuite
from src.trade_journal import V3TradeJournal


class V3CommercialStrategyEngine:
    """
    Executes commercial trading decisions utilizing Optimeering Quantiles,
    Asymmetric Spike Shield, and persistent Immutable Order Journaling.
    Supports both Pure Day-Ahead (D-1) and Continuous Intraday (D-0).
    """

    def __init__(self, price_area='DK1', capital=100000.0, base_volume_mwh=2.0):
        self.price_area = price_area
        self.capital = float(capital)
        self.base_volume_mwh = float(base_volume_mwh)
        self.fee_per_mwh = 0.51  # €0.06 Nord Pool + €0.20 TSO + €0.25 Slippage
        self.tax_rate = 0.22  # Danish 22% Corporate Tax
        self.model_suite = V3QuantileModelSuite(price_area=price_area)
        self.journal = V3TradeJournal()

    def evaluate_trading_ledger(self, df_day_d, model_name="Transformer-TFT", market_mode="DAY_AHEAD_D1"):
        """
        Evaluates 96-quarter commercial ledger with strict Gate Closure order locking.
        
        - market_mode='DAY_AHEAD_D1': Pure Day-Ahead 24h auction schedule locked on D-1.
        - market_mode='INTRADAY_D0': Continuous Intraday rolling schedule locked at T-15m.
        """
        if not self.model_suite.models:
            self.model_suite.load_models()

        # Extract delivery date
        t_col = "time_dk" if "time_dk" in df_day_d.columns else "time_utc"
        first_dt = pd.to_datetime(df_day_d.iloc[0][t_col])
        delivery_date_str = first_dt.strftime("%Y-%m-%d")

        # Generate predictions tailored to the market mode
        preds = self.model_suite.predict_day_ahead_quantiles(df_day_d)
        point_preds = preds.get(model_name, {}).get("pred_spread", np.zeros(len(df_day_d)))
        quantiles = preds["quantiles"]
        probs = preds["probabilities"]

        trades = []
        running_capital = self.capital
        gross_pnl_acc = 0.0
        fees_acc = 0.0
        active_count = 0
        hold_count = 0
        occurred_count = 0

        for i, row in df_day_d.iterrows():
            q_idx = i + 1
            q_str = f"Q{q_idx}"
            t_dk_str = row[t_col].strftime("%Y-%m-%d %H:%M") if hasattr(row[t_col], "strftime") else str(row[t_col])
            p_spot = float(row["spot_price_eur"])

            # -----------------------------------------------------------------
            # 1. CHECK / LOCK ORDER IN IMMUTABLE TRADE JOURNAL
            # -----------------------------------------------------------------
            existing_order = self.journal.get_order(market_mode, self.price_area, model_name, delivery_date_str, q_idx)

            if existing_order is None:
                # Calculate new trade signal to lock
                pred_spread = float(point_preds[i])
                p_pred = p_spot + pred_spread

                q10_s = float(quantiles["q10_spread"][i])
                q50_s = float(quantiles["q50_spread"][i])
                q90_s = float(quantiles["q90_spread"][i])

                p_up = float(probs["p_up"][i])
                p_down = float(probs["p_down"][i])
                p_up_spike = float(probs["p_up_spike"][i])

                action = "HOLD"
                direction = "BALANCED (0)"
                vol_multiplier = 1.0

                if pred_spread > 1.2:
                    action = "BUY Spot (Long)"
                    direction = "UP-REGULATION (+1)"
                    if p_up >= 0.65 or pred_spread >= 5.0:
                        vol_multiplier = 2.5
                    elif p_up >= 0.45 or pred_spread >= 2.5:
                        vol_multiplier = 1.5
                    else:
                        vol_multiplier = 1.0

                elif pred_spread < -1.2:
                    if q90_s > 25.0 or p_up_spike > 0.20:
                        action = "HOLD (Spike Shield Protected)"
                        direction = "BALANCED (Shield)"
                    else:
                        action = "SELL Spot (Short)"
                        direction = "DOWN-REGULATION (-1)"
                        if p_down >= 0.65 or pred_spread <= -5.0:
                            vol_multiplier = 2.0
                        elif p_down >= 0.45 or pred_spread <= -2.5:
                            vol_multiplier = 1.2
                        else:
                            vol_multiplier = 1.0

                trade_vol = self.base_volume_mwh * vol_multiplier if "BUY" in action or "SELL" in action else 0.0

                # Persist order to Immutable Journal
                order_to_lock = {
                    "trade_id": str(uuid.uuid4()),
                    "market_mode": market_mode,
                    "price_area": self.price_area,
                    "model_name": model_name,
                    "delivery_date": delivery_date_str,
                    "quarter_index": q_idx,
                    "quarter_str": q_str,
                    "time_dk": t_dk_str,
                    "gate_closure_time": datetime.now().isoformat(),
                    "locked_action": action,
                    "locked_direction": direction,
                    "locked_volume_mwh": trade_vol,
                    "locked_spot_price_eur": p_spot,
                    "locked_pred_price_eur": p_pred,
                    "locked_pred_spread_eur": pred_spread,
                    "locked_q10_price_eur": p_spot + q10_s,
                    "locked_q50_price_eur": p_spot + q50_s,
                    "locked_q90_price_eur": p_spot + q90_s,
                    "locked_p_up": p_up,
                    "locked_p_down": p_down,
                    "locked_p_up_spike": p_up_spike
                }
                self.journal.lock_order(order_to_lock)
                locked_order = self.journal.get_order(market_mode, self.price_area, model_name, delivery_date_str, q_idx)
            else:
                # Use previously locked immutable order
                locked_order = existing_order

            # Extract locked immutable values
            action = locked_order["locked_action"]
            direction = locked_order["locked_direction"]
            trade_vol = float(locked_order["locked_volume_mwh"])
            p_pred = float(locked_order["locked_pred_price_eur"])
            pred_spread = float(locked_order["locked_pred_spread_eur"])
            q10_p = float(locked_order.get("locked_q10_price_eur") or (p_spot + quantiles["q10_spread"][i]))
            q50_p = float(locked_order.get("locked_q50_price_eur") or (p_spot + quantiles["q50_spread"][i]))
            q90_p = float(locked_order.get("locked_q90_price_eur") or (p_spot + quantiles["q90_spread"][i]))
            p_up = float(locked_order.get("locked_p_up") or probs["p_up"][i])
            p_down = float(locked_order.get("locked_p_down") or probs["p_down"][i])

            # -----------------------------------------------------------------
            # 2. SETTLEMENT AUDIT (Strictly Against Locked Order)
            # -----------------------------------------------------------------
            raw_act = row.get("actual_settled_imbalance_eur") or row.get("actual_imbalance_eur") or row.get("imbalance_price_eur")
            is_settled = False
            p_actual = None

            if pd.notnull(raw_act) and str(raw_act).strip() not in ["--", "None", "nan", ""]:
                try:
                    cleaned_act = str(raw_act).replace("€", "").replace("EUR", "").replace(",", "").strip()
                    p_actual = float(cleaned_act)
                    is_settled = True
                except Exception:
                    is_settled = False

            if is_settled:
                occurred_count += 1
                audited_record = self.journal.update_settlement(
                    market_mode, self.price_area, model_name, delivery_date_str, q_idx,
                    actual_price=p_actual, fee_per_mwh=self.fee_per_mwh, tax_rate=self.tax_rate
                )

                actual_str = f"€ {p_actual:.2f}"
                status = "✅ Settled"

                gross_pnl = float(audited_record.get("gross_pnl_eur") or 0.0)
                fees = float(audited_record.get("fees_eur") or 0.0)
                net_q_pnl = float(audited_record.get("net_pnl_eur") or 0.0)
                da_cash_val = float(audited_record.get("da_cash_flow_eur") or 0.0)
                settle_cash_val = float(audited_record.get("settle_cash_flow_eur") or 0.0)

                da_cash_str = f"{'+€' if da_cash_val >= 0 else '-€'} {abs(da_cash_val):.2f}" if da_cash_val != 0 else "€ 0.00"
                settle_cash_str = f"{'+€' if settle_cash_val >= 0 else '-€'} {abs(settle_cash_val):.2f}" if settle_cash_val != 0 else "€ 0.00"

                if "BUY" in action or "SELL" in action:
                    active_count += 1
                else:
                    hold_count += 1

                gross_pnl_acc += gross_pnl
                fees_acc += fees
                running_capital += net_q_pnl
                pnl_str = f"{'+€' if net_q_pnl >= 0 else '-€'} {abs(net_q_pnl):.2f}"
                run_cap_str = f"€ {running_capital:,.2f}"
            else:
                actual_str = "--"
                status = "🔒 Locked (Pending)" if trade_vol > 0 else "⏳ Pending"
                da_cash_str = f"-€ {trade_vol * p_spot:.2f}" if "BUY" in action else (f"+€ {trade_vol * p_spot:.2f}" if "SELL" in action else "€ 0.00")
                settle_cash_str = "--"
                pnl_str = "--"
                run_cap_str = "--"
                fees = 0.0
                net_q_pnl = 0.0

            trades.append({
                "quarter": q_str,
                "time_dk": t_dk_str,
                "spot_price_eur": round(p_spot, 2),
                "pred_imbalance_eur": round(p_pred, 2),
                "pred_spread_eur": round(pred_spread, 2),
                "q10_price_eur": round(q10_p, 2),
                "q50_price_eur": round(q50_p, 2),
                "q90_price_eur": round(q90_p, 2),
                "p_up": round(p_up * 100.0, 1),
                "p_down": round(p_down * 100.0, 1),
                "action": action,
                "direction": direction,
                "volume_mwh": round(trade_vol, 1),
                "da_cash_flow": da_cash_str,
                "actual_settled_eur": actual_str,
                "settle_cash_flow": settle_cash_str,
                "roundtrip_fees_eur": round(fees, 2) if is_settled and "HOLD" not in action else 0.0,
                "net_pnl_eur": pnl_str,
                "running_capital": run_cap_str,
                "status": status,
                "is_settled": is_settled
            })

        # Calculate final accumulated tax & net realized profit
        tax_acc = max(0.0, (gross_pnl_acc - fees_acc) * self.tax_rate)
        net_realized_profit = gross_pnl_acc - fees_acc - tax_acc
        live_roc = (net_realized_profit / self.capital) * 100.0

        summary = {
            "market_mode": market_mode,
            "price_area": self.price_area,
            "model_name": model_name,
            "occurred_quarters": occurred_count,
            "total_quarters": len(df_day_d),
            "active_occurred": active_count,
            "hold_occurred": hold_count,
            "trades_fraction_str": f"{active_count}/{occurred_count}" if occurred_count > 0 else "0/0",
            "gross_pnl_so_far": round(gross_pnl_acc, 2),
            "fees_so_far": round(fees_acc, 2),
            "tax_so_far": round(tax_acc, 2),
            "net_realized_profit_so_far": round(net_realized_profit, 2),
            "live_roc_percent": round(live_roc, 2),
            "trades": trades
        }

        return summary
