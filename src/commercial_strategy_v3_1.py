# ==============================================================================
# src/commercial_strategy_v3_1.py
# V3.1 Commercial Strategy Engine: Dual Architecture (Trees + PyTorch Deep Sequence)
# Optimeering Quantile Spreads, Asymmetric Spike Shield, Real-Time PnL Tracking
# ==============================================================================

import os
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from src.model_trainer_v3_1 import V31QuantileModelSuite
from src.trade_journal import V3TradeJournal


VOLUME_PROFILES = {
    "tier1_conservative": {"v_min": 1.0, "v_max": 15.0, "label": "Tier 1: Conservative (1.0 - 15.0 MW)"},
    "tier2_standard":     {"v_min": 2.0, "v_max": 20.0, "label": "Tier 2: Standard (2.0 - 20.0 MW)"},
    "tier3_aggressive":   {"v_min": 0.5, "v_max": 25.0, "label": "Tier 3: Aggressive (0.5 - 25.0 MW)"},
    "fixed_2mwh":         {"v_min": 2.0, "v_max": 2.0,  "label": "Fixed: 2.0 MWh / Trade"},
    "fixed_5mwh":         {"v_min": 5.0, "v_max": 5.0,  "label": "Fixed: 5.0 MWh / Trade"},
    "fixed_10mwh":        {"v_min": 10.0, "v_max": 10.0, "label": "Fixed: 10.0 MWh / Trade"},
}


class V31CommercialStrategyEngine:
    """
    Executes commercial trading decisions utilizing Optimeering Quantiles,
    Asymmetric Spike Shield, and persistent Immutable Order Journaling for V3.1.
    """

    def __init__(self, price_area='DK1', capital=20000.0, base_volume_mwh=2.0, profile="tier1_conservative", v_min=None, v_max=None):
        self.price_area = price_area
        self.capital = float(capital)
        self.base_volume_mwh = float(base_volume_mwh)
        self.profile = profile
        self.v_min = v_min
        self.v_max = v_max
        self.fee_per_mwh = 0.51  # €0.06 Nord Pool + €0.20 TSO + €0.25 Slippage
        self.tax_rate = 0.22  # Danish 22% Corporate Tax
        self.model_suite = V31QuantileModelSuite(price_area=price_area, model_dir='models_v3_1')
        self.journal = V3TradeJournal(db_path="data/v3_1_trade_journal.db")

    def compute_conviction_volume(self, pred_spread: float, p_up: float, p_down: float, action: str,
                                  profile: str = "tier1_conservative", v_min: float = None, v_max: float = None) -> float:
        if "BUY" not in action and "SELL" not in action:
            return 0.0

        target_profile = profile or self.profile or "tier1_conservative"
        tier_cfg = VOLUME_PROFILES.get(target_profile, VOLUME_PROFILES["tier1_conservative"])
        resolved_v_min = v_min if v_min is not None else (self.v_min if self.v_min is not None else tier_cfg["v_min"])
        resolved_v_max = v_max if v_max is not None else (self.v_max if self.v_max is not None else tier_cfg["v_max"])

        if target_profile.startswith("fixed_"):
            return float(resolved_v_min)

        p_up_norm = p_up / 100.0 if p_up > 1.0 else p_up
        p_down_norm = p_down / 100.0 if p_down > 1.0 else p_down
        p_target = p_up_norm if "BUY" in action else p_down_norm

        spread_mag = abs(pred_spread)
        m_spread = min(max((spread_mag - 1.20) / (8.00 - 1.20), 0.0), 1.0)

        if p_target < 0.20:
            m_prob = 0.0
        elif p_target < 0.50:
            m_prob = ((p_target - 0.20) / (0.50 - 0.20)) * 0.60
        else:
            m_prob = 0.60 + ((p_target - 0.50) / (1.00 - 0.50)) * 0.40

        m_prob = min(max(m_prob, 0.0), 1.0)
        conviction = m_spread * m_prob
        vol = resolved_v_min + (resolved_v_max - resolved_v_min) * conviction
        return round(vol, 1)

    def evaluate_trading_ledger(self, df_day_d, model_name="Transfer-LightGBM", market_mode="DAY_AHEAD_D1",
                                profile: str = None, v_min: float = None, v_max: float = None,
                                approach: str = "A", df_prev_day=None):
        if not self.model_suite.models:
            self.model_suite.load_models()

        t_col = "time_dk" if "time_dk" in df_day_d.columns else "time_utc"
        first_dt = pd.to_datetime(df_day_d.iloc[0][t_col])
        delivery_date_str = first_dt.strftime("%Y-%m-%d")

        preds = self.model_suite.predict_day_ahead_quantiles(df_day_d, market_mode=market_mode, approach=approach, df_prev_day=df_prev_day)
        point_preds = preds.get(model_name, {}).get("pred_spread", np.zeros(len(df_day_d)))
        quantiles = preds["quantiles"]
        probs = preds["probabilities"]

        journal_mode = f"{market_mode}_{approach}" if approach and market_mode == "INTRADAY_D0" else market_mode

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
            t_dk_str = str(row[t_col])[:16]
            p_spot = float(row.get("spot_price_eur") or row.get("DayAheadPriceEUR") or row.get("SpotPriceEUR") or 0.0)

            existing_order = self.journal.get_order(journal_mode, self.price_area, model_name, delivery_date_str, q_idx)

            if existing_order is None:
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

                if pred_spread > 1.2:
                    action = "BUY Spot (Long)"
                    direction = "UP-REGULATION (+1)"
                elif pred_spread < -1.2:
                    if (q90_s > 80.0 and p_up_spike > 0.35) or p_up_spike > 0.45:
                        action = "HOLD (Spike Shield Protected)"
                        direction = "BALANCED (Shield)"
                    else:
                        action = "SELL Spot (Short)"
                        direction = "DOWN-REGULATION (-1)"

                trade_vol = self.compute_conviction_volume(pred_spread, p_up, p_down, action,
                                                           profile=profile, v_min=v_min, v_max=v_max)

                order_to_lock = {
                    "trade_id": str(uuid.uuid4()),
                    "market_mode": journal_mode,
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
                trade_record = order_to_lock
            else:
                trade_record = existing_order
                action = trade_record["locked_action"]
                direction = trade_record["locked_direction"]
                trade_vol = trade_record["locked_volume_mwh"]
                pred_spread = trade_record["locked_pred_spread_eur"]
                p_pred = trade_record["locked_pred_price_eur"]

            p_actual = row.get("actual_settled_imbalance_eur") or row.get("ImbalancePriceEUR")
            is_settled = pd.notnull(p_actual) and str(p_actual) not in ["--", "nan", ""]

            gross_pnl = 0.0
            fees = 0.0
            net_pnl = 0.0
            p_actual_val = None

            if is_settled:
                occurred_count += 1
                p_actual_val = float(str(p_actual).replace("€", "").replace("EUR", "").strip())
                actual_spread = p_actual_val - p_spot

                if "BUY" in action:
                    active_count += 1
                    gross_pnl = trade_vol * actual_spread
                    fees = trade_vol * self.fee_per_mwh
                    net_pnl = gross_pnl - fees
                elif "SELL" in action:
                    active_count += 1
                    gross_pnl = trade_vol * (-actual_spread)
                    fees = trade_vol * self.fee_per_mwh
                    net_pnl = gross_pnl - fees
                else:
                    hold_count += 1

                gross_pnl_acc += gross_pnl
                fees_acc += fees
                running_capital += net_pnl

            da_cash_str = "--"
            settle_cash_str = "--"
            if trade_vol > 0:
                if "BUY" in action:
                    da_cash_str = f"-€ {p_spot * trade_vol:,.2f}"
                    if is_settled and p_actual_val is not None:
                        settle_cash_str = f"+€ {p_actual_val * trade_vol:,.2f}"
                elif "SELL" in action:
                    da_cash_str = f"+€ {p_spot * trade_vol:,.2f}"
                    if is_settled and p_actual_val is not None:
                        settle_cash_str = f"-€ {p_actual_val * trade_vol:,.2f}"

            status_str = "Settled & Audited" if is_settled else "Order Locked"
            pnl_str = f"{'+€' if net_pnl >= 0 else '-€'} {abs(net_pnl):,.2f}" if is_settled and trade_vol > 0 else "€ 0.00"

            trades.append({
                "quarter": q_str,
                "time_dk": t_dk_str,
                "spot_price_eur": p_spot,
                "pred_imbalance_eur": p_pred,
                "pred_spread_eur": pred_spread,
                "q10_price_eur": trade_record.get("locked_q10_price_eur", p_spot),
                "q90_price_eur": trade_record.get("locked_q90_price_eur", p_spot),
                "p_up": trade_record.get("locked_p_up", 50.0),
                "p_down": trade_record.get("locked_p_down", 50.0),
                "p_up_spike": trade_record.get("locked_p_up_spike", 0.0),
                "action": action,
                "direction": direction,
                "volume_mwh": trade_vol,
                "is_settled": is_settled,
                "actual_settled_eur": f"€ {p_actual_val:.2f}" if p_actual_val is not None else "--",
                "gross_pnl_eur": gross_pnl,
                "fees_eur": fees,
                "roundtrip_fees_eur": fees,
                "net_pnl_eur": pnl_str,
                "net_pnl_val": net_pnl,
                "da_cash_flow": da_cash_str,
                "settle_cash_flow": settle_cash_str,
                "running_capital": f"€ {running_capital:,.2f}",
                "running_capital_eur": running_capital,
                "status": status_str
            })

        net_profit = running_capital - self.capital
        roc = (net_profit / self.capital) * 100.0 if self.capital > 0 else 0.0
        trades_str = f"{active_count}/{occurred_count}" if occurred_count > 0 else "0/0"
        tax_val = max(0.0, (gross_pnl_acc - fees_acc) * self.tax_rate)

        midnight_trades = [t for t in trades[:8] if t.get("volume_mwh", 0) > 0]
        midnight_pnl = sum(t.get("net_pnl_val", 0.0) for t in trades[:8])
        active_trades_list = [t for t in trades if t.get("volume_mwh", 0) > 0]
        win_trades = [t for t in active_trades_list if t.get("net_pnl_val", 0.0) > 0]
        win_rate = (len(win_trades) / len(active_trades_list) * 100.0) if active_trades_list else 0.0

        return {
            "model_name": model_name,
            "market_mode": market_mode,
            "approach": approach,
            "price_area": self.price_area,
            "profile": profile or self.profile,
            "delivery_date": delivery_date_str,
            "initial_capital": self.capital,
            "capital": self.capital,
            "ending_capital": running_capital,
            "net_pnl_eur": net_profit,
            "gross_pnl_eur": gross_pnl_acc,
            "fees_eur": fees_acc,
            "tax_eur": tax_val,
            "total_trades": len(trades),
            "occurred_trades": occurred_count,
            "active_trades": active_count,
            "hold_trades": hold_count,
            "trades_fraction_str": trades_str,
            "net_realized_profit_so_far": round(net_profit, 2),
            "net_realized_profit_so_far_str": f"{'+€' if net_profit >= 0 else '-€'} {abs(net_profit):,.2f}",
            "gross_pnl_so_far": round(gross_pnl_acc, 2),
            "fees_so_far": round(fees_acc, 2),
            "tax_so_far": round(tax_val, 2),
            "live_roc_percent": round(roc, 2),
            "roc_percent": round(roc, 2),
            "midnight_pnl_eur": round(midnight_pnl, 2),
            "midnight_pnl_str": f"{'+€' if midnight_pnl >= 0 else '-€'} {abs(midnight_pnl):,.2f}",
            "midnight_active_trades": len(midnight_trades),
            "win_rate_pct": round(win_rate, 1),
            "trades": trades
        }
