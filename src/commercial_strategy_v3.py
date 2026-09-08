# ==============================================================================
# src/commercial_strategy_v3.py
# V3 Commercial Strategy Engine: Asymmetric Spike Shield & Conviction Sizing
# Zero Synthetic Data / 100% Genuine Settlement Audits
# ==============================================================================

import numpy as np
import pandas as pd
from src.tournament_tables_v2 import TournamentTableGenerator
from src.model_trainer_v3 import V3QuantileModelSuite


class V3CommercialStrategyEngine:
    """
    Executes commercial trading decisions utilizing Optimeering Quantiles
    and the Asymmetric Spike Shield to prevent disastrous short-position drawdowns.
    """

    def __init__(self, price_area='DK1', capital=100000.0, base_volume_mwh=2.0):
        self.price_area = price_area
        self.capital = float(capital)
        self.base_volume_mwh = float(base_volume_mwh)
        self.fee_per_mwh = 0.51  # €0.06 Nord Pool + €0.20 TSO + €0.25 Slippage
        self.tax_rate = 0.22  # Danish 22% Corporate Tax
        self.model_suite = V3QuantileModelSuite(price_area=price_area)

    def evaluate_trading_ledger(self, df_day_d, model_name="Transformer-TFT"):
        """
        Calculates trade decisions, cash flows, and realized PnL across all 96 quarters.
        Applies:
        1. Asymmetric Spike Shield on Short positions
        2. Dynamic Conviction Sizing based on Quantile Width and Direction Probability
        """
        # Ensure models are loaded and generate predictions
        if not self.model_suite.models:
            self.model_suite.load_models()

        preds = self.model_suite.predict_day_ahead_quantiles(df_day_d)
        point_preds = preds[model_name]["pred_spread"]
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
            t_dk_str = row["time_dk"].strftime("%Y-%m-%d %H:%M") if hasattr(row["time_dk"], "strftime") else str(row["time_dk"])
            p_spot = float(row["spot_price_eur"])
            pred_spread = float(point_preds[i])
            p_pred = p_spot + pred_spread

            q10_s = float(quantiles["q10_spread"][i])
            q50_s = float(quantiles["q50_spread"][i])
            q90_s = float(quantiles["q90_spread"][i])

            p_up = float(probs["p_up"][i])
            p_down = float(probs["p_down"][i])
            p_up_spike = float(probs["p_up_spike"][i])

            # -----------------------------------------------------------------
            # TRADING DECISION MATRIX WITH ASYMMETRIC SPIKE SHIELD
            # -----------------------------------------------------------------
            action = "HOLD"
            direction = "BALANCED (0)"
            vol_multiplier = 1.0

            # 1. LONG SIGNAL (Expecting Up-Regulation / Imbalance > Spot)
            if pred_spread > 1.2 and p_up > 0.45:
                action = "BUY Spot (Long)"
                direction = "UP-REGULATION (+1)"
                # Conviction sizing: scale up if high probability
                if p_up >= 0.70 and pred_spread >= 5.0:
                    vol_multiplier = 2.5  # Scale to max growth sizing
                elif p_up >= 0.55:
                    vol_multiplier = 1.5

            # 2. SHORT SIGNAL (Expecting Down-Regulation / Imbalance < Spot)
            elif pred_spread < -1.2 and p_down > 0.45:
                # ASYMMETRIC SPIKE SHIELD GATE
                # If upper quantile q90 indicates upside deficit risk, DO NOT SHORT!
                if q90_s > 20.0 or p_up_spike > 0.15:
                    action = "HOLD (Spike Shield Protected)"
                    direction = "BALANCED (Shield)"
                else:
                    action = "SELL Spot (Short)"
                    direction = "DOWN-REGULATION (-1)"
                    if p_down >= 0.70 and pred_spread <= -5.0:
                        vol_multiplier = 2.0
                    elif p_down >= 0.55:
                        vol_multiplier = 1.2

            trade_vol = self.base_volume_mwh * vol_multiplier if "BUY" in action or "SELL" in action else 0.0

            # -----------------------------------------------------------------
            # SETTLEMENT & CASH FLOW EVALUATION
            # -----------------------------------------------------------------
            actual_imb = row.get("actual_imbalance_eur") or row.get("imbalance_price_eur")
            is_settled = pd.notnull(actual_imb) and str(actual_imb) != "--"

            if is_settled:
                occurred_count += 1
                p_actual = float(actual_imb)
                actual_str = f"€ {p_actual:.2f}"
                status = "✅ Settled"

                if "BUY" in action:
                    active_count += 1
                    da_cash_str = f"-€ {trade_vol * p_spot:.2f}"
                    da_cash_val = -(trade_vol * p_spot)
                    settle_cash_str = f"+€ {trade_vol * p_actual:.2f}"
                    settle_cash_val = trade_vol * p_actual
                    fees = trade_vol * self.fee_per_mwh
                    gross_pnl = da_cash_val + settle_cash_val
                    net_q_pnl = gross_pnl - fees
                elif "SELL" in action:
                    active_count += 1
                    da_cash_str = f"+€ {trade_vol * p_spot:.2f}"
                    da_cash_val = trade_vol * p_spot
                    settle_cash_str = f"-€ {trade_vol * p_actual:.2f}"
                    settle_cash_val = -(trade_vol * p_actual)
                    fees = trade_vol * self.fee_per_mwh
                    gross_pnl = da_cash_val + settle_cash_val
                    net_q_pnl = gross_pnl - fees
                else:
                    hold_count += 1
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
                da_cash_str = f"-€ {trade_vol * p_spot:.2f}" if "BUY" in action else (f"+€ {trade_vol * p_spot:.2f}" if "SELL" in action else "€ 0.00")
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
                "q10_price_eur": round(p_spot + q10_s, 2),
                "q50_price_eur": round(p_spot + q50_s, 2),
                "q90_price_eur": round(p_spot + q90_s, 2),
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
