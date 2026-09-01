# ==============================================================================
# src/trading_agent_v2.py
# V2 Commercial Simulation Trading Agent for Danish Electricity Markets
#
# Simulates autonomous commercial trading decisions (Buy Spot/Sell Imbalance,
# Sell Spot/Buy Imbalance, Hold) across 96 quarters of Day D.
#
# Accounting Includes:
# - Nord Pool / Exchange Clearing Fee: €0.06 / MWh
# - Energinet TSO Balancing Tariff: €0.20 / MWh
# - Execution Slippage Friction: €0.25 / MWh
# - Danish Corporate Profit Tax: 22%
# ==============================================================================

import numpy as np
import pandas as pd


class CommercialTradingAgent:
    """
    Autonomous Commercial Energy Trading Agent operating under Danish market rules.
    Takes multi-model predictions and executes capital-constrained arbitrage trades.
    """

    def __init__(self, initial_capital=100000.0, exchange_fee=0.06, tso_fee=0.20, slippage=0.25, tax_rate=0.22):
        self.initial_capital = initial_capital
        self.exchange_fee = exchange_fee
        self.tso_fee = tso_fee
        self.slippage = slippage
        self.total_friction = exchange_fee + tso_fee + slippage  # ~0.51 EUR/MWh
        self.tax_rate = tax_rate
        self.max_mw_per_trade = 20.0  # Max 20 MWh per 15-minute quarter
        self.hurdle_buffer = 0.50     # Margin of safety above fee

    def simulate_day_trading(self, val_df, pred_spread, pred_direction, pred_q10=None, pred_q90=None, model_label="Model"):
        """
        Executes simulated trading across all 96 quarters of the validation day.
        """
        df = val_df.copy().reset_index(drop=True)
        n_steps = len(df)

        capital = self.initial_capital
        capital_history = [capital]
        trades = []

        gross_pnl_total = 0.0
        fees_total = 0.0
        slippage_total = 0.0

        for i in range(n_steps):
            row = df.iloc[i]
            time_str = str(row.get("time_utc", f"Q{i+1}"))
            p_spot = row["spot_price_eur"]
            p_imb = row["imbalance_price_eur"]
            actual_spread = p_imb - p_spot
            actual_dir = row.get("target_direction", 0)

            exp_spread = pred_spread[i]
            exp_dir = pred_direction[i]

            # Uncertainty width penalty
            uncertainty = 1.0
            if pred_q10 is not None and pred_q90 is not None:
                interval_width = max(pred_q90[i] - pred_q10[i], 1.0)
                uncertainty = np.clip(10.0 / interval_width, 0.2, 1.5)

            # Decision Logic
            action = "HOLD"
            volume_mwh = 0.0
            gross_pnl = 0.0
            fee_cost = 0.0
            slip_cost = 0.0
            net_pnl = 0.0

            # 1. LONG SPOT / SELL IMBALANCE (Up-regulation arbitrage)
            if exp_dir == 1 and exp_spread > (self.total_friction + self.hurdle_buffer):
                action = "LONG_SPOT"
                # Sizing based on spread conviction and uncertainty
                conviction = min(abs(exp_spread) / 10.0, 1.0)
                volume_mwh = np.round(self.max_mw_per_trade * conviction * uncertainty, 2)
                volume_mwh = max(volume_mwh, 1.0)

                # Realized outcome: Bought at Spot, Sold at Imbalance
                gross_pnl = volume_mwh * actual_spread
                fee_cost = volume_mwh * (self.exchange_fee + self.tso_fee)
                slip_cost = volume_mwh * self.slippage
                net_pnl = gross_pnl - fee_cost - slip_cost

            # 2. SHORT SPOT / BUY IMBALANCE (Down-regulation arbitrage)
            elif exp_dir == -1 and exp_spread < -(self.total_friction + self.hurdle_buffer):
                action = "SHORT_SPOT"
                conviction = min(abs(exp_spread) / 10.0, 1.0)
                volume_mwh = np.round(self.max_mw_per_trade * conviction * uncertainty, 2)
                volume_mwh = max(volume_mwh, 1.0)

                # Realized outcome: Sold at Spot, Bought back at Imbalance
                gross_pnl = volume_mwh * (-actual_spread)
                fee_cost = volume_mwh * (self.exchange_fee + self.tso_fee)
                slip_cost = volume_mwh * self.slippage
                net_pnl = gross_pnl - fee_cost - slip_cost

            capital += net_pnl
            capital_history.append(capital)

            gross_pnl_total += gross_pnl
            fees_total += fee_cost
            slippage_total += slip_cost

            trades.append({
                "quarter": i + 1,
                "time": time_str,
                "spot_price": p_spot,
                "actual_imbalance": p_imb,
                "actual_spread": actual_spread,
                "pred_spread": exp_spread,
                "action": action,
                "volume_mwh": volume_mwh,
                "gross_pnl": gross_pnl,
                "fees": fee_cost,
                "slippage": slip_cost,
                "net_pnl": net_pnl,
                "capital": capital
            })

        df_trades = pd.DataFrame(trades)

        # Tax calculation
        net_operating_profit = gross_pnl_total - fees_total - slippage_total
        tax_deduction = max(net_operating_profit * self.tax_rate, 0.0)
        final_net_profit = net_operating_profit - tax_deduction
        final_capital = self.initial_capital + final_net_profit
        return_pct = (final_net_profit / self.initial_capital) * 100.0

        active_trades = df_trades[df_trades["action"] != "HOLD"]
        win_trades = active_trades[active_trades["net_pnl"] > 0]
        win_rate = (len(win_trades) / len(active_trades) * 100.0) if len(active_trades) > 0 else 0.0

        summary = {
            "model_label": model_label,
            "initial_capital": self.initial_capital,
            "final_capital": final_capital,
            "net_profit": final_net_profit,
            "return_pct": return_pct,
            "gross_pnl": gross_pnl_total,
            "fees_paid": fees_total,
            "slippage_paid": slippage_total,
            "operating_profit": net_operating_profit,
            "tax_paid": tax_deduction,
            "total_trades": len(active_trades),
            "win_rate": win_rate,
            "total_volume_mwh": active_trades["volume_mwh"].sum() if not active_trades.empty else 0.0,
            "trade_log": df_trades,
            "capital_curve": capital_history
        }

        return summary

    def print_commercial_statement(self, summary):
        """Prints formatted commercial statement."""
        print("\n" + "=" * 80)
        print(f"      COMMERCIAL TRADING AGENT REPORT — {summary['model_label'].upper()}")
        print("=" * 80)
        print(f"  Initial Capital:              € {summary['initial_capital']:>12,.2f}")
        print(f"  Ending Portfolio Capital:     € {summary['final_capital']:>12,.2f}")
        print(f"  Daily Net Return (ROC):         {summary['return_pct']:>11.2f} %")
        print(f"  Total Trades (Active / Hold):   {summary['total_trades']} / {96 - summary['total_trades']}")
        print(f"  Win Rate on Active Trades:      {summary['win_rate']:>11.1f} %")
        print(f"  Total Traded Volume:            {summary['total_volume_mwh']:>11.1f} MWh")
        print("-" * 80)
        print(f"  Gross Trading PnL:            € {summary['gross_pnl']:>12,.2f}")
        print(f"  Exchange & TSO Fees:          -€ {summary['fees_paid']:>11,.2f}")
        print(f"  Execution Slippage:           -€ {summary['slippage_paid']:>11,.2f}")
        print(f"  Net Operating Profit (EBIT):  € {summary['operating_profit']:>12,.2f}")
        print(f"  Danish Corporate Tax (22%):   -€ {summary['tax_paid']:>11,.2f}")
        print("-" * 80)
        print(f"  NET REALIZED PROFIT:          EUR {summary['net_profit']:>12,.2f}")
        print("=" * 80)
