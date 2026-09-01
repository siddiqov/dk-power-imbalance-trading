# ==============================================================================
# run_v2_commercial_tournament.py
# V2 Master Execution & Commercial Backtesting Tournament
#
# Runs an end-to-end tournament across:
# - 3 Real-Data Paradigms (Zero Synthetic Data)
# - Multiple Machine Learning Architectures (LightGBM, XGBoost, CatBoost, Ensemble)
# - Commercial Simulation Trading Agent with Danish Fees, Slippage & 22% Tax
# - Complete 96-Quarter Day-Ahead Backtest & Monetary Leaderboard
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.data_ingestion_v2 import V2DataEngine
from src.model_trainer_v2 import V2ModelTournament
from src.trading_agent_v2 import CommercialTradingAgent


def run_tournament(price_area="DK1", initial_capital=100000.0):
    print("\n" + "#" * 80)
    print(f"      STARTING V2 COMMERCIAL TOURNAMENT & 96-QUARTER BACKTEST ({price_area})")
    print("#" * 80)

    # 1. Initialize Engine & Load Data
    engine = V2DataEngine()
    engine.sync_modern_15min()

    # Load data for each paradigm
    df_p1_hourly, df_p1_15m = engine.load_paradigm1_transfer_learning(price_area)
    df_p2_macro, df_p2_micro = engine.load_paradigm2_hierarchical(price_area)
    df_p3_full_1h, df_p3_pure_15m = engine.load_paradigm3_dual_models(price_area)

    print(f"\n  Data Loaded for {price_area}:")
    print(f"    - Historical Hourly Rows:     {len(df_p1_hourly):,}")
    print(f"    - Modern 15-Minute Rows:      {len(df_p1_15m):,}")

    if len(df_p1_15m) < 192:
        print("  ⚠ Insufficient 15m data points to run full 96-quarter backtest.")
        return

    # 2. Train Paradigms & Models
    tournament = V2ModelTournament(price_area=price_area)

    # Paradigm 1
    res_p1 = tournament.train_paradigm1_transfer_learning(df_p1_hourly, df_p1_15m, val_days=1)

    # Paradigm 2
    res_p2 = tournament.train_paradigm2_hierarchical(df_p2_macro, df_p2_micro, val_days=1)

    # Paradigm 3
    res_p3 = tournament.train_paradigm3_dual_models(df_p3_full_1h, df_p3_pure_15m, val_days=1)

    # Meta-Ensemble
    res_ensemble = tournament.build_stacking_ensemble([res_p1, res_p2, res_p3])

    all_models = [res_p1, res_p2, res_p3, res_ensemble]

    # 3. Autonomous Commercial Trading Agent Simulation
    agent = CommercialTradingAgent(initial_capital=initial_capital)
    trading_summaries = []

    print("\n" + "=" * 80)
    print("      EXECUTING COMMERCIAL TRADING AGENT ACROSS ALL MODELS (96 QUARTERS)")
    print("=" * 80)

    for model_res in all_models:
        summary = agent.simulate_day_trading(
            val_df=model_res["val_df"],
            pred_spread=model_res["pred_spread"],
            pred_direction=model_res["pred_direction"],
            pred_q10=model_res.get("pred_q10"),
            pred_q90=model_res.get("pred_q90"),
            model_label=f"{model_res['paradigm']} | {model_res['model_name']}"
        )
        agent.print_commercial_statement(summary)
        trading_summaries.append({
            "paradigm": model_res["paradigm"],
            "model_name": model_res["model_name"],
            "mae": model_res["mae"],
            "acc": model_res["acc"],
            "f1": model_res["f1"],
            "summary": summary
        })

    # 4. Generate Final Leaderboard
    leaderboard = []
    for item in trading_summaries:
        s = item["summary"]
        leaderboard.append({
            "Paradigm": item["paradigm"],
            "Model Architecture": item["model_name"],
            "Dir Accuracy": f"{item['acc']*100:.1f}%",
            "Spread MAE": f"€{item['mae']:.2f}",
            "Trades": f"{s['total_trades']}/96",
            "Win Rate": f"{s['win_rate']:.1f}%",
            "Volume (MWh)": f"{s['total_volume_mwh']:.1f}",
            "Gross PnL": f"€{s['gross_pnl']:,.2f}",
            "Fees & Slip": f"€{(s['fees_paid'] + s['slippage_paid']):,.2f}",
            "Danish Tax (22%)": f"€{s['tax_paid']:,.2f}",
            "Net Realized Profit": s["net_profit"],
            "Daily Return": f"{s['return_pct']:+.2f}%"
        })

    df_lb = pd.DataFrame(leaderboard).sort_values(by="Net Realized Profit", ascending=False)
    df_lb["Net Realized Profit (€)"] = df_lb["Net Realized Profit"].map("€{:,.2f}".format)
    df_lb.drop(columns=["Net Realized Profit"], inplace=True)
    df_lb.reset_index(drop=True, inplace=True)
    df_lb.index += 1

    print("\n" + "#" * 100)
    print(f"                   FINAL COMMERCIAL TOURNAMENT LEADERBOARD ({price_area})")
    print("#" * 100)
    print(df_lb.to_string())
    print("#" * 100)

    # Save leaderboard to CSV
    os.makedirs("results", exist_ok=True)
    df_lb.to_csv(f"results/v2_commercial_leaderboard_{price_area}.csv", index=True)

    # 5. Plot Comprehensive 96-Quarter Backtest Dashboard
    plot_backtest_dashboard(all_models, trading_summaries, price_area)

    return df_lb


def plot_backtest_dashboard(all_models, trading_summaries, price_area):
    """Plots the 96-quarter price execution chart, equity curve, and cost waterfall."""
    best_item = max(trading_summaries, key=lambda x: x["summary"]["net_profit"])
    best_summary = best_item["summary"]
    trade_log = best_summary["trade_log"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1.2, 1.2]})
    plt.subplots_adjust(hspace=0.35)

    # Subplot 1: 96-Quarter Price Trajectory & Trade Executions
    ax1 = axes[0]
    quarters = trade_log["quarter"]
    ax1.plot(quarters, trade_log["spot_price"], label="Day-Ahead Spot Price (€/MWh)", color="#1f77b4", linewidth=2.0, alpha=0.8)
    ax1.plot(quarters, trade_log["actual_imbalance"], label="Actual Imbalance Price (€/MWh)", color="#2ca02c", linewidth=2.2)
    ax1.plot(quarters, trade_log["spot_price"] + trade_log["pred_spread"], label=f"Best Pred ({best_item['model_name']})", color="#ff7f0e", linestyle="--", linewidth=1.8)

    # Execution markers
    longs = trade_log[trade_log["action"] == "LONG_SPOT"]
    shorts = trade_log[trade_log["action"] == "SHORT_SPOT"]

    if not longs.empty:
        ax1.scatter(longs["quarter"], longs["spot_price"], marker="^", color="green", s=80, label=f"BUY Spot / LONG ({len(longs)})", zorder=5)
    if not shorts.empty:
        ax1.scatter(shorts["quarter"], shorts["spot_price"], marker="v", color="red", s=80, label=f"SELL Spot / SHORT ({len(shorts)})", zorder=5)

    ax1.set_title(f"Day-Ahead 96-Quarter Imbalance Arbitrage & Trade Executions ({price_area})", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price (EUR/MWh)")
    ax1.set_xlim(1, 96)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", frameon=True)

    # Subplot 2: Cumulative Portfolio Equity Curve
    ax2 = axes[1]
    for item in trading_summaries:
        s = item["summary"]
        ax2.plot(range(len(s["capital_curve"])), s["capital_curve"], label=f"{item['model_name']} ({s['return_pct']:+.2f}%)", linewidth=1.8)

    ax2.axhline(best_summary["initial_capital"], color="black", linestyle=":", alpha=0.6, label="Starting Capital (€100k)")
    ax2.set_title("Intraday Capital & Equity Curve across 96 Quarters (€)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Portfolio Capital (€)")
    ax2.set_xlim(0, 96)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", frameon=True)

    # Subplot 3: Fee, Slippage & Tax Waterfall Breakdown
    ax3 = axes[2]
    categories = ["Gross PnL", "Exchange & TSO Fees", "Slippage", "Danish Tax (22%)", "NET REALIZED PROFIT"]
    values = [
        best_summary["gross_pnl"],
        -best_summary["fees_paid"],
        -best_summary["slippage_paid"],
        -best_summary["tax_paid"],
        best_summary["net_profit"]
    ]
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#1f77b4"]

    bars = ax3.bar(categories, values, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        y_pos = height if height >= 0 else height - (abs(height) * 0.15)
        ax3.text(bar.get_x() + bar.get_width()/2.0, y_pos, f"€{val:,.2f}", ha='center', va='bottom' if height < 0 else 'bottom', fontweight='bold', fontsize=10)

    ax3.axhline(0, color="black", linewidth=1.0)
    ax3.set_title(f"Financial Waterfall Analysis for Best Performer: {best_item['model_name']}", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Amount (EUR)")
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    chart_path = f"results/v2_commercial_backtest_{price_area}.png"
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved 96-Quarter Backtesting Dashboard to: {chart_path}")


if __name__ == '__main__':
    run_tournament(price_area="DK1")
