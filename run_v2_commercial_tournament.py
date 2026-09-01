# ==============================================================================
# run_v2_commercial_tournament.py
# V2 Master Execution & Commercial Backtesting Tournament
#
# Runs an end-to-end tournament across:
# - The 4 Real-Data Paradigms (Zero Synthetic Data)
# - Tabular Boosters + Deep Sequence Models (Bi-LSTM, TFT Attention, LGBM, XGB, CatBoost)
# - Commercial Simulation Trading Agent with Danish Fees, Slippage & 22% Tax
# - Enhanced 4-Panel 96-Quarter Visual Dashboard
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    df_p1_hourly, df_p1_15m = engine.load_paradigm1_transfer_learning(price_area)
    df_p2_macro, df_p2_micro = engine.load_paradigm2_hierarchical(price_area)
    df_p3_full_1h, df_p3_pure_15m = engine.load_paradigm3_dual_models(price_area)

    print(f"\n  Data Loaded for {price_area}:")
    print(f"    - Historical Hourly Rows:     {len(df_p1_hourly):,}")
    print(f"    - Modern 15-Minute Rows:      {len(df_p1_15m):,}")

    if len(df_p1_15m) < 192:
        print("  [Notice] Insufficient 15m data points to run full 96-quarter backtest.")
        return None

    # 2. Train All Models across the 4 Paradigms
    tournament = V2ModelTournament(price_area=price_area)

    # Paradigm 1: Transfer-LGBM & Deep-BiLSTM
    models_p1 = tournament.train_paradigm1_transfer_learning(df_p1_hourly, df_p1_15m, val_days=1)

    # Paradigm 2: Hierarchical-LGBM+XGB
    models_p2 = tournament.train_paradigm2_hierarchical(df_p2_macro, df_p2_micro, val_days=1)

    # Paradigm 3: Pure15m-CatBoost & Transformer-TFT
    models_p3 = tournament.train_paradigm3_dual_models(df_p3_full_1h, df_p3_pure_15m, val_days=1)

    # Paradigm 4: Stacking Meta-Ensemble
    base_candidates = models_p1 + models_p2 + models_p3
    models_p4 = tournament.build_stacking_ensemble(base_candidates)

    all_models = base_candidates + models_p4

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
            "summary": summary,
            "pred_spread": model_res["pred_spread"],
            "val_df": model_res["val_df"]
        })

    # 4. Generate Final Leaderboard
    leaderboard = []
    for item in trading_summaries:
        s = item["summary"]
        leaderboard.append({
            "Paradigm": item["paradigm"],
            "Model Architecture": item["model_name"],
            "Dir Accuracy": f"{item['acc']*100:.1f}%",
            "Spread MAE": f"EUR {item['mae']:.2f}",
            "Trades": f"{s['total_trades']}/96",
            "Win Rate": f"{s['win_rate']:.1f}%",
            "Volume (MWh)": f"{s['total_volume_mwh']:.1f}",
            "Gross PnL": f"EUR {s['gross_pnl']:,.2f}",
            "Fees & Slip": f"EUR {(s['fees_paid'] + s['slippage_paid']):,.2f}",
            "Danish Tax (22%)": f"EUR {s['tax_paid']:,.2f}",
            "Net Realized Profit": s["net_profit"],
            "Daily Return": f"{s['return_pct']:+.2f}%"
        })

    df_lb = pd.DataFrame(leaderboard).sort_values(by="Net Realized Profit", ascending=False)
    df_lb["Net Realized Profit (EUR)"] = df_lb["Net Realized Profit"].map("EUR {:,.2f}".format)
    df_lb.drop(columns=["Net Realized Profit"], inplace=True)
    df_lb.reset_index(drop=True, inplace=True)
    df_lb.index += 1

    print("\n" + "#" * 105)
    print(f"                   FINAL COMMERCIAL TOURNAMENT LEADERBOARD ({price_area})")
    print("#" * 105)
    print(df_lb.to_string())
    print("#" * 105)

    os.makedirs("results", exist_ok=True)
    df_lb.to_csv(f"results/v2_commercial_leaderboard_{price_area}.csv", index=True)

    # 5. Render Enhanced 4-Panel 96-Quarter Backtesting Dashboard
    plot_enhanced_4panel_dashboard(all_models, trading_summaries, price_area)

    return df_lb


def plot_enhanced_4panel_dashboard(all_models, trading_summaries, price_area):
    """Plots the Enhanced 4-Panel 96-Quarter Diagnostic & Trading Dashboard."""
    best_item = max(trading_summaries, key=lambda x: x["summary"]["net_profit"])
    best_summary = best_item["summary"]
    trade_log = best_summary["trade_log"]
    quarters = trade_log["quarter"]

    fig, axes = plt.subplots(4, 1, figsize=(15, 16), gridspec_kw={'height_ratios': [2.2, 1.4, 1.4, 1.3]})
    plt.subplots_adjust(hspace=0.38)

    # -------------------------------------------------------------------------
    # PANEL 1: Multi-Model 96-Quarter Trajectory Comparison vs Spot & Actual
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(quarters, trade_log["spot_price"], label="Day-Ahead Spot Baseline (EUR/MWh)", color="#2563eb", linewidth=2.2, alpha=0.9)
    ax1.plot(quarters, trade_log["actual_imbalance"], label="Actual Imbalance Settlement Price (EUR/MWh)", color="#10b981", linewidth=2.4)

    # Plot top model predictions
    colors = ["#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"]
    for idx, item in enumerate(trading_summaries[:4]):
        pred_line = trade_log["spot_price"] + item["pred_spread"][:len(quarters)]
        ax1.plot(quarters, pred_line, label=f"{item['model_name']} (Spread MAE: €{item['mae']:.1f})", color=colors[idx % len(colors)], linestyle="--", linewidth=1.7, alpha=0.85)

    ax1.set_title(f"PANEL 1: 96-Quarter Day-Ahead Imbalance Forecasts vs. Actual Market Clearing ({price_area})", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Price (EUR/MWh)")
    ax1.set_xlim(1, 96)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", frameon=True, ncol=2)

    # -------------------------------------------------------------------------
    # PANEL 2: Buy & Sell Action Map & Execution Points
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(quarters, trade_log["actual_spread"], label="Actual Spread (Imbalance - Spot)", color="#64748b", linewidth=1.5, alpha=0.7)
    ax2.axhline(0, color="black", linestyle="-", alpha=0.5)

    longs = trade_log[trade_log["action"] == "LONG_SPOT"]
    shorts = trade_log[trade_log["action"] == "SHORT_SPOT"]

    if not longs.empty:
        ax2.scatter(longs["quarter"], longs["actual_spread"], marker="^", color="#10b981", s=100, label=f"BUY Spot / LONG ({len(longs)} trades)", zorder=5, edgecolor="black")
    if not shorts.empty:
        ax2.scatter(shorts["quarter"], shorts["actual_spread"], marker="v", color="#ef4444", s=100, label=f"SELL Spot / SHORT ({len(shorts)} trades)", zorder=5, edgecolor="black")

    ax2.set_title(f"PANEL 2: Commercial Agent Arbitrage Actions (BUY Low / SELL High) — {best_item['model_name']}", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Spread (EUR/MWh)")
    ax2.set_xlim(1, 96)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", frameon=True)

    # -------------------------------------------------------------------------
    # PANEL 3: Intraday Cumulative Equity Curves (€) across All Models
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    for idx, item in enumerate(trading_summaries):
        s = item["summary"]
        ax3.plot(range(len(s["capital_curve"])), s["capital_curve"], label=f"{item['model_name']} (ROC: {s['return_pct']:+.2f}%)", linewidth=2.0)

    ax3.axhline(best_summary["initial_capital"], color="black", linestyle=":", alpha=0.7, label="Initial Capital (€100k)")
    ax3.set_title("PANEL 3: Cumulative Intraday Portfolio Capital Curve from Q1 to Q96 (€)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Portfolio Value (€)")
    ax3.set_xlim(0, 96)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", frameon=True, ncol=2)

    # -------------------------------------------------------------------------
    # PANEL 4: Danish Cost, Fee & Tax Waterfall Analysis
    # -------------------------------------------------------------------------
    ax4 = axes[3]
    categories = ["Gross PnL", "Exchange Fees", "TSO Tariffs", "Slippage", "Danish Tax (22%)", "NET PROFIT"]
    values = [
        best_summary["gross_pnl"],
        -(best_summary["fees_paid"] * (0.06 / 0.26)),
        -(best_summary["fees_paid"] * (0.20 / 0.26)),
        -best_summary["slippage_paid"],
        -best_summary["tax_paid"],
        best_summary["net_profit"]
    ]
    bar_colors = ["#10b981", "#ef4444", "#f59e0b", "#6366f1", "#8b5cf6", "#2563eb"]

    bars = ax4.bar(categories, values, color=bar_colors, width=0.55, edgecolor="black", linewidth=1.1)
    for bar, val in zip(bars, values):
        h = bar.get_height()
        y_pos = h if h >= 0 else h - (abs(h) * 0.20)
        ax4.text(bar.get_x() + bar.get_width()/2.0, y_pos, f"€{val:,.2f}", ha='center', va='bottom' if h < 0 else 'bottom', fontweight='bold', fontsize=9.5)

    ax4.axhline(0, color="black", linewidth=1.0)
    ax4.set_title(f"PANEL 4: Danish Market Cost & Tax Reconciliation for Winner: {best_item['model_name']}", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Amount (EUR)")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    chart_path = f"results/v2_commercial_backtest_{price_area}.png"
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved Enhanced 4-Panel Dashboard to: {chart_path}")


if __name__ == '__main__':
    run_tournament(price_area="DK1")
