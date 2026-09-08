# ==============================================================================
# src/visualization_v3.py
# V3 Advanced Commercial Visualizer: Optimeering Quantile Horizon Fan Charts &
# Tri-State Direction Probability Stacks
# ==============================================================================

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_v3_optimeering_dashboard(df_day_d, preds, ledger_summary, price_area='DK1', date_str=None, is_backtest=False):
    """
    Generates the V3 Optimeering Commercial Dashboard Chart:
    - Panel 1: Quantile Horizon Fan Chart (q10 to q90 cloud) vs Day-Ahead Spot & Actual
    - Panel 2: Direction Probabilities (P(Up), P(Down), P(Balanced))
    - Panel 3: Trading Decisions & Spike Shield Executions
    - Panel 4: Cumulative Portfolio Capital Curve (€)
    """
    n_q = len(df_day_d)
    quarters = np.arange(1, n_q + 1)

    spot_arr = df_day_d["spot_price_eur"].values
    quantiles = preds["quantiles"]
    probs = preds["probabilities"]

    q10_price = quantiles["q10_price"][:n_q]
    q50_price = quantiles["q50_price"][:n_q]
    q90_price = quantiles["q90_price"][:n_q]

    fig, axes = plt.subplots(4, 1, figsize=(16, 17), gridspec_kw={'height_ratios': [2.4, 1.3, 1.4, 1.4]})
    plt.subplots_adjust(hspace=0.38)

    title_date = f" ({date_str})" if date_str else ""

    # -------------------------------------------------------------------------
    # PANEL 1: Optimeering Quantile Horizon Fan Chart (Confidence Cloud)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.fill_between(quarters, q10_price, q90_price, color="#38bdf8", alpha=0.25, label="80% Quantile Horizon Band (q10 - q90)")
    ax1.plot(quarters, spot_arr, label="Day-Ahead Spot Baseline (EUR/MWh)", color="#2563eb", linewidth=2.4, zorder=4)
    ax1.plot(quarters, q50_price, label="Median Imbalance Forecast q50 (EUR/MWh)", color="#f59e0b", linestyle="--", linewidth=2.0, zorder=5)

    # If actual imbalance prices are available, plot them
    actual_vals = np.full(n_q, np.nan)
    for i, row in df_day_d.iterrows():
        raw_val = row.get("actual_settled_imbalance_eur") or row.get("actual_imbalance_eur") or row.get("imbalance_price_eur")
        if pd.notnull(raw_val) and str(raw_val).strip() not in ["--", "None", "nan", ""]:
            try:
                actual_vals[i] = float(str(raw_val).replace("€", "").replace("EUR", "").replace(",", "").strip())
            except Exception:
                pass

    valid_idx = ~np.isnan(actual_vals)
    if np.any(valid_idx):
        ax1.plot(quarters[valid_idx], actual_vals[valid_idx], label="Actual Energinet Settlement (EUR/MWh)", color="#10b981", linewidth=2.6, marker='o', markersize=3, zorder=6)

    mode_label = "BACKTEST" if is_backtest else "LIVE DAY D"
    ax1.set_title(f"PANEL 1: OPTIMEERING QUANTILE HORIZON & SPREAD ENVELOPE [{mode_label}] ({price_area}){title_date}", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Price (EUR/MWh)", fontweight="bold")
    ax1.set_xlim(1, 96)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", frameon=True, ncol=2)

    # -------------------------------------------------------------------------
    # PANEL 2: Direction Probabilities Stack (P(Up), P(Down), P(Balanced))
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    p_up = probs["p_up"][:n_q] * 100.0
    p_down = probs["p_down"][:n_q] * 100.0
    p_bal = probs["p_balanced"][:n_q] * 100.0

    ax2.bar(quarters, p_up, label="P(Up-Regulation)", color="#10b981", width=0.85, alpha=0.85)
    ax2.bar(quarters, p_down, bottom=p_up, label="P(Down-Regulation)", color="#ef4444", width=0.85, alpha=0.85)
    ax2.bar(quarters, p_bal, bottom=p_up+p_down, label="P(Balanced)", color="#64748b", width=0.85, alpha=0.6)

    ax2.set_title("PANEL 2: 96-Quarter Imbalance Direction Probabilities (Tri-State Softmax)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Probability (%)", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_xlim(1, 96)
    ax2.grid(True, alpha=0.2, axis="y")
    ax2.legend(loc="upper right", frameon=True, ncol=3)

    # -------------------------------------------------------------------------
    # PANEL 3: Trading Decisions & Asymmetric Spike Shield Shielding
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    trades = ledger_summary.get("trades", [])
    if trades:
        long_q = [i+1 for i, t in enumerate(trades) if "BUY" in t["action"]]
        short_q = [i+1 for i, t in enumerate(trades) if "SELL" in t["action"]]
        shield_q = [i+1 for i, t in enumerate(trades) if "Shield" in t["action"]]

        pred_spreads = [t["pred_spread_eur"] for t in trades]
        ax3.plot(quarters, pred_spreads, label="Predicted Spread (EUR/MWh)", color="#8b5cf6", linewidth=1.8, alpha=0.8)
        ax3.axhline(0, color="black", linestyle="-", alpha=0.4)

        if long_q:
            ax3.scatter(long_q, [pred_spreads[q-1] for q in long_q], marker="^", color="#10b981", s=110, label=f"BUY Spot / LONG ({len(long_q)} trades)", zorder=5, edgecolor="black")
        if short_q:
            ax3.scatter(short_q, [pred_spreads[q-1] for q in short_q], marker="v", color="#ef4444", s=110, label=f"SELL Spot / SHORT ({len(short_q)} trades)", zorder=5, edgecolor="black")
        if shield_q:
            ax3.scatter(shield_q, [pred_spreads[q-1] for q in shield_q], marker="s", color="#f59e0b", s=90, label=f"Spike Shield Protected ({len(shield_q)} blocks)", zorder=6, edgecolor="black")

    ax3.set_title("PANEL 3: Execution Map & Asymmetric Spike Shield Interventions", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Spread (EUR/MWh)", fontweight="bold")
    ax3.set_xlim(1, 96)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", frameon=True)

    # -------------------------------------------------------------------------
    # PANEL 4: Cumulative Portfolio Capital Curve (€)
    # -------------------------------------------------------------------------
    ax4 = axes[3]
    cap_curve = []
    c = ledger_summary.get("capital", 100000.0)
    for t in trades:
        run_cap_str = t.get("running_capital", "--")
        if run_cap_str != "--":
            val = float(str(run_cap_str).replace("€", "").replace(",", "").strip())
            cap_curve.append(val)
        else:
            cap_curve.append(c if not cap_curve else cap_curve[-1])

    if cap_curve:
        ax4.plot(quarters[:len(cap_curve)], cap_curve, color="#38bdf8", linewidth=2.5, label=f"V3 Net Portfolio Equity (ROC: {ledger_summary.get('live_roc_percent', 0.0):+.2f}%)")
        ax4.axhline(ledger_summary.get("capital", 100000.0), color="white", linestyle=":", alpha=0.7, label="Initial Capital (€100k)")

    ax4.set_title("PANEL 4: Real-Time Cumulative Portfolio Equity Curve from Q1 to Q96 (€)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Portfolio Value (€)", fontweight="bold")
    ax4.set_xlim(1, 96)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper left", frameon=True)

    os.makedirs("results", exist_ok=True)
    if is_backtest and date_str:
        save_path = f"results/v3_backtest_{price_area}_{date_str}.png"
    else:
        save_path = f"results/v3_optimeering_dashboard_{price_area}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()
    return save_path
