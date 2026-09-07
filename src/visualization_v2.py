import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_enhanced_4panel_dashboard(trading_summaries, price_area, mode='backtest', date_str=None):
    """
    Plots the Enhanced 4-Panel 96-Quarter Diagnostic & Trading Dashboard.
    Mode can be 'backtest' or 'live'.
    """
    if not trading_summaries:
        print("No summaries to plot.")
        return

    best_item = max(trading_summaries, key=lambda x: x["summary"]["net_profit"])
    best_summary = best_item["summary"]
    trade_log = best_summary["trade_log"]
    quarters = np.asarray(trade_log["quarter"])
    n_q = len(quarters)

    fig, axes = plt.subplots(4, 1, figsize=(15, 16), gridspec_kw={'height_ratios': [2.2, 1.4, 1.4, 1.3]})
    plt.subplots_adjust(hspace=0.38)

    title_date = f" for {date_str}" if date_str else ""
    title_prefix = "LIVE REAL-TIME EXECUTION" if mode == 'live' else "HISTORICAL BACKTEST"

    # -------------------------------------------------------------------------
    # PANEL 1: Multi-Model 96-Quarter Trajectory Comparison vs Spot & Actual
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    spot_arr = np.asarray(trade_log["spot_price"], dtype=float)[:n_q]
    imbalance_arr = np.asarray(trade_log["actual_imbalance"], dtype=float)[:n_q]
    ax1.plot(quarters, spot_arr, label="Day-Ahead Spot Baseline (EUR/MWh)", color="#2563eb", linewidth=2.2, alpha=0.9)
    ax1.plot(quarters, imbalance_arr, label="Actual Imbalance Settlement Price (EUR/MWh)", color="#10b981", linewidth=2.4)

    # Plot top model predictions
    colors = ["#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16", "#ef4444"]
    for idx, item in enumerate(trading_summaries[:4]):
        pred_spread_arr = np.asarray(item["pred_spread"], dtype=float)[:n_q]
        pred_line = spot_arr + pred_spread_arr
        ax1.plot(quarters, pred_line, label=f"{item['model_name']} Forecast", color=colors[idx % len(colors)], linestyle="--", linewidth=1.7, alpha=0.85)

    ax1.set_title(f"PANEL 1: {title_prefix} - 96-Quarter Forecasts vs. Actuals ({price_area}){title_date}", fontsize=12, fontweight="bold")
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

    ax2.set_title(f"PANEL 2: Commercial Arbitrage Actions — {best_item['model_name']}", fontsize=12, fontweight="bold")
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
        ax3.plot(range(1, len(s["capital_curve"]) + 1), s["capital_curve"], label=f"{item['model_name']} (ROC: {s['return_pct']:+.2f}%)", linewidth=2.0)

    ax3.axhline(best_summary["initial_capital"], color="black", linestyle=":", alpha=0.7, label=f"Initial Capital (€{int(best_summary['initial_capital']/1000)}k)")
    ax3.set_title("PANEL 3: Cumulative Intraday Portfolio Capital Curve from Q1 to Q96 (€)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Portfolio Value (€)")
    ax3.set_xlim(1, 96)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", frameon=True, ncol=2)

    # -------------------------------------------------------------------------
    # PANEL 4: Danish Cost, Fee & Tax Waterfall Analysis
    # -------------------------------------------------------------------------
    ax4 = axes[3]
    categories = ["Gross PnL", "Exchange Fees", "TSO Tariffs", "Slippage", "Danish Tax (22%)", "NET PROFIT"]
    values = [
        best_summary["gross_pnl"],
        -(best_summary["fees_paid"] * (0.06 / 0.26)) if best_summary["fees_paid"] > 0 else 0,
        -(best_summary["fees_paid"] * (0.20 / 0.26)) if best_summary["fees_paid"] > 0 else 0,
        -best_summary["slippage_paid"],
        -best_summary["tax_paid"],
        best_summary["net_profit"]
    ]
    colors_bar = ["#10b981" if v > 0 else "#ef4444" for v in values]
    colors_bar[-1] = "#2563eb" if values[-1] > 0 else "#ef4444"

    bars = ax4.bar(categories, values, color=colors_bar, width=0.6, edgecolor="black")
    ax4.axhline(0, color="black", linewidth=1.2)
    ax4.set_title(f"PANEL 4: Danish Market Cost & Tax Reconciliation for Winner: {best_item['model_name']}", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Amount (EUR)")
    ax4.grid(True, alpha=0.2, axis="y")

    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'€{height:,.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3 if height >= 0 else -12),
                     textcoords="offset points", ha='center', va='bottom', fontweight="bold", fontsize=9)

    os.makedirs("results", exist_ok=True)
    if mode == 'live':
        save_path = f"results/live_commercial_dashboard_{price_area}.png"
    else:
        save_path = f"results/backtest_{price_area}_{date_str}.png" if date_str else f"results/v2_commercial_backtest_{price_area}.png"
        
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()
    return save_path

def generate_live_dashboard_plot(price_area, capital=100000.0):
    """Generates the live rolling horizon plot directly from the LiveIntradayLedger."""
    from src.live_intraday_ledger import LiveIntradayLedger
    from datetime import datetime
    
    ledger = LiveIntradayLedger(price_area=price_area, capital=capital)
    leaderboard = ledger.get_live_today_leaderboard()
    
    if not leaderboard:
        return None
        
    # We need to construct trading_summaries from the ledger
    trading_summaries = []
    
    # We only have the leaderboard. We need the full log to plot.
    from src.tournament_tables_v2 import TournamentTableGenerator
    table_gen = TournamentTableGenerator(price_area=price_area)
    df_future = table_gen.get_future_table()
    if df_future.empty:
        return None
        
    # Build trade logs for each model
    model_col_map = {
        "Transformer-TFT": "transformer_tft_eur",
        "Transfer-LightGBM": "transfer_lgb_eur",
        "Hierarchical-LGBM+XGB": "hierarchical_eur",
        "Pure15m-CatBoost": "pure15m_catboost_eur",
        "Deep-BiLSTM": "deep_bilstm_eur",
        "Stacking-MetaEnsemble": "meta_ensemble_eur"
    }
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    for item in leaderboard:
        model_name = item["Model Architecture"]
        pred_col = model_col_map.get(model_name)
        if not pred_col: continue
        
        trade_log = []
        cap = capital
        gross_pnl = 0
        fees = 0
        
        for i, row in df_future.iterrows():
            q_num = i + 1
            t_str = str(row["time_dk"])
            
            spot = row["spot_price_eur"]
            pred_imb = row[pred_col]
            pred_spread = pred_imb - spot
            
            act_str = str(row.get("actual_settled_imbalance_eur", "--")).replace("€", "").replace(",", "").strip()
            try:
                actual_imb = float(act_str)
            except:
                actual_imb = np.nan
                
            actual_spread = actual_imb - spot if pd.notnull(actual_imb) else np.nan
            
            action = "HOLD"
            threshold = 1.20 # default
            if pred_spread > threshold:
                action = "LONG_SPOT"
            elif pred_spread < -threshold:
                action = "SHORT_SPOT"
                
            q_pnl = 0
            q_gross = 0
            q_fees = 0
            
            if pd.notnull(actual_imb) and action != "HOLD":
                if action == "LONG_SPOT":
                    q_gross = (actual_imb - spot) * 2.0
                else:
                    q_gross = (spot - actual_imb) * 2.0
                
                q_fees = 1.02
                q_pnl = q_gross - q_fees
                
            cap += q_pnl
            gross_pnl += q_gross
            fees += q_fees
            
            trade_log.append({
                "quarter": q_num,
                "spot_price": spot,
                "actual_imbalance": actual_imb,
                "actual_spread": actual_spread,
                "action": action
            })
            
        trade_df = pd.DataFrame(trade_log)
        
        # Calculate net profit and tax
        tax = 0
        net_profit = cap - capital
        if net_profit > 0:
            tax = net_profit * 0.22
            net_profit -= tax
            cap -= tax
            
        return_pct = (net_profit / capital) * 100
        
        # Build capital curve from trades
        live_ledger_summary = ledger.get_live_today_ledger(model_name=model_name)
        live_trades = live_ledger_summary.get("trades", [])
        capital_curve = [capital]
        for t in live_trades:
            if t.get("is_settled"):
                # "€ 100,123.45" -> float
                run_cap_str = t.get("running_capital", "0")
                if run_cap_str != "--":
                    clean_str = run_cap_str.replace("€", "").replace(",", "").strip()
                    try:
                        capital_curve.append(float(clean_str))
                    except:
                        capital_curve.append(capital_curve[-1])
        
        summary = {
            "initial_capital": capital,
            "trade_log": trade_df,
            "capital_curve": capital_curve,
            "gross_pnl": gross_pnl,
            "fees_paid": fees,
            "slippage_paid": 0,
            "tax_paid": tax,
            "net_profit": net_profit,
            "return_pct": return_pct
        }
        
        trading_summaries.append({
            "model_name": model_name,
            "pred_spread": df_future[pred_col] - df_future["spot_price_eur"],
            "summary": summary
        })
        
    return plot_enhanced_4panel_dashboard(trading_summaries, price_area, mode='live', date_str=today_str)
