# ==============================================================================
# benchmark_v2_vs_v3.py
# Side-by-Side Performance Comparison: V2 Baseline vs V3 Optimeering
# ==============================================================================

import requests
import json
import pandas as pd


def run_benchmark():
    models = ["Transformer-TFT", "Hierarchical-LGBM+XGB", "Transfer-LightGBM", "Pure15m-CatBoost"]
    areas = ["DK1", "DK2"]

    print("\n" + "=" * 95)
    print("           NUREX TRADING: V2 BASELINE (Port 5000) vs V3 OPTIMEERING (Port 5001)")
    print("                     SIDE-BY-SIDE REALIZED PERFORMANCE AUDIT")
    print("=" * 95)

    for area in areas:
        print(f"\n" + "-" * 95)
        print(f"  PRICE AREA: {area} (Committed Volume: 2.0 MWh / Capital: €100,000)")
        print("-" * 95)

        # 1. Day D (Live Market Settled Quarters)
        print(f"\n  [1] TODAY'S LIVE DELIVERED QUARTERS (Day D Settlement):")
        rows_live = []
        for m in models:
            try:
                r2 = requests.get(f"http://127.0.0.1:5000/api/model_trades_ledger?area={area}&model={m}&volume=2.0", timeout=5).json()
            except Exception:
                r2 = {}
            try:
                r3 = requests.get(f"http://127.0.0.1:5001/api/model_trades_ledger?area={area}&model={m}&volume=2.0&tab=live", timeout=5).json()
            except Exception:
                r3 = {}

            v2_pnl = r2.get("net_realized_profit_so_far", 0.0)
            v3_pnl = r3.get("net_realized_profit_so_far", 0.0)
            v2_trades = r2.get("trades_fraction_str", "--")
            v3_trades = r3.get("trades_fraction_str", "--")
            shield_count = len([t for t in r3.get("trades", []) if "Shield" in t.get("action", "")])

            pnl_diff = v3_pnl - v2_pnl
            diff_str = f"+EUR {pnl_diff:,.2f}" if pnl_diff >= 0 else f"-EUR {abs(pnl_diff):,.2f}"

            rows_live.append({
                "Model Architecture": m,
                "V2 Trades": v2_trades,
                "V2 Net PnL": f"EUR {v2_pnl:,.2f}",
                "V3 Trades": v3_trades,
                "V3 Net PnL": f"EUR {v3_pnl:,.2f}",
                "V3 Spike Shield Blocks": f"{shield_count} Qs",
                "V3 Alpha Delta": diff_str
            })

        df_l = pd.DataFrame(rows_live)
        print(df_l.to_string(index=False))

        # 2. Backtest (Full 96Q Benchmark)
        print(f"\n  [2] FULL 96-QUARTER BACKTEST BENCHMARK:")
        rows_bt = []
        for m in models:
            try:
                r3_bt = requests.get(f"http://127.0.0.1:5001/api/model_trades_ledger?area={area}&model={m}&volume=2.0&tab=backtest", timeout=5).json()
            except Exception:
                r3_bt = {}

            v3_pnl = r3_bt.get("net_realized_profit_so_far", 0.0)
            v3_trades = r3_bt.get("trades_fraction_str", "--")
            shield_count = len([t for t in r3_bt.get("trades", []) if "Shield" in t.get("action", "")])
            roc = r3_bt.get("live_roc_percent", 0.0)

            rows_bt.append({
                "Model Architecture": m,
                "Backtest Trades": v3_trades,
                "Gross PnL": f"EUR {r3_bt.get('gross_pnl_so_far', 0.0):,.2f}",
                "Fees (0.51/MWh)": f"EUR {r3_bt.get('fees_so_far', 0.0):,.2f}",
                "Net Profit (EUR)": f"EUR {v3_pnl:,.2f}",
                "Return on Capital": f"{roc:+.2f}%",
                "Spike Shield Blocks": f"{shield_count} Qs"
            })

        df_b = pd.DataFrame(rows_bt)
        print(df_b.to_string(index=False))

    print("\n" + "=" * 95 + "\n")


if __name__ == '__main__':
    run_benchmark()
