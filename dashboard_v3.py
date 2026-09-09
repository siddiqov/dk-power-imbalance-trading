# ==============================================================================
# dashboard_v3.py
# V3 Optimeering Commercial Dashboard Server (Port 5001 Default)
# Quantile Horizon Fan Charts, Direction Probabilities, Spike Shield Ledgers
# Dual-Market Support: Pure Day-Ahead (D-1) & Continuous Intraday (D-0)
# ==============================================================================

import os
import sys
import time
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

from src.tournament_tables_v2 import TournamentTableGenerator
from src.commercial_strategy_v3 import V3CommercialStrategyEngine
from src.model_trainer_v3 import V3QuantileModelSuite
from src.visualization_v3 import plot_v3_optimeering_dashboard

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/static/results/<path:filename>')
def serve_static_results(filename):
    return send_from_directory('results', filename)


@app.route('/')
def index():
    price_area = request.args.get('area', 'DK1')
    initial_capital = float(request.args.get('capital', 100000.0))
    trade_volume = float(request.args.get('volume', 2.0))
    active_tab = request.args.get('tab', 'live')
    raw_mode = request.args.get('mode', 'day_ahead')  # 'day_ahead' or 'intraday'
    market_mode = "DAY_AHEAD_D1" if raw_mode == 'day_ahead' else "INTRADAY_D0"
    selected_date = request.args.get('date', '2026-08-31')

    # Load 96-quarter genuine table from Energi Data Service
    table_gen = TournamentTableGenerator(price_area=price_area)
    if active_tab == 'backtest':
        target_df = table_gen.get_backtest_table(date_str=selected_date)
    else:
        target_df = table_gen.get_future_table()

    # Run V3 Strategy Engine with selected Market Mode
    strategy = V3CommercialStrategyEngine(price_area=price_area, capital=initial_capital, base_volume_mwh=trade_volume)
    
    # Evaluate for default / winner model
    ledger_summary = strategy.evaluate_trading_ledger(target_df, model_name="Transformer-TFT", market_mode=market_mode)
    
    # Generate Optimeering Predictions & Chart
    preds = strategy.model_suite.predict_day_ahead_quantiles(target_df, market_mode=market_mode)
    chart_path = plot_v3_optimeering_dashboard(
        target_df, preds, ledger_summary, 
        price_area=price_area, 
        date_str=selected_date if active_tab == 'backtest' else None, 
        is_backtest=(active_tab == 'backtest')
    )
    chart_filename = os.path.basename(chart_path)
    chart_exists = os.path.exists(chart_path)

    # Build multi-model leaderboard for V3
    leaderboard = []
    models = ["Transformer-TFT", "Hierarchical-LGBM+XGB", "Transfer-LightGBM", "Pure15m-CatBoost", "Deep-BiLSTM", "Stacking-MetaEnsemble"]
    for m in models:
        s = strategy.evaluate_trading_ledger(target_df, model_name=m, market_mode=market_mode)
        settled = [t for t in s["trades"] if t["is_settled"]]
        active = [t for t in settled if "BUY" in t["action"] or "SELL" in t["action"]]
        win_trades = [t for t in active if t["net_pnl_eur"].startswith("+€")]
        win_rate = (len(win_trades) / len(active) * 100.0) if active else 0.0

        leaderboard.append({
            "Paradigm": "Optimeering V3",
            "Model Architecture": m,
            "Trades": s["trades_fraction_str"],
            "Win Rate": f"{win_rate:.1f}%",
            "Volume (MWh)": f"{len(active) * trade_volume:.1f}",
            "Gross PnL": f"{'+EUR' if s['gross_pnl_so_far'] >= 0 else '-EUR'} {abs(s['gross_pnl_so_far']):,.2f}",
            "Fees & Slip": f"EUR {s['fees_so_far']:,.2f}",
            "Danish Tax (22%)": f"EUR {s['tax_so_far']:,.2f}",
            "Net Realized Profit (EUR)": f"{'+EUR' if s['net_realized_profit_so_far'] >= 0 else '-EUR'} {abs(s['net_realized_profit_so_far']):,.2f}",
            "Daily Return": f"{s['live_roc_percent']:+.2f}%",
            "_net_profit": s["net_realized_profit_so_far"]
        })

    leaderboard.sort(key=lambda x: x["_net_profit"], reverse=True)
    for i, r in enumerate(leaderboard, 1):
        r["Rank"] = f"#{i}"

    top_model = leaderboard[0] if leaderboard else {}
    ts = int(time.time())

    return render_template(
        'dashboard_v3.html',
        price_area=price_area,
        capital=initial_capital,
        trade_volume=trade_volume,
        active_tab=active_tab,
        raw_mode=raw_mode,
        market_mode=market_mode,
        selected_date=selected_date,
        leaderboard=leaderboard,
        top_model=top_model,
        summary=ledger_summary,
        chart_filename=chart_filename,
        chart_exists=chart_exists,
        today_str=datetime.now().strftime("%d %B %Y"),
        ts=ts
    )


@app.route('/api/model_trades_ledger')
def api_model_trades_ledger():
    """Returns quarter-by-quarter financial ledger with Quantiles & Spike Shield statuses."""
    price_area = request.args.get('area', 'DK1')
    model_name = request.args.get('model', 'Transformer-TFT')
    if ' ' in model_name:
        model_name = model_name.replace(' ', '+')
    capital = float(request.args.get('capital', 100000.0))
    trade_volume = float(request.args.get('volume', 2.0))
    tab = request.args.get('tab', 'live')
    raw_mode = request.args.get('mode', 'day_ahead')
    market_mode = "DAY_AHEAD_D1" if raw_mode == 'day_ahead' else "INTRADAY_D0"
    selected_date = request.args.get('date', '2026-08-31')

    table_gen = TournamentTableGenerator(price_area=price_area)
    target_df = table_gen.get_future_table() if tab == 'live' else table_gen.get_backtest_table(date_str=selected_date)

    strategy = V3CommercialStrategyEngine(price_area=price_area, capital=capital, base_volume_mwh=trade_volume)
    summary = strategy.evaluate_trading_ledger(target_df, model_name=model_name, market_mode=market_mode)
    return jsonify(summary)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print("\n" + "=" * 80)
    print(f"  V3 OPTIMEERING COMMERCIAL TRADING SIMULATOR RUNNING ON http://127.0.0.1:{port}")
    print("=" * 80, flush=True)
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
