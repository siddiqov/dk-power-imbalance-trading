# ==============================================================================
# dashboard_v2.py
# V2 Interactive Web Dashboard & Real-Time Commercial Trading Simulator
#
# Features:
# - Dual Zone Switching: DK1 (West Denmark) & DK2 (East Denmark)
# - Dual-Tab 96-Quarter Tournament Views:
#   * Tab 1: 96-Quarter Backtesting Tournament (31st August / Day D-1 Test Set)
#   * Tab 2: 96-Quarter Future Day-Ahead Forecasts (Tomorrow / Day D Pending)
# - Full Danish Fee & Tax Accounting (TSO, Exchange, Slippage, 22% Tax)
# - Enhanced 4-Panel 96-Quarter Visualization Dashboard
# ==============================================================================

import os
import sys
import json
import base64
from io import BytesIO
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify, request, send_from_directory

from src.data_ingestion_v2 import V2DataEngine
from run_v2_commercial_tournament import run_tournament
from src.tournament_tables_v2 import TournamentTableGenerator
from src.live_intraday_ledger import LiveIntradayLedger

app = Flask(__name__)


@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory('results', filename)


@app.route('/static/results/<path:filename>')
def serve_static_results(filename):
    return send_from_directory('results', filename)


def load_leaderboard(price_area="DK1", capital=100000.0, trade_volume_mwh=2.0, mode='live'):
    """Loads the real-time occurred quarters leaderboard or historical backtest CSV."""
    if mode == 'live':
        try:
            ledger_calc = LiveIntradayLedger(price_area=price_area, capital=capital, trade_volume_mwh=trade_volume_mwh)
            live_lb = ledger_calc.get_live_today_leaderboard()
            if live_lb:
                return live_lb
        except Exception as e:
            print(f"  [WARN] Live leaderboard calculation fallback: {e}")

    csv_path = f"results/v2_commercial_leaderboard_{price_area}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        return df.to_dict(orient="records")
    return []


from src.visualization_v2 import generate_live_dashboard_plot, plot_enhanced_4panel_dashboard
import time

@app.route('/')
def index():
    price_area = request.args.get('area', 'DK1')
    initial_capital = float(request.args.get('capital', 100000.0))
    trade_volume_mwh = float(request.args.get('volume', 2.0))
    active_tab = request.args.get('tab', 'live')
    date_str = request.args.get('date', None)

    leaderboard = load_leaderboard(price_area, capital=initial_capital, trade_volume_mwh=trade_volume_mwh, mode='live')
    
    # Generate the live plot right before rendering
    live_plot_path = generate_live_dashboard_plot(price_area, capital=initial_capital)
    chart_exists = True if live_plot_path else os.path.exists(f"results/v2_commercial_backtest_{price_area}.png")
    
    top_model = leaderboard[0] if leaderboard else {}

    # Retrieve cached 96-Quarter Tables (Instant Response)
    table_gen = TournamentTableGenerator(price_area=price_area)
    df_backtest = table_gen.get_backtest_table()
    df_future = table_gen.get_future_table()

    if not date_str:
        import sqlite3
        from src.db_manager import DB_PATH
        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute('SELECT MAX(date) FROM trade_ledger WHERE price_area = ?', (price_area,))
            row = cur.fetchone()
            if row and row[0]:
                date_str = row[0]
            conn.close()

    if date_str:
        print(f"DEBUG: date_str is {date_str}, price_area is {price_area}")
        import sqlite3
        import pandas as pd
        from src.db_manager import DB_PATH
        if os.path.exists(DB_PATH):
            print("DEBUG: DB_PATH exists")
            conn = sqlite3.connect(DB_PATH)
            df_hist = pd.read_sql('SELECT * FROM trade_ledger WHERE date = ? AND price_area = ?', conn, params=(date_str, price_area))
            conn.close()
            print(f"DEBUG: df_hist empty? {df_hist.empty}")
            if not df_hist.empty:
                # Map columns to what the HTML expects
                df_hist['time_dk'] = df_hist['time_dk']
                df_hist['deep_bilstm_eur'] = df_hist.get('deep_bilstm_eur', '--').fillna('--')
                df_hist['transformer_tft_eur'] = df_hist.get('transformer_tft_eur', '--').fillna('--')
                df_hist['transfer_lgb_eur'] = df_hist.get('transfer_lgb_eur', '--').fillna('--')
                df_hist['hierarchical_eur'] = df_hist.get('hierarchical_eur', '--').fillna('--')
                df_hist['pure15m_catboost_eur'] = df_hist.get('pure15m_catboost_eur', '--').fillna('--')
                df_hist['meta_ensemble_eur'] = df_hist['model_prediction_eur'].round(2)
                df_hist['meta_ensemble_dkk'] = (df_hist['model_prediction_eur'] * 7.45).round(2)
                df_hist['actual_settled_imbalance_eur'] = df_hist['actual_settled_eur'].round(2)
                # Safely calculate error spread handling NaNs
                df_hist['error_spread_eur'] = abs(df_hist['model_prediction_eur'] - df_hist['actual_settled_eur']).round(2)
                df_hist['error_spread_eur'] = df_hist['error_spread_eur'].fillna('--')
                df_hist['actual_settled_imbalance_eur'] = df_hist['actual_settled_imbalance_eur'].fillna('--')
                
                df_hist['agent_action'] = df_hist['action']
                df_hist['volume'] = df_hist['volume_mwh']
                if 'net_pnl_eur' in df_hist.columns:
                    df_hist['running_net_pnl_eur'] = df_hist['net_pnl_eur'].cumsum().round(2)
                backtest_records = df_hist.to_dict(orient="records")
                backtest_total_pnl = round(df_hist['net_pnl_eur'].sum(), 2) if 'net_pnl_eur' in df_hist.columns else None
                print(f"DEBUG: Success! Populated {len(backtest_records)} from DB.")
            else:
                print("DEBUG: df_hist is empty. Falling back to CSV.")
                backtest_records = df_backtest.to_dict(orient="records") if not df_backtest.empty else []
                backtest_total_pnl = None
        else:
            print("DEBUG: DB_PATH does not exist. Falling back to CSV.")
            backtest_records = df_backtest.to_dict(orient="records") if not df_backtest.empty else []
            backtest_total_pnl = None
    else:
        print("DEBUG: No date_str. Falling back to CSV.")
        backtest_records = df_backtest.to_dict(orient="records") if not df_backtest.empty else []
        backtest_total_pnl = None

    future_records = df_future.to_dict(orient="records") if not df_future.empty else []
    today_str = datetime.now().strftime("%d %B %Y")
    ts = int(time.time())

    return render_template(
        'dashboard_v2.html',
        price_area=price_area,
        capital=initial_capital,
        trade_volume=trade_volume_mwh,
        active_tab=active_tab,
        leaderboard=leaderboard,
        top_model=top_model,
        backtest_records=backtest_records,
        backtest_total_pnl=backtest_total_pnl,
        future_records=future_records,
        chart_exists=chart_exists,
        today_str=today_str,
        ts=ts,
        date_str=date_str
    )

@app.route('/api/run_historical_backtest', methods=['POST'])
def api_run_historical_backtest():
    try:
        data = request.get_json() or {}
        price_area = data.get('area', 'DK1')
        date_str = data.get('date')
        if not date_str:
            return jsonify({"status": "error", "message": "Date is required"}), 400
            
        capital = float(data.get('capital', 100000.0))
        trade_volume_mwh = float(data.get('volume', 2.0))
        
        from src.db_manager import fetch_daily_ledger
        from src.visualization_v2 import plot_enhanced_4panel_dashboard
        
        trading_summaries = fetch_daily_ledger(date_str, price_area)
        
        if not trading_summaries:
            return jsonify({"status": "error", "message": f"No archived trades found for {date_str} in {price_area}."})
            
        # Ensure correct capital base
        # Since DB stores relative PnL, we recreate capital curve
        for s in trading_summaries:
            diff = capital - 100000.0
            s["summary"]["capital_curve"] = [x + diff for x in s["summary"]["capital_curve"]]
            s["summary"]["initial_capital"] = capital
            
        plot_enhanced_4panel_dashboard(trading_summaries, price_area, mode='backtest', date_str=date_str)
        
        return jsonify({"status": "success", "message": f"Historical Backtest loaded for {date_str}!"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/run_tournament', methods=['POST'])
def api_run_tournament():
    try:
        data = request.get_json() or {}
        price_area = data.get('area', 'DK1')
        capital = float(data.get('capital', 100000.0))
        trade_volume_mwh = float(data.get('volume', 2.0))

        df_lb = run_tournament(price_area=price_area, initial_capital=capital)
        leaderboard = load_leaderboard(price_area, capital=capital, trade_volume_mwh=trade_volume_mwh, mode='live')
        return jsonify({
            "status": "success",
            "message": f"Full 4-Paradigm Tournament completed for {price_area}!",
            "leaderboard": leaderboard
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/model_trades_ledger')
def api_model_trades_ledger():
    """Returns quarter-by-quarter financial trade audit ledger for an occurred/active model."""
    price_area = request.args.get('area', 'DK1')
    model_name = request.args.get('model', 'Transformer-TFT')
    capital = float(request.args.get('capital', 100000.0))
    trade_volume_mwh = float(request.args.get('volume', 2.0))

    from src.live_intraday_ledger import LiveIntradayLedger
    ledger_calc = LiveIntradayLedger(price_area=price_area, capital=capital, trade_volume_mwh=trade_volume_mwh)
    summary = ledger_calc.get_live_today_ledger(model_name=model_name)
    return jsonify(summary)


@app.route('/api/export_csv')
def api_export_csv():
    """Exports either the 96-quarter backtest or future forecast table to CSV."""
    price_area = request.args.get('area', 'DK1')
    table_type = request.args.get('type', 'backtest')
    table_gen = TournamentTableGenerator(price_area=price_area)

    if table_type == 'backtest':
        df = table_gen.get_backtest_table()
        filename = f"96Q_backtest_table_{price_area}.csv"
    else:
        df = table_gen.get_future_table()
        filename = f"96Q_future_table_{price_area}.csv"

    return send_from_directory("results", filename, as_attachment=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 80)
    print(f"  V2 COMMERCIAL TRADING SIMULATOR RUNNING ON http://127.0.0.1:{port}")
    print("=" * 80)
    app.run(host='0.0.0.0', port=port, debug=False)
