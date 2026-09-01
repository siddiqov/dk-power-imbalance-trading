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

app = Flask(__name__)


@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory('results', filename)


@app.route('/static/results/<path:filename>')
def serve_static_results(filename):
    return send_from_directory('results', filename)


def load_leaderboard(price_area="DK1"):
    """Loads the commercial leaderboard CSV."""
    csv_path = f"results/v2_commercial_leaderboard_{price_area}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        return df.to_dict(orient="records")
    return []


@app.route('/')
def index():
    price_area = request.args.get('area', 'DK1')
    initial_capital = float(request.args.get('capital', 100000.0))
    active_tab = request.args.get('tab', 'backtest')

    leaderboard = load_leaderboard(price_area)
    chart_exists = os.path.exists(f"results/v2_commercial_backtest_{price_area}.png")
    top_model = leaderboard[0] if leaderboard else {}

    # Retrieve cached 96-Quarter Tables (Instant Response)
    table_gen = TournamentTableGenerator(price_area=price_area)
    df_backtest = table_gen.get_backtest_table()
    df_future = table_gen.get_future_table()

    backtest_records = df_backtest.to_dict(orient="records") if not df_backtest.empty else []
    future_records = df_future.to_dict(orient="records") if not df_future.empty else []

    return render_template(
        'dashboard_v2.html',
        price_area=price_area,
        capital=initial_capital,
        active_tab=active_tab,
        leaderboard=leaderboard,
        top_model=top_model,
        backtest_records=backtest_records,
        future_records=future_records,
        chart_exists=chart_exists
    )


@app.route('/api/run_tournament', methods=['POST'])
def api_run_tournament():
    try:
        data = request.get_json() or {}
        price_area = data.get('area', 'DK1')
        capital = float(data.get('capital', 100000.0))

        df_lb = run_tournament(price_area=price_area, initial_capital=capital)
        leaderboard = load_leaderboard(price_area)
        return jsonify({
            "status": "success",
            "message": f"Full 4-Paradigm Tournament completed for {price_area}!",
            "leaderboard": leaderboard
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


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
