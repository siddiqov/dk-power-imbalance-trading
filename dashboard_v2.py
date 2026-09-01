# ==============================================================================
# dashboard_v2.py
# V2 Interactive Web Dashboard & Commercial Trading Simulator
#
# Features:
# - Dual Zone Switching: DK1 (West Denmark) & DK2 (East Denmark)
# - Interactive Commercial Leaderboard ranked by Net Monetary Profit
# - Full Danish Fee & Tax Accounting (TSO, Exchange, Slippage, 22% Tax)
# - 96-Quarter Day-Ahead Arbitrage Execution Chart & Cumulative Equity Curve
# - Run On-Demand Tournaments across the 3 Real-Data Paradigms
# ==============================================================================

import os
import sys
import json
import base64
from io import BytesIO
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify, request, send_from_directory

from src.data_ingestion_v2 import V2DataEngine
from run_v2_commercial_tournament import run_tournament

app = Flask(__name__)


@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory('results', filename)


@app.route('/static/results/<path:filename>')
def serve_static_results(filename):
    return send_from_directory('results', filename)


def load_leaderboard(price_area="DK1"):
    """Loads the commercial leaderboard CSV or generates a default if missing."""
    csv_path = f"results/v2_commercial_leaderboard_{price_area}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Drop index column if present
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        return df.to_dict(orient="records")
    return []


@app.route('/')
def index():
    price_area = request.args.get('area', 'DK1')
    initial_capital = float(request.args.get('capital', 100000.0))
    leaderboard = load_leaderboard(price_area)
    chart_exists = os.path.exists(f"results/v2_commercial_backtest_{price_area}.png")
    
    return render_template(
        'dashboard_v2.html',
        price_area=price_area,
        capital=initial_capital,
        leaderboard=leaderboard,
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
            "message": f"Tournament completed for {price_area}!",
            "leaderboard": leaderboard
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/leaderboard')
def api_leaderboard():
    price_area = request.args.get('area', 'DK1')
    leaderboard = load_leaderboard(price_area)
    return jsonify({"area": price_area, "leaderboard": leaderboard})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 80)
    print(f"  V2 COMMERCIAL TRADING SIMULATOR DASHBOARD RUNNING ON http://localhost:{port}")
    print("=" * 80)
    app.run(host='0.0.0.0', port=port, debug=False)
