# ==============================================================================
# dashboard_v2_1.py
# V2.1 Commercial Day-Ahead Auction & Bidding Dashboard (Port 5002 Default)
#
# Dedicated server for Day-Ahead Pre-Auction 96-Quarter Bidding Orders (11:00 CET)
# Zero synthetic data: 100% genuine Energi Data Service & Nord Pool datasets.
# ==============================================================================

import os
import sys
import json
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.day_ahead_auction_engine import DayAheadAuctionEngine
from src.tournament_tables_v2 import TournamentTableGenerator
from src.commercial_strategy_v3 import V3CommercialStrategyEngine

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

def get_danish_now():
    try:
        return datetime.now(ZoneInfo("Europe/Copenhagen"))
    except Exception:
        from datetime import timezone
        return datetime.now(timezone.utc) + timedelta(hours=2)


@app.route('/static/results/<path:filename>')
@app.route('/results/<path:filename>')
def serve_static_results(filename):
    return send_from_directory('results', filename)


@app.route('/download/technical_report')
def download_technical_report():
    pdf_path = "results/V3_Intraday_Technical_Report.pdf"
    if not os.path.exists(pdf_path):
        import generate_intraday_technical_report
        generate_intraday_technical_report.build_pdf(pdf_path)
    return send_from_directory('results', 'V3_Intraday_Technical_Report.pdf', as_attachment=True)


@app.route('/download/decision_feature_guide')
def download_decision_feature_guide():
    pdf_path = "results/V3_Decision_Hierarchy_and_14D_Features_Guide.pdf"
    if not os.path.exists(pdf_path):
        from generate_decision_feature_whitepaper import build_pdf
        build_pdf(pdf_path)
    return send_from_directory('results', 'V3_Decision_Hierarchy_and_14D_Features_Guide.pdf', as_attachment=True)


@app.route('/')
def index():
    price_area = request.args.get('area', 'DK1')
    volume = float(request.args.get('volume', 2.0))
    capital = float(request.args.get('capital', 100000.0))
    date_str = request.args.get('date', None)

    engine = DayAheadAuctionEngine(price_area=price_area, capital=capital, base_volume_mwh=volume)
    res = engine.generate_fixed_auction_bids(delivery_date=date_str)

    return render_template(
        'day_ahead_auction.html',
        price_area=price_area,
        base_volume=volume,
        capital=capital,
        delivery_date=res["delivery_date"],
        summary=res["summary_metrics"],
        bids=res["bids"],
        tsv_payload=res["tsv_payload"],
        csv_payload=res["csv_payload"]
    )


@app.route('/day_ahead_auction')
def day_ahead_auction_view():
    return index()


@app.route('/api/day_ahead_auction')
def api_day_ahead_auction():
    price_area = request.args.get('area', 'DK1')
    volume = float(request.args.get('volume', 2.0))
    capital = float(request.args.get('capital', 100000.0))
    date_str = request.args.get('date', None)

    engine = DayAheadAuctionEngine(price_area=price_area, capital=capital, base_volume_mwh=volume)
    res = engine.generate_fixed_auction_bids(delivery_date=date_str)
    return jsonify(res)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    print("\n" + "=" * 80)
    print(f"  V2.1 DAY-AHEAD AUCTION COMMERCIAL SERVER RUNNING ON http://127.0.0.1:{port}")
    print("=" * 80, flush=True)
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
