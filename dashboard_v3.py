# ==============================================================================
# dashboard_v3.py
# V3 Optimeering Commercial Dashboard Server (Port 5001 Default)
# Quantile Horizon Fan Charts, Direction Probabilities, Spike Shield Ledgers
# Dual-Market Support: Pure Day-Ahead (D-1) & Continuous Intraday (D-0)
# ==============================================================================

import os
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

from src.tournament_tables_v2 import TournamentTableGenerator
from src.commercial_strategy_v3 import V3CommercialStrategyEngine
from src.model_trainer_v3 import V3QuantileModelSuite
from src.visualization_v3 import plot_v3_optimeering_dashboard
from src.deep_analysis_engine import V3DeepAnalysisEngine
from src.day_ahead_auction_engine import DayAheadAuctionEngine

from zoneinfo import ZoneInfo

def get_danish_now():
    try:
        return datetime.now(ZoneInfo("Europe/Copenhagen"))
    except Exception:
        from datetime import timezone
        return datetime.now(timezone.utc) + timedelta(hours=2)


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/static/results/<path:filename>')
def serve_static_results(filename):
    return send_from_directory('results', filename)


@app.route('/download/technical_report')
def download_technical_report():
    pdf_path = "results/V3_Intraday_Technical_Report.pdf"
    if not os.path.exists(pdf_path):
        import generate_intraday_technical_report
        generate_intraday_technical_report.build_pdf(pdf_path)
    return send_from_directory('results', 'V3_Intraday_Technical_Report.pdf', as_attachment=True)


@app.route('/download/dispatch_workflow_guide')
def download_dispatch_workflow_guide():
    pdf_path = "results/Intraday_2Hour_Dispatch_Workflow_Guide.pdf"
    if not os.path.exists(pdf_path):
        import generate_intraday_technical_report
        generate_intraday_technical_report.build_2hour_dispatch_workflow_pdf(pdf_path)
    return send_from_directory('results', 'Intraday_2Hour_Dispatch_Workflow_Guide.pdf', as_attachment=True)


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
    initial_capital = float(request.args.get('capital', 100000.0))
    trade_volume = float(request.args.get('volume', 2.0))
    active_tab = request.args.get('tab', 'live')
    raw_mode = request.args.get('mode', 'day_ahead')  # 'day_ahead' or 'intraday'
    market_mode = "DAY_AHEAD_D1" if raw_mode == 'day_ahead' else "INTRADAY_D0"
    
    dk_now = get_danish_now()
    default_date = (dk_now - timedelta(days=1)).strftime("%Y-%m-%d")
    selected_date = request.args.get('date', default_date)
    today_date_str = dk_now.strftime("%Y-%m-%d")

    # Load 96-quarter genuine table from Energi Data Service
    table_gen = TournamentTableGenerator(price_area=price_area)
    if active_tab == 'backtest':
        target_df = table_gen.get_backtest_table(date_str=selected_date)
    else:
        target_df = table_gen.get_future_table(date_str=today_date_str)

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
    tomorrow_dt = dk_now + timedelta(days=1)
    tomorrow_str = tomorrow_dt.strftime("%Y-%m-%d")
    tomorrow_display = tomorrow_dt.strftime("%d %B %Y")
    today_str = dk_now.strftime("%d %B %Y")

    return render_template(
        'dashboard_v3.html',
        price_area=price_area,
        capital=initial_capital,
        trade_volume=trade_volume,
        active_tab=active_tab,
        raw_mode=raw_mode,
        market_mode=market_mode,
        selected_date=selected_date,
        tomorrow_str=tomorrow_str,
        tomorrow_display=tomorrow_display,
        leaderboard=leaderboard,
        top_model=top_model,
        summary=ledger_summary,
        chart_filename=chart_filename,
        chart_exists=chart_exists,
        today_str=today_str,
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
    
    dk_now = get_danish_now()
    default_date = (dk_now - timedelta(days=1)).strftime("%Y-%m-%d")
    selected_date = request.args.get('date', default_date)
    today_date_str = dk_now.strftime("%Y-%m-%d")

    table_gen = TournamentTableGenerator(price_area=price_area)
    target_df = table_gen.get_future_table(date_str=today_date_str) if tab == 'live' else table_gen.get_backtest_table(date_str=selected_date)

    strategy = V3CommercialStrategyEngine(price_area=price_area, capital=capital, base_volume_mwh=trade_volume)
    summary = strategy.evaluate_trading_ledger(target_df, model_name=model_name, market_mode=market_mode)
    return jsonify(summary)


@app.route('/api/dispatch_batch_1')
def api_dispatch_batch_1():
    """Generates the Q1 to Q9 (00:00 to 02:15 CET) Intraday Dispatch Batch for Google Docs/Sheets."""
    price_area = request.args.get('area', 'DK1')
    model_name = request.args.get('model', 'Transformer-TFT')
    if ' ' in model_name:
        model_name = model_name.replace(' ', '+')
    capital = float(request.args.get('capital', 100000.0))
    trade_volume = float(request.args.get('volume', 2.0))
    tab = request.args.get('tab', 'live')
    num_quarters = int(request.args.get('num_quarters', 9))

    dk_now = get_danish_now()
    today_date_str = dk_now.strftime("%Y-%m-%d")
    tomorrow_str = (dk_now + timedelta(days=1)).strftime("%Y-%m-%d")
    table_gen = TournamentTableGenerator(price_area=price_area)

    req_date = request.args.get('date')
    if req_date:
        target_delivery_date = req_date
    elif tab == 'live':
        target_delivery_date = tomorrow_str
    else:
        target_delivery_date = (dk_now - timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        target_df = table_gen.generate_and_save_future_table(date_str=target_delivery_date)
    except Exception as e:
        print(f"[Dispatch Batch 1] Future table generation fallback: {e}")
        target_df = table_gen.get_backtest_table(date_str=target_delivery_date)

    if target_df.empty:
        target_df = table_gen.get_future_table()
        
    strategy = V3CommercialStrategyEngine(price_area=price_area, capital=capital, base_volume_mwh=trade_volume)
    summary = strategy.evaluate_trading_ledger(target_df, model_name=model_name, market_mode="INTRADAY_D0")
    
    trades = summary.get("trades", [])[:num_quarters]
    
    # Build formatted tab-delimited text for 1-click clipboard paste into Google Docs / Sheets
    headers = [
        "Quarter", "Delivery Time (CET)", "Spot Price (€/MWh)", "Pred Imbalance (€/MWh)", 
        "Pred Spread (€/MWh)", "Quantiles (q10-q90)", "P(Up) / P(Dn)", "Recommended Action", 
        "Volume (MW)", "Status"
    ]
    tsv_lines = ["\t".join(headers)]
    csv_lines = [",".join([f'"{h}"' for h in headers])]
    
    formatted_trades = []
    for t in trades:
        action_clean = t['action'].replace(' (Long)', '').replace(' (Short)', '')
        q_spread = f"€{t['q10_price_eur']:.1f} - €{t['q90_price_eur']:.1f}"
        p_ratio = f"{t['p_up']}% Up / {t['p_down']}% Dn"
        
        row_tsv = [
            t['quarter'],
            t['time_dk'],
            f"€{t['spot_price_eur']:.2f}",
            f"€{t['pred_imbalance_eur']:.2f}",
            f"{'+' if t['pred_spread_eur']>=0 else ''}€{t['pred_spread_eur']:.2f}",
            q_spread,
            p_ratio,
            action_clean,
            f"{t['volume_mwh']:.1f} MW" if t['volume_mwh'] > 0 else "0.0 MW",
            "Batch 1 Dispatched"
        ]
        tsv_lines.append("\t".join(row_tsv))
        csv_lines.append(",".join([f'"{c}"' for c in row_tsv]))
        
        formatted_trades.append({
            "quarter": t['quarter'],
            "time_dk": t['time_dk'],
            "spot_price_eur": round(t['spot_price_eur'], 2),
            "pred_imbalance_eur": round(t['pred_imbalance_eur'], 2),
            "pred_spread_eur": round(t['pred_spread_eur'], 2),
            "q10_price_eur": round(t['q10_price_eur'], 2),
            "q90_price_eur": round(t['q90_price_eur'], 2),
            "p_up": t['p_up'],
            "p_down": t['p_down'],
            "action": action_clean,
            "volume_mwh": t['volume_mwh'],
            "status": "Batch 1 (00:00 - 02:15 CET)"
        })
        
    tsv_payload = "\n".join(tsv_lines)
    csv_payload = "\n".join(csv_lines)
    actual_delivery_date = formatted_trades[0]['time_dk'][:10] if formatted_trades else target_delivery_date
    
    return jsonify({
        "success": True,
        "price_area": price_area,
        "model_name": model_name,
        "date": actual_delivery_date,
        "delivery_window": "00:00 - 02:15 CET (Q1-Q9)",
        "gate_closure_cutoff": "D-1 21:45 CET",
        "num_quarters": len(formatted_trades),
        "trades": formatted_trades,
        "tsv_payload": tsv_payload,
        "csv_payload": csv_payload
    })


@app.route('/trade_ledger_v3')
def trade_ledger_v3():
    model_name = request.args.get('model', 'Transformer-TFT')
    price_area = request.args.get('area', 'DK1')
    capital = float(request.args.get('capital', 100000.0))
    trade_volume = float(request.args.get('volume', 2.0))
    tab = request.args.get('tab', 'live')
    raw_mode = request.args.get('mode', 'day_ahead')
    dk_now = get_danish_now()
    default_date = (dk_now - timedelta(days=1)).strftime("%Y-%m-%d")
    selected_date = request.args.get('date', default_date)

    return render_template(
        'trade_ledger_v3.html',
        model_name=model_name,
        price_area=price_area,
        capital=capital,
        trade_volume=trade_volume,
        active_tab=tab,
        raw_mode=raw_mode,
        selected_date=selected_date,
        today_str=dk_now.strftime("%d %B %Y")
    )


@app.route('/api/deep_analysis_data')
def api_deep_analysis_data():
    price_area = request.args.get('area', 'DK1')
    capital = float(request.args.get('capital', 100000.0))
    trade_volume = float(request.args.get('volume', 2.0))
    tab = request.args.get('tab', 'live')
    raw_mode = request.args.get('mode', 'day_ahead')
    market_mode = "DAY_AHEAD_D1" if raw_mode == 'day_ahead' else "INTRADAY_D0"
    dk_now = get_danish_now()
    default_date = (dk_now - timedelta(days=1)).strftime("%Y-%m-%d")
    selected_date = request.args.get('date', default_date)

    engine = V3DeepAnalysisEngine(price_area=price_area, capital=capital, base_volume_mwh=trade_volume)
    diagnostics = engine.compute_full_diagnostics(active_tab=tab, market_mode=market_mode, selected_date=selected_date)
    return jsonify(diagnostics)


@app.route('/deep_analysis')
def deep_analysis():
    price_area = request.args.get('area', 'DK1')
    capital = float(request.args.get('capital', 100000.0))
    trade_volume = float(request.args.get('volume', 2.0))
    tab = request.args.get('tab', 'live')
    raw_mode = request.args.get('mode', 'day_ahead')
    market_mode = "DAY_AHEAD_D1" if raw_mode == 'day_ahead' else "INTRADAY_D0"
    dk_now = get_danish_now()
    default_date = (dk_now - timedelta(days=1)).strftime("%Y-%m-%d")
    selected_date = request.args.get('date', default_date)

    return render_template(
        'deep_analysis.html',
        price_area=price_area,
        capital=capital,
        trade_volume=trade_volume,
        active_tab=tab,
        raw_mode=raw_mode,
        market_mode=market_mode,
        selected_date=selected_date,
        today_str=dk_now.strftime("%d %B %Y")
    )


@app.route('/dispatch_batch_1')
def dispatch_batch_1_view():
    price_area = request.args.get('area', 'DK1')
    model_name = request.args.get('model', 'Transformer-TFT')
    capital = float(request.args.get('capital', 100000.0))
    trade_volume = float(request.args.get('volume', 2.0))
    tab = request.args.get('tab', 'live')
    raw_mode = request.args.get('mode', 'intraday')
    dk_now = get_danish_now()
    today_str = dk_now.strftime("%Y-%m-%d")
    selected_date = request.args.get('date', today_str if tab == 'live' else (dk_now - timedelta(days=1)).strftime("%Y-%m-%d"))
    num_quarters = int(request.args.get('num_quarters', 9))

    return render_template(
        'dispatch_batch_1.html',
        price_area=price_area,
        model_name=model_name,
        capital=capital,
        trade_volume=trade_volume,
        active_tab=tab,
        raw_mode=raw_mode,
        selected_date=selected_date,
        tomorrow_str=today_str,
        num_quarters=num_quarters
    )


@app.route('/day_ahead_auction')
def day_ahead_auction_view():
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
    port = int(os.environ.get('PORT', 5001))
    print("\n" + "=" * 80)
    print(f"  V3 OPTIMEERING COMMERCIAL TRADING SIMULATOR RUNNING ON http://127.0.0.1:{port}")
    print("=" * 80, flush=True)
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
