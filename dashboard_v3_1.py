# ==============================================================================
# dashboard_v3_1.py
# V3.1 Optimeering Commercial Dashboard Server (Port 5003 Default)
# Multi-Paradigm Tournament: Trees (Optuna) vs PyTorch Deep Sequence Networks
# Real-Time Financial Tournament Leaderboard, Quantile Fan Charts, Side-by-Side V3 vs V3.1 Comparison
# ==============================================================================

import os
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

from src.tournament_tables_v2 import TournamentTableGenerator
from src.commercial_strategy_v3_1 import V31CommercialStrategyEngine
from src.model_trainer_v3_1 import V31QuantileModelSuite
from src.tournament_engine_v3_1 import V31RealTimeTournamentEngine
from src.visualization_v3 import plot_v3_optimeering_dashboard
from src.deep_analysis_engine import V3DeepAnalysisEngine
from src.day_ahead_auction_engine import DayAheadAuctionEngine

# Import V3 strategy engine for side-by-side comparison
from src.commercial_strategy_v3 import V3CommercialStrategyEngine

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
    initial_capital = float(request.args.get('capital', 20000.0))
    trade_volume = float(request.args.get('volume', 2.0))
    profile = request.args.get('profile', 'tier2_standard')
    active_tab = request.args.get('tab', 'live')
    raw_mode = request.args.get('mode', 'day_ahead')  # 'day_ahead' or 'intraday'
    market_mode = "DAY_AHEAD_D1" if raw_mode == 'day_ahead' else "INTRADAY_D0"
    
    dk_now = get_danish_now()
    default_date = (dk_now - timedelta(days=1)).strftime("%Y-%m-%d")
    selected_date = request.args.get('date', default_date)
    today_date_str = dk_now.strftime("%Y-%m-%d")

    # Load 96-quarter table from Energi Data Service
    table_gen = TournamentTableGenerator(price_area=price_area)
    if active_tab == 'backtest':
        target_df = table_gen.get_backtest_table(date_str=selected_date)
    else:
        target_df = table_gen.get_future_table(date_str=today_date_str)

    # Run V3.1 Real-Time Financial Tournament across all 6 models
    tournament_engine = V31RealTimeTournamentEngine(price_area=price_area, initial_capital=initial_capital, profile=profile)
    tournament_date = selected_date if active_tab == 'backtest' else today_date_str
    tournament_res = tournament_engine.run_tournament(date_str=tournament_date)
    leaderboard = tournament_res.get("leaderboard", [])
    champion = tournament_res.get("champion", {})

    # Selected model (defaults to the tournament champion)
    default_model = champion.get("model_name", "Hierarchical-LGBM+XGB") if champion else "Hierarchical-LGBM+XGB"
    selected_model = request.args.get('model', default_model)
    if ' ' in selected_model:
        selected_model = selected_model.replace(' ', '+')

    # Run V3.1 Strategy Engine for the selected model
    strategy = V31CommercialStrategyEngine(price_area=price_area, capital=initial_capital, base_volume_mwh=trade_volume, profile=profile)
    ledger_summary = strategy.evaluate_trading_ledger(target_df, model_name=selected_model, market_mode=market_mode, profile=profile)

    # Generate Optimeering Predictions & Quantile Fan Chart
    preds = strategy.model_suite.predict_day_ahead_quantiles(target_df, market_mode=market_mode)
    chart_path = plot_v3_optimeering_dashboard(
        target_df, preds, ledger_summary, 
        price_area=price_area, 
        date_str=selected_date if active_tab == 'backtest' else None, 
        is_backtest=(active_tab == 'backtest')
    )
    chart_filename = os.path.basename(chart_path)
    chart_exists = os.path.exists(chart_path)

    ts = int(time.time())
    tomorrow_dt = dk_now + timedelta(days=1)
    tomorrow_str = tomorrow_dt.strftime("%Y-%m-%d")
    tomorrow_display = tomorrow_dt.strftime("%d %B %Y")
    today_str = dk_now.strftime("%d %B %Y")

    return render_template(
        'dashboard_v3_1.html',
        price_area=price_area,
        capital=initial_capital,
        trade_volume=trade_volume,
        profile=profile,
        active_tab=active_tab,
        raw_mode=raw_mode,
        market_mode=market_mode,
        selected_date=selected_date,
        tomorrow_str=tomorrow_str,
        tomorrow_display=tomorrow_display,
        leaderboard=leaderboard,
        champion=champion,
        selected_model=selected_model,
        summary=ledger_summary,
        chart_filename=chart_filename,
        chart_exists=chart_exists,
        today_str=today_str,
        ts=ts
    )


@app.route('/api/tournament_leaderboard')
def api_tournament_leaderboard():
    """Returns real-time tournament leaderboard JSON comparing all 6 models."""
    price_area = request.args.get('area', 'DK1')
    capital = float(request.args.get('capital', 20000.0))
    profile = request.args.get('profile', 'tier2_standard')
    date_str = request.args.get('date', None)

    engine = V31RealTimeTournamentEngine(price_area=price_area, initial_capital=capital, profile=profile)
    res = engine.run_tournament(date_str=date_str)
    return jsonify(res)


@app.route('/api/model_trades_ledger')
def api_model_trades_ledger():
    """Returns quarter-by-quarter financial ledger with Quantiles & Spike Shield statuses."""
    price_area = request.args.get('area', 'DK1')
    model_name = request.args.get('model', 'Hierarchical-LGBM+XGB')
    if ' ' in model_name:
        model_name = model_name.replace(' ', '+')
    capital = float(request.args.get('capital', 20000.0))
    trade_volume = float(request.args.get('volume', 2.0))
    profile = request.args.get('profile', 'tier2_standard')
    tab = request.args.get('tab', 'live')
    raw_mode = request.args.get('mode', 'day_ahead')
    market_mode = "DAY_AHEAD_D1" if raw_mode == 'day_ahead' else "INTRADAY_D0"
    
    dk_now = get_danish_now()
    default_date = (dk_now - timedelta(days=1)).strftime("%Y-%m-%d")
    selected_date = request.args.get('date', default_date)
    today_date_str = dk_now.strftime("%Y-%m-%d")

    table_gen = TournamentTableGenerator(price_area=price_area)
    target_df = table_gen.get_future_table(date_str=today_date_str) if tab == 'live' else table_gen.get_backtest_table(date_str=selected_date)

    strategy = V31CommercialStrategyEngine(price_area=price_area, capital=capital, base_volume_mwh=trade_volume, profile=profile)
    summary = strategy.evaluate_trading_ledger(target_df, model_name=model_name, market_mode=market_mode, profile=profile)
    return jsonify(summary)


@app.route('/compare_v3_vs_v3_1')
def compare_v3_vs_v3_1_view():
    """Side-by-side comparison page between V3 baseline and V3.1 tournament suite."""
    price_area = request.args.get('area', 'DK1')
    capital = float(request.args.get('capital', 20000.0))
    profile = request.args.get('profile', 'tier2_standard')
    dk_now = get_danish_now()
    default_date = (dk_now - timedelta(days=1)).strftime("%Y-%m-%d")
    selected_date = request.args.get('date', default_date)

    return render_template(
        'compare_v3_vs_v3_1.html',
        price_area=price_area,
        capital=capital,
        profile=profile,
        selected_date=selected_date,
        today_str=dk_now.strftime("%d %B %Y")
    )


@app.route('/api/compare_v3_vs_v3_1')
def api_compare_v3_vs_v3_1():
    """Calculates side-by-side financial and model metrics between V3 baseline and V3.1 suite."""
    price_area = request.args.get('area', 'DK1')
    capital = float(request.args.get('capital', 20000.0))
    profile = request.args.get('profile', 'tier2_standard')
    date_str = request.args.get('date', None)

    table_gen = TournamentTableGenerator(price_area=price_area)
    if date_str:
        target_df = table_gen.get_backtest_table(date_str=date_str)
    else:
        target_df = table_gen.get_backtest_table()
        if target_df.empty:
            target_df = table_gen.get_future_table()

    if target_df.empty:
        return jsonify({"success": False, "message": "No settlement data available"})

    # 1. Run V3 Baseline (Transformer-TFT default)
    strat_v3 = V3CommercialStrategyEngine(price_area=price_area, capital=capital, profile=profile)
    summary_v3 = strat_v3.evaluate_trading_ledger(target_df, model_name="Transformer-TFT", market_mode="INTRADAY_D0", profile=profile)

    # 2. Run V3.1 Real-Time Tournament Engine
    tournament_engine = V31RealTimeTournamentEngine(price_area=price_area, initial_capital=capital, profile=profile)
    tourn_res = tournament_engine.run_tournament(date_str=date_str)
    champ = tourn_res.get("champion", {})
    champ_model = champ.get("model_name", "Hierarchical-LGBM+XGB")

    strat_v3_1 = V31CommercialStrategyEngine(price_area=price_area, capital=capital, profile=profile)
    summary_v3_1 = strat_v3_1.evaluate_trading_ledger(target_df, model_name=champ_model, market_mode="INTRADAY_D0", profile=profile)

    # 3. Compute Deltas & Improvements
    pnl_v3 = float(summary_v3.get("net_realized_profit_so_far", summary_v3.get("net_pnl_eur", 0.0)))
    pnl_v3_1 = float(summary_v3_1.get("net_realized_profit_so_far", summary_v3_1.get("net_pnl_eur", 0.0)))
    delta_pnl = round(pnl_v3_1 - pnl_v3, 2)
    delta_pnl_pct = round((delta_pnl / abs(pnl_v3) * 100), 2) if abs(pnl_v3) > 0 else 0.0

    trades_v3 = summary_v3.get("trades", [])
    trades_v3_1 = summary_v3_1.get("trades", [])
    settled_v3 = [t for t in trades_v3 if t.get("is_settled")]
    settled_v3_1 = [t for t in trades_v3_1 if t.get("is_settled")]

    win_v3 = len([t for t in settled_v3 if str(t.get("net_pnl_eur", "")).startswith("+€") or (isinstance(t.get("net_pnl_eur"), (int, float)) and t.get("net_pnl_eur") > 0)])
    win_v3_1 = len([t for t in settled_v3_1 if str(t.get("net_pnl_eur", "")).startswith("+€") or (isinstance(t.get("net_pnl_eur"), (int, float)) and t.get("net_pnl_eur") > 0)])
    wr_v3 = round((win_v3 / len(settled_v3) * 100), 1) if settled_v3 else 0.0
    wr_v3_1 = round((win_v3_1 / len(settled_v3_1) * 100), 1) if settled_v3_1 else 0.0

    return jsonify({
        "success": True,
        "price_area": price_area,
        "date": summary_v3_1.get("delivery_date", date_str or "Live"),
        "profile": profile,
        "v3_baseline": {
            "version": "V3.0 (Heuristic Baseline)",
            "model_name": "Transformer-TFT (Baseline)",
            "hyperparameters": "Fixed defaults (depth=6, lr=0.05, leaves=31)",
            "architectures": "LightGBM, CatBoost, XGBoost, Torch TFT",
            "net_pnl_eur": round(pnl_v3, 2),
            "win_rate_pct": wr_v3,
            "roc_pct": summary_v3.get("live_roc_percent", summary_v3.get("roc_percent", 0.0)),
            "fees_eur": round(summary_v3.get("fees_so_far", summary_v3.get("fees_eur", 0.0)), 2),
            "tax_eur": round(summary_v3.get("tax_so_far", summary_v3.get("tax_eur", 0.0)), 2),
            "total_trades": len(trades_v3),
            "active_trades": summary_v3.get("active_occurred", summary_v3.get("active_trades", 0))
        },
        "v3_1_suite": {
            "version": "V3.1 (Optuna Bayesian + PyTorch Dual Suites)",
            "model_name": f"{champ_model} (Tournament Champion)",
            "hyperparameters": "Optuna Bayesian Optimization + TimeSeriesSplit CV",
            "architectures": "6 Models: 3 Trees + 2 PyTorch Deep Networks + 1 Stacking Blend",
            "net_pnl_eur": round(pnl_v3_1, 2),
            "win_rate_pct": wr_v3_1,
            "roc_pct": summary_v3_1.get("live_roc_percent", summary_v3_1.get("roc_percent", 0.0)),
            "fees_eur": round(summary_v3_1.get("fees_so_far", summary_v3_1.get("fees_eur", 0.0)), 2),
            "tax_eur": round(summary_v3_1.get("tax_so_far", summary_v3_1.get("tax_eur", 0.0)), 2),
            "total_trades": len(trades_v3_1),
            "active_trades": summary_v3_1.get("active_occurred", summary_v3_1.get("active_trades", 0))
        },
        "comparison_delta": {
            "delta_net_pnl_eur": delta_pnl,
            "delta_pnl_pct": delta_pnl_pct,
            "delta_win_rate_pct": round(wr_v3_1 - wr_v3, 1),
            "winner": "V3.1 Suite" if delta_pnl >= 0 else "V3.0 Baseline"
        },
        "tournament_leaderboard": tourn_res.get("leaderboard", [])
    })


@app.route('/dispatch_96quarter')
@app.route('/intraday_96quarter')
def dispatch_96quarter_view_v3_1():
    price_area = request.args.get('area', 'DK1')
    model_name = request.args.get('model', 'Hierarchical-LGBM+XGB')
    profile = request.args.get('profile', 'tier2_standard')
    date_str = request.args.get('date', None)
    capital = float(request.args.get('capital', 20000.0))

    return render_template(
        'dispatch_96quarter.html',
        price_area=price_area,
        model_name=model_name,
        profile=profile,
        selected_date=date_str,
        capital=capital,
        version="v3_1"
    )


@app.route('/trade_ledger_v3_1')
@app.route('/trade_ledger_v3')
def trade_ledger_v3_1():
    model_name = request.args.get('model', 'Hierarchical-LGBM+XGB')
    if ' ' in model_name:
        model_name = model_name.replace(' ', '+')
    price_area = request.args.get('area', 'DK1')
    capital = float(request.args.get('capital', 20000.0))
    trade_volume = float(request.args.get('volume', 2.0))
    profile = request.args.get('profile', 'tier2_standard')
    tab = request.args.get('tab', 'live')
    raw_mode = request.args.get('mode', 'intraday')
    dk_now = get_danish_now()
    default_date = (dk_now - timedelta(days=1)).strftime("%Y-%m-%d")
    selected_date = request.args.get('date', default_date)

    return render_template(
        'trade_ledger_v3.html',
        model_name=model_name,
        price_area=price_area,
        capital=capital,
        trade_volume=trade_volume,
        profile=profile,
        active_tab=tab,
        raw_mode=raw_mode,
        selected_date=selected_date,
        today_str=dk_now.strftime("%d %B %Y")
    )


@app.route('/api/dispatch_96quarter')
def api_dispatch_96quarter_v3_1():
    price_area = request.args.get('area', 'DK1')
    model_name = request.args.get('model', 'Hierarchical-LGBM+XGB')
    profile = request.args.get('profile', 'tier2_standard')
    date_str = request.args.get('date', None)
    capital = float(request.args.get('capital', 20000.0))

    from src.intraday_dispatch_engine import Intraday2HourDispatchEngine
    engine = Intraday2HourDispatchEngine(price_area=price_area, capital=capital, profile=profile)
    res = engine.generate_full_day_96q(date_str=date_str, model_name=model_name)
    return jsonify(res)


@app.route('/api/dispatch_batch_1')
def api_dispatch_batch_1_v3_1():
    price_area = request.args.get('area', 'DK1')
    model_name = request.args.get('model', 'Hierarchical-LGBM+XGB')
    if ' ' in model_name:
        model_name = model_name.replace(' ', '+')
    capital = float(request.args.get('capital', 20000.0))
    trade_volume = float(request.args.get('volume', 2.0))
    profile = request.args.get('profile', 'tier2_standard')
    tab = request.args.get('tab', 'live')
    num_quarters = int(request.args.get('num_quarters', 9))

    dk_now = get_danish_now()
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
    except Exception:
        target_df = table_gen.get_backtest_table(date_str=target_delivery_date)

    if target_df.empty:
        target_df = table_gen.get_future_table()

    strategy = V31CommercialStrategyEngine(price_area=price_area, capital=capital, base_volume_mwh=trade_volume, profile=profile)
    summary = strategy.evaluate_trading_ledger(target_df, model_name=model_name, market_mode="INTRADAY_D0", profile=profile)

    trades = summary.get("trades", [])[:num_quarters]
    headers = [
        "Quarter", "Delivery Time (CET)", "Spot Price (€/MWh)", "Pred Imbalance (€/MWh)", 
        "Pred Spread (€/MWh)", "Quantiles (q10-q90)", "P(Up) / P(Dn)", "Recommended Action", 
        "Volume (MW)", "Status"
    ]
    tsv_lines = ["	".join(headers)]
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
        tsv_lines.append("	".join(row_tsv))
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

    actual_delivery_date = formatted_trades[0]['time_dk'][:10] if formatted_trades else target_delivery_date

    return jsonify({
        "success": True,
        "price_area": price_area,
        "model_name": model_name,
        "profile": profile,
        "date": actual_delivery_date,
        "delivery_window": "00:00 - 02:15 CET (Q1-Q9)",
        "gate_closure_cutoff": "D-1 21:45 CET",
        "num_quarters": len(formatted_trades),
        "trades": formatted_trades,
        "tsv_payload": "\n".join(tsv_lines),
        "csv_payload": "\n".join(csv_lines)
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    print("\n" + "=" * 80)
    print(f"  V3.1 FINANCIAL TOURNAMENT & DEEP LEARNING DASHBOARD RUNNING ON http://127.0.0.1:{port}")
    print("=" * 80, flush=True)
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
