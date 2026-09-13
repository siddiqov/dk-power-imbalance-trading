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
    today_date_str = dk_now.strftime("%Y-%m-%d")
    is_after_13 = (dk_now.hour >= 13)

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
        today_date_str=today_date_str,
        tomorrow_str=tomorrow_str,
        tomorrow_display=tomorrow_display,
        is_after_13=is_after_13,
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
    """Side-by-side pairwise model-vs-model comparison between V3 and V3.1 for DK1 & DK2."""
    capital = float(request.args.get('capital', 20000.0))
    profile = request.args.get('profile', 'tier2_standard')
    dk_now = get_danish_now()
    selected_date = request.args.get('date', dk_now.strftime("%Y-%m-%d"))

    return render_template(
        'compare_v3_vs_v3_1.html',
        capital=capital,
        profile=profile,
        selected_date=selected_date,
        today_str=dk_now.strftime("%d %B %Y")
    )


def _build_model_vs_model(price_area, capital, profile, date_str):
    """Runs all 6 models through both V3 and V3.1 engines for a given area and returns pairwise results."""
    # Model name mapping: V3 name -> V3.1 name
    MODEL_PAIRS = [
        {"v3_name": "Transformer-TFT",       "v31_name": "Transformer-TFT",       "paradigm": "P3. Pure 15m Native",        "type": "Deep Neural"},
        {"v3_name": "Transfer-LightGBM",      "v31_name": "Transfer-LightGBM",     "paradigm": "P1. Transfer Learning",      "type": "Tree"},
        {"v3_name": "Hierarchical-LGBM+XGB",  "v31_name": "Hierarchical-LGBM+XGB", "paradigm": "P2. Hierarchical Residual",  "type": "Tree"},
        {"v3_name": "Pure15m-CatBoost",       "v31_name": "Pure15m-CatBoost",      "paradigm": "P3. Pure 15m Native",        "type": "Tree"},
        {"v3_name": "Deep-BiLSTM",            "v31_name": "Transfer-BiLSTM",       "paradigm": "P1. Transfer Learning",      "type": "Deep Neural"},
        {"v3_name": "Stacking-MetaEnsemble",  "v31_name": "Stacking-MetaEnsemble", "paradigm": "P4. Stacking Meta-Ensemble", "type": "Hybrid"},
    ]

    table_gen = TournamentTableGenerator(price_area=price_area)
    try:
        target_df = table_gen.generate_and_save_future_table(date_str=date_str)
    except Exception:
        target_df = table_gen.get_backtest_table(date_str=date_str)
    if target_df.empty:
        target_df = table_gen.get_future_table()
    if target_df.empty:
        return []

    strat_v3 = V3CommercialStrategyEngine(price_area=price_area, capital=capital, profile=profile)
    strat_v31 = V31CommercialStrategyEngine(price_area=price_area, capital=capital, profile=profile)

    rows = []
    for pair in MODEL_PAIRS:
        # V3 evaluation
        s_v3 = strat_v3.evaluate_trading_ledger(target_df, model_name=pair["v3_name"], market_mode="INTRADAY_D0", profile=profile)
        pnl_v3 = float(s_v3.get("net_realized_profit_so_far", s_v3.get("net_pnl_eur", 0.0)))
        trades_v3 = s_v3.get("trades", [])
        settled_v3 = [t for t in trades_v3 if t.get("is_settled")]
        active_v3 = [t for t in settled_v3 if "BUY" in t.get("action", "") or "SELL" in t.get("action", "")]
        win_v3 = [t for t in active_v3 if str(t.get("net_pnl_eur", "")).startswith("+€") or (isinstance(t.get("net_pnl_eur"), (int, float)) and t.get("net_pnl_eur") > 0) or t.get("net_pnl_val", 0) > 0]
        wr_v3 = round((len(win_v3) / len(active_v3) * 100), 1) if active_v3 else 0.0
        trades_str_v3 = s_v3.get("trades_fraction_str", f"{len(active_v3)}/{len(trades_v3)}")

        # V3.1 evaluation
        s_v31 = strat_v31.evaluate_trading_ledger(target_df, model_name=pair["v31_name"], market_mode="INTRADAY_D0", profile=profile)
        pnl_v31 = float(s_v31.get("net_realized_profit_so_far", s_v31.get("net_pnl_eur", 0.0)))
        trades_v31 = s_v31.get("trades", [])
        settled_v31 = [t for t in trades_v31 if t.get("is_settled")]
        active_v31 = [t for t in settled_v31 if t.get("net_pnl_val", 0) != 0 or "BUY" in t.get("action", "") or "SELL" in t.get("action", "")]
        if not active_v31:
            active_v31 = [t for t in settled_v31 if str(t.get("action", "")).upper() not in ("HOLD", "")]
        win_v31 = [t for t in active_v31 if t.get("net_pnl_val", 0) > 0]
        wr_v31 = round((len(win_v31) / len(active_v31) * 100), 1) if active_v31 else 0.0
        trades_str_v31 = s_v31.get("trades_fraction_str", f"{len(active_v31)}/{len(trades_v31)}")

        delta = round(pnl_v31 - pnl_v3, 2)

        rows.append({
            "model_name": pair["v3_name"],
            "v31_model_name": pair["v31_name"],
            "paradigm": pair["paradigm"],
            "model_type": pair["type"],
            "v3_pnl": round(pnl_v3, 2),
            "v3_win_rate": wr_v3,
            "v3_trades": trades_str_v3,
            "v3_active": len(active_v3),
            "v31_pnl": round(pnl_v31, 2),
            "v31_win_rate": wr_v31,
            "v31_trades": trades_str_v31,
            "v31_active": len(active_v31),
            "delta_pnl": delta,
            "winner": "V3.1" if delta > 0 else ("V3" if delta < 0 else "Tie")
        })

    # Sort by V3.1 PnL descending
    rows.sort(key=lambda x: x["v31_pnl"], reverse=True)
    return rows


@app.route('/api/compare_v3_vs_v3_1')
def api_compare_v3_vs_v3_1():
    """Pairwise model-vs-model Net Realized Profit comparison for DK1 and DK2."""
    capital = float(request.args.get('capital', 20000.0))
    profile = request.args.get('profile', 'tier2_standard')
    date_str = request.args.get('date', None)
    if not date_str:
        dk_now = get_danish_now()
        date_str = dk_now.strftime("%Y-%m-%d")

    dk1_rows = _build_model_vs_model("DK1", capital, profile, date_str)
    dk2_rows = _build_model_vs_model("DK2", capital, profile, date_str)

    # Aggregate totals
    dk1_total_v3 = round(sum(r["v3_pnl"] for r in dk1_rows), 2)
    dk1_total_v31 = round(sum(r["v31_pnl"] for r in dk1_rows), 2)
    dk2_total_v3 = round(sum(r["v3_pnl"] for r in dk2_rows), 2)
    dk2_total_v31 = round(sum(r["v31_pnl"] for r in dk2_rows), 2)

    return jsonify({
        "success": True,
        "date": date_str,
        "profile": profile,
        "capital": capital,
        "dk1": {
            "models": dk1_rows,
            "total_v3_pnl": dk1_total_v3,
            "total_v31_pnl": dk1_total_v31,
            "total_delta": round(dk1_total_v31 - dk1_total_v3, 2)
        },
        "dk2": {
            "models": dk2_rows,
            "total_v3_pnl": dk2_total_v3,
            "total_v31_pnl": dk2_total_v31,
            "total_delta": round(dk2_total_v31 - dk2_total_v3, 2)
        }
    })


@app.route('/dispatch_96quarter')
@app.route('/intraday_96quarter')
def dispatch_96quarter_view_v3_1():
    """Renders 16-column D+1 Intraday 96-quarter trade ledger for V3.1."""
    price_area = request.args.get('area', 'DK1')
    model_name = request.args.get('model', 'Transfer-LightGBM')
    if ' ' in model_name:
        model_name = model_name.replace(' ', '+')
    profile = request.args.get('profile', 'tier2_standard')
    date_str = request.args.get('date', None)
    capital = float(request.args.get('capital', 20000.0))

    dk_now = get_danish_now()
    is_after_13 = (dk_now.hour >= 13)
    tomorrow_str = (dk_now + timedelta(days=1)).strftime("%Y-%m-%d")
    today_str = dk_now.strftime("%Y-%m-%d")

    if not date_str:
        selected_date = tomorrow_str if is_after_13 else today_str
    else:
        selected_date = date_str

    return render_template(
        'intraday_96quarter_v3_1.html',
        price_area=price_area,
        model_name=model_name,
        profile=profile,
        selected_date=selected_date,
        capital=capital,
        is_after_13=is_after_13,
        tomorrow_str=tomorrow_str,
        today_str=today_str,
        version="v3_1"
    )


@app.route('/api/intraday_96quarter_ledger')
def api_intraday_96quarter_ledger_v3_1():
    """Generates the 96-quarter trade ledger with all 16 audited columns for D+1 or chosen date, supporting Approach A (Midnight Boundary Bridge) and Approach B (Forward Quantile Trajectory)."""
    price_area = request.args.get('area', 'DK1')
    model_name = request.args.get('model', 'Transfer-LightGBM')
    if ' ' in model_name:
        model_name = model_name.replace(' ', '+')
    capital = float(request.args.get('capital', 20000.0))
    profile = request.args.get('profile', 'tier2_standard')
    date_str = request.args.get('date', None)
    req_approach = request.args.get('approach', 'both')

    dk_now = get_danish_now()
    today_str = dk_now.strftime("%Y-%m-%d")
    tomorrow_str = (dk_now + timedelta(days=1)).strftime("%Y-%m-%d")
    if not date_str:
        if dk_now.hour >= 13:
            date_str = tomorrow_str
        else:
            date_str = today_str

    table_gen = TournamentTableGenerator(price_area=price_area)
    try:
        target_df = table_gen.generate_and_save_future_table(date_str=date_str)
    except Exception:
        target_df = table_gen.get_backtest_table(date_str=date_str)
    if target_df.empty:
        target_df = table_gen.get_future_table(date_str=date_str)

    # Load Day D table for Midnight Bridge if evaluating D+1
    try:
        t_col = "time_dk" if "time_dk" in target_df.columns else "time_utc"
        first_dt = pd.to_datetime(target_df.iloc[0][t_col])
        prev_date_str = (first_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        if prev_date_str == today_str:
            df_prev_day = table_gen.get_future_table(date_str=today_str)
        else:
            df_prev_day = table_gen.get_backtest_table(date_str=prev_date_str)
    except Exception:
        df_prev_day = None

    strategy = V31CommercialStrategyEngine(price_area=price_area, capital=capital, profile=profile)
    summary_A = strategy.evaluate_trading_ledger(target_df, model_name=model_name, market_mode="INTRADAY_D0", profile=profile, approach="A", df_prev_day=df_prev_day)
    summary_B = strategy.evaluate_trading_ledger(target_df, model_name=model_name, market_mode="INTRADAY_D0", profile=profile, approach="B", df_prev_day=df_prev_day)

    pnl_A = float(summary_A.get("net_realized_profit_so_far", 0.0))
    pnl_B = float(summary_B.get("net_realized_profit_so_far", 0.0))
    mid_A = float(summary_A.get("midnight_pnl_eur", 0.0))
    mid_B = float(summary_B.get("midnight_pnl_eur", 0.0))

    if pnl_A > pnl_B:
        recommended = "Approach A (Midnight Boundary Bridge)"
        reason = f"Approach A delivers superior net return (+€ {pnl_A - pnl_B:,.2f}) by capitalizing on continuous physical grid inertia carried across midnight."
    elif pnl_B > pnl_A:
        recommended = "Approach B (Forward Quantile Trajectory)"
        reason = f"Approach B delivers superior net return (+€ {pnl_B - pnl_A:,.2f}) by eliminating stale prior-day inertia and responding purely to forward market forces."
    else:
        recommended = "Approach A & B Tied"
        reason = "Both models converged on consistent dispatch orders across the delivery horizon."

    comparison = {
        "pnl_A": pnl_A,
        "pnl_B": pnl_B,
        "pnl_diff": round(pnl_A - pnl_B, 2),
        "pnl_A_str": summary_A.get("net_realized_profit_so_far_str", "€ 0.00"),
        "pnl_B_str": summary_B.get("net_realized_profit_so_far_str", "€ 0.00"),
        "midnight_pnl_A": mid_A,
        "midnight_pnl_B": mid_B,
        "midnight_diff": round(mid_A - mid_B, 2),
        "midnight_pnl_A_str": summary_A.get("midnight_pnl_str", "€ 0.00"),
        "midnight_pnl_B_str": summary_B.get("midnight_pnl_str", "€ 0.00"),
        "win_rate_A": summary_A.get("win_rate_pct", 0.0),
        "win_rate_B": summary_B.get("win_rate_pct", 0.0),
        "active_trades_A": summary_A.get("active_trades", 0),
        "active_trades_B": summary_B.get("active_trades", 0),
        "recommended_approach": recommended,
        "recommendation_reason": reason
    }

    active_summary = summary_B if req_approach == "B" else summary_A
    response_data = dict(active_summary)
    response_data["approach_a"] = summary_A
    response_data["approach_b"] = summary_B
    response_data["comparison"] = comparison
    response_data["active_approach"] = req_approach if req_approach in ["A", "B"] else "A"

    return jsonify(response_data)


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

    dk_now = get_danish_now()
    if not date_str:
        if dk_now.hour >= 13:
            date_str = (dk_now + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            date_str = dk_now.strftime("%Y-%m-%d")

    from src.intraday_dispatch_engine import Intraday2HourDispatchEngine
    engine = Intraday2HourDispatchEngine(price_area=price_area, capital=capital, profile=profile, version="v3_1")
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
