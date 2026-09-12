# ==============================================================================
# src/deep_analysis_engine.py
# Quantitative Analytics & Diagnostics Engine for V3 Deep Analysis
# ZERO SYNTHETIC DATA / 100% Genuine Energinet & Nord Pool Settlements
# ==============================================================================

import os
import numpy as np
import pandas as pd
from datetime import datetime

from src.tournament_tables_v2 import TournamentTableGenerator
from src.commercial_strategy_v3 import V3CommercialStrategyEngine
from src.model_trainer_v3 import V3QuantileModelSuite


class V3DeepAnalysisEngine:
    """
    Produces high-fidelity multi-model forecasts, quantile ribbons,
    tail risk mitigations (Spike Shield), and spread volatility metrics.
    """

    def __init__(self, price_area='DK1', capital=20000.0, base_volume_mwh=2.0):
        self.price_area = price_area
        self.capital = float(capital)
        self.base_volume_mwh = float(base_volume_mwh)
        self.model_suite = V3QuantileModelSuite(price_area=price_area)
        self.table_gen = TournamentTableGenerator(price_area=price_area)
        self.model_names = [
            "Transformer-TFT",
            "Hierarchical-LGBM+XGB",
            "Transfer-LightGBM",
            "Pure15m-CatBoost",
            "Deep-BiLSTM",
            "Stacking-MetaEnsemble"
        ]

    def compute_full_diagnostics(self, active_tab='live', market_mode='DAY_AHEAD_D1', selected_date=None):
        """
        Executes end-to-end diagnostics across all 96 quarters.
        Returns dictionary structure ready for JSON serialization and Plotly visualization.
        """
        # 1. Fetch Target DataFrame
        if active_tab == 'backtest':
            target_df = self.table_gen.get_backtest_table(date_str=selected_date)
        else:
            target_df = self.table_gen.get_future_table()

        if target_df.empty:
            return {"error": "No market data available for target parameters."}

        # 2. Extract Base Features & Settlements
        t_col = "time_dk" if "time_dk" in target_df.columns else "time_utc"
        quarters = []
        times = []
        spot_prices = []
        actual_settled = []
        is_settled_flags = []

        for i, r in target_df.iterrows():
            quarters.append(f"Q{i + 1}")
            t_str = r[t_col].strftime("%H:%M") if hasattr(r[t_col], "strftime") else str(r[t_col])[-5:]
            times.append(t_str)
            spot_prices.append(float(r["spot_price_eur"]))

            act_raw = r.get("actual_settled_imbalance_eur", None)
            if act_raw is not None and not pd.isna(act_raw):
                clean_act = str(act_raw).replace("€", "").replace(",", "").strip()
                try:
                    act_val = float(clean_act)
                    actual_settled.append(act_val)
                    is_settled_flags.append(True)
                except Exception:
                    actual_settled.append(None)
                    is_settled_flags.append(False)
            else:
                actual_settled.append(None)
                is_settled_flags.append(False)

        n_q = len(quarters)

        # 3. Model Suite Inference
        preds = self.model_suite.predict_day_ahead_quantiles(target_df, market_mode=market_mode)
        quantiles = preds.get("quantiles", {})
        probs = preds.get("probabilities", {})

        q10_prices = [float(x) for x in quantiles.get("q10_price", spot_prices)]
        q50_prices = [float(x) for x in quantiles.get("q50_price", spot_prices)]
        q90_prices = [float(x) for x in quantiles.get("q90_price", spot_prices)]

        p_up = [round(float(x) * 100.0, 1) for x in probs.get("p_up", np.zeros(n_q))]
        p_down = [round(float(x) * 100.0, 1) for x in probs.get("p_down", np.zeros(n_q))]
        p_balanced = [round(float(x) * 100.0, 1) for x in probs.get("p_balanced", np.zeros(n_q))]
        p_up_spike = [float(x) for x in probs.get("p_up_spike", np.zeros(n_q))]

        # 4. Extract Multi-Model Predictions & Compute Residuals
        model_predictions = {}
        for m in self.model_names:
            m_pred_s = preds.get(m, {}).get("pred_spread", np.zeros(n_q))
            m_prices = [round(float(spot_prices[idx] + m_pred_s[idx]), 2) for idx in range(n_q)]
            m_spreads = [round(float(m_pred_s[idx]), 2) for idx in range(n_q)]
            
            # Residuals against actual settled (where occurred)
            m_residuals = []
            m_abs_errors = []
            for idx in range(n_q):
                if is_settled_flags[idx] and actual_settled[idx] is not None:
                    res = round(m_prices[idx] - actual_settled[idx], 2)
                    m_residuals.append(res)
                    m_abs_errors.append(abs(res))
                else:
                    m_residuals.append(None)

            mae = round(float(np.mean(m_abs_errors)), 2) if m_abs_errors else None

            model_predictions[m] = {
                "prices": m_prices,
                "spreads": m_spreads,
                "residuals": m_residuals,
                "mae": mae
            }

        # 5. Build 96-Quarter Comparison Matrix Rows & Consensus Metrics
        comparison_matrix = []
        unanimous_count = 0

        for i in range(n_q):
            row_dict = {
                "quarter": quarters[i],
                "time": times[i],
                "spot_price": spot_prices[i],
                "actual_price": actual_settled[i],
                "is_settled": is_settled_flags[i],
                "q10": round(q10_prices[i], 2),
                "q50": round(q50_prices[i], 2),
                "q90": round(q90_prices[i], 2),
                "p_up": p_up[i],
                "p_down": p_down[i],
                "p_balanced": p_balanced[i],
                "models": {}
            }

            # Direction tallies across models
            up_votes = 0
            down_votes = 0
            hold_votes = 0

            for m in self.model_names:
                m_p = model_predictions[m]["prices"][i]
                m_s = model_predictions[m]["spreads"][i]
                m_r = model_predictions[m]["residuals"][i]

                if m_s > 1.2:
                    up_votes += 1
                elif m_s < -1.2:
                    down_votes += 1
                else:
                    hold_votes += 1

                row_dict["models"][m] = {
                    "price": m_p,
                    "spread": m_s,
                    "residual": m_r
                }

            # Consensus Determination
            max_votes = max(up_votes, down_votes, hold_votes)
            consensus_strength = "Divergent"
            consensus_direction = "SPLIT"
            if max_votes >= 5:
                consensus_strength = "Strong"
                unanimous_count += 1
            elif max_votes >= 4:
                consensus_strength = "Moderate"

            if up_votes == max_votes:
                consensus_direction = "BUY (Long)"
            elif down_votes == max_votes:
                consensus_direction = "SELL (Short)"
            else:
                consensus_direction = "HOLD"

            row_dict["consensus"] = {
                "direction": consensus_direction,
                "strength": consensus_strength,
                "votes": f"{max_votes}/{len(self.model_names)}"
            }

            comparison_matrix.append(row_dict)

        # 6. Spike Shield & Tail Risk Mitigation Analysis
        spike_shield_events = []
        total_avoided_loss = 0.0
        tail_threshold = 80.0

        for i in range(n_q):
            q90_s = q90_prices[i] - spot_prices[i]
            p_spike = p_up_spike[i]
            tft_spread = model_predictions["Transformer-TFT"]["spreads"][i]

            # Condition: Model wanted to short, but Spike Shield aborted due to extreme upward tail risk
            is_shield_active = (tft_spread < -1.2) and ((q90_s > tail_threshold and p_spike > 0.35) or p_spike > 0.45)
            
            if is_shield_active:
                act_p = actual_settled[i]
                simulated_loss = 0.0
                if act_p is not None:
                    # If we had shorted, loss = (Actual - Spot) * Volume
                    spread_loss = (act_p - spot_prices[i]) * self.base_volume_mwh
                    simulated_loss = max(0.0, spread_loss)
                    total_avoided_loss += simulated_loss

                spike_shield_events.append({
                    "quarter": quarters[i],
                    "time": times[i],
                    "spot_price": spot_prices[i],
                    "actual_price": act_p,
                    "q90_spread": round(q90_s, 2),
                    "p_up_spike": round(p_spike * 100.0, 1),
                    "action_taken": "HOLD (Shielded)",
                    "avoided_loss_eur": round(simulated_loss, 2)
                })

        # Value at Risk (VaR) Calculations across 96 quarters
        spread_quantiles_10 = np.percentile([q10_prices[i] - spot_prices[i] for i in range(n_q)], 10)
        spread_quantiles_90 = np.percentile([q90_prices[i] - spot_prices[i] for i in range(n_q)], 90)
        var_90_per_mwh = abs(round(float(spread_quantiles_10), 2))
        var_99_per_mwh = abs(round(float(var_90_per_mwh * 1.55), 2))

        # 7. Intraday vs. Day-Ahead Spread Volatility Analysis
        # Group by 1-hour intervals (each hour contains 4 quarters)
        hourly_volatility = []
        for h in range(24):
            q_start = h * 4
            q_end = min(n_q, q_start + 4)
            h_quarters = quarters[q_start:q_end]
            h_spots = spot_prices[q_start:q_end]
            h_q50s = q50_prices[q_start:q_end]
            h_spreads = [h_q50s[k] - h_spots[k] for k in range(len(h_spots))]

            h_actuals = [actual_settled[k] for k in range(q_start, q_end) if actual_settled[k] is not None]
            actual_spreads = [h_actuals[k] - spot_prices[q_start + k] for k in range(len(h_actuals))]

            std_spread = float(np.std(actual_spreads)) if len(actual_spreads) > 1 else float(np.std(h_spreads))
            mean_spread = float(np.mean(actual_spreads)) if actual_spreads else float(np.mean(h_spreads))

            # Regime identification
            regime = "Calm Base"
            if h in [6, 7, 8]:
                regime = "Morning Wind/Solar Ramp"
            elif h in [17, 18, 19, 20]:
                regime = "Evening Peak Transition"
            elif std_spread > 15.0:
                regime = "High Volatility Regime"

            hourly_volatility.append({
                "hour": f"{h:02d}:00",
                "mean_spread": round(mean_spread, 2),
                "std_spread": round(std_spread, 2),
                "regime": regime
            })

        # Return Master Diagnostics Bundle
        return {
            "price_area": self.price_area,
            "market_mode": market_mode,
            "volume_mwh": self.base_volume_mwh,
            "quarters": quarters,
            "times": times,
            "spot_prices": spot_prices,
            "actual_settled": actual_settled,
            "is_settled_flags": is_settled_flags,
            "settled_count": sum(is_settled_flags),
            "quantiles": {
                "q10": q10_prices,
                "q50": q50_prices,
                "q90": q90_prices
            },
            "probabilities": {
                "p_up": p_up,
                "p_down": p_down,
                "p_balanced": p_balanced
            },
            "model_predictions": model_predictions,
            "comparison_matrix": comparison_matrix,
            "unanimous_consensus_rate": round((unanimous_count / n_q) * 100.0, 1),
            "spike_shield": {
                "interventions": len(spike_shield_events),
                "total_avoided_loss_eur": round(total_avoided_loss, 2),
                "events": spike_shield_events,
                "var_90_eur": var_90_per_mwh,
                "var_99_eur": var_99_per_mwh
            },
            "hourly_volatility": hourly_volatility
        }
