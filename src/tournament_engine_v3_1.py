# ==============================================================================
# src/tournament_engine_v3_1.py
# Real-Time Financial PnL Tournament Engine for V3.1
# Dynamically compares all 6 candidate models (Trees vs PyTorch Deep Networks)
# Ranks by Realized Net PnL (€), Win Rate %, ROC %, and flags the Live Champion.
# 100% Genuine Energinet & Nord Pool Settled Data
# ==============================================================================

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.tournament_tables_v2 import TournamentTableGenerator
from src.commercial_strategy_v3_1 import V31CommercialStrategyEngine


class V31RealTimeTournamentEngine:
    """
    Evaluates all V3.1 models in real time based on actual realized trading PnL
    under realistic Danish power market settlement rules.
    """

    def __init__(self, price_area="DK1", initial_capital=20000.0, profile="tier2_standard"):
        self.price_area = price_area
        self.capital = float(initial_capital)
        self.profile = profile
        self.table_gen = TournamentTableGenerator(price_area=price_area)
        self.strategy = V31CommercialStrategyEngine(
            price_area=price_area, capital=self.capital, profile=self.profile
        )

        self.candidate_models = [
            {"name": "Transfer-LightGBM",     "type": "Tree",        "paradigm": "P1. Transfer Learning",      "badge": "🌳 LightGBM (Optuna)"},
            {"name": "Transfer-BiLSTM",       "type": "Deep Neural", "paradigm": "P1. Transfer Learning",      "badge": "🧠 PyTorch BiLSTM"},
            {"name": "Hierarchical-LGBM+XGB", "type": "Tree",        "paradigm": "P2. Hierarchical Residual",  "badge": "🌳 Macro+Micro Residual"},
            {"name": "Pure15m-CatBoost",      "type": "Tree",        "paradigm": "P3. Pure 15m Native",        "badge": "🌳 CatBoost (Optuna)"},
            {"name": "Transformer-TFT",       "type": "Deep Neural", "paradigm": "P3. Pure 15m Native",        "badge": "🧠 PyTorch Self-Attention"},
            {"name": "Stacking-MetaEnsemble", "type": "Hybrid",      "paradigm": "P4. Stacking Meta-Ensemble", "badge": "⚡ Hybrid Tree+Neural Blend"}
        ]

    def run_tournament(self, date_str=None):
        """
        Executes real-time PnL simulation across all candidate models for a given date.
        """
        if not date_str:
            target_df = self.table_gen.get_backtest_table()
            if target_df.empty:
                target_df = self.table_gen.get_future_table()
        else:
            try:
                target_df = self.table_gen.generate_and_save_future_table(date_str=date_str)
            except Exception:
                target_df = self.table_gen.get_backtest_table(date_str=date_str)

        if target_df.empty:
            return {"success": False, "message": "No settlement data available for tournament"}

        results = []

        for candidate in self.candidate_models:
            m_name = candidate["name"]
            summary = self.strategy.evaluate_trading_ledger(
                target_df, model_name=m_name, market_mode="INTRADAY_D0", profile=self.profile
            )

            trades = summary.get("trades", [])
            settled_trades = [t for t in trades if t.get("is_settled")]

            net_pnl = summary.get("net_pnl_eur", 0.0)
            gross_pnl = summary.get("gross_pnl_eur", 0.0)
            fees = summary.get("fees_eur", 0.0)

            # Performance Metrics
            win_trades = [t for t in settled_trades if t.get("net_pnl_val", 0.0) > 0]
            loss_trades = [t for t in settled_trades if t.get("net_pnl_val", 0.0) < 0]
            active_settled = len(win_trades) + len(loss_trades)

            win_rate = round((len(win_trades) / active_settled * 100), 1) if active_settled > 0 else 0.0
            gross_win = sum(t.get("net_pnl_val", 0.0) for t in win_trades)
            gross_loss = abs(sum(t.get("net_pnl_val", 0.0) for t in loss_trades))
            profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else (9.99 if gross_win > 0 else 1.0)
            roc_pct = round((net_pnl / self.capital) * 100, 2)

            # Error metrics
            errors = []
            for t in settled_trades:
                act = t.get("actual_settled_eur")
                pred = t.get("pred_imbalance_eur")
                if act is not None and pred is not None:
                    try:
                        act_val = float(str(act).replace("€", "").replace(",", "").strip())
                        errors.append(abs(pred - act_val))
                    except Exception:
                        pass
            mae = round(np.mean(errors), 2) if errors else 0.0

            results.append({
                "model_name": m_name,
                "model_type": candidate["type"],
                "paradigm": candidate["paradigm"],
                "badge": candidate["badge"],
                "net_pnl_eur": round(net_pnl, 2),
                "gross_pnl_eur": round(gross_pnl, 2),
                "fees_eur": round(fees, 2),
                "roc_pct": roc_pct,
                "win_rate_pct": win_rate,
                "profit_factor": profit_factor,
                "mae_eur": mae,
                "total_trades": len(trades),
                "active_trades": summary.get("active_trades", 0),
                "hold_trades": summary.get("hold_trades", 0),
                "win_count": len(win_trades),
                "loss_count": len(loss_trades)
            })

        # Rank strictly by Net Realized PnL (€) descending
        results.sort(key=lambda x: x["net_pnl_eur"], reverse=True)

        for rank, item in enumerate(results, 1):
            item["rank"] = rank
            item["is_champion"] = (rank == 1)

        champion = results[0] if results else None

        return {
            "success": True,
            "price_area": self.price_area,
            "profile": self.profile,
            "date": summary.get("delivery_date", date_str or "Live"),
            "initial_capital": self.capital,
            "champion": champion,
            "leaderboard": results
        }
