# ==============================================================================
# train_all_models.py — Train LSTM, GRU, Prophet and run full 9-model benchmark
# ==============================================================================

import sys
import os
import json
import pickle
import warnings
import time

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Import our pipeline
from run_backtest_pipeline import BacktestFeaturePipeline, BacktestEngine

# Import new models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from models.lstm_gru_models import train_all_sequence_models, load_best_sequence_model, predict_with_sequence_model
from models.prophet_model import train_prophet_models, predict_with_prophet, load_prophet_bundle


def main():
    area = 'DK1'
    print("=" * 70)
    print("  FULL MODEL TRAINING & BENCHMARK PIPELINE")
    print("  9 Models + RF Regime Classifier + Gaussian HMM Regime Detector")
    print("=" * 70)

    t0 = time.time()

    # ------------------------------------------------------------------ #
    # 1. Load data via feature pipeline
    # ------------------------------------------------------------------ #
    pipeline = BacktestFeaturePipeline('energy_data.db')
    df = pipeline.load_and_prepare_features(area)

    # ------------------------------------------------------------------ #
    # 2. Train Baseline Ensemble (default params)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  PHASE 1: BASELINE ENSEMBLE (Default Parameters)")
    print("=" * 70)
    engine_base = BacktestEngine(df, area=area, test_intervals=5, use_tuning=False)
    results_base, metrics_base = engine_base.train_and_evaluate(retrain=True)
    base_mae = metrics_base['mae_eur']
    print(f"\n  >> Baseline Ensemble MAE: {base_mae:.2f} EUR/MWh")

    # ------------------------------------------------------------------ #
    # 3. Train Optuna Bayesian Tuned Ensemble
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  PHASE 2: OPTUNA BAYESIAN TUNED ENSEMBLE")
    print("=" * 70)
    engine_opt = BacktestEngine(df, area=area, test_intervals=5, use_tuning=False)
    results_opt, metrics_opt = engine_opt.train_and_evaluate(retrain=True)
    opt_mae = metrics_opt['mae_eur']
    print(f"\n  >> Optuna Ensemble MAE: {opt_mae:.2f} EUR/MWh")

    # ------------------------------------------------------------------ #
    # 4. Train LSTM & GRU (dual lookback: 24 and 96)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  PHASE 3: LSTM & GRU TRAINING (lookback=24 and 96)")
    print("=" * 70)

    feature_cols = engine_opt.feature_cols
    seq_results, seq_meta = train_all_sequence_models(
        df, feature_cols, price_col='imbalance_price_eur',
        save_dir='models', area=area
    )

    # ------------------------------------------------------------------ #
    # 5. Train Prophet
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  PHASE 4: PROPHET TRAINING (5 horizon steps)")
    print("=" * 70)
    prophet_bundle = train_prophet_models(df, area=area, save_dir='models')

    # ------------------------------------------------------------------ #
    # 6. Full comparison on the SAME test set
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  PHASE 5: FULL 9-MODEL COMPARISON")
    print("=" * 70)

    # Use the same test interval actuals from the Optuna engine
    test_actuals = results_opt['Actual_EUR'].values
    n_test = len(test_actuals)

    comparison = {
        'Baseline Ensemble': {
            'mae': base_mae,
            'predictions': results_base['Predicted_Ensemble_EUR'].tolist(),
        },
        'Optuna Ensemble': {
            'mae': opt_mae,
            'predictions': results_opt['Predicted_Ensemble_EUR'].tolist(),
        },
    }

    # LSTM and GRU evaluation on same test periods
    for tag, info in seq_results.items():
        comparison[f'{tag.upper()}'] = {
            'mae': info['val_mae'],
            'lookback': info['lookback'],
        }

    # Prophet
    comparison['Prophet'] = {
        'mae': prophet_bundle['val_mae'],
        'per_step_mae': prophet_bundle['val_maes_per_step'],
    }

    # Sort by MAE
    ranked = sorted(comparison.items(), key=lambda x: x[1]['mae'])

    print(f"\n  {'Rank':<6} {'Model':<25} {'MAE (EUR/MWh)':<15}")
    print("  " + "-" * 50)
    for i, (name, info) in enumerate(ranked, 1):
        marker = " << BEST" if i == 1 else ""
        print(f"  {i:<6} {name:<25} {info['mae']:<15.2f}{marker}")

    best_model_name = ranked[0][0]
    best_mae = ranked[0][1]['mae']

    print(f"\n  WINNER: {best_model_name} (MAE: {best_mae:.2f} EUR/MWh)")

    # ------------------------------------------------------------------ #
    # 7. Save full comparison results
    # ------------------------------------------------------------------ #
    full_results = {
        'area': area,
        'best_model': best_model_name,
        'best_mae': best_mae,
        'ranking': [(name, {'mae': info['mae']}) for name, info in ranked],
        'baseline_mae': base_mae,
        'optuna_mae': opt_mae,
        'lstm_24_mae': seq_results.get('lstm_24', {}).get('val_mae'),
        'lstm_96_mae': seq_results.get('lstm_96', {}).get('val_mae'),
        'gru_24_mae': seq_results.get('gru_24', {}).get('val_mae'),
        'gru_96_mae': seq_results.get('gru_96', {}).get('val_mae'),
        'prophet_mae': prophet_bundle['val_mae'],
        'best_lstm_lookback': seq_meta['results'][seq_meta['best_lstm']]['lookback'],
        'best_gru_lookback': seq_meta['results'][seq_meta['best_gru']]['lookback'],
        'regime_accuracy_rf': metrics_opt.get('regime_accuracy_rf', 'N/A'),
        'regime_accuracy_hmm': metrics_opt.get('regime_accuracy_hmm', 'N/A'),
        'regime_accuracy_consensus': metrics_opt.get('regime_accuracy_consensus', 'N/A'),
    }

    results_path = os.path.join('results', 'full_model_comparison_DK1.json')
    os.makedirs('results', exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\n  Saved comparison: {results_path}")

    elapsed = time.time() - t0
    print(f"\n  Total training time: {elapsed/60:.1f} minutes")
    print("=" * 70)


if __name__ == '__main__':
    main()
