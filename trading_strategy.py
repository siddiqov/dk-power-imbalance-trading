# ============================================
# trading_strategy.py - IMPROVED VERSION
# Smart Regime-Switching Trading
# ============================================

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')


class TradingStrategy:
    """Advanced Regime-Switching Trading Strategy"""

    def __init__(self, trainer, db):
        self.trainer = trainer
        self.db = db
        self.regime_names = ['Downward', 'Neutral', 'Upward']

        # Regime-based trading logic (CORRECT, not inverted)
        self.regime_positions = {
            0: {'signal': -1, 'size': 0.6},  # Downward → SELL (prices fall)
            1: {'signal': 0, 'size': 0.1},  # Neutral → HOLD
            2: {'signal': 1, 'size': 0.8}  # Upward → BUY (prices rise)
        }

    def generate_signals(self, features_df, model_name=None):
        """Generate trading signals with smart position sizing"""
        if model_name is None:
            model_name = list(self.trainer.models.keys())[0]

        # Get predictions and probabilities
        predictions = self.trainer.predict(features_df, model_name)
        probabilities = self.trainer.predict_proba(features_df, model_name)

        features_df = features_df.copy()
        features_df['predicted_regime'] = predictions
        features_df['regime_prob_0'] = probabilities[:, 0]
        features_df['regime_prob_1'] = probabilities[:, 1]
        features_df['regime_prob_2'] = probabilities[:, 2]
        features_df['confidence'] = np.max(probabilities, axis=1)

        # Generate base signals
        features_df['signal'] = 0
        features_df['base_position'] = 0.0

        for regime, params in self.regime_positions.items():
            mask = features_df['predicted_regime'] == regime
            if np.sum(mask) == 0:
                continue

            features_df.loc[mask, 'signal'] = params['signal']
            features_df.loc[mask, 'base_position'] = params['size']

        # ============================================
        # SMART POSITION SIZING
        # ============================================

        # 1. Confidence adjustment
        confidence_factor = features_df['confidence']

        # 2. Regime strength (based on price spread)
        price_spread = np.abs(features_df['imbalance_price'] - features_df['day_ahead_price'])
        regime_strength = price_spread / 100
        regime_strength = regime_strength.clip(0.2, 1.0)

        # 3. Volatility adjustment
        volatility = features_df['day_ahead_price'].rolling(24, min_periods=1).std().fillna(0)
        volatility_factor = 1 / (1 + volatility / 100)
        volatility_factor = volatility_factor.clip(0.3, 1.0)

        # 4. Trend strength
        price_ma_short = features_df['day_ahead_price'].rolling(8, min_periods=1).mean()
        price_ma_long = features_df['day_ahead_price'].rolling(32, min_periods=1).mean()
        trend_diff = np.abs(price_ma_short - price_ma_long) / (price_ma_long + 1)
        trend_factor = 0.5 + 0.5 * trend_diff.clip(0, 1)

        # Combined position size
        features_df['position_size'] = (
                features_df['base_position'] *
                confidence_factor *
                regime_strength *
                volatility_factor *
                trend_factor
        )
        features_df['position_size'] = features_df['position_size'].clip(0.1, 1.0)

        # ============================================
        # EXIT RULES
        # ============================================

        features_df = self.apply_exit_rules(features_df)

        print(f"\n   Strategy: Smart Regime-Switching Trading")
        print(f"   - Upward Regulation → BUY (with dynamic sizing)")
        print(f"   - Downward Regulation → SELL (with dynamic sizing)")
        print(f"   - Neutral → HOLD")
        print(f"   - Position sizing: Confidence × Regime Strength × Volatility × Trend")

        signal_counts = features_df['signal'].value_counts().sort_index()
        signal_names = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}
        for signal, count in signal_counts.items():
            pct = count / len(features_df) * 100
            print(f"      {signal_names.get(signal, signal)}: {count:,} ({pct:.1f}%)")

        # Regime distribution
        regime_counts = features_df['predicted_regime'].value_counts().sort_index()
        print(f"\n   Predicted Regime Distribution:")
        for r, count in regime_counts.items():
            pct = count / len(features_df) * 100
            print(f"      {self.regime_names[r]}: {count:,} ({pct:.1f}%)")

        # Average confidence and position size
        print(f"\n   Average Confidence: {features_df['confidence'].mean():.2%}")
        print(f"   Average Position Size: {features_df['position_size'].mean():.2%}")

        return features_df

    def apply_exit_rules(self, df):
        """Apply exit rules to protect capital"""
        df = df.copy()

        # Track position changes
        df['position_change'] = df['signal'].diff().abs()
        df['entry_price'] = df['day_ahead_price'].shift(1)

        # Calculate profit/loss since entry
        df['bars_since_entry'] = df.groupby((df['signal'] != df['signal'].shift()).cumsum()).cumcount()

        # Exit conditions
        exit_mask = (
            # Time stop: exit after 24 periods (6 hours)
                (df['bars_since_entry'] > 24) |
                # Regime change: exit when neutral predicted
                (df['predicted_regime'] == 1)
        )

        # Apply exits
        df.loc[exit_mask & (df['signal'] != 0), 'signal'] = 0
        df.loc[exit_mask & (df['position_size'] > 0.01), 'position_size'] = 0.0

        return df

    def backtest(self, signals_df, area):
        """Backtest with realistic costs"""
        df = signals_df.copy()

        # Calculate returns
        df['price_returns'] = df['day_ahead_price'].pct_change() * 100
        df['price_returns'] = df['price_returns'].fillna(0)
        df['price_returns'] = df['price_returns'].clip(-3, 3)

        # Position
        df['position'] = df['signal'] * df['position_size']

        # Gross returns
        df['gross_returns'] = df['position'].shift(1) * df['price_returns']

        # Transaction costs
        df['position_change'] = df['position'].diff().abs()
        df['transaction_cost'] = df['position_change'] * 0.1
        df['transaction_cost'] = df['transaction_cost'].clip(0, 1)

        # Net returns
        df['strategy_returns'] = df['gross_returns'] - df['transaction_cost']

        # Cumulative
        df['cumulative_returns'] = df['strategy_returns'].cumsum()
        df['buy_hold_returns'] = df['price_returns'].cumsum()

        # Count trades
        df['trade_entry'] = (df['position'].diff().abs() > 0.01).astype(int)
        total_trades = df['trade_entry'].sum()
        total_costs = df['transaction_cost'].sum()

        # Metrics
        clean_returns = df['strategy_returns'].dropna()
        clean_returns = clean_returns[np.isfinite(clean_returns)]
        clean_returns = clean_returns[abs(clean_returns) < 10]

        metrics = {
            'area': area,
            'total_return': df['cumulative_returns'].iloc[-1] if not df.empty else 0,
            'buy_hold_return': df['buy_hold_returns'].iloc[-1] if not df.empty else 0,
            'total_trades': total_trades,
            'total_costs': total_costs,
            'sharpe_ratio': 0,
            'win_rate': 0,
            'max_drawdown': 0,
            'profit_factor': 0
        }

        if len(clean_returns) > 10:
            metrics['sharpe_ratio'] = (clean_returns.mean() / clean_returns.std()) * np.sqrt(
                24 * 365) if clean_returns.std() > 0 else 0
            metrics['win_rate'] = (clean_returns > 0).mean() * 100

            cummax = df['cumulative_returns'].expanding().max()
            drawdown = (cummax - df['cumulative_returns'])
            metrics['max_drawdown'] = drawdown.max() if not drawdown.empty else 0

            wins = clean_returns[clean_returns > 0]
            losses = clean_returns[clean_returns < 0]
            if len(wins) > 0 and len(losses) > 0:
                metrics['profit_factor'] = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else 0

        # Regime performance
        regime_performance = {}
        for regime in [0, 1, 2]:
            mask = df['predicted_regime'] == regime
            if np.sum(mask) > 50:
                regime_returns = df.loc[mask, 'strategy_returns']
                regime_performance[regime] = {
                    'total_return': regime_returns.sum() if not regime_returns.empty else 0,
                    'win_rate': (regime_returns > 0).mean() * 100 if len(regime_returns) > 0 else 0,
                    'count': np.sum(mask),
                    'trades': df.loc[mask, 'trade_entry'].sum(),
                    'avg_position': df.loc[mask, 'position'].abs().mean(),
                    'avg_confidence': df.loc[mask, 'confidence'].mean()
                }

        metrics['regime_performance'] = regime_performance

        # Store in database
        self._store_results(metrics, df, area)

        return df, metrics

    def _store_results(self, metrics, df, area):
        """Store results"""
        pred_df = df[['time_utc', 'time_dk', 'price_area',
                      'imbalance_price', 'predicted_regime',
                      'regime_prob_0', 'regime_prob_1', 'regime_prob_2',
                      'confidence', 'signal', 'position',
                      'strategy_returns', 'cumulative_returns']].copy()
        pred_df['model_name'] = list(self.trainer.models.keys())[0]
        self.db.insert_dataframe('model_predictions', pred_df)


# ============================================
# PLOT FUNCTIONS
# ============================================

def plot_results(signals_df, metrics, area, save_path=None):
    """Plot regime-based trading performance"""
    if signals_df.empty:
        print("  No data to plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{area} - Smart Regime-Switching Trading Strategy', fontsize=16, fontweight='bold')

    # 1. Cumulative Returns
    ax1 = axes[0, 0]
    ax1.plot(signals_df['time_utc'], signals_df['cumulative_returns'], 'g-', linewidth=2, label='Strategy')
    ax1.plot(signals_df['time_utc'], signals_df['buy_hold_returns'], 'b--', linewidth=1.5, alpha=0.7,
             label='Buy & Hold')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cumulative Returns (%)')
    ax1.set_title('Strategy vs Buy & Hold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax1.text(0.02, 0.95,
             f"Return: {metrics['total_return']:.2f}%\nSharpe: {metrics['sharpe_ratio']:.2f}\nWin Rate: {metrics['win_rate']:.1f}%",
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. Regime Predictions Over Time
    ax2 = axes[0, 1]
    subset = signals_df.tail(500)
    ax2.plot(subset['time_utc'], subset['predicted_regime'], 'r-', alpha=0.7, linewidth=1)
    ax2.fill_between(subset['time_utc'], 0, subset['predicted_regime'],
                     where=subset['predicted_regime'] == 2, color='green', alpha=0.3, label='Upward')
    ax2.fill_between(subset['time_utc'], 0, subset['predicted_regime'],
                     where=subset['predicted_regime'] == 0, color='red', alpha=0.3, label='Downward')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Regime')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Downward', 'Neutral', 'Upward'])
    ax2.set_title('Predicted Regimes (Last 500 periods)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Regime Distribution
    ax3 = axes[0, 2]
    regime_counts = signals_df['predicted_regime'].value_counts().sort_index()
    regime_names = ['Downward', 'Neutral', 'Upward']
    colors = ['red', 'gray', 'green']
    ax3.bar(regime_names, [regime_counts.get(i, 0) for i in range(3)], color=colors, alpha=0.7)
    ax3.set_xlabel('Regime')
    ax3.set_ylabel('Count')
    ax3.set_title('Predicted Regime Distribution')
    ax3.grid(True, alpha=0.3)

    # 4. Signals Distribution
    ax4 = axes[1, 0]
    signal_counts = signals_df['signal'].value_counts().sort_index()
    signal_names = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}
    labels = [signal_names.get(s, f'Signal {s}') for s in signal_counts.index]
    colors = ['red' if s == -1 else 'gray' if s == 0 else 'green' for s in signal_counts.index]
    ax4.bar(labels, signal_counts.values, color=colors, alpha=0.7)
    ax4.set_xlabel('Signal')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Signal Distribution\nTotal: {len(signals_df):,} periods')
    ax4.grid(True, alpha=0.3)

    # 5. Confidence Distribution
    ax5 = axes[1, 1]
    ax5.hist(signals_df['confidence'], bins=20, edgecolor='black', alpha=0.7, color='purple')
    ax5.set_xlabel('Confidence')
    ax5.set_ylabel('Frequency')
    ax5.set_title(f'Confidence Distribution\nMean: {signals_df["confidence"].mean():.2f}')
    ax5.grid(True, alpha=0.3)

    # 6. Positions by Regime
    ax6 = axes[1, 2]
    regime_positions = signals_df.groupby('predicted_regime')['position'].mean()
    ax6.bar(regime_names, [regime_positions.get(i, 0) for i in range(3)], color=colors, alpha=0.7)
    ax6.set_xlabel('Predicted Regime')
    ax6.set_ylabel('Average Position')
    ax6.set_title('Average Position by Regime')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs('results', exist_ok=True)
        filename = f'{area}_smart_regime_performance.png'
        plt.savefig(f'results/{filename}', dpi=300, bbox_inches='tight')
        print(f"  Plot saved to results/{filename}")

    plt.show()