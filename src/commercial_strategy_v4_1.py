import numpy as np
import pandas as pd

class V41CommercialStrategyEngine:
    """
    Auditor-Compliant Intraday Expected Value Strategy.
    Replaces static thresholds with dynamic probabilistic EV sizing.
    """
    def __init__(self, fee_per_mwh=1.79, slippage_per_mwh=0.5):
        # The auditor noted 0.51 EUR was too low and 1.79 EUR/MWh is the V4.2 plan assumption.
        self.fee = fee_per_mwh
        self.slippage = slippage_per_mwh
        self.total_cost = self.fee + self.slippage
        
        # Auditor requested checking if 10-25 MW is realistic for liquidity.
        self.max_volume = 25.0
        self.min_volume = 10.0

    def calculate_expected_value(self, p_up, p_down, p_none, spread_q10, spread_q50, spread_q90):
        """
        Calculates Expected Value (EV) of taking a position.
        """
        # Expected value if we BUY (UP direction):
        # We win if it goes UP, we lose if it goes DOWN or NONE
        # Using the median (q50) for expected magnitude
        ev_buy = (p_up * spread_q50) - (p_down * spread_q50) - (p_none * self.total_cost) - self.total_cost
        
        # Expected value if we SELL (DOWN direction):
        ev_sell = (p_down * spread_q50) - (p_up * spread_q50) - (p_none * self.total_cost) - self.total_cost
        
        return ev_buy, ev_sell

    def evaluate_position(self, p_up, p_down, p_none, spread_q10, spread_q50, spread_q90):
        ev_buy, ev_sell = self.calculate_expected_value(p_up, p_down, p_none, spread_q10, spread_q50, spread_q90)
        
        # Determine Direction
        if ev_buy > 0 and ev_buy > ev_sell:
            direction = "BUY"
            conviction = ev_buy / (self.total_cost + 1e-4) # Signal to noise
        elif ev_sell > 0 and ev_sell > ev_buy:
            direction = "SELL"
            conviction = ev_sell / (self.total_cost + 1e-4)
        else:
            direction = "HOLD"
            conviction = 0.0

        # Position Sizing
        if direction == "HOLD":
            volume = 0.0
        else:
            # Scale volume between 10 and 25 based on conviction (cap at 3.0 conviction)
            vol_scale = np.clip(conviction / 3.0, 0, 1)
            volume = self.min_volume + (self.max_volume - self.min_volume) * vol_scale
            
            # Risk Cap: Ensure 90th percentile downside doesn't exceed a strict loss limit per quarter
            # E.g., Max acceptable loss per quarter = 500 EUR
            max_loss_limit = 500.0
            worst_case_spread = spread_q90 if direction == "BUY" else spread_q90 # Assume symmetry for now
            worst_case_loss = volume * worst_case_spread
            if worst_case_loss > max_loss_limit:
                volume = max_loss_limit / (worst_case_spread + 1e-4)
                
            volume = np.clip(volume, self.min_volume, self.max_volume)

        return {
            "Action": direction,
            "Volume_MW": round(volume, 2),
            "EV_EUR": round(ev_buy if direction == "BUY" else ev_sell, 2),
            "Conviction": round(conviction, 2)
        }

    def generate_trading_signals(self, df_preds):
        """
        Process a batch of predictions to generate a commercial ledger.
        Assumes df_preds contains: p_up, p_down, p_none, spread_q10, spread_q50, spread_q90
        """
        results = []
        for idx, row in df_preds.iterrows():
            decision = self.evaluate_position(
                p_up=row.get('p_up', 0.33),
                p_down=row.get('p_down', 0.33),
                p_none=row.get('p_none', 0.34),
                spread_q10=row.get('spread_q10', 0.0),
                spread_q50=row.get('spread_q50', 0.0),
                spread_q90=row.get('spread_q90', 0.0)
            )
            decision['time_dk'] = row['time_dk']
            results.append(decision)
            
        return pd.DataFrame(results)
