# ==============================================================================
# src/trade_journal.py
# Immutable Trade Execution Journal & Audit Log (Zero Hindsight Bias)
# Guarantees that trade orders locked at Gate Closure can NEVER be mutated
# ==============================================================================

import os
import sqlite3
import pandas as pd
from datetime import datetime


class V3TradeJournal:
    """
    Manages persistent, immutable trade order logs for both
    Pure Day-Ahead (D-1) and Continuous Intraday (D-0) markets.
    """

    def __init__(self, db_path="trades_journal.db"):
        self.db_path = db_path
        self._initialize_schema()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _initialize_schema(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_orders (
                    trade_id TEXT PRIMARY KEY,
                    market_mode TEXT NOT NULL,       -- 'DAY_AHEAD_D1' or 'INTRADAY_D0'
                    price_area TEXT NOT NULL,        -- 'DK1' or 'DK2'
                    model_name TEXT NOT NULL,        -- 'Transformer-TFT', etc.
                    delivery_date TEXT NOT NULL,     -- 'YYYY-MM-DD'
                    quarter_index INTEGER NOT NULL,  -- 1 to 96
                    quarter_str TEXT NOT NULL,       -- 'Q1', 'Q2', etc.
                    time_dk TEXT NOT NULL,           -- 'YYYY-MM-DD HH:MM'
                    gate_closure_time TEXT NOT NULL, -- ISO timestamp when locked
                    locked_action TEXT NOT NULL,     -- 'BUY Spot (Long)', 'SELL Spot (Short)', 'HOLD', 'HOLD (Spike Shield Protected)'
                    locked_direction TEXT NOT NULL,  -- 'UP (+1)', 'DOWN (-1)', 'BALANCED (0)'
                    locked_volume_mwh REAL NOT NULL, -- Committed Volume
                    locked_spot_price_eur REAL NOT NULL,
                    locked_pred_price_eur REAL NOT NULL,
                    locked_pred_spread_eur REAL NOT NULL,
                    locked_q10_price_eur REAL,
                    locked_q50_price_eur REAL,
                    locked_q90_price_eur REAL,
                    locked_p_up REAL,
                    locked_p_down REAL,
                    locked_p_up_spike REAL,
                    actual_settled_price_eur REAL,
                    da_cash_flow_eur REAL,
                    settle_cash_flow_eur REAL,
                    gross_pnl_eur REAL,
                    fees_eur REAL,
                    tax_eur REAL,
                    net_pnl_eur REAL,
                    status TEXT NOT NULL,            -- 'LOCKED_PENDING', 'SETTLED_AUDITED'
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    settled_at TIMESTAMP,
                    UNIQUE(market_mode, price_area, model_name, delivery_date, quarter_index)
                )
            """)
            conn.commit()

    def get_order(self, market_mode, price_area, model_name, delivery_date, quarter_index):
        """Retrieves a previously locked order if it exists."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM trade_orders
                WHERE market_mode = ? AND price_area = ? AND model_name = ? 
                  AND delivery_date = ? AND quarter_index = ?
            """, (market_mode, price_area, model_name, delivery_date, quarter_index))
            row = cursor.fetchone()
            if row:
                col_names = [desc[0] for desc in cursor.description]
                return dict(zip(col_names, row))
            return None

    def lock_order(self, order_dict):
        """
        Locks an order before Gate Closure.
        Once written, the action, volume, and predicted prices are IMMUTABLE.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO trade_orders (
                    trade_id, market_mode, price_area, model_name, delivery_date,
                    quarter_index, quarter_str, time_dk, gate_closure_time,
                    locked_action, locked_direction, locked_volume_mwh,
                    locked_spot_price_eur, locked_pred_price_eur, locked_pred_spread_eur,
                    locked_q10_price_eur, locked_q50_price_eur, locked_q90_price_eur,
                    locked_p_up, locked_p_down, locked_p_up_spike,
                    status
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    'LOCKED_PENDING'
                )
            """, (
                order_dict["trade_id"],
                order_dict["market_mode"],
                order_dict["price_area"],
                order_dict["model_name"],
                order_dict["delivery_date"],
                int(order_dict["quarter_index"]),
                order_dict["quarter_str"],
                order_dict["time_dk"],
                order_dict.get("gate_closure_time", datetime.now().isoformat()),
                order_dict["locked_action"],
                order_dict["locked_direction"],
                float(order_dict["locked_volume_mwh"]),
                float(order_dict["locked_spot_price_eur"]),
                float(order_dict["locked_pred_price_eur"]),
                float(order_dict["locked_pred_spread_eur"]),
                float(order_dict.get("locked_q10_price_eur", 0.0)),
                float(order_dict.get("locked_q50_price_eur", 0.0)),
                float(order_dict.get("locked_q90_price_eur", 0.0)),
                float(order_dict.get("locked_p_up", 0.0)),
                float(order_dict.get("locked_p_down", 0.0)),
                float(order_dict.get("locked_p_up_spike", 0.0))
            ))
            conn.commit()

    def update_settlement(self, market_mode, price_area, model_name, delivery_date, quarter_index, actual_price, fee_per_mwh=0.51, tax_rate=0.22):
        """
        Audits post-delivery settlement strictly against the LOCKED order.
        Calculates genuine gross PnL, fees, tax, and net PnL (True Win or True Loss).
        """
        order = self.get_order(market_mode, price_area, model_name, delivery_date, quarter_index)
        if not order:
            return None

        # If already settled, return existing audited record
        if order["status"] == "SETTLED_AUDITED" and order["actual_settled_price_eur"] is not None:
            return order

        action = order["locked_action"]
        vol = float(order["locked_volume_mwh"])
        p_spot = float(order["locked_spot_price_eur"])
        p_act = float(actual_price)

        if "BUY" in action and vol > 0:
            da_cash = -(vol * p_spot)
            settle_cash = +(vol * p_act)
            fees = vol * fee_per_mwh
            gross_pnl = da_cash + settle_cash
            net_pnl = gross_pnl - fees
        elif "SELL" in action and vol > 0:
            da_cash = +(vol * p_spot)
            settle_cash = -(vol * p_act)
            fees = vol * fee_per_mwh
            gross_pnl = da_cash + settle_cash
            net_pnl = gross_pnl - fees
        else:
            da_cash = 0.0
            settle_cash = 0.0
            fees = 0.0
            gross_pnl = 0.0
            net_pnl = 0.0

        tax = max(0.0, net_pnl * tax_rate)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE trade_orders
                SET actual_settled_price_eur = ?,
                    da_cash_flow_eur = ?,
                    settle_cash_flow_eur = ?,
                    gross_pnl_eur = ?,
                    fees_eur = ?,
                    tax_eur = ?,
                    net_pnl_eur = ?,
                    status = 'SETTLED_AUDITED',
                    settled_at = CURRENT_TIMESTAMP
                WHERE market_mode = ? AND price_area = ? AND model_name = ? 
                  AND delivery_date = ? AND quarter_index = ?
            """, (
                p_act, da_cash, settle_cash, gross_pnl, fees, tax, net_pnl,
                market_mode, price_area, model_name, delivery_date, quarter_index
            ))
            conn.commit()

        return self.get_order(market_mode, price_area, model_name, delivery_date, quarter_index)

    def get_full_day_orders(self, market_mode, price_area, model_name, delivery_date):
        """Retrieves all 96 orders for a specific day and market mode."""
        with self._get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT * FROM trade_orders
                WHERE market_mode = ? AND price_area = ? AND model_name = ? AND delivery_date = ?
                ORDER BY quarter_index ASC
            """, conn, params=(market_mode, price_area, model_name, delivery_date))
            return df
