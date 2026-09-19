"""DuckDB storage: raw market data + simulation ledger.

Raw tables hold exactly what the public APIs return (mapped to snake_case),
aggregated to 15-minute buckets where the source is finer. Missing values stay
NULL - nothing is ever filled with defaults.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

SCHEMA = {
    # ---------------- raw market data ----------------
    "imbalance": """
        area VARCHAR, time_utc TIMESTAMP,
        spot_eur DOUBLE, imbalance_eur DOUBLE, satisfied_demand_mw DOUBLE,
        dominating_direction DOUBLE,
        afrr_up_mw DOUBLE, afrr_down_mw DOUBLE, afrr_vwa_up_eur DOUBLE, afrr_vwa_down_eur DOUBLE,
        mfrr_price_up_eur DOUBLE, mfrr_price_down_eur DOUBLE,
        source VARCHAR, ingested_at TIMESTAMP,
        PRIMARY KEY (area, time_utc)""",
    "dayahead": """
        area VARCHAR, time_utc TIMESTAMP, price_eur DOUBLE,
        source VARCHAR, ingested_at TIMESTAMP,
        PRIMARY KEY (area, time_utc)""",
    "forecast": """
        area VARCHAR, time_utc TIMESTAMP, ftype VARCHAR,
        f_da DOUBLE, f_5h DOUBLE, f_1h DOUBLE,
        source VARCHAR, ingested_at TIMESTAMP,
        PRIMARY KEY (area, time_utc, ftype)""",
    "psrn": """
        time_utc TIMESTAMP,
        prod_ge100_mw DOUBLE, prod_lt100_mw DOUBLE, solar_mw DOUBLE,
        offshore_mw DOUBLE, onshore_mw DOUBLE, exch_sum_mw DOUBLE,
        exch_dk1_de DOUBLE, exch_dk1_nl DOUBLE, exch_dk1_gb DOUBLE, exch_dk1_no DOUBLE,
        exch_dk1_se DOUBLE, exch_dk1_dk2 DOUBLE, exch_dk2_de DOUBLE, exch_dk2_se DOUBLE,
        afrr_act_dk1 DOUBLE, afrr_act_dk2 DOUBLE,
        n_minutes INTEGER, ingested_at TIMESTAMP,
        PRIMARY KEY (time_utc)""",
    "mfrr_market": """
        area VARCHAR, time_utc TIMESTAMP,
        sa_up_req_mw DOUBLE, sa_up_eur DOUBLE, sa_down_req_mw DOUBLE, sa_down_eur DOUBLE,
        da_up_mw DOUBLE, da_up_eur DOUBLE, da_down_mw DOUBLE, da_down_eur DOUBLE,
        total_up_mw DOUBLE, total_down_mw DOUBLE,
        ingested_at TIMESTAMP,
        PRIMARY KEY (area, time_utc)""",
    "entsoe_series": """
        series VARCHAR, time_utc TIMESTAMP, value DOUBLE, ingested_at TIMESTAMP,
        PRIMARY KEY (series, time_utc)""",
    # ---------------- simulation ledger ----------------
    "runs": """
        run_id VARCHAR PRIMARY KEY, book VARCHAR, kind VARCHAR, area VARCHAR,
        as_of_utc TIMESTAMP, model_id VARCHAR, n_quarters INTEGER, created_at TIMESTAMP""",
    "plan": """
        run_id VARCHAR, book VARCHAR, area VARCHAR, quarter_utc TIMESTAMP, batch_start_utc TIMESTAMP,
        zone VARCHAR, action VARCHAR, mwh DOUBLE, exp_spread DOUBLE, p_up DOUBLE, p_down DOUBLE,
        q10 DOUBLE, q50 DOUBLE, q90 DOUBLE, edge DOUBLE, reason VARCHAR,
        PRIMARY KEY (run_id, area, quarter_utc)""",
    "positions": """
        book VARCHAR, area VARCHAR, quarter_utc TIMESTAMP, batch_start_utc TIMESTAMP,
        action VARCHAR, mwh DOUBLE, exp_spread DOUBLE, p_up DOUBLE, p_down DOUBLE,
        q10 DOUBLE, q50 DOUBLE, q90 DOUBLE, run_id VARCHAR, locked_at TIMESTAMP, reason VARCHAR,
        PRIMARY KEY (book, area, quarter_utc)""",
    "batches": """
        book VARCHAR, area VARCHAR, batch_start_utc TIMESTAMP, deadline_utc TIMESTAMP,
        run_id VARCHAR, locked_at TIMESTAMP, file_path VARCHAR, n_trades INTEGER, mwh_total DOUBLE,
        PRIMARY KEY (book, area, batch_start_utc)""",
    "settlements": """
        book VARCHAR, area VARCHAR, quarter_utc TIMESTAMP, action VARCHAR, mwh DOUBLE,
        spot_eur DOUBLE, imbalance_eur DOUBLE, spread_eur DOUBLE,
        gross_eur DOUBLE, cost_eur DOUBLE, net_eur DOUBLE, settled_at TIMESTAMP,
        PRIMARY KEY (book, area, quarter_utc)""",
}


class Store:
    def __init__(self, path: str | Path, read_only: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        self.con = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:
            self.init_schema()

    def init_schema(self) -> None:
        for name, cols in SCHEMA.items():
            self.con.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols})")

    def close(self) -> None:
        self.con.close()

    def upsert(self, table: str, df: pd.DataFrame) -> int:
        """Insert or replace rows by primary key. Columns not supplied become NULL."""
        if df is None or df.empty:
            return 0
        cols = [r[0] for r in self.con.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name='{table}' "
            f"ORDER BY ordinal_position").fetchall()]
        d = df.copy()
        if "ingested_at" in cols and "ingested_at" not in d.columns:
            d["ingested_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
        for c in cols:
            if c not in d.columns:
                d[c] = None
        d = d[cols]
        self.con.register("_up", d)
        try:
            self.con.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _up")
        finally:
            self.con.unregister("_up")
        return len(d)

    def df(self, sql: str, params=None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    def max_time(self, table: str, where: str = "") -> pd.Timestamp | None:
        w = f"WHERE {where}" if where else ""
        v = self.con.execute(f"SELECT max(time_utc) FROM {table} {w}").fetchone()[0]
        return pd.Timestamp(v) if v is not None else None

    def min_time(self, table: str, where: str = "") -> pd.Timestamp | None:
        w = f"WHERE {where}" if where else ""
        v = self.con.execute(f"SELECT min(time_utc) FROM {table} {w}").fetchone()[0]
        return pd.Timestamp(v) if v is not None else None

    def coverage(self) -> pd.DataFrame:
        rows = []
        for t, key in [("imbalance", "area"), ("dayahead", "area"), ("forecast", "area || ':' || ftype"),
                       ("mfrr_market", "area"), ("entsoe_series", "series")]:
            q = (f"SELECT '{t}' AS tbl, {key} AS key, count(*) AS n_rows, min(time_utc) AS first, "
                 f"max(time_utc) AS last FROM {t} GROUP BY 2 ORDER BY 2")
            rows.append(self.df(q))
        rows.append(self.df("SELECT 'psrn' AS tbl, 'all' AS key, count(*) AS n_rows, min(time_utc) AS first, "
                            "max(time_utc) AS last FROM psrn"))
        out = pd.concat(rows, ignore_index=True)
        out["expected_quarters"] = ((pd.to_datetime(out["last"]) - pd.to_datetime(out["first"]))
                                    / pd.Timedelta(minutes=15) + 1).round()
        out["completeness_pct"] = (100 * out["n_rows"] / out["expected_quarters"]).round(1)
        return out
