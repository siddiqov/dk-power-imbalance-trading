"""pytest for Nurex V4.1 Intraday.   Run from Basic_Approach:  python -m pytest tests_v41 -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v4_1_intraday import leakage, settings as S, pipeline_id as P  # noqa: E402
from v4_1_intraday.features_id import IntradayFeatureBuilder, danish_holidays  # noqa: E402
from nurex42.storage import Store  # noqa: E402
from datetime import date  # noqa: E402


@pytest.fixture(scope="module")
def env():
    cfg = S.load()
    if not cfg.db_path.exists():
        pytest.skip("no V4.2 database")
    st = Store(cfg.db_path, read_only=True)
    yield cfg, st
    st.close()


def test_no_future_information(env):
    cfg, st = env
    assert leakage.run(st, cfg, n_samples=3) == []


def test_target_not_in_features(env):
    cfg, st = env
    fb = IntradayFeatureBuilder(st, cfg)
    t = P.target_frame(fb, "DK1", cfg).iloc[-4000:]
    X = P.build_X(fb, "DK1", t["quarter_utc"], t["as_of_trade"])
    y = t["spread"].values
    for c in X.columns:
        v = X[c].values
        m = ~np.isnan(v)
        if m.sum() > 200 and np.nanstd(v[m]) > 0:
            assert abs(np.corrcoef(v[m], y[m])[0, 1]) < 0.9, c


def test_gate_closure_times(env):
    cfg, st = env
    fb = IntradayFeatureBuilder(st, cfg)
    t = P.target_frame(fb, "DK1", cfg).iloc[:10]
    assert ((t["quarter_utc"] - t["as_of_trade"]) == pd.Timedelta(minutes=60)).all()
    assert (t["label_known_at"] > t["as_of_trade"]).all()


def test_holidays():
    h = danish_holidays([2026])
    assert date(2026, 4, 5) in h and date(2026, 5, 14) in h and date(2026, 12, 25) in h


def test_risk_overlay_never_exceeds_cap():
    cfg = S.load()
    n = 200
    q = pd.date_range("2026-01-01", periods=n, freq="15min")
    df = pd.DataFrame({"quarter_utc": q, "batch_start_utc": q, "as_of_trade": q - pd.Timedelta(minutes=60),
                       "label_known_at": q + pd.Timedelta(minutes=45), "spread": -500.0,
                       "action": "BUY", "mwh": 10.0, "edge": 5.0, "reason": "edge"})
    out = P.apply_risk_overlays(df, cfg)
    assert out["mwh"].max() <= cfg["risk"]["max_mwh_per_quarter"]
    assert (out["action"] == "HOLD").any()          # daily stop / drawdown guard kicked in
