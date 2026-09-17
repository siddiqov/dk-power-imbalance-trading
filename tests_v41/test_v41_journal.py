"""Journal: lock only before the gate, missed gates stay HOLD, settlement uses the published spread."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4_1_intraday import journal as J, settings as S, pipeline_id as P  # noqa: E402
from v4_1_intraday.features_id import IntradayFeatureBuilder  # noqa: E402
from nurex42.storage import Store  # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    cfg = S.load()
    if not cfg.db_path.exists() or not P.model_path(cfg, "DK1").exists():
        pytest.skip("needs V4.2 store and trained V4.1 models")
    cfg.raw["journal_path"] = str(tmp_path_factory.mktemp("j") / "journal.sqlite")
    cfg.raw["areas"] = ["DK1"]
    st = Store(cfg.db_path, read_only=True)
    fb = IntradayFeatureBuilder(st, cfg)
    b = {"DK1": P.IntradayBundle.load(P.model_path(cfg, "DK1"))}
    yield cfg, st, fb, b
    st.close()


def test_lock_missed_and_settle(env):
    cfg, st, fb, b = env
    t0 = pd.Timestamp("2026-09-16 09:07")                     # first run: nothing missed
    r0 = J.lock(st, cfg, now=t0, fb=fb, bundles=b)
    assert r0["DK1"]["missed"] == 0 and r0["DK1"]["locked"] >= 1
    j = J.read(cfg, "DK1")
    assert (j["as_of_utc"] < j["gate_utc"]).all()             # decided strictly before gate closure
    assert (j["as_of_utc"] == t0).all()
    # system down for 2 hours -> gates in between are MISSED, never decided late
    t1 = t0 + pd.Timedelta(hours=2)
    r1 = J.lock(st, cfg, now=t1, fb=fb, bundles=b)
    j = J.read(cfg, "DK1")
    missed = j[j["status"] == "MISSED"]
    assert len(missed) == r1["DK1"]["missed"] > 0
    assert (missed["action"] == "HOLD").all()
    assert (missed["gate_utc"] <= t1).all()
    # locking again does not change existing rows
    before = J.read(cfg, "DK1")
    J.lock(st, cfg, now=t1 + pd.Timedelta(minutes=1), fb=fb, bundles=b)
    after = J.read(cfg, "DK1")
    pd.testing.assert_frame_equal(before, after.iloc[:len(before)].reset_index(drop=True))
    n = J.settle(st, cfg, now=pd.Timestamp("2026-09-17 00:00"))
    j = J.read(cfg, "DK1")
    assert n == len(j) and j["spread_actual"].notna().all()
    tr = j[j["action"] != "HOLD"]
    exp = (tr["action"].map({"BUY": 1, "SELL": -1}) * tr["mwh"] * tr["spread_actual"]
           - tr["mwh"] * tr["cost_eur_mwh"])
    assert (abs(exp - tr["pnl_eur"]) < 1e-6).all()
    assert (j.loc[j["action"] == "HOLD", "pnl_eur"] == 0).all()
    s = J.summary(cfg)
    assert s.loc[0, "missed_gates"] == len(missed)
