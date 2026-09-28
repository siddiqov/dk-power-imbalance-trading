"""Hourly client batches (intraday.mode = batch): schedule, point-in-time decisions, journal locking."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4_1_intraday import journal as J, settings as S, pipeline_id as P  # noqa: E402
from v4_1_intraday.features_id import IntradayFeatureBuilder  # noqa: E402
from nurex42.storage import Store  # noqa: E402

TZ = "Europe/Copenhagen"


def utc(local: str) -> pd.Timestamp:
    return pd.Timestamp(local).tz_localize(TZ).tz_convert("UTC").tz_localize(None)


def loc(ts) -> pd.Series:
    return pd.to_datetime(pd.Series(ts)).dt.tz_localize("UTC").dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %H:%M")


# ------------------------------------------------------------------ schedule (no data needed)
def test_client_schedule_examples():
    cfg = S.load(mode="batch")
    q = pd.Series([utc("2026-09-28 00:00"), utc("2026-09-28 00:45"), utc("2026-09-28 01:00"),
                   utc("2026-09-28 10:00"), utc("2026-09-28 10:45")])
    assert list(loc(cfg.deadlines(q))) == ["2026-09-27 21:45", "2026-09-27 21:45", "2026-09-27 22:45",
                                          "2026-09-28 07:45", "2026-09-28 07:45"]
    assert list(loc(cfg.decision_times(q))) == ["2026-09-27 21:30", "2026-09-27 21:30", "2026-09-27 22:30",
                                               "2026-09-28 07:30", "2026-09-28 07:30"]
    assert list(J.quarter_number(q, TZ)) == [1, 4, 5, 41, 44]


def test_every_hour_is_one_batch_of_four_including_clock_changes():
    cfg = S.load(mode="batch")
    from nurex42.timeutil import local_day_quarters
    for day, n in (("2026-09-28", 96), ("2026-10-25", 100), ("2026-03-29", 92)):
        q = pd.Series(local_day_quarters(day, TZ))
        assert len(q) == n
        assert list(J.quarter_number(q, TZ)) == list(range(1, n + 1))
        sizes = cfg.batch_starts(q).value_counts()
        assert (sizes == 4).all() and len(sizes) == n // 4
        lead = (q - cfg.deadlines(q)).dt.total_seconds() / 60
        assert set(lead) == {135, 150, 165, 180}             # 2h15 before the first quarter of its hour
        assert ((cfg.deadlines(q) - cfg.decision_times(q)) == pd.Timedelta(minutes=15)).all()


def test_gate_mode_unchanged():
    cfg = S.load(mode="gate")
    q = pd.Series([utc("2026-09-28 10:00"), utc("2026-09-28 10:15")])
    assert (cfg.decision_times(q) == q - pd.Timedelta(minutes=60)).all()
    assert (cfg.deadlines(q) == cfg.decision_times(q)).all()
    assert (cfg.batch_starts(q) == q).all()
    assert P.model_path(cfg, "DK1").name == "v4_1_intraday_DK1.pkl"
    assert P.model_path(S.load(mode="batch"), "DK1").name == "v4_1_batch_DK1.pkl"


# ------------------------------------------------------------------ with data
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    cfg = S.load(mode="batch")
    gate = S.load(mode="gate")
    if not cfg.db_path.exists() or not P.model_path(gate, "DK1").exists():
        pytest.skip("needs V4.2 store and a trained V4.1 model")
    tmp = tmp_path_factory.mktemp("b")
    cfg.raw["journal_path"] = str(tmp / "journal.sqlite")
    cfg.raw["batch_schedule"]["batch_files_dir"] = str(tmp / "batches")
    cfg.raw["areas"] = ["DK1"]
    st = Store(cfg.db_path, read_only=True)
    fb = IntradayFeatureBuilder(st, cfg)
    # mechanics only: any trained bundle can drive the journal (batch models are trained on the PC)
    b = {"DK1": P.IntradayBundle.load(P.model_path(gate, "DK1"))}
    yield cfg, st, fb, b, tmp
    st.close()


def test_batch_rows_are_point_in_time(env):
    cfg, st, fb, _, _ = env
    t = P.target_frame(fb, "DK1", cfg, start="2026-09-01", end="2026-09-03")
    assert (t["as_of_trade"] < t["deadline_utc"]).all()
    assert (t.groupby("batch_start_utc")["as_of_trade"].nunique() == 1).all()     # one decision per batch
    lead = (t["quarter_utc"] - t["as_of_trade"]).dt.total_seconds() / 60
    assert lead.min() == 150 and lead.max() == 195
    X = P.build_X(fb, "DK1", t["quarter_utc"], t["as_of_trade"])
    assert (X["lead_min"].values == lead.values).all()


def test_overnight_cycle_locks_hour_by_hour(env):
    cfg, st, fb, b, tmp = env
    # a 15-minute cycle from 21:25 to 07:55 (local) on a day with data in the store
    day = pd.Timestamp(st.df("SELECT max(time_utc) AS t FROM imbalance")["t"].iloc[0]).normalize() - pd.Timedelta(days=3)
    d0 = pd.Timestamp(day.date())
    t = utc(f"{(d0 - pd.Timedelta(days=1)).date()} 21:25")
    end = utc(f"{d0.date()} 07:55")
    runs = []
    while t <= end:
        runs.append(J.lock(st, cfg, now=t, fb=fb, bundles=b)["DK1"])
        t += pd.Timedelta(minutes=15)
    j = J.read(cfg, area="DK1")
    lk = j[j["status"] == "LOCKED"]
    # Q1..Q44 of the day are locked (07:45 deadline covers 10:00-10:45), nothing later
    qn = J.quarter_number(lk["quarter_utc"], TZ)
    day_rows = lk[loc(lk["quarter_utc"]).str.startswith(str(d0.date())).values]
    assert sorted(J.quarter_number(day_rows["quarter_utc"], TZ)) == list(range(1, 45))
    assert (lk["as_of_utc"] <= lk["deadline_utc"] - pd.Timedelta(minutes=15)).all()   # decided with data <= cut-off
    assert (lk["locked_at_utc"] < lk["deadline_utc"]).all()                            # locked before the deadline
    assert (lk.groupby("batch_start_utc")["locked_at_utc"].nunique() == 1).all()        # 4 quarters together
    assert (lk.groupby("batch_start_utc").size() == 4).all()
    assert (j["status"] == "MISSED").sum() == 0
    # one CSV + JSON per batch for the BRP
    files = sorted((tmp / "batches" / "DK1").rglob("batch_*.csv"))
    assert len(files) == lk["batch_start_utc"].nunique()
    f = pd.read_csv(files[-1])
    assert len(f) == 4 and set(f["action"]) <= {"BUY", "SELL", "HOLD"}


def test_missed_batch_is_never_decided_late(env):
    cfg, st, fb, b, _ = env
    j0 = J.read(cfg, area="DK1")
    last = j0["quarter_utc"].max()
    # system down for 3 hours after the last lock -> those batches become MISSED/HOLD
    now = pd.Timestamp(j0["locked_at_utc"].max()) + pd.Timedelta(hours=3)
    J.lock(st, cfg, now=now, fb=fb, bundles=b)
    j = J.read(cfg, area="DK1")
    new = j[j["quarter_utc"] > last]
    missed = new[new["status"] == "MISSED"]
    assert len(missed) >= 8 and (missed["action"] == "HOLD").all()
    assert (missed["deadline_utc"] <= now).all()
    assert (new.loc[new["status"] == "LOCKED", "deadline_utc"] > now).all()
