"""Tests for the new V4.1 data sources: parsers (mocked API payloads) and end-to-end
point-in-time features + leakage test on a COPY of the database filled with synthetic data.

Run from Basic_Approach:  python -m pytest tests_v41 -q
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v4_1_intraday import settings as S  # noqa: E402
from v4_1_intraday.collectors import common, entsoe_ext, frequency, nordpool_id as npid, umm, weather  # noqa: E402
from nurex42.storage import Store  # noqa: E402


# ----------------------------------------------------------------------------- parsers
def test_umm_parse_versions_and_units():
    items = [{
        "messageId": "M1", "version": 2, "publicationDate": "2026-09-01T10:00:00Z",
        "unavailabilityType": "Unplanned", "eventStatus": "Active", "messageType": 1,
        "productionUnits": [{"name": "Esbjergvaerket", "areaEic": "10YDK-1--------W", "installedCapacity": 400,
                             "fuelType": 14, "timePeriods": [
                                 {"eventStart": "2026-09-01T12:00:00Z", "eventStop": "2026-09-02T00:00:00Z",
                                  "unavailableCapacity": 400, "availableCapacity": 0}]}],
        "transmissionUnits": [{"name": "Skagerrak 4", "inAreaEic": "10YDK-1--------W", "outAreaEic": "10YNO-2--------T",
                               "timePeriods": [{"eventStart": "2026-09-01T12:00:00+02:00",
                                                "eventStop": "2026-09-01T18:00:00+02:00",
                                                "unavailableCapacity": 700}]}],
    }]
    df = umm.parse_messages(items)
    assert len(df) == 2
    p = df[df["asset_kind"] == "production"].iloc[0]
    assert p["area"] == "DK1" and p["unavailable_mw"] == 400 and p["version"] == 2
    t = df[df["asset_kind"] == "transmission"].iloc[0]
    assert t["area"] == "DK1>NO2" and t["event_start"] == pd.Timestamp("2026-09-01 10:00")


def test_weather_parse_multi_location():
    payload = [{"hourly": {"time": ["2026-09-01T00:00", "2026-09-01T01:00"],
                           "temperature_2m_previous_day1": [10.0, 11.0],
                           "wind_speed_100m_previous_day1": [8.0, None]}},
               {"hourly": {"time": ["2026-09-01T00:00", "2026-09-01T01:00"],
                           "temperature_2m_previous_day1": [12.0, 13.0],
                           "wind_speed_100m_previous_day1": [10.0, 12.0]}}]
    df = weather.parse(payload, "DK1")
    t = df[df["var"] == "temperature_2m"].set_index("time_utc")["value"]
    assert t[pd.Timestamp("2026-09-01 00:00")] == 11.0
    assert t[pd.Timestamp("2026-09-01 01:45")] == 12.0          # hourly value on all four quarters
    w = df[df["var"] == "wind_speed_100m"].set_index("time_utc")["value"]
    assert w[pd.Timestamp("2026-09-01 01:00")] == 12.0          # mean ignores missing


def test_frequency_to_quarters():
    raw = pd.DataFrame({"startTime": pd.date_range("2026-09-01", periods=10, freq="3min", tz="UTC").astype(str),
                        "value": [50.0, 50.1, 49.9, 50.0, 50.02, 49.95, 50.0, 50.0, 50.0, 50.0]})
    q = frequency.to_quarters(raw)
    assert list(q["n"]) == [5, 5]
    assert abs(q["f_min"].iloc[0] - 49.9) < 1e-9


def test_entsoe_to_15min_and_net_flows():
    idx = pd.date_range("2026-09-01", periods=3, freq="h", tz="Europe/Copenhagen")
    s = entsoe_ext.to_15min(pd.Series([1.0, 2.0, 3.0], index=idx))
    assert len(s) == 12 and s.iloc[5] == 2.0 and s.index[0] == pd.Timestamp("2026-08-31 22:00")

    class Fake:
        def query_scheduled_exchanges(self, a, b, start, end, dayahead=False):
            i = pd.date_range(start, end, freq="15min", inclusive="left")
            return pd.Series(100.0 if a == "DK_1" else 30.0, index=i)
        query_crossborder_flows = query_scheduled_exchanges

        def query_load(self, z, start, end):
            i = pd.date_range(start, end, freq="15min", inclusive="left")
            return pd.DataFrame({"Actual Load": 3000.0}, index=i)
        query_load_forecast = query_load

        def query_net_transfer_capacity_dayahead(self, a, b, start, end):
            raise RuntimeError("No matching data")

        def query_wind_and_solar_forecast(self, z, start, end):
            i = pd.date_range(start, end, freq="h", inclusive="left")
            return pd.DataFrame({"Solar": 10.0, "Wind Onshore": 20.0}, index=i)

    st = pd.Timestamp("2026-09-01", tz="UTC")
    df = entsoe_ext.fetch_window(Fake(), st, st + pd.Timedelta(hours=2))
    v = df[df["series"] == "sched:DK1>DE_LU"]["value"]
    assert (v == 70.0).all()                        # 100 export - 30 import
    assert set(df["series"]) >= {"load:DK1", "loadfc:DE_LU", "phys:DK1>NO_2", "wsfc:DE_LU:Solar"}
    assert not any(df["series"].str.startswith("ntc:"))


def test_stomp_frames_and_router():
    f = npid.stomp_frame("SUBSCRIBE", {"destination": "/user/u/v1/streaming/ticker", "id": "sub-1"})
    cmd, h, body = npid.parse_frame(f)
    assert cmd == "SUBSCRIBE" and h["id"] == "sub-1" and body == ""
    assert npid.parse_frame("\n") is None
    r = npid.Router(dict(npid.DEFAULTS))
    now = pd.Timestamp("2026-09-01 10:00", tz="UTC")
    r.handle("/user/u/v1/streaming/deliveryAreas", {}, [{"deliveryAreaId": 7, "eicCode": "10YDK-1--------W"}], now)
    assert r.area_ids == {7: "DK1"}
    r.handle("/user/u/v1/conflated/localview/7", {"x-nps-snapshot": "true"},
             [{"contractId": "C1", "deliveryAreaId": 7, "revisionNo": 1,
               "buyOrders": [{"orderId": "b1", "price": 10000, "qty": 5000}, {"orderId": "b2", "price": 9800, "qty": 1000}],
               "sellOrders": [{"orderId": "s1", "price": 10300, "qty": 2000}]}], now)
    r.handle("/user/u/v1/conflated/localview/7", {},
             [{"contractId": "C1", "deliveryAreaId": 7, "revisionNo": 2,
               "buyOrders": [{"orderId": "b1", "deleted": True}], "sellOrders": []}], now)
    book = r.rows["book"][-1]
    assert book["best_bid"] == 9800 and book["best_ask"] == 10300 and book["n_bid"] == 1
    r.handle("/user/u/v1/streaming/ticker", {}, [{"tradeId": "T1", "tradeTime": "2026-09-01T09:59:00Z", "legs": [
        {"contractId": "C1", "deliveryAreaId": 7, "side": "BUY", "unitPrice": 10100, "quantity": 1000, "aggressor": True},
        {"contractId": "C1", "deliveryAreaId": 99, "side": "SELL", "unitPrice": 10100, "quantity": 1000, "aggressor": False}]}], now)
    assert len(r.rows["trades"]) == 1                       # leg in a non-DK area filtered out
    assert npid.decode_body(json.dumps({"a": 1})) == {"a": 1}


# ----------------------------------------------------------------------------- end-to-end on a copy
def _synthetic(db: Path, idroot: Path, area_start: pd.Timestamp, area_end: pd.Timestamp):
    st = Store(db)
    common.ensure_tables(st)
    rng = np.random.default_rng(1)
    idx = pd.date_range(area_start, area_end, freq="15min")
    rows = []
    for name, base in [("load:DK1", 3000), ("loadfc:DK1", 3000), ("load:DK2", 1800), ("loadfc:DK2", 1800),
                       ("load:DE_LU", 55000), ("loadfc:DE_LU", 55000), ("sched:DK1>DE_LU", 500),
                       ("phys:DK1>DE_LU", 500), ("ntc:DK1>DE_LU", 2500), ("wsfc:DE_LU:Solar", 8000),
                       ("wsfc:DE_LU:Wind Onshore", 15000), ("sched:DK2>SE_4", -300), ("phys:DK2>SE_4", -300)]:
        rows.append(pd.DataFrame({"series": name, "time_utc": idx, "value": base + rng.normal(0, 100, len(idx))}))
    st.upsert("entsoe_series", pd.concat(rows))
    ev = []
    for k in range(60):
        t0 = area_start + pd.Timedelta(hours=int(rng.integers(0, int((area_end - area_start) / pd.Timedelta(hours=1)) - 30)))
        for v in (1, 2):
            ev.append({"message_id": f"M{k}", "version": v, "publication_utc": t0 + pd.Timedelta(hours=2 * v),
                       "message_type": "1", "unavailability_type": "Unplanned" if k % 2 else "Planned",
                       "event_status": "Active", "asset_kind": "production", "asset_name": f"U{k}",
                       "area": "DK1" if k % 3 else "DK2", "area_eic": "", "fuel": "", "installed_mw": 400.0,
                       "available_mw": 0.0, "unavailable_mw": 100.0 * v,
                       "event_start": t0 + pd.Timedelta(hours=3), "event_stop": t0 + pd.Timedelta(hours=20)})
    st.upsert("umm_events", pd.DataFrame(ev))
    wx = []
    for a in ("DK1", "DK2"):
        for v in weather.VARS:
            wx.append(pd.DataFrame({"area": a, "time_utc": idx, "var": v, "run": "d1",
                                    "value": rng.normal(10, 3, len(idx))}))
    st.upsert("weather_fc", pd.concat(wx))
    st.upsert("frequency", pd.DataFrame({"area": "NORDIC", "time_utc": idx, "f_mean": 50 + rng.normal(0, .02, len(idx)),
                                         "f_std": .01, "f_min": 49.95, "f_max": 50.05, "n": 5}))
    st.close()
    # intraday market (parquet)
    recv = pd.Timestamp("2026-09-10")
    contracts, stats, trades, book = [], [], [], []
    for i, q in enumerate(pd.date_range(area_end - pd.Timedelta(days=5), area_end, freq="15min")):
        cid = f"Q{i}"
        contracts.append({"recv_utc": recv, "contract_id": cid, "name": cid, "product_type": "X", "product_id": "P",
                          "dlvry_start": q, "dlvry_end": q + pd.Timedelta(minutes=15), "duration_s": 900,
                          "deleted": False, "area_ids": "[7]"})
        for m in range(0, 180, 20):
            t = q - pd.Timedelta(minutes=180 - m)
            stats.append({"recv_utc": t, "sent_at": t, "contract_id": cid, "area_id": 7, "last_price": 10000 + m,
                          "last_qty": 1000, "last_trade_time": t, "high": 11000, "low": 9000, "vwap": 10000 + m,
                          "turnover": 5000, "da_price": 10000, "deleted": False, "updated_at": t})
            trades.append({"recv_utc": t, "trade_id": f"{cid}-{m}", "trade_time": t, "state": "", "deleted": False,
                           "contract_id": cid, "area_id": 7, "side": "BUY" if m % 40 else "SELL", "price": 10000 + m,
                           "qty": 1000, "aggressor": True})
            book.append({"recv_utc": t, "contract_id": cid, "area_id": 7, "revision": m, "best_bid": 9900 + m,
                         "best_ask": 10100 + m, "bid_qty_best": 1000, "ask_qty_best": 500, "bid_qty_depth": 3000,
                         "ask_qty_depth": 1000, "n_bid": 3, "n_ask": 2})
    tables = {"areas": [{"recv_utc": recv, "area_id": 7, "eic": "10YDK-1--------W", "area_code": "DK1", "dk": "DK1"}],
              "contracts": contracts, "stats": stats, "trades": trades, "book": book}
    npid.write_rows(tables, idroot)


@pytest.fixture(scope="module")
def synth_env(tmp_path_factory):
    cfg0 = S.load()
    if not cfg0.db_path.exists():
        pytest.skip("no V4.2 database")
    tmp = tmp_path_factory.mktemp("v41src")
    db = tmp / "copy.duckdb"
    shutil.copy(cfg0.db_path, db)
    idroot = tmp / "nordpool_id"
    s0 = Store(cfg0.db_path, read_only=True)
    end = pd.Timestamp(s0.df("SELECT max(time_utc) t FROM imbalance WHERE imbalance_eur IS NOT NULL")["t"].iloc[0])
    s0.close()
    _synthetic(db, idroot, end - pd.Timedelta(days=40), end)
    cfg = S.load(db_path=str(db))
    cfg.raw["intraday_data_dir"] = str(idroot)
    orig = common.data_dir
    common.data_dir = lambda: idroot
    npid.data_dir = lambda: idroot
    yield cfg, db, end
    common.data_dir = orig
    npid.data_dir = orig


def test_new_features_present_and_point_in_time(synth_env):
    from v4_1_intraday import pipeline_id as P
    from v4_1_intraday.features_id import IntradayFeatureBuilder
    cfg, db, end = synth_env
    st = Store(db, read_only=True)
    fb = IntradayFeatureBuilder(st, cfg)
    q = pd.Series(pd.date_range(end - pd.Timedelta(days=2), end - pd.Timedelta(hours=2), freq="15min"))
    X = P.build_X(fb, "DK1", q, q - pd.Timedelta(minutes=60))
    for c in ["e_loadfc_own", "e_loaderr_own_last", "e_sched_DE_LU", "e_headroom_DE_LU", "e_physdev_DE_LU_last",
              "e_wsfc_de", "wx_temperature_2m", "freq_f_dev", "umm_prod_unplanned_own",
              "id15_vwap_minus_da", "book15_mid_minus_da", "trd15_n_60m"]:
        assert c in X.columns, c
        assert X[c].notna().any(), c
    # synthetic snapshots arrive at q-180 ... q-20 min with vwap 10000+m cents; at a = q-60 the
    # latest allowed one is m=120 -> 101.20 EUR (a later snapshot would give > 101.2)
    v = X["id15_vwap"].dropna()
    assert len(v) and np.allclose(v, 101.2)
    assert (X["id15_age_min"].dropna() == 0).all()
    st.close()


def test_leakage_with_all_sources(synth_env):
    from v4_1_intraday import leakage, pipeline_id as P
    from v4_1_intraday.features_id import IntradayFeatureBuilder
    cfg, db, end = synth_env
    st = Store(db, read_only=True)
    clean = IntradayFeatureBuilder(st, cfg)
    rng = np.random.default_rng(3)
    qs = pd.date_range(end - pd.Timedelta(days=4), end - pd.Timedelta(hours=3), freq="15min")
    for qq in rng.choice(qs, 4, replace=False):
        qq = pd.Timestamp(qq)
        for lead in (60, 90):
            a = qq - pd.Timedelta(minutes=lead)
            tgt = pd.DataFrame({"quarter_utc": [qq], "as_of_utc": [a]})
            x0 = clean.build("DK1", tgt)
            x1 = leakage._CorruptedBuilder(st, cfg, a).build("DK1", tgt)
            bad = [c for c in x0.columns if not (np.isnan(x0[c].iloc[0]) and np.isnan(x1[c].iloc[0]))
                   and not np.isclose(x0[c].iloc[0], x1[c].iloc[0])]
            assert not bad, f"{qq} lead {lead}: {bad}"
    st.close()
