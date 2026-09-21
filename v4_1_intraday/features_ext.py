"""Point-in-time features from the new sources (ENTSO-E, UMM, weather, frequency, Nord Pool intraday).

Used by IntradayFeatureBuilder. Every block follows the same rule as the rest of V4.1:
a raw value is used only if it was public at the decision time `a`; missing -> NaN.

  ENTSO-E day-ahead   (known D-1 13:00 local; wind/solar forecast D-1 18:00 local)
      load forecast own/DE, scheduled exchanges per border, NTC headroom, DE wind+solar forecast,
      DE residual-load forecast, own renewable share
  ENTSO-E realised    (quarter end + entsoe_actual_lag_minutes)
      actual load, load forecast error (+ 1h mean), physical flow - scheduled per border
  Live flow deviation (PowerSystemRightNow flow at the latest bucket - day-ahead schedule, sign-calibrated)
  UMM outages         (message version valid at `a`, event active at the delivery quarter)
      unavailable MW own zone (planned / unplanned), transmission, neighbours, new unplanned in last 2h
  Weather             (Open-Meteo previous-day run: known 24h before the hour)
  Frequency           (Fingrid 3-min -> 15-min stats, bucket end + frequency_lag_minutes)
  Intraday market     (Nord Pool recorder, receive time <= a)
      15-min and hourly contract VWAP / last / range / turnover, ID-DA spread, trades of the last
      hour (count, signed aggressor volume), order book mid / spread / depth imbalance
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from nurex42.features import _grid, _local, _local_hhmm_on_prev_day

log = logging.getLogger("nurex41id.features")
Q = pd.Timedelta(minutes=15)
JUNK = 9999.0

# border name in entsoe_series  ->  PowerSystemRightNow column (as seen from the zone)
PSRN_BORDER = {
    "DK1": {"DE_LU": "exch_dk1_de", "NL": "exch_dk1_nl", "GB": "exch_dk1_gb", "NO_2": "exch_dk1_no",
            "SE_3": "exch_dk1_se", "DK2": "exch_dk1_dk2"},
    "DK2": {"DE_LU": "exch_dk2_de", "SE_4": "exch_dk2_se", "DK1": "exch_dk1_dk2"},
}
NEIGHBOUR_AREAS = {"DK1": ["DE_LU", "NO2", "SE3", "NL"], "DK2": ["DE_LU", "SE4"]}


def _unplanned(s: pd.Series) -> pd.Series:
    # Nord Pool UMM: unavailabilityType 1 = unplanned, 2 = planned (verified on live data 17 Sep 2026)
    t = s.astype(str).str.lower().str.replace(".0", "", regex=False)
    return t.str.contains("unplan") | (t == "1")


def _dismissed(s: pd.Series) -> pd.Series:
    # Nord Pool UMM: eventStatus 1 = active, 3 = dismissed / replaced
    t = s.astype(str).str.lower().str.replace(".0", "", regex=False)
    return t.str.contains("dismiss") | t.str.contains("cancel") | t.str.contains("withdraw") | (t == "3")


class ExtMixin:
    """Mixed into IntradayFeatureBuilder. Call _load_ext(store) at load time and _build_ext() in build."""

    # hook for the leakage test: corrupt raw frames before anything is derived from them
    def _before_derive(self, raw: dict) -> None:
        pass

    # ------------------------------------------------------------------ loading
    def _load_ext(self, store) -> None:
        av = self.cfg["availability"]
        self.lag_entsoe = pd.Timedelta(minutes=av.get("entsoe_actual_lag_minutes", 90))
        self.lag_freq = pd.Timedelta(minutes=av.get("frequency_lag_minutes", 5))
        self.weather_before = pd.Timedelta(minutes=av.get("weather_known_before_minutes", 1440))
        start, end = self.start, self.end
        raw = {}

        def q(sql, params=()):
            try:
                return store.df(sql, list(params))
            except Exception:           # table not created yet
                return pd.DataFrame()

        raw["ent"] = q("SELECT series, time_utc, value FROM entsoe_series WHERE time_utc >= ?", [start])
        raw["umm"] = q("SELECT * FROM umm_events WHERE event_stop >= ?", [start])
        raw["wx"] = q("SELECT area, time_utc, var, value FROM weather_fc WHERE time_utc >= ? AND run = 'd1'", [start])
        raw["freq"] = q("SELECT * FROM frequency WHERE time_utc >= ?", [start])
        raw["id"] = self._load_intraday_market()
        self._before_derive(raw)

        # ENTSO-E: one wide 15-min grid
        e = raw["ent"]
        self.ent2 = (_grid(e.pivot_table(index="time_utc", columns="series", values="value"), start, end)
                     if len(e) else None)
        self.umm = raw["umm"]
        self._umm_cache = {}
        w = raw["wx"]
        self.wx = {}
        for a in ("DK1", "DK2"):
            d = w[w["area"] == a] if len(w) else w
            if len(d):
                g = _grid(d.pivot_table(index="time_utc", columns="var", values="value"), start, end)
                for c in list(g.columns):
                    g[f"{c}_dev7d"] = g[c] - g[c].rolling(96 * 7, min_periods=96).mean()
                    g[f"{c}_d1h"] = g[c].shift(-4) - g[c]           # forecast change over the next hour
                self.wx[a] = g
        f = raw["freq"]
        if len(f):
            g = _grid(f[f["area"] == "NORDIC"].set_index("time_utc")[["f_mean", "f_std", "f_min", "f_max"]],
                      start, end)
            g["f_dev"] = g["f_mean"] - 50.0
            g["f_dev_m4"] = g["f_dev"].rolling(4, min_periods=1).mean()
            g["f_range"] = g["f_max"] - g["f_min"]
            self.freq = g
        else:
            self.freq = None
        self.idm = raw["id"]
        self._psrn_sign = {}

    def _load_intraday_market(self) -> dict:
        try:
            from .collectors import nordpool_id as npid
        except Exception:
            return {}
        try:
            root = npid.data_dir()
            out = {t: npid.load(t, root=root) for t in ("areas", "contracts", "stats", "trades", "book")}
            self.id_api = npid.cfg_api()
            return out
        except Exception as e:
            log.warning("intraday market data not loaded: %s", e)
            return {}

    # ------------------------------------------------------------------ helpers
    def _look_grid(self, frame, times, cols):
        cols = [c for c in cols if c in frame.columns]
        out = frame[cols].reindex(pd.DatetimeIndex(times))
        out.index = range(len(times))
        return out

    def _psrn_sign_for(self, sched: str, psrn_col: str):
        """Sign convention of the PSRN flow column relative to the ENTSO-E schedule direction.

        Calibrated ONLY on a fixed warm-up window at the very start of the store
        (`sign_calibration_days`, default 45). Using the full history would let data published
        after the decision time decide whether a feature exists at all - a look-ahead that also
        made the feature set unstable (leaktest KeyError 'live_flowdev_*', 17 Sep 2026).
        The convention is a static property of the feed, so an early window is enough."""
        key = (sched, psrn_col)
        if key not in self._psrn_sign:
            s = None
            if self.psrn is not None and psrn_col in self.psrn.columns and sched in self.ent2.columns:
                d = pd.concat([self.psrn[psrn_col], self.ent2[sched]], axis=1).dropna()
                if len(d):
                    days = int(self.cfg.get("sign_calibration_days", 45) or 45)
                    d = d.loc[: d.index.min() + pd.Timedelta(days=days)]
                if len(d) > 500:
                    c = np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1]
                    s = float(np.sign(c)) if abs(c) > 0.3 else None
            self._psrn_sign[key] = s
        return self._psrn_sign[key]

    # ------------------------------------------------------------------ build
    def _build_ext(self, X: pd.DataFrame, area: str, q: pd.Series, a: pd.Series) -> dict:
        cols = {}
        tz = self.tz
        av = self.cfg["availability"]
        # ---------------- ENTSO-E
        if self.ent2 is not None:
            E = self.ent2
            da_ok = (a >= _local_hhmm_on_prev_day(q, av.get("entsoe_dayahead_publish_local", "13:00"), tz)).values
            ws_ok = (a >= _local_hhmm_on_prev_day(q, av.get("entsoe_wsfc_publish_local", "18:00"), tz)).values
            v = self._look_grid(E, q, list(E.columns))
            nan = np.full(len(q), np.nan)
            g = lambda c, ok: np.where(ok, v[c].values, np.nan) if c in v.columns else nan
            cols["e_loadfc_own"] = g(f"loadfc:{area}", da_ok)
            cols["e_loadfc_de"] = g("loadfc:DE_LU", da_ok)
            ws_de = [c for c in v.columns if c.startswith("wsfc:DE_LU:")]
            if ws_de:
                tot = np.where(ws_ok, v[ws_de].sum(axis=1, min_count=1).values, np.nan)
                cols["e_wsfc_de"] = tot
                cols["e_resload_fc_de"] = cols["e_loadfc_de"] - tot
            ws_own = [c for c in v.columns if c.startswith(f"wsfc:{area}:")]
            if ws_own:
                cols["e_res_share_own"] = (np.where(ws_ok, v[ws_own].sum(axis=1, min_count=1).values, np.nan)
                                           / cols["e_loadfc_own"])
            # V4.2's collector names the other Danish zone DK_1/DK_2; keep one name per border
            sched_cols = [c for c in v.columns if c.startswith(f"sched:{area}>") and not c.endswith(("DK_1", "DK_2"))]
            if sched_cols:
                cols["e_sched_net"] = np.where(da_ok, v[sched_cols].sum(axis=1, min_count=1).values, np.nan)
            for sc in sched_cols:
                nb = sc.split(">", 1)[1]
                cols[f"e_sched_{nb}"] = g(sc, da_ok)
                ntc = f"ntc:{area}>{nb}"
                if ntc in v.columns:
                    cols[f"e_headroom_{nb}"] = g(ntc, da_ok) - np.abs(cols[f"e_sched_{nb}"])
            # realised, lagged
            te = (a - self.lag_entsoe).dt.floor("15min") - Q
            r = self._look_grid(E, te, list(E.columns))
            gr = lambda c: r[c].values if c in r.columns else np.full(len(q), np.nan)
            cols["e_load_own_last"] = gr(f"load:{area}")
            cols["e_loaderr_own_last"] = gr(f"load:{area}") - gr(f"loadfc:{area}")
            cols["e_loaderr_de_last"] = gr("load:DE_LU") - gr("loadfc:DE_LU")
            if f"load:{area}" in E.columns and f"loadfc:{area}" in E.columns:
                err = (E[f"load:{area}"] - E[f"loadfc:{area}"]).rolling(4, min_periods=2).mean()
                cols["e_loaderr_own_m4"] = err.reindex(pd.DatetimeIndex(te)).values
            devs = []
            for sc in sched_cols:
                nb = sc.split(">", 1)[1]
                ph = f"phys:{area}>{nb}"
                if ph in r.columns:
                    d = gr(ph) - gr(sc)
                    cols[f"e_physdev_{nb}_last"] = d
                    devs.append(d)
            cols["e_physdev_sum_last"] = (np.nansum(np.vstack(devs), axis=0) if devs
                                          else np.full(len(q), np.nan))
            # live flow deviation (PSRN, 5 min lag) vs schedule
            if self.psrn is not None:
                tp = (a - self.lag_psrn).dt.floor("15min") - Q
                ldev = []
                for nb, pcol in PSRN_BORDER.get(area, {}).items():
                    sc = f"sched:{area}>{nb}"
                    s = self._psrn_sign_for(sc, pcol) if sc in E.columns else None
                    if s is None:
                        # column still emitted (all-NaN) so the feature set never depends on data
                        cols[f"live_flowdev_{nb}"] = np.full(len(q), np.nan)
                        continue
                    live = self._look_grid(self.psrn, tp, [pcol])[pcol].values
                    sch = E[sc].reindex(pd.DatetimeIndex(tp)).values
                    ok = (a >= _local_hhmm_on_prev_day(tp, av.get("entsoe_dayahead_publish_local", "13:00"), tz)).values
                    d = np.where(ok, s * live - sch, np.nan)
                    cols[f"live_flowdev_{nb}"] = d
                    ldev.append(d)
                cols["live_flowdev_sum"] = (np.nansum(np.vstack(ldev), axis=0) if ldev
                                            else np.full(len(q), np.nan))
        # ---------------- UMM
        if self.umm is not None and len(self.umm):
            cols.update(self._umm_features(area, q, a))
        # ---------------- weather
        wx = self.wx.get(area)
        if wx is not None:
            v = self._look_grid(wx, q, list(wx.columns))
            known = pd.DatetimeIndex(q) - self.weather_before <= pd.DatetimeIndex(a)
            known_next = pd.DatetimeIndex(q) + pd.Timedelta(hours=1) - self.weather_before <= pd.DatetimeIndex(a)
            for c in v.columns:
                ok = known_next if c.endswith("_d1h") else known
                cols[f"wx_{c}"] = np.where(ok, v[c].values, np.nan)
        # ---------------- frequency
        if self.freq is not None:
            tf = (a - self.lag_freq).dt.floor("15min") - Q
            v = self._look_grid(self.freq, tf, ["f_dev", "f_std", "f_range", "f_dev_m4"])
            for c in v.columns:
                cols[f"freq_{c}"] = v[c].values
        # ---------------- intraday market
        if self.idm:
            cols.update(self._id_features(area, q, a, X))
        return cols

    # ------------------------------------------------------------------ UMM
    def _umm_versions(self) -> pd.DataFrame:
        if "versions" in self._umm_cache:
            return self._umm_cache["versions"]
        u = self.umm.copy()
        pubs = u.groupby(["message_id", "version"])["publication_utc"].min().reset_index()
        pubs = pubs.sort_values(["message_id", "version"])
        pubs["next_pub"] = pubs.groupby("message_id")["publication_utc"].shift(-1)
        first = pubs.groupby("message_id")["publication_utc"].min().rename("first_pub")
        u = u.merge(pubs[["message_id", "version", "next_pub"]], on=["message_id", "version"], how="left")
        u = u.merge(first, on="message_id", how="left")
        u["next_pub"] = u["next_pub"].fillna(pd.Timestamp("2100-01-01"))
        u = u[~_dismissed(u["event_status"])]
        u["unplanned"] = _unplanned(u["unavailability_type"])
        u["mw"] = pd.to_numeric(u["unavailable_mw"], errors="coerce").fillna(0.0).clip(lower=0)
        self._umm_cache["versions"] = u
        return u

    def _umm_groups(self, area: str) -> dict:
        u = self._umm_versions()
        own = u["area"] == area
        trans = (u["asset_kind"] == "transmission") & u["area"].astype(str).str.contains(area)
        prod = u["asset_kind"].isin(["production", "generation"])
        nb = u["area"].isin(NEIGHBOUR_AREAS.get(area, []))
        return {
            "umm_prod_unplanned_own": u[own & prod & u["unplanned"]],
            "umm_prod_planned_own": u[own & prod & ~u["unplanned"]],
            "umm_trans_own": u[trans],
            "umm_prod_unplanned_nb": u[nb & prod & u["unplanned"]],
            "umm_cons_own": u[own & (u["asset_kind"] == "consumption")],
        }

    def _umm_interval_series(self, ev: pd.DataFrame, lead: pd.Timedelta, new_window=None) -> pd.Series:
        """MW active at q with a = q - lead: q in [max(start, pub+lead), min(stop, next_pub+lead))."""
        idx = pd.date_range(self.start, self.end, freq="15min")
        diff = np.zeros(len(idx) + 1)
        if len(ev):
            vals = idx.values
            # version valid at a = q - lead  <=>  q in [pub + lead, next_pub + lead)
            # event overlaps quarter q        <=>  q in (start - 15min, stop)
            i0 = np.maximum(np.searchsorted(vals, (ev["publication_utc"] + lead).values, side="left"),
                            np.searchsorted(vals, (ev["event_start"] - Q).values, side="right"))
            hi = np.minimum(ev["event_stop"].values, (ev["next_pub"] + lead).values)
            if new_window is not None:   # first published within the last `new_window` before a
                hi = np.minimum(hi, (ev["first_pub"] + lead + new_window).values)
            i1 = np.searchsorted(vals, hi, side="left")
            ok = i1 > i0
            np.add.at(diff, i0[ok], ev["mw"].values[ok])
            np.add.at(diff, i1[ok], -ev["mw"].values[ok])
        return pd.Series(np.cumsum(diff)[:-1], index=idx)

    def _umm_features(self, area: str, q: pd.Series, a: pd.Series) -> dict:
        groups = self._umm_groups(area)
        leads = (q - a)
        out = {k: np.full(len(q), np.nan) for k in list(groups) + ["umm_new_unplanned_own_2h"]}
        for lead in leads.unique():
            m = (leads == lead).values
            key = (area, lead)
            if key not in self._umm_cache:
                ser = {k: self._umm_interval_series(ev, pd.Timedelta(lead)) for k, ev in groups.items()}
                ser["umm_new_unplanned_own_2h"] = self._umm_interval_series(
                    groups["umm_prod_unplanned_own"], pd.Timedelta(lead), new_window=pd.Timedelta(hours=2))
                self._umm_cache[key] = ser
            for k, s in self._umm_cache[key].items():
                out[k][m] = s.reindex(pd.DatetimeIndex(q[m])).values
        return out

    # ------------------------------------------------------------------ intraday market
    def _id_features(self, area: str, q: pd.Series, a: pd.Series, X: pd.DataFrame) -> dict:
        d = self.idm
        c = getattr(self, "id_api", {}) or {}
        pdiv = float(c.get("price_divisor", 100.0))
        qdiv = float(c.get("qty_divisor", 1000.0))
        out = {}
        areas = d.get("areas", pd.DataFrame())
        con = d.get("contracts", pd.DataFrame())
        if areas is None or con is None or areas.empty or con.empty:
            return out
        ids = areas.dropna(subset=["dk"])
        ids = ids[ids["dk"] == area]["area_id"].dropna().astype(int).unique()
        if len(ids) == 0:
            return out
        aid = int(ids[0])
        con = con.dropna(subset=["contract_id", "dlvry_start", "dlvry_end"]).drop_duplicates("contract_id", keep="last")
        con = con.assign(dur=(con["dlvry_end"] - con["dlvry_start"]).dt.total_seconds())
        # FIX (audit #5): str.contains(str(aid)) is a substring match — area id 1 also
        # matches "10", "11", "110", etc.  Parse the JSON list instead so only exact
        # membership is tested.  An empty list ([]) means no area restriction.
        def _aid_in(s, _aid=aid):
            try:
                lst = __import__("json").loads(s)
                return len(lst) == 0 or _aid in lst
            except Exception:
                return False
        con = con[con["area_ids"].astype(str).apply(_aid_in)]
        c15 = con[con["dur"] == 900].drop_duplicates("dlvry_start", keep="last").set_index("dlvry_start")["contract_id"]
        c60 = con[con["dur"] == 3600].drop_duplicates("dlvry_start", keep="last").set_index("dlvry_start")["contract_id"]
        base = pd.DataFrame({"i": np.arange(len(q)), "a": a.values,
                             "cid15": c15.reindex(pd.DatetimeIndex(q)).values,
                             "cid60": c60.reindex(pd.DatetimeIndex(q).floor("h")).values})
        da = X["da"].values if "da" in X.columns else np.full(len(q), np.nan)

        def asof(frame, cid_col, cols, prefix):
            f = frame
            if f is None or f.empty:
                return
            f = f[(f["area_id"] == aid)].dropna(subset=["contract_id"])
            if f.empty:
                return
            f = f.sort_values("recv_utc")
            left = base.dropna(subset=[cid_col]).rename(columns={cid_col: "contract_id"}).sort_values("a")
            if left.empty:
                return
            m = pd.merge_asof(left, f[["recv_utc", "contract_id"] + cols], left_on="a", right_on="recv_utc",
                              by="contract_id", direction="backward")
            for col in cols:
                arr = np.full(len(q), np.nan)
                arr[m["i"].values] = pd.to_numeric(m[col], errors="coerce").values
                out[f"{prefix}_{col}"] = arr
            age = np.full(len(q), np.nan)
            age[m["i"].values] = (m["a"] - m["recv_utc"]).dt.total_seconds().values / 60.0
            out[f"{prefix}_age_min"] = age

        st = d.get("stats")
        if st is not None and not st.empty:
            st = st[st["deleted"] != True] if "deleted" in st.columns else st  # noqa: E712
            for cid, p in (("cid15", "id15"), ("cid60", "id60")):
                asof(st, cid, ["vwap", "last_price", "high", "low", "turnover"], p)
                for k in ("vwap", "last_price", "high", "low"):
                    if f"{p}_{k}" in out:
                        out[f"{p}_{k}"] = out[f"{p}_{k}"] / pdiv
                if f"{p}_vwap" in out:
                    out[f"{p}_vwap_minus_da"] = out[f"{p}_vwap"] - da
                    out[f"{p}_last_minus_da"] = out[f"{p}_last_price"] - da
                    out[f"{p}_range"] = out[f"{p}_high"] - out[f"{p}_low"]
        bk = d.get("book")
        if bk is not None and not bk.empty:
            asof(bk, "cid15", ["best_bid", "best_ask", "bid_qty_depth", "ask_qty_depth"], "book15")
            if "book15_best_bid" in out:
                bb, ba = out["book15_best_bid"] / pdiv, out["book15_best_ask"] / pdiv
                out["book15_mid_minus_da"] = (bb + ba) / 2 - da
                out["book15_spread"] = ba - bb
                tb, ta = out.pop("book15_bid_qty_depth"), out.pop("book15_ask_qty_depth")
                out["book15_depth_imb"] = (tb - ta) / np.where((tb + ta) > 0, tb + ta, np.nan)
                out.pop("book15_best_bid"); out.pop("book15_best_ask")
        tr = d.get("trades")
        if tr is not None and not tr.empty:
            t = tr[(tr["area_id"] == aid) & (tr["aggressor"] == True)].dropna(subset=["contract_id"])  # noqa: E712
            if len(t):
                t = t.assign(sign=np.where(t["side"].astype(str).str.upper().str.contains("BUY"), 1.0, -1.0),
                             q=pd.to_numeric(t["qty"], errors="coerce") / qdiv,
                             p=pd.to_numeric(t["price"], errors="coerce") / pdiv).sort_values("recv_utc")
                n = np.full(len(q), np.nan)
                sv = np.full(len(q), np.nan)
                vw = np.full(len(q), np.nan)
                groups = {k: g for k, g in t.groupby("contract_id")}
                for i, (cid, aa) in enumerate(zip(base["cid15"].values, base["a"].values)):
                    g = groups.get(cid)
                    if g is None:
                        continue
                    tt = g["recv_utc"].values
                    j1 = np.searchsorted(tt, aa, side="right")
                    j0 = np.searchsorted(tt, aa - np.timedelta64(60, "m"), side="right")
                    w = g.iloc[j0:j1]
                    n[i] = len(w)
                    sv[i] = float((w["sign"] * w["q"]).sum())
                    vol = w["q"].sum()
                    vw[i] = float((w["p"] * w["q"]).sum() / vol) if vol > 0 else np.nan
                out["trd15_n_60m"] = n
                out["trd15_signed_mw_60m"] = sv
                out["trd15_vwap_60m_minus_da"] = vw - da
        return out
