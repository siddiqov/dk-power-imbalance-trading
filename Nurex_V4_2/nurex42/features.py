"""Point-in-time feature builder.

Every feature for delivery quarter q is computed "as of" a decision time a.
A raw value is used only if the rules in config.availability say it was
published at or before a. Missing information stays NaN (LightGBM handles NaN
natively) - there are no default fills.

Main entry point:
    fb = FeatureBuilder(store, cfg)
    X = fb.build("DK1", targets)   # targets: DataFrame[quarter_utc, as_of_utc]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .storage import Store

Q = pd.Timedelta(minutes=15)
DA_15MIN_START = pd.Timestamp("2025-09-30 22:00")  # 1 Oct 2025 00:00 CET, first 15-min day-ahead delivery
FTYPES = {"Offshore Wind": "off", "Onshore Wind": "on", "Solar": "sol"}
NEIGHBOURS = {"DK1": ["DE", "NO2", "SE3", "NL", "DK2"], "DK2": ["DE", "SE4", "DK1"]}
PSRN_AREA = {
    "DK1": {"exch_dk1_de": "flow_de", "exch_dk1_nl": "flow_nl", "exch_dk1_gb": "flow_gb",
            "exch_dk1_no": "flow_no", "exch_dk1_se": "flow_se", "exch_dk1_dk2": "flow_dk2",
            "afrr_act_dk1": "afrr_act"},
    "DK2": {"exch_dk2_de": "flow_de", "exch_dk2_se": "flow_se", "exch_dk1_dk2": "flow_dk1_neg",
            "afrr_act_dk2": "afrr_act"},
}


def _grid(df: pd.DataFrame, start, end) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="15min")
    return df.reindex(idx)


def _local(ts: pd.Series, tz: str) -> pd.Series:
    return pd.to_datetime(ts).dt.tz_localize("UTC").dt.tz_convert(tz)


def _local_hhmm_on_prev_day(q: pd.Series, hhmm: str, tz: str) -> pd.Series:
    """UTC-naive time of HH:MM local on the local day before each quarter's delivery day."""
    loc = _local(q, tz)
    day = loc.dt.tz_localize(None).dt.normalize() - pd.Timedelta(days=1)
    h, m = (int(x) for x in hhmm.split(":"))
    t = (day + pd.Timedelta(hours=h, minutes=m)).dt.tz_localize(tz, nonexistent="shift_forward")
    return t.dt.tz_convert("UTC").dt.tz_localize(None)


class FeatureBuilder:
    def __init__(self, store: Store, cfg, areas: list[str] | None = None):
        self.cfg = cfg
        self.tz = cfg["local_tz"]
        av = cfg["availability"]
        self.lag_imb = pd.Timedelta(minutes=av["imbalance_lag_minutes"])
        self.lag_psrn = pd.Timedelta(minutes=av["psrn_lag_minutes"])
        self.f5h_before = pd.Timedelta(minutes=av["forecast_5h_minutes_before"])
        self.areas = areas or cfg["areas"]
        self._load(store)

    # ------------------------------------------------------------------ loading
    def _load(self, store: Store) -> None:
        start = pd.Timestamp(self.cfg["history_start"]) - pd.Timedelta(days=10)
        imb = store.df("SELECT * FROM imbalance WHERE time_utc >= ?", [start])
        da = store.df("SELECT area, time_utc, price_eur FROM dayahead WHERE time_utc >= ?", [start])
        fc = store.df("SELECT area, time_utc, ftype, f_da, f_5h FROM forecast WHERE time_utc >= ?", [start])
        ps = store.df("SELECT * FROM psrn WHERE time_utc >= ?", [start])
        en = store.df("SELECT series, time_utc, value FROM entsoe_series WHERE time_utc >= ?", [start])

        ends = [x["time_utc"].max() for x in (imb, da, fc) if not x.empty]
        self.end = (max(ends) if ends else pd.Timestamp.now()) + pd.Timedelta(days=2)
        self.start = start

        self.imb = {}
        for a in self.areas:
            d = imb[imb["area"] == a].drop_duplicates("time_utc").set_index("time_utc").sort_index()
            d = d.drop(columns=[c for c in ["area", "source", "ingested_at"] if c in d.columns])
            d["spread"] = d["imbalance_eur"] - d["spot_eur"]
            g = _grid(d, start, self.end)
            s = g["spread"]
            for w in (4, 8, 16, 96):
                g[f"spread_mean_{w}"] = s.rolling(w, min_periods=max(2, w // 4)).mean()
                g[f"spread_absmean_{w}"] = s.abs().rolling(w, min_periods=max(2, w // 4)).mean()
                g[f"pos_frac_{w}"] = (s > 0).astype(float).where(s.notna()).rolling(
                    w, min_periods=max(2, w // 4)).mean()
            g["spread_std_16"] = s.rolling(16, min_periods=4).std()
            g["sd_mean_4"] = g["satisfied_demand_mw"].rolling(4, min_periods=2).mean()
            self.imb[a] = g

        dap = _grid(da.pivot_table(index="time_utc", columns="area", values="price_eur"), start, self.end)
        # Before 1 Oct 2025 day-ahead prices were hourly: the hourly price applies to all four quarters.
        early = dap.index < DA_15MIN_START
        dap.loc[early] = dap.loc[early].ffill(limit=3)
        # The imbalance dataset repeats the day-ahead price of the quarter; use it where the
        # day-ahead table has a gap (same published number, not an estimate).
        for a in self.areas:
            sp = self.imb[a]["spot_eur"].reindex(dap.index)
            dap[a] = dap[a].combine_first(sp) if a in dap.columns else sp
        self.da = dap

        self.fc = {}
        for a in ["DK1", "DK2"]:
            d = fc[fc["area"] == a]
            if d.empty:
                self.fc[a] = pd.DataFrame(index=pd.date_range(start, self.end, freq="15min"))
                continue
            p = d.pivot_table(index="time_utc", columns="ftype", values=["f_da", "f_5h"])
            p.columns = [f"{v}_{FTYPES.get(t, t)}" for v, t in p.columns]
            p = _grid(p, start, self.end)
            for v in ("f_da", "f_5h"):
                cols = [c for c in (f"{v}_off", f"{v}_on") if c in p.columns]
                p[f"{v}_wind"] = p[cols].sum(axis=1, min_count=len(cols)) if cols else np.nan
            for k in ("wind", "off", "on", "sol"):
                if f"f_5h_{k}" in p.columns and f"f_da_{k}" in p.columns:
                    p[f"rev_{k}"] = p[f"f_5h_{k}"] - p[f"f_da_{k}"]
            self.fc[a] = p

        if not ps.empty:
            ps = ps.drop_duplicates("time_utc").set_index("time_utc").sort_index()
            self.psrn = _grid(ps.drop(columns=[c for c in ["ingested_at", "n_minutes"] if c in ps.columns]),
                              start, self.end)
        else:
            self.psrn = None

        self.ent = (_grid(en.pivot_table(index="time_utc", columns="series", values="value"), start, self.end)
                    if not en.empty else None)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _look(frame: pd.DataFrame, times: pd.Series, cols) -> pd.DataFrame:
        cols = [c for c in cols if c in frame.columns]
        out = frame[cols].reindex(pd.DatetimeIndex(times))
        out.index = range(len(times))
        return out

    def targets_available(self, area: str) -> pd.DataFrame:
        g = self.imb[area]
        s = g["spread"].dropna()
        return pd.DataFrame({"quarter_utc": s.index, "spread": s.values})

    # ------------------------------------------------------------------ build
    def build(self, area: str, targets: pd.DataFrame) -> pd.DataFrame:
        t = targets.reset_index(drop=True)
        q = pd.to_datetime(t["quarter_utc"])
        a = pd.to_datetime(t["as_of_utc"])
        X = pd.DataFrame(index=range(len(t)))
        tz = self.tz
        av = self.cfg["availability"]

        # ---- calendar (always known)
        loc = _local(q, tz)
        X["lead_min"] = (q - a).dt.total_seconds() / 60.0
        X["hour"] = loc.dt.hour
        X["qod"] = loc.dt.hour * 4 + loc.dt.minute // 15
        X["q_in_hour"] = loc.dt.minute // 15
        X["dow"] = loc.dt.dayofweek
        X["month"] = loc.dt.month
        X["weekend"] = (loc.dt.dayofweek >= 5).astype(int)

        # ---- day-ahead prices (published D-1 ~13:00 local)
        da_ok = (a >= _local_hhmm_on_prev_day(q, av["dayahead_publish_local"], tz)).values
        own = self._look(self.da, q, [area])[area] if area in self.da.columns else pd.Series(np.nan, index=X.index)
        X["da"] = own
        for k, name in [(-1, "da_prev1"), (1, "da_next1"), (-4, "da_prev4"), (4, "da_next4")]:
            tq = q + k * Q
            v = self._look(self.da, tq, [area])[area] if area in self.da.columns else np.nan
            # a neighbouring quarter on another delivery day may not be published yet
            ok = (a >= _local_hhmm_on_prev_day(tq, av["dayahead_publish_local"], tz)).values
            X[name] = np.where(ok, v, np.nan)
        X["da_jump_prev"] = X["da"] - X["da_prev1"]
        X["da_jump_next"] = X["da_next1"] - X["da"]
        # delivery-day statistics (whole day is published at once)
        if area in self.da.columns:
            dd = self.da[[area]].copy()
            dd["day"] = _local(pd.Series(dd.index), tz).dt.date.values
            stats = dd.groupby("day")[area].agg(["mean", "std", "min", "max"])
            day = loc.dt.date
            st = stats.reindex(day.values)
            X["da_day_mean"] = st["mean"].values
            X["da_day_std"] = st["std"].values
            X["da_day_range"] = (st["max"] - st["min"]).values
            X["da_dev_day"] = X["da"] - X["da_day_mean"]
            X["da_z_day"] = X["da_dev_day"] / X["da_day_std"].replace(0, np.nan)
        for nb in NEIGHBOURS.get(area, []):
            if nb in self.da.columns:
                X[f"da_spread_{nb}"] = X["da"] - self._look(self.da, q, [nb])[nb]
        X.loc[~da_ok, [c for c in X.columns if c.startswith("da")]] = np.nan

        # ---- wind/solar forecasts
        f = self.fc.get(area)
        if f is not None and len(f.columns):
            fda_ok = (a >= _local_hhmm_on_prev_day(q, av["forecast_da_publish_local"], tz)).values
            cols_da = [c for c in f.columns if c.startswith("f_da_")]
            v = self._look(f, q, cols_da)
            for c in v.columns:
                X[c] = np.where(fda_ok, v[c], np.nan)
            f5_ok = (q - self.f5h_before <= a).values
            cols_5h = [c for c in f.columns if c.startswith("f_5h_") or c.startswith("rev_")]
            v = self._look(f, q, cols_5h)
            for c in v.columns:
                X[c] = np.where(f5_ok & fda_ok, v[c], np.nan)
            # most recent forecast revision known at a: quarters in [a+4h, a+5h)
            t_last = (a + self.f5h_before).dt.floor("15min") - Q
            ok = (a >= _local_hhmm_on_prev_day(t_last, av["forecast_da_publish_local"], tz)).values
            for rc in [c for c in f.columns if c.startswith("rev_")]:
                rr = f[rc].rolling(4, min_periods=2).mean()
                X[f"{rc}_recent"] = np.where(ok, rr.reindex(pd.DatetimeIndex(t_last)).values, np.nan)
            wcol = "f_da_wind" if "f_da_wind" in X.columns and X["f_da_wind"].notna().any() else "f_da_off"
            if wcol in X.columns and "da" in X.columns:
                X["da_x_wind"] = X["da"] * X[wcol] / 1000.0

        # ---- last published imbalance state
        g = self.imb[area]
        t0 = (a - self.lag_imb).dt.floor("15min") - Q            # last quarter whose end+lag <= a
        cols = ["spread", "imbalance_eur", "satisfied_demand_mw", "dominating_direction", "afrr_up_mw",
                "afrr_down_mw", "mfrr_price_up_eur", "mfrr_price_down_eur", "spot_eur",
                "spread_mean_4", "spread_mean_8", "spread_mean_16", "spread_mean_96",
                "spread_absmean_16", "spread_absmean_96", "pos_frac_16", "pos_frac_96",
                "spread_std_16", "sd_mean_4"]
        v = self._look(g, t0, cols)
        for c in v.columns:
            X[f"last_{c}"] = v[c].values
        X["last_mfrr_up_minus_spot"] = X.get("last_mfrr_price_up_eur") - X.get("last_spot_eur")
        X["last_mfrr_down_minus_spot"] = X.get("last_mfrr_price_down_eur") - X.get("last_spot_eur")
        X["age_last_min"] = (q - t0).dt.total_seconds() / 60.0
        # same quarter on previous days (only if already published at a)
        for d in (1, 2, 7):
            tq = q - pd.Timedelta(days=d)
            ok = (tq + Q + self.lag_imb <= a).values
            X[f"spread_d{d}"] = np.where(ok, self._look(g, tq, ["spread"])["spread"].values, np.nan)
        # mean of the same quarter-of-day over days 2..8 back
        vals = []
        for d in range(2, 9):
            tq = q - pd.Timedelta(days=d)
            ok = (tq + Q + self.lag_imb <= a).values
            vals.append(np.where(ok, self._look(g, tq, ["spread"])["spread"].values, np.nan))
        arr = np.vstack(vals)
        import warnings
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            X["spread_qod_mean7"] = np.nanmean(arr, axis=0)
            X["pos_qod_frac7"] = np.nanmean(np.where(np.isnan(arr), np.nan, arr > 0), axis=0)

        # ---- live system state (PowerSystemRightNow, aggregated)
        if self.psrn is not None:
            tp = (a - self.lag_psrn).dt.floor("15min") - Q
            m = PSRN_AREA[area]
            v = self._look(self.psrn, tp, list(m) + ["offshore_mw", "onshore_mw", "solar_mw", "exch_sum_mw"])
            for src, dst in m.items():
                if src in v.columns:
                    X[f"live_{dst}"] = v[src].values
            v1 = self._look(self.psrn, tp - 4 * Q, list(m))
            for src, dst in m.items():
                if src in v1.columns:
                    X[f"live_{dst}_d1h"] = v[src].values - v1[src].values
            if {"offshore_mw", "onshore_mw"} <= set(v.columns):
                live_wind = v["offshore_mw"].values + v["onshore_mw"].values
                X["live_wind_dk"] = live_wind
                fda_both = []
                for ar in ("DK1", "DK2"):
                    ff = self.fc.get(ar)
                    if ff is not None and "f_da_wind" in ff.columns:
                        fda_both.append(self._look(ff, tp, ["f_da_wind"])["f_da_wind"].values)
                if len(fda_both) == 2:
                    X["live_wind_err_dk"] = live_wind - (fda_both[0] + fda_both[1])

        # ---- ENTSO-E day-ahead schedules and load forecast
        if self.ent is not None:
            ent_ok = (a >= _local_hhmm_on_prev_day(q, av["entsoe_dayahead_publish_local"], tz)).values
            cols = [c for c in self.ent.columns if c.startswith(f"sched:{area}>") or c == f"loadfc:{area}"]
            v = self._look(self.ent, q, cols)
            for c in v.columns:
                name = c.replace(":", "_").replace(">", "_")
                X[name] = np.where(ent_ok, v[c].values, np.nan)
            if f"loadfc_{area}" in X.columns and "f_da_wind" in X.columns:
                X["res_share_da"] = X["f_da_wind"] / X[f"loadfc_{area}"]

        return X.astype(float)
