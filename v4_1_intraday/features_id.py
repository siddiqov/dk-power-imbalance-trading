"""Point-in-time intraday features for Nurex V4.1.

Extends the V4.2 FeatureBuilder (day-ahead prices and neighbour spreads, D-1/5h
wind and solar forecasts, last published imbalance state, same-quarter history,
live PowerSystemRightNow flows) with information that only matters close to
delivery:

  * fast system state   satisfied demand (NRV), dominating direction and aFRR
                        volumes, published minutes after the quarter ends
                        (own, shorter lag than the imbalance price)
  * 1-hour forecasts    Forecast1Hour for wind/solar, known 60 min before delivery
  * live momentum       aFRR and flows over the last hour, live wind vs 1h forecast
  * other zone          the other Danish zone's latest NRV and spread
  * calendar            Danish public holidays, local time (DST-safe)

Rule: a value is used only if it was published at or before the decision time
`as_of_utc`. Missing information stays NaN - no default fills, no bfill.
"""
from __future__ import annotations

from datetime import date, timedelta

import warnings

import numpy as np
import pandas as pd

from . import settings as _s  # noqa: F401  (puts Nurex_V4_2 on sys.path)
from nurex42.features import FeatureBuilder, _grid, _local
from .features_ext import ExtMixin

Q = pd.Timedelta(minutes=15)
FTYPES = {"Offshore Wind": "off", "Onshore Wind": "on", "Solar": "sol"}
OTHER = {"DK1": "DK2", "DK2": "DK1"}
FAST_COLS = ["satisfied_demand_mw", "dominating_direction", "afrr_up_mw", "afrr_down_mw"]


def _easter(y: int) -> date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return date(y, month, day)


def danish_holidays(years) -> set:
    out = set()
    for y in years:
        e = _easter(y)
        out |= {date(y, 1, 1), date(y, 6, 5), date(y, 12, 24), date(y, 12, 25), date(y, 12, 26), date(y, 12, 31)}
        out |= {e + timedelta(days=d) for d in (-3, -2, 0, 1, 39, 40, 49, 50)}  # Maundy Thu .. Whit Monday
    return out


class IntradayFeatureBuilder(ExtMixin, FeatureBuilder):
    def __init__(self, store, cfg, areas=None):
        av = cfg["availability"]
        self.lag_fast = pd.Timedelta(minutes=av["fast_state_lag_minutes"])
        self.f1h_before = pd.Timedelta(minutes=av["forecast_1h_minutes_before"])
        super().__init__(store, cfg, areas)
        self._load_intraday(store)

    # ------------------------------------------------------------------ loading
    def _load_intraday(self, store) -> None:
        start = self.start
        fc = store.df("SELECT area, time_utc, ftype, f_da, f_1h FROM forecast WHERE time_utc >= ?", [start])
        self.fc1 = {}
        for a in ("DK1", "DK2"):
            d = fc[fc["area"] == a]
            if d.empty:
                continue
            p = d.pivot_table(index="time_utc", columns="ftype", values=["f_1h", "f_da"])
            p.columns = [f"{v}_{FTYPES.get(t, t)}" for v, t in p.columns]
            p = _grid(p, start, self.end)
            for v in ("f_1h", "f_da"):
                cols = [c for c in (f"{v}_off", f"{v}_on") if c in p.columns]
                p[f"{v}_wind"] = p[cols].sum(axis=1, min_count=len(cols)) if cols else np.nan
            for k in ("wind", "sol"):
                if f"f_1h_{k}" in p.columns and f"f_da_{k}" in p.columns:
                    p[f"rev1h_{k}"] = p[f"f_1h_{k}"] - p[f"f_da_{k}"]
            self.fc1[a] = p[[c for c in p.columns if c.startswith("f_1h") or c.startswith("rev1h")]]

        # fast state (NRV / direction / aFRR volumes) with rolling summaries on the regular grid
        self.fast = {}
        for a, g in self.imb.items():
            f = g[FAST_COLS].copy()
            sd = f["satisfied_demand_mw"]
            f["afrr_net"] = f["afrr_up_mw"] - f["afrr_down_mw"]
            for w in (2, 4, 8):
                f[f"sd_mean_{w}"] = sd.rolling(w, min_periods=1).mean()
                f[f"dir_mean_{w}"] = f["dominating_direction"].rolling(w, min_periods=1).mean()
                f[f"afrr_net_mean_{w}"] = f["afrr_net"].rolling(w, min_periods=1).mean()
            f["sd_d1"] = sd.diff(1)
            f["sd_d4"] = sd.diff(4)
            # length of the current run of the same dominating direction (capped at 16)
            d = f["dominating_direction"].fillna(0)
            run = np.zeros(len(d))
            prev, cnt = None, 0
            for i, v in enumerate(d.values):
                cnt = cnt + 1 if v == prev else 1
                prev = v
                run[i] = min(cnt, 16) * (1 if v > 0 else (-1 if v < 0 else 0))
            f["dir_run"] = run
            f.loc[g["dominating_direction"].isna(), "dir_run"] = np.nan
            # REC-02: upward regulation fraction over rolling windows
            # 0.0 = all downward, 0.5 = neutral, 1.0 = all upward activation
            afrr_up = f["afrr_up_mw"].fillna(0)
            afrr_dn = f["afrr_down_mw"].fillna(0)
            for _w in (3, 6, 12):
                _total = (afrr_up + afrr_dn).rolling(_w, min_periods=1).sum()
                f[f"up_reg_frac_{_w}q"] = (
                    afrr_up.rolling(_w, min_periods=1).sum() / (_total + 1e-6)
                ).clip(0.0, 1.0)
            self.fast[a] = f

        # live PSRN rolling summaries
        if self.psrn is not None:
            p = self.psrn
            extra = pd.DataFrame(index=p.index)
            for k in ("afrr_act_dk1", "afrr_act_dk2"):
                if k in p.columns:
                    extra[f"{k}_m4"] = p[k].rolling(4, min_periods=1).mean()
                    extra[f"{k}_abs_m4"] = p[k].abs().rolling(4, min_periods=1).mean()
            if {"offshore_mw", "onshore_mw", "solar_mw"} <= set(p.columns):
                extra["res_dk"] = p["offshore_mw"] + p["onshore_mw"] + p["solar_mw"]
                extra["res_dk_d1h"] = extra["res_dk"].diff(4)
            if {"prod_ge100_mw", "prod_lt100_mw"} <= set(p.columns):
                extra["thermal_dk"] = p["prod_ge100_mw"] + p["prod_lt100_mw"]
                extra["thermal_dk_d1h"] = extra["thermal_dk"].diff(4)
            self.psrn_x = extra
        else:
            self.psrn_x = None

        yrs = range(self.start.year - 1, self.end.year + 2)
        self.holidays = danish_holidays(yrs)

        # new sources: ENTSO-E, UMM outages, weather, frequency, Nord Pool intraday market
        self._load_ext(store)

    # ------------------------------------------------------------------ build
    def build(self, area: str, targets: pd.DataFrame) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            return self._build(area, targets)

    def _build(self, area: str, targets: pd.DataFrame) -> pd.DataFrame:
        X = super().build(area, targets).copy()
        t = targets.reset_index(drop=True)
        q = pd.to_datetime(t["quarter_utc"])
        a = pd.to_datetime(t["as_of_utc"])

        # calendar
        loc = _local(q, self.tz)
        ld = loc.dt.date
        X["holiday"] = ld.isin(self.holidays).astype(float).values
        X["pre_holiday"] = (ld + timedelta(days=1)).isin(self.holidays).astype(float).values if len(ld) else []
        X["minute_of_day"] = (loc.dt.hour * 60 + loc.dt.minute).values
        # REC-03: cyclical hour encoding — trees handle midnight wrap-around correctly
        _hour_frac = loc.dt.hour + loc.dt.minute / 60.0
        X["hour_sin"] = np.sin(2 * np.pi * _hour_frac / 24.0).values
        X["hour_cos"] = np.cos(2 * np.pi * _hour_frac / 24.0).values
        X["is_evening_peak"] = ((_hour_frac >= 16.0) & (_hour_frac < 22.0)).astype(float).values

        # fast state for own and other zone
        tf = (a - self.lag_fast).dt.floor("15min") - Q
        X["age_fast_min"] = ((q - tf).dt.total_seconds() / 60.0).values
        for zone, tag in ((area, "fast"), (OTHER[area], "oz_fast")):
            f = self.fast.get(zone)
            if f is None:
                continue
            v = self._look(f, tf, list(f.columns))
            for c in v.columns:
                X[f"{tag}_{c}"] = v[c].values
        # other zone's last published spread (same lag as own spread)
        oz = self.imb.get(OTHER[area])
        if oz is not None:
            t0 = (a - self.lag_imb).dt.floor("15min") - Q
            v = self._look(oz, t0, ["spread", "spread_mean_4"])
            X["oz_last_spread"] = v["spread"].values
            X["oz_last_spread_mean_4"] = v["spread_mean_4"].values

        # 1-hour forecasts: only once published (60 min before delivery)
        f1 = self.fc1.get(area)
        if f1 is not None:
            ok = (q - self.f1h_before <= a).values
            v = self._look(f1, q, list(f1.columns))
            for c in v.columns:
                X[c] = np.where(ok, v[c].values, np.nan)

        # live PSRN momentum + live wind vs 1h forecast of the same (past) bucket
        if self.psrn_x is not None:
            tp = (a - self.lag_psrn).dt.floor("15min") - Q
            v = self._look(self.psrn_x, tp, list(self.psrn_x.columns))
            for c in v.columns:
                X[f"live_{c}"] = v[c].values
            if "live_wind_dk" in X.columns and all(z in self.fc1 for z in ("DK1", "DK2")):
                f1_both = sum(self._look(self.fc1[z], tp, ["f_1h_wind"])["f_1h_wind"].values for z in ("DK1", "DK2"))
                ok = (tp - self.f1h_before <= a).values
                X["live_wind_err_1h_dk"] = np.where(ok, X["live_wind_dk"].values - f1_both, np.nan)
            own = "afrr_act_dk1" if area == "DK1" else "afrr_act_dk2"
            if f"live_{own}_m4" in X.columns:
                X["live_afrr_own_m4"] = X[f"live_{own}_m4"]

        # new sources (all point-in-time, NaN when not collected yet)
        ext = self._build_ext(X, area, q, a)
        if ext:
            X = pd.concat([X, pd.DataFrame(ext, index=X.index)], axis=1)

        # simple interactions the trees find hard to build
        if "da_spread_DE" in X.columns:
            X["coupled_de"] = (X["da_spread_DE"].abs() < 0.5).astype(float).where(X["da_spread_DE"].notna())
        if "fast_satisfied_demand_mw" in X.columns and "da" in X.columns:
            X["nrv_x_da"] = X["fast_satisfied_demand_mw"] * X["da"] / 100.0
        return X.copy().astype(float)
