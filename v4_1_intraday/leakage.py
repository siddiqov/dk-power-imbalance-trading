"""Leakage test for the V4.1 intraday feature builder.

For sampled (quarter, decision time) pairs, every raw value that was published AFTER
the decision time is overwritten with garbage before the builder derives anything
from it. If a single feature changes, the builder used information it could not
have had at gate closure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features_id import FAST_COLS, IntradayFeatureBuilder

Q = pd.Timedelta(minutes=15)
JUNK = 9999.0


class _CorruptedBuilder(IntradayFeatureBuilder):
    def __init__(self, store, cfg, as_of: pd.Timestamp):
        self._as_of = pd.Timestamp(as_of)
        self._cfg0 = cfg
        super().__init__(store, cfg)

    def _load_intraday(self, store) -> None:
        a, cfg = self._as_of, self._cfg0
        for area, g in self.imb.items():
            slow = g.index + Q + self.lag_imb > a
            fast = g.index + Q + self.lag_fast > a
            g.loc[slow, [c for c in g.columns if c not in FAST_COLS]] = JUNK
            g.loc[fast, [c for c in FAST_COLS if c in g.columns]] = JUNK
        if self.psrn is not None:
            self.psrn.loc[self.psrn.index + Q + self.lag_psrn > a] = JUNK
        for area, f in self.fc.items():
            f5 = [c for c in f.columns if c.startswith("f_5h") or c.startswith("rev_")]
            f.loc[f.index - self.f5h_before > a, f5] = JUNK
        loc = pd.Series(self.da.index).dt.tz_localize("UTC").dt.tz_convert(cfg["local_tz"])
        day = loc.dt.tz_localize(None).dt.normalize()
        pub = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=13)).dt.tz_localize(cfg["local_tz"]) \
            .dt.tz_convert("UTC").dt.tz_localize(None)
        self.da.loc[(pub > a).values] = JUNK
        super()._load_intraday(store)
        for area, f in self.fc1.items():
            f.loc[f.index - self.f1h_before > a] = JUNK


    def _before_derive(self, raw: dict) -> None:
        a, cfg = self._as_of, self._cfg0
        tz = cfg["local_tz"]
        e = raw.get("ent")
        if e is not None and len(e):
            t = pd.to_datetime(e["time_utc"])
            realised = e["series"].str.startswith(("load:", "phys:"))
            loc = t.dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None).dt.normalize() - pd.Timedelta(days=1)
            hh = np.where(e["series"].str.startswith("wsfc:"), 18, 13)
            pub = (loc + pd.to_timedelta(hh, unit="h")).dt.tz_localize(tz, nonexistent="shift_forward",
                                                                          ambiguous=True).dt.tz_convert("UTC").dt.tz_localize(None)
            future = np.where(realised, t + Q + self.lag_entsoe > a, pub > a)
            e.loc[future, "value"] = JUNK
        u = raw.get("umm")
        if u is not None and len(u):
            u.loc[u["publication_utc"] > a, ["unavailable_mw", "event_start", "event_stop"]] = [
                JUNK, pd.Timestamp("2000-01-01"), pd.Timestamp("2100-01-01")]
        w = raw.get("wx")
        if w is not None and len(w):
            w.loc[pd.to_datetime(w["time_utc"]) - self.weather_before > a, "value"] = JUNK
        f = raw.get("freq")
        if f is not None and len(f):
            f.loc[pd.to_datetime(f["time_utc"]) + Q + self.lag_freq > a, ["f_mean", "f_std", "f_min", "f_max"]] = JUNK
        for name, d in (raw.get("id") or {}).items():
            if d is not None and len(d) and name in ("stats", "trades", "book"):
                num = [c for c in d.columns if c not in ("recv_utc", "contract_id", "area_id", "side", "aggressor",
                                                          "deleted", "state", "trade_id") and pd.api.types.is_numeric_dtype(d[c])]
                d.loc[d["recv_utc"] > a, num] = JUNK


def run(store, cfg, n_samples: int = 8, seed: int = 7, areas=None) -> list[str]:
    from .pipeline_id import target_frame
    rng = np.random.default_rng(seed)
    clean = IntradayFeatureBuilder(store, cfg)
    problems = []
    for area in areas or cfg["areas"]:
        t = target_frame(clean, area, cfg)
        idx = rng.choice(np.arange(5000, len(t)), size=n_samples, replace=False)
        for _, row in t.iloc[idx].iterrows():
            leads = [cfg["intraday"]["gate_lead_minutes"]] + list(cfg["intraday"].get("train_extra_leads_minutes") or [])
            for lead in leads:
                a = row["quarter_utc"] - pd.Timedelta(minutes=int(lead))
                tgt = pd.DataFrame({"quarter_utc": [row["quarter_utc"]], "as_of_utc": [a]})
                x0 = clean.build(area, tgt)
                x1 = _CorruptedBuilder(store, cfg, a).build(area, tgt)
                bad = [c for c in x0.columns
                       if not (np.isnan(x0[c].iloc[0]) and np.isnan(x1[c].iloc[0]))
                       and not np.isclose(x0[c].iloc[0], x1[c].iloc[0])]
                if bad:
                    problems.append(f"{area} q={row['quarter_utc']} as_of={a}: {bad}")
    return problems
