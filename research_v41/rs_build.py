"""Step 1: build the point-in-time research dataset from a PRIVATE COPY of the store.
Uses the production feature builder unchanged (pipeline_id.target_frame / design), so the
features are exactly what the live model sees. Adds candidate 'forecast drift' features
computed only from forecasts already published at each decision time.
Output: results/research_v41/cache/{t,Xt,Xe,drift}_<area>.parquet"""
import sys
import duckdb
import numpy as np
import pandas as pd
from rs_common import CACHE, COPY, P, cfg, log, Q
from nurex42.storage import Store
from v4_1_intraday.features_id import IntradayFeatureBuilder

H1 = pd.Timedelta(minutes=60)     # Forecast1Hour known 60 min before its quarter (config availability)


def drift_features(con, area, quarters: pd.Series, asof: pd.Series) -> pd.DataFrame:
    f = con.sql("select area, time_utc, ftype, f_da, f_5h, f_1h from forecast").df()
    out = pd.DataFrame(index=quarters.index)
    L = (asof + H1).dt.floor("15min")          # newest quarter whose 1-hour forecast is published
    for zone, pre in ((area, "fd_"), ("DK2" if area == "DK1" else "DK1", "fd_oz_")):
        g = f[f["area"] == zone]
        wind = g[g["ftype"].str.contains("Wind")].groupby("time_utc")[["f_da", "f_5h", "f_1h"]].sum(min_count=1)
        sol = g[g["ftype"] == "Solar"].groupby("time_utc")[["f_da", "f_5h", "f_1h"]].sum(min_count=1)
        grid = pd.date_range(min(wind.index.min(), sol.index.min()), max(wind.index.max(), sol.index.max()), freq="15min")
        wind, sol = wind.reindex(grid), sol.reindex(grid)
        d15 = wind["f_1h"] - wind["f_5h"]
        feats = {
            "wind_1h5h_4": d15.rolling(4, min_periods=2).mean(),
            "wind_1h5h_8": d15.rolling(8, min_periods=4).mean(),
            "wind_1hda_4": (wind["f_1h"] - wind["f_da"]).rolling(4, min_periods=2).mean(),
            "wind_1h_slope_4": wind["f_1h"].diff(4),
            "solar_1h5h_4": (sol["f_1h"] - sol["f_5h"]).rolling(4, min_periods=2).mean(),
        }
        for k, s in feats.items():
            out[pre + k] = s.reindex(L.values).values
        # corrected forecast for the traded quarter: its 5-hour forecast + the recent 1h-vs-5h drift
        f5_target = wind["f_5h"].reindex(quarters.values).values
        out[pre + "wind_5h_corrected"] = f5_target + out[pre + "wind_1h5h_4"].values
        lvl = wind["f_5h"].reindex(L.values).abs().values + 50.0
        out[pre + "wind_drift_rel"] = out[pre + "wind_1h5h_4"].values / lvl
    return out


def main(areas):
    c = cfg()
    st = Store(COPY, read_only=True)
    fb = IntradayFeatureBuilder(st, c)
    con = duckdb.connect(str(COPY), read_only=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    for area in areas:
        log(f"[build {area}] targets + point-in-time features (production builder)")
        t = P.target_frame(fb, area, c)
        Xt, extras = P.design(fb, area, t, c)
        Xe = extras[0]
        t[["quarter_utc", "as_of_trade", "label_known_at", "batch_start_utc", "deadline_utc", "spread"]] \
            .to_parquet(CACHE / f"t_{area}.parquet")
        Xt.to_parquet(CACHE / f"Xt_{area}.parquet")
        Xe.to_parquet(CACHE / f"Xe_{area}.parquet")
        lead_e = pd.Timedelta(minutes=int(c.train_extra_leads[0]))
        dt = drift_features(con, area, t["quarter_utc"], t["as_of_trade"])
        de = drift_features(con, area, t["quarter_utc"], t["quarter_utc"] - lead_e)
        d = pd.concat([dt.add_prefix("t_"), de.add_prefix("e_")], axis=1)
        d.to_parquet(CACHE / f"drift_{area}.parquet")
        log(f"[build {area}] rows={len(t)} features={Xt.shape[1]} drift={dt.shape[1]} "
            f"range {t['quarter_utc'].min()} .. {t['quarter_utc'].max()}")
    con.close(); st.close()


if __name__ == "__main__":
    main(sys.argv[1:] or ["DK1", "DK2"])
