"""Replay report for Nurex V4.1 Intraday (reuses the V4.2 metrics)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import settings as _s  # noqa: F401
from nurex42 import decision as dec
from nurex42 import evaluation as ev


def direction_skill(r: pd.DataFrame) -> dict:
    """Probability skill vs climatology on the test window (UP vs not-UP and DOWN vs not-DOWN)."""
    y = r["spread"].values
    out = {}
    for name, col, lab in (("up", "p_up", y > 0.5), ("down", "p_down", y < -0.5)):
        p = r[col].values
        base = lab.mean()
        bs = np.mean((p - lab) ** 2)
        bs0 = np.mean((base - lab) ** 2)
        out[f"brier_{name}"] = bs
        out[f"brier_skill_{name}_pct"] = 100 * (1 - bs / bs0) if bs0 > 0 else np.nan
        try:
            from sklearn.metrics import roc_auc_score
            out[f"auc_{name}"] = roc_auc_score(lab, p)
        except Exception:
            pass
    return out


def extra_baselines(r: pd.DataFrame, cfg) -> pd.DataFrame:
    cost = cfg.cost_per_mwh
    rows = []
    if "fast_satisfied_demand_mw" in r:
        sd = r["fast_satisfied_demand_mw"]
        act = pd.Series(np.where(sd > 0, dec.BUY, np.where(sd < 0, dec.SELL, dec.HOLD)), index=r.index)
        m = pd.Series(np.where(act == dec.HOLD, 0.0, 1.0), index=r.index)
        _, _, n = dec.pnl(act, m, r["spread"], cost)
        rows.append({"strategy": "sign_of_latest_NRV (t-6)", "trades": int((act != dec.HOLD).sum()),
                     "net_eur": n.sum(), "net_eur_per_mwh": n.sum() / max(m.sum(), 1e-9)})
    return pd.DataFrame(rows)


def write(out: dict, cfg, path: Path) -> dict:
    r = out["results"].copy()
    r["deadline_utc"] = r["as_of_trade"]
    rep = ev.write_report(out, cfg, path)            # writes markdown + csv
    ds = direction_skill(r)
    eb = extra_baselines(r, cfg)
    txt = path.read_text(encoding="utf-8")
    txt = txt.replace("# Nurex V4.2 walk-forward replay", "# Nurex V4.1 Intraday walk-forward replay")
    txt = txt.replace("at its batch decision time (deadline minus 15 min)",
                      f"at intraday gate closure (delivery start minus {cfg['intraday']['gate_lead_minutes']} min)")
    txt = txt.replace(f"retrained every {cfg['replay']['retrain_every_days']} days",
                      f"retrained every {cfg['intraday']['retrain_every_days']} days")
    raw = {}
    if "net_raw_eur" in r:
        traded = r["mwh_raw"] > 0
        net = r.loc[traded, "net_raw_eur"].sort_values(ascending=False)
        raw = {"signals": int(traded.sum()), "mwh": float(r.loc[traded, "mwh_raw"].sum()),
               "net_eur": float(net.sum()),
               "net_eur_per_mwh": float(net.sum() / max(r.loc[traded, "mwh_raw"].sum(), 1e-9)),
               "net_without_top20_eur": float(net.iloc[20:].sum()),
               "top20_share_pct": float(100 * net.head(20).sum() / net.sum()) if net.sum() != 0 else float("nan"),
               "win_rate_pct": float(100 * (net > 0).mean()) if len(net) else float("nan")}
        clipped = np.clip(r["spread"], -300, 300)
        tm_side = np.where(r["action"] == "BUY", 1.0, np.where(r["action"] == "SELL", -1.0, 0.0))
        raw["net_after_overlay_spreads_clipped_300_eur"] = float(
            (tm_side * r["mwh"] * clipped - np.where(tm_side != 0, r["mwh"] * cfg.cost_per_mwh, 0)).sum())
    mon = r.assign(m=r["quarter_utc"].dt.strftime("%Y-%m")).groupby("m").agg(
        trades=("action", lambda x: int((x != "HOLD").sum())), net_eur=("net_eur", "sum"),
        raw_signals=("mwh_raw", lambda x: int((x > 0).sum())) if "mwh_raw" in r else ("net_eur", "size"),
        raw_net_eur=("net_raw_eur", "sum") if "net_raw_eur" in r else ("net_eur", "sum")).reset_index()
    add = ["", "## Robustness", ev._fmt(raw) if raw else "n/a", "",
           "## Monthly (after risk overlay vs raw model signals)", mon.to_markdown(index=False, floatfmt=",.0f"), "",
           "## Direction skill vs climatology", ev._fmt(ds), "",
           "## Extra intraday baseline (1 MWh, same costs)",
           eb.to_markdown(index=False, floatfmt=",.2f") if len(eb) else "n/a", ""]
    path.write_text(txt + "\n".join(add), encoding="utf-8")
    rep["direction"] = ds
    rep["raw"] = raw
    rep["monthly_raw"] = mon
    rep["extra_baselines"] = eb
    return rep
