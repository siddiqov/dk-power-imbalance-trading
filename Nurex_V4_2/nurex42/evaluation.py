"""Evaluation of a replay: trading metrics, forecast metrics and naive baselines."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import decision as dec


def trading_metrics(r: pd.DataFrame, cfg, net_col: str = "net_eur") -> dict:
    tz = cfg["local_tz"]
    tr = r[r["action"] != dec.HOLD]
    day = r["quarter_utc"].dt.tz_localize("UTC").dt.tz_convert(tz).dt.date
    daily = r.groupby(day)[net_col].sum()
    eq = daily.cumsum()
    dd = (eq - eq.cummax()).min() if len(eq) else 0.0
    mwh = tr["mwh"].sum()
    wins = (tr[net_col] > 0).mean() if len(tr) else np.nan
    sharpe = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else np.nan
    return {
        "quarters": len(r), "trades": len(tr), "trade_rate_pct": 100 * len(tr) / max(len(r), 1),
        "buy": int((tr["action"] == dec.BUY).sum()), "sell": int((tr["action"] == dec.SELL).sum()),
        "mwh_traded": mwh, "net_eur": r[net_col].sum(), "gross_eur": r.get("gross_eur", pd.Series(0)).sum(),
        "cost_eur": r.get("cost_eur", pd.Series(0)).sum(),
        "net_eur_per_mwh": r[net_col].sum() / mwh if mwh else np.nan,
        "win_rate_pct": 100 * wins if wins == wins else np.nan,
        "days": len(daily), "positive_days_pct": 100 * (daily > 0).mean() if len(daily) else np.nan,
        "worst_day_eur": daily.min() if len(daily) else np.nan,
        "max_drawdown_eur": dd,
        "max_drawdown_pct_collateral": 100 * dd / cfg.collateral_eur,
        "sharpe_daily_ann": sharpe,
    }


def forecast_metrics(r: pd.DataFrame) -> dict:
    y = r["spread"].values
    e = r["exp_spread"].values
    out = {
        "mae_model": np.mean(np.abs(y - e)),
        "mae_zero": np.mean(np.abs(y)),
        "mae_median_q50": np.mean(np.abs(y - r["q50"].values)),
        "dir_acc_model_pct": 100 * np.mean(np.sign(e) == np.sign(y)),
        "dir_acc_always_negative_pct": 100 * np.mean(y < 0),
        "share_positive_spread_pct": 100 * np.mean(y > 0),
    }
    up = (y > 0).astype(float)
    out["brier_up_model"] = np.mean((r["p_up"].values - up) ** 2)
    out["brier_up_climatology"] = np.mean((up.mean() - up) ** 2)
    for q, col in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        d = y - r[col].values
        out[f"pinball_{col}"] = np.mean(np.maximum(q * d, (q - 1) * d))
        out[f"coverage_{col}_pct"] = 100 * np.mean(y <= r[col].values)
    if "prov_exp_spread" in r:
        p = r["prov_exp_spread"].values
        out["mae_provisional"] = np.mean(np.abs(y - p))
        out["dir_acc_provisional_pct"] = 100 * np.mean(np.sign(p) == np.sign(y))
        out["provisional_vs_final_same_sign_pct"] = 100 * np.mean(np.sign(p) == np.sign(e))
    return out


def baselines(r: pd.DataFrame, cfg, mwh: float = 1.0) -> pd.DataFrame:
    """Naive strategies with a fixed 1 MWh per quarter, same costs.
    Signals only use information available at the decision time (columns from the feature matrix)."""
    cost = cfg.cost_per_mwh
    s = r["spread"]
    strategies = {
        "always_sell": pd.Series(dec.SELL, index=r.index),
        "always_buy": pd.Series(dec.BUY, index=r.index),
        "sign_of_last_published": pd.Series(np.where(r["last_spread"] > 0, dec.BUY, dec.SELL), index=r.index),
        "sign_of_same_quarter_7d_mean": pd.Series(np.where(r["spread_qod_mean7"] > 0, dec.BUY,
                                                           np.where(r["spread_qod_mean7"] < 0, dec.SELL,
                                                                    dec.HOLD)), index=r.index),
    }
    rows = []
    for name, act in strategies.items():
        m = pd.Series(np.where(act == dec.HOLD, 0.0, mwh), index=r.index)
        g, c, n = dec.pnl(act, m, s, cost)
        rows.append({"strategy": name, "trades": int((act != dec.HOLD).sum()), "net_eur": n.sum(),
                     "net_eur_per_mwh": n.sum() / max(m.sum(), 1e-9)})
    return pd.DataFrame(rows)


def monthly(r: pd.DataFrame, cfg) -> pd.DataFrame:
    tz = cfg["local_tz"]
    mth = r["quarter_utc"].dt.tz_localize("UTC").dt.tz_convert(tz).dt.strftime("%Y-%m")
    g = r.assign(month=mth, traded=r["action"] != dec.HOLD)
    return g.groupby("month").agg(trades=("traded", "sum"), mwh=("mwh", "sum"), net_eur=("net_eur", "sum"),
                                  mean_abs_spread=("spread", lambda x: x.abs().mean())).reset_index()


def _fmt(d: dict) -> str:
    lines = ["| metric | value |", "|---|---|"]
    for k, v in d.items():
        lines.append(f"| {k} | {v:,.2f} |" if isinstance(v, (float, np.floating)) else f"| {k} | {v} |")
    return "\n".join(lines)


def write_report(out: dict, cfg, path: Path) -> dict:
    r = out["results"]
    tm = trading_metrics(r, cfg)
    stress_cost = cfg.cost_per_mwh * float(cfg["costs"]["stress_multiplier"])
    r2 = r.copy()
    r2["net_stress"] = r2["gross_eur"] - np.where(r2["action"] != dec.HOLD, r2["mwh"] * stress_cost, 0.0)
    tm["net_eur_at_stress_costs"] = r2["net_stress"].sum()
    fm = forecast_metrics(r)
    base = baselines(r, cfg) if "last_spread" in r else pd.DataFrame()
    mon = monthly(r, cfg)
    per = out["periods"]
    imp = out["feature_importance"].head(25).round(2)
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = [f"# Nurex V4.2 walk-forward replay — {out['area']}", "",
           f"Book: `{out['book']}`  ",
           f"Test window: {r['quarter_utc'].min()} → {r['quarter_utc'].max()} UTC  ",
           f"Cost model: {cfg.cost_per_mwh:.2f} EUR/MWh (stress x{cfg['costs']['stress_multiplier']})  ",
           f"Collateral: {cfg.collateral_eur:,.0f} EUR  ",
           "Every quarter was forecast with information available at its batch decision time "
           "(deadline minus 15 min); models retrained every "
           f"{cfg['replay']['retrain_every_days']} days on labels known at that time.", "",
           "## Trading", _fmt(tm), "", "## Forecast quality", _fmt(fm), "",
           "## Naive baselines (1 MWh every quarter, same costs)",
           base.to_markdown(index=False, floatfmt=",.2f") if len(base) else "n/a", "",
           "## Monthly", mon.to_markdown(index=False, floatfmt=",.2f"), "",
           "## Retraining periods and chosen thresholds", per.to_markdown(index=False), "",
           "## Feature importance (last model, % gain)", imp.to_frame("pct").to_markdown(), ""]
    path.write_text("\n".join(txt), encoding="utf-8")
    r.to_csv(path.with_suffix(".csv"), index=False)
    return {"trading": tm, "forecast": fm, "baselines": base, "monthly": mon}
