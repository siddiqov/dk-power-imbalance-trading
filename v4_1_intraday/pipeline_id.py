"""Nurex V4.1 Intraday: design matrix, walk-forward replay, final training and prediction.

Trading model (simulation):
  Decision time (information cut-off) a per delivery quarter q, see settings.py:
    gate mode   a = q - gate_lead (60 min), every quarter on its own
    batch mode  a = batch deadline - final_run; batch = local wall-clock hour (4 quarters),
                deadline = first quarter - 2h15  ->  a = first quarter - 2h30 for all 4 quarters
  BUY  x MWh -> pnl = x * (imbalance - day-ahead) - x * cost
  SELL x MWh -> pnl = x * (day-ahead - imbalance) - x * cost
  (the intraday fill price is approximated by the day-ahead price + slippage in the cost,
   because no Nord Pool intraday price history is available yet)
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import settings as S
from .features_id import IntradayFeatureBuilder
from nurex42 import decision as dec
from nurex42.models import SpreadModel
from .model_v41 import SpreadModelV41, recency_weights
from .model_logreg import SpreadModelLogReg
from nurex42.storage import Store

log = logging.getLogger("nurex41id")
Q = pd.Timedelta(minutes=15)
VERSION = "4.1-intraday-2026-09-17"
VERSION_BATCH = "4.1-batch-2026-09-27"      # hourly client batches, locked 2h15 ahead

# ----------------------------------------------------------------------------- model families
# 2026-10-04: two V4.1 models per zone.
#   "lgbm"   = the current V4.1 LightGBM model (unchanged)
#   "logreg" = logistic regression, all features, C / class weight chosen every retrain on the
#              validation window (model_logreg.py)
# config `models.recommended` decides, per zone, whose decisions are LOCKED and sent to the
# client. Zones whose recommended model is not LightGBM also run LightGBM in a shadow journal,
# so the dashboard tournament compares real locked decisions of both.
FAMILIES = ("lgbm", "logreg")
FAMILY_LABEL = {"lgbm": "V4.1 LightGBM", "logreg": "V4.1 Logistic regression"}


def recommended_family(cfg, area: str) -> str:
    rec = ((cfg.raw.get("models") or {}).get("recommended") or {})
    f = str(rec.get(area, "lgbm")).lower()
    return f if f in FAMILIES else "lgbm"


def families_for(cfg, area: str) -> list[str]:
    """Models trained / run for a zone: the recommended one first, plus LightGBM for comparison."""
    rec = recommended_family(cfg, area)
    return [rec] if rec == "lgbm" else [rec, "lgbm"]


def family_version(cfg, family: str) -> str:
    v = VERSION_BATCH if cfg.batch_mode else VERSION
    return v if family == "lgbm" else f"{v}+{family}"


# ----------------------------------------------------------------------------- data
def target_frame(fb: IntradayFeatureBuilder, area: str, cfg, start=None, end=None) -> pd.DataFrame:
    t = fb.targets_available(area)
    t = t[t["quarter_utc"] >= pd.Timestamp(cfg["history_start"])]
    if start is not None:
        t = t[t["quarter_utc"] >= pd.Timestamp(start)]
    if end is not None:
        t = t[t["quarter_utc"] < pd.Timestamp(end)]
    t = t.reset_index(drop=True)
    t["as_of_trade"] = cfg.decision_times(t["quarter_utc"]).values
    t["label_known_at"] = t["quarter_utc"] + Q + fb.lag_imb
    t["batch_start_utc"] = cfg.batch_starts(t["quarter_utc"]).values   # gate mode: the quarter itself
    t["deadline_utc"] = cfg.deadlines(t["quarter_utc"]).values
    return t


def build_X(fb, area, quarters: pd.Series, as_of: pd.Series) -> pd.DataFrame:
    return fb.build(area, pd.DataFrame({"quarter_utc": quarters.values, "as_of_utc": as_of.values}))


def design(fb, area, t, cfg):
    """X at the trading lead, plus extra-lead copies used only for training."""
    Xt = build_X(fb, area, t["quarter_utc"], t["as_of_trade"])
    extras = []
    for m in cfg.train_extra_leads:
        extras.append(build_X(fb, area, t["quarter_utc"], t["quarter_utc"] - pd.Timedelta(minutes=int(m))))
    cols = sorted(set(Xt.columns).union(*[set(e.columns) for e in extras]))
    return Xt.reindex(columns=cols), [e.reindex(columns=cols) for e in extras]


def usable_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if X[c].notna().mean() > 0.02 and X[c].nunique(dropna=True) > 1]


ID_MARKET_PREFIXES = ("id15_", "id60_", "book15_", "trd15_")   # Nord Pool intraday market inputs


def intraday_market_allowed(Xt_rows: pd.DataFrame, cfg) -> list[str]:
    """Safeguard (2026-09-28): Nord Pool intraday-market inputs enter training only when
    switched on (model.intraday_market_features, after a replay shows they help) AND each has
    values for at least model.intraday_market_min_days x 96 quarters at the trading lead.
    Returns the id-market columns that are allowed; all others of that family are dropped."""
    mc = cfg["model"]
    cols = [c for c in Xt_rows.columns if c.startswith(ID_MARKET_PREFIXES)]
    if not cols or not mc.get("intraday_market_features", False):
        return []
    need = int(mc.get("intraday_market_min_days", 90)) * 96
    return [c for c in cols if int(Xt_rows[c].notna().sum()) >= need]


def fit_model(Xt, extras, y, rows, cfg, times=None) -> SpreadModel:
    X = pd.concat([Xt.loc[rows]] + [e.loc[rows] for e in extras], ignore_index=True)
    yy = pd.concat([y.loc[rows]] * (1 + len(extras)), ignore_index=True)
    allowed = set(intraday_market_allowed(Xt.loc[rows], cfg))
    cols = [c for c in usable_columns(X) if not c.startswith(ID_MARKET_PREFIXES) or c in allowed]
    mc = cfg["model"]
    if not mc.get("enhanced"):
        m = SpreadModel(params=mc["lgbm"], deadband=mc["direction_deadband_eur"], quantiles=tuple(mc["quantiles"]))
        return m.fit(X[cols], yy, stress_q=cfg["risk"]["stress_quantile"])
    # 2026-09-28 upgrade: Huber size loss, recency weights, spike classifiers (see model_v41.py)
    w = None
    if times is not None and mc.get("recency_half_life_days"):
        tt = pd.concat([pd.Series(times.loc[rows].values)] * (1 + len(extras)), ignore_index=True)
        w = recency_weights(tt, mc["recency_half_life_days"])
    m = SpreadModelV41(params=mc["lgbm"], deadband=mc["direction_deadband_eur"], quantiles=tuple(mc["quantiles"]),
                       size_objective=mc.get("size_objective", "huber"), huber_delta=float(mc.get("huber_delta", 40.0)),
                       spike_level=float(mc.get("spike_level_eur", 150.0)))
    return m.fit(X[cols], yy, stress_q=cfg["risk"]["stress_quantile"], sample_weight=w)


def _design_rows(Xt, extras, y, rows, cfg):
    """Training matrix exactly as fit_model builds it (same rows, same usable columns)."""
    X = pd.concat([Xt.loc[rows]] + [e.loc[rows] for e in extras], ignore_index=True)
    yy = pd.concat([y.loc[rows]] * (1 + len(extras)), ignore_index=True)
    allowed = set(intraday_market_allowed(Xt.loc[rows], cfg))
    cols = [c for c in usable_columns(X) if not c.startswith(ID_MARKET_PREFIXES) or c in allowed]
    return X[cols], yy


def logreg_candidates(cfg, area: str | None = None) -> list[dict]:
    """Settings tried every retrain, in a fixed order (ties keep the first).
    config logreg.C_grid x logreg.class_weight_grid (a list, or a dict per zone)."""
    lc = cfg.raw.get("logreg") or {}
    cs = [float(c) for c in (lc.get("C_grid") or [0.003, 0.01, 0.03, 0.1, 0.3, 1.0])]
    cw = lc.get("class_weight_grid", ["balanced", None])
    if isinstance(cw, dict):
        cw = cw.get(area, ["balanced", None])
    cw = [None if (w is None or str(w).lower() in ("none", "null")) else str(w) for w in cw]
    return [{"C": c, "class_weight": w} for c in sorted(cs) for w in cw]


def fit_logreg(Xt, extras, y, rows, cfg, C, class_weight, with_quantiles=True) -> SpreadModelLogReg:
    X, yy = _design_rows(Xt, extras, y, rows, cfg)
    lc = cfg.raw.get("logreg") or {}
    mc = cfg["model"]
    m = SpreadModelLogReg(C=float(C), class_weight=class_weight, deadband=mc["direction_deadband_eur"],
                          lgbm_params=dict(mc["lgbm"]), quantiles=tuple(mc["quantiles"]),
                          with_quantiles=with_quantiles, ridge_alpha=float(lc.get("ridge_alpha", 10.0)),
                          max_iter=int(lc.get("max_iter", 1000)))
    return m.fit(X, yy, stress_q=cfg["risk"]["stress_quantile"])


def validation_pnl(pred, spread, quarters, params, area, cfg, stress) -> float:
    """Raw validation PnL of a tuned rule (decide + fallback BUY, no risk overlay) - the score
    used to pick the logistic settings each retrain (identical to the Stage 4 research)."""
    d = dec.decide(pred, quarters, stress, cfg, params)
    d = apply_fallback_buy(d, pred, params, area, cfg)
    s = np.where(d["action"] == dec.BUY, 1.0, np.where(d["action"] == dec.SELL, -1.0, 0.0))
    mwh = d["mwh"].to_numpy(float)
    return float(np.sum(s * mwh * spread.to_numpy(float) - np.where(s != 0, mwh * cfg.cost_per_mwh, 0.0)))


# ----------------------------------------------------------------------------- guards (2026-09-28)
def apply_guards(pred: pd.DataFrame, d: pd.DataFrame, guards: dict | None) -> pd.DataFrame:
    """HOLD a trade when the model sees a likely spike against it or a likely flat quarter.
      spike_up_max  : SELL -> HOLD when P(spread > +spike_level) >= this
      spike_down_max: BUY  -> HOLD when P(spread < -spike_level) >= this
      flat_max      : any  -> HOLD when P(flat: imbalance price == spot) >= this (costs only)
    `pred` and `d` share the same index. Missing columns (old models) = guard off."""
    if not guards:
        return d
    d = d.copy()
    rules = (("spike_up_max", "p_spike_up", "SELL", "spike guard: likely up-spike"),
             ("spike_down_max", "p_spike_down", "BUY", "spike guard: likely down-spike"),
             ("flat_max", "p_flat", None, "flat guard: imbalance likely = spot"))
    for key, col, side, why in rules:
        lim = guards.get(key)
        if lim is None or col not in pred.columns:
            continue
        hit = (pred[col].values >= float(lim)) & (d["action"].values != "HOLD")
        if side:
            hit &= d["action"].values == side
        if hit.any():
            d.loc[hit, ["action", "reason"]] = ["HOLD", why]
            d.loc[hit, ["mwh", "edge"]] = 0.0
    return d


def tune_guards(pred: pd.DataFrame, spread: pd.Series, keys: pd.Series, stress: dict, params: dict,
                cfg) -> dict | None:
    """Pick guard thresholds on the validation window (after the BUY/SELL rule is tuned).
    Kept only if net PnL improves and stays positive in both halves; else no guard."""
    gc = cfg["decision"].get("guards") or {}
    if not gc.get("enabled", True) or "p_flat" not in pred.columns:
        return None
    d0 = dec.decide(pred, keys, stress, cfg, params)
    cost = cfg.cost_per_mwh
    half = np.arange(len(pred)) < len(pred) // 2

    def score(d):
        _, _, net = dec.pnl(d["action"], d["mwh"], spread, cost)
        net = np.asarray(net, dtype=float)
        return float(net.sum()), net[half].sum() > 0 and net[~half].sum() > 0

    best_net, _ = score(d0)
    best = None
    import itertools
    grid_s = [None] + list(gc.get("spike_grid", [0.10, 0.15, 0.20, 0.30]))
    grid_f = [None] + list(gc.get("flat_grid", [0.40, 0.50, 0.60, 0.70]))
    for su, sd, fm in itertools.product(grid_s, grid_s, grid_f):
        g = {"spike_up_max": su, "spike_down_max": sd, "flat_max": fm}
        if su is None and sd is None and fm is None:
            continue
        net, stable = score(apply_guards(pred, d0, g))
        if stable and net > best_net + 1e-6:
            best, best_net = g, net
    return best


def apply_fallback_buy(out: pd.DataFrame, pred: pd.DataFrame, decision_params: dict, area: str, cfg,
                       ok: bool = True) -> pd.DataFrame:
    """Fallback BUY rule (config decision.fallback_buy): only while the model has no BUY rule."""
    fb_cfg = (cfg["decision"].get("fallback_buy") or {}) if "decision" in cfg.raw else {}
    if not (ok and fb_cfg.get("enabled") and not (decision_params or {}).get("buy")
            and area in (fb_cfg.get("areas") or [area])):
        return out
    edge = pred["exp_spread"] - cfg.cost_per_mwh
    fbuy = ((out["action"] == "HOLD") & (pred["p_up"] >= float(fb_cfg.get("pmin", 0.40)))
            & (edge > float(fb_cfg.get("margin_eur", 10.0))))
    # never a fallback BUY into a likely down-spike or a likely flat quarter
    g = (decision_params or {}).get("guards") or {}
    if g.get("spike_down_max") is not None and "p_spike_down" in pred.columns:
        fbuy &= pred["p_spike_down"] < float(g["spike_down_max"])
    if g.get("flat_max") is not None:
        fbuy &= pred["p_flat"] < float(g["flat_max"])
    if fbuy.any():
        out = out.copy()
        out.loc[fbuy, "action"] = "BUY"
        out.loc[fbuy, "mwh"] = float(fb_cfg.get("mwh", 0.2))
        out.loc[fbuy, "edge"] = edge[fbuy]
        out.loc[fbuy, "reason"] = "fallback BUY rule"
    return out


def train_with_validation(Xt, extras, t, cfg, cut: pd.Timestamp, family: str = "lgbm", area: str | None = None):
    """Fit on labels known before `cut`; tune BUY/SELL thresholds on the last validation_days
    (profitable in both halves, >= 1x costs per MWh, else that side is switched off)."""
    known = t["label_known_at"] <= cut
    val_start = cut - pd.Timedelta(days=cfg["intraday"]["validation_days"])
    fit_rows = t.index[known & (t["label_known_at"] <= val_start)]
    val_rows = t.index[known & (t["as_of_trade"] >= val_start)]
    all_rows = t.index[known]
    times = t["quarter_utc"]
    if family == "logreg":
        return _train_logreg_with_validation(Xt, extras, t, cfg, fit_rows, val_rows, all_rows, area)
    if len(fit_rows) < 3000 or len(val_rows) < 1000:
        params = {"no_trade": True, "val_net_eur": 0.0, "val_trades": 0,
                  "note": f"insufficient history (fit={len(fit_rows)}, val={len(val_rows)})"}
        val_pred = None
    else:
        mv = fit_model(Xt, extras, t["spread"], fit_rows, cfg, times=times)
        val_pred = mv.predict(Xt.loc[val_rows])
        # risk budget key = the quarter (as before); the hourly MWh cap limits a whole batch/hour
        params = dec.tune(val_pred, t.loc[val_rows, "spread"], t.loc[val_rows, "quarter_utc"], mv.stress, cfg)
        if cfg["model"].get("enhanced") and not params.get("no_trade"):
            params["guards"] = tune_guards(val_pred, t.loc[val_rows, "spread"], t.loc[val_rows, "quarter_utc"],
                                           mv.stress, params, cfg)
    if len(all_rows) < 3000:
        raise RuntimeError(f"only {len(all_rows)} labelled quarters before {cut}")
    m = fit_model(Xt, extras, t["spread"], all_rows, cfg, times=times)
    return m, params, len(all_rows)


def _train_logreg_with_validation(Xt, extras, t, cfg, fit_rows, val_rows, all_rows, area):
    """Logistic regression with its settings chosen on the validation window.

    Every candidate (C x class weight) is fitted on the rows before the validation window,
    its BUY/SELL thresholds are tuned on the validation window with dec.tune, and it is scored
    by the validation PnL of that tuned rule. The best candidate is refitted on all labelled
    rows and keeps its own tuned thresholds. Only data known before the cut is used."""
    cands = logreg_candidates(cfg, area)
    tried = []
    if len(fit_rows) < 3000 or len(val_rows) < 1000:
        params = {"no_trade": True, "val_net_eur": 0.0, "val_trades": 0,
                  "note": f"insufficient history (fit={len(fit_rows)}, val={len(val_rows)})"}
        best = cands[0]
    else:
        spread_v, q_v = t.loc[val_rows, "spread"], t.loc[val_rows, "quarter_utc"]
        best, params, best_pnl = None, None, None
        for cand in cands:
            mv = fit_logreg(Xt, extras, t["spread"], fit_rows, cfg, cand["C"], cand["class_weight"],
                            with_quantiles=False)
            vp = mv.predict(Xt.loc[val_rows])
            p = dec.tune(vp, spread_v, q_v, mv.stress, cfg)
            pnl = validation_pnl(vp, spread_v, q_v, p, area, cfg, mv.stress)
            tried.append({**cand, "val_pnl_eur": round(pnl, 2), "buy": p.get("buy"), "sell": p.get("sell")})
            log.info("[%s] logreg C=%s class_weight=%s val_pnl=%.0f buy=%s sell=%s", area, cand["C"],
                     cand["class_weight"], pnl, p.get("buy"), p.get("sell"))
            if best_pnl is None or pnl > best_pnl:
                best, params, best_pnl = cand, p, pnl
    if len(all_rows) < 3000:
        raise RuntimeError(f"only {len(all_rows)} labelled quarters")
    m = fit_logreg(Xt, extras, t["spread"], all_rows, cfg, best["C"], best["class_weight"], with_quantiles=True)
    params = dict(params)
    params["logreg"] = {"C": best["C"], "class_weight": best["class_weight"], "candidates": tried}
    return m, params, len(all_rows)


# ----------------------------------------------------------------------------- risk overlay
def apply_risk_overlays(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Per-quarter walk in decision-time order using only PnL settled at that time:
    daily loss stop, half size beyond 25% drawdown, stop beyond 50% drawdown.
    The drawdown is measured against the equity peak of the last `drawdown_window_days`
    (a permanent all-time peak would halt trading forever once flat, because equity can no
    longer recover)."""
    r = cfg["risk"]
    coll = cfg.collateral_eur
    stop = -float(r["daily_loss_stop_fraction"]) * coll
    half, halt = float(r["drawdown_half_fraction"]) * coll, float(r["drawdown_stop_fraction"]) * coll
    step, mn = float(r["mwh_step"]), float(r["min_mwh_per_quarter"])
    cost = cfg.cost_per_mwh
    df = df.sort_values("as_of_trade").reset_index(drop=True)
    df["local_day"] = df["quarter_utc"].dt.tz_localize("UTC").dt.tz_convert(cfg["local_tz"]).dt.date
    sign0 = np.where(df["action"] == dec.BUY, 1.0, np.where(df["action"] == dec.SELL, -1.0, 0.0))
    mwh = df["mwh"].to_numpy(float).copy()
    action = df["action"].to_numpy(object).copy()
    reason = df["reason"].to_numpy(object).copy()
    spread = df["spread"].to_numpy(float)
    net = np.zeros(len(df))
    order = np.argsort(df["label_known_at"].to_numpy())
    known_at = df["label_known_at"].to_numpy()[order]
    asof = df["as_of_trade"].to_numpy()
    days = df["local_day"].to_numpy()
    from collections import deque
    win = np.timedelta64(int(float(r.get("drawdown_window_days", 7)) * 86400), "s")
    df["mwh_raw"] = df["mwh"].astype(float)
    j, equity = 0, 0.0
    pts_t, pts_eq = [], []          # settled equity path
    dq: deque = deque()             # indices into pts, decreasing equity (sliding-window max)
    exp_ptr, base_eq = 0, 0.0       # equity level at the start of the window
    day_net: dict = {}
    for i in range(len(df)):
        while j < len(order) and known_at[j] <= asof[i]:
            k = order[j]
            equity += net[k]
            pts_t.append(known_at[j]); pts_eq.append(equity)
            while dq and pts_eq[dq[-1]] <= equity:
                dq.pop()
            dq.append(len(pts_t) - 1)
            day_net[days[k]] = day_net.get(days[k], 0.0) + net[k]
            j += 1
        lo = asof[i] - win
        while exp_ptr < len(pts_t) and pts_t[exp_ptr] < lo:
            base_eq = pts_eq[exp_ptr]
            exp_ptr += 1
        while dq and dq[0] < exp_ptr:
            dq.popleft()
        peak = max(base_eq, pts_eq[dq[0]] if dq else base_eq, equity)
        if sign0[i] != 0:
            dd = peak - equity
            mult = 0.0 if dd >= halt else (0.5 if dd >= half else 1.0)
            why = "drawdown guard" if mult < 1 else None
            if day_net.get(days[i], 0.0) <= stop:
                mult, why = 0.0, "daily loss stop"
            if mult < 1.0:
                m = np.floor(mwh[i] * mult / step) * step
                mwh[i] = m if m + 1e-9 >= mn else 0.0
                reason[i] = why
                if mwh[i] <= 0:
                    action[i] = dec.HOLD
        s = 1.0 if action[i] == dec.BUY else (-1.0 if action[i] == dec.SELL else 0.0)
        net[i] = s * mwh[i] * spread[i] - (mwh[i] * cost if s != 0 else 0.0)
    df["mwh"], df["action"], df["reason"] = mwh, action, reason
    s = np.where(df["action"] == dec.BUY, 1.0, np.where(df["action"] == dec.SELL, -1.0, 0.0))
    df["gross_eur"] = s * df["mwh"] * df["spread"]
    df["cost_eur"] = np.where(s != 0, df["mwh"] * cost, 0.0)
    df["net_eur"] = df["gross_eur"] - df["cost_eur"]
    df["net_raw_eur"] = sign0 * df["mwh_raw"] * df["spread"] - np.where(sign0 != 0, df["mwh_raw"] * cost, 0.0)
    return df.drop(columns=["local_day"]).sort_values("quarter_utc").reset_index(drop=True)


# ----------------------------------------------------------------------------- replay
def walk_forward(store: Store, cfg, area: str, start=None, end=None, fb=None, family: str | None = None) -> dict:
    family = family or recommended_family(cfg, area)
    fb = fb or IntradayFeatureBuilder(store, cfg)
    t = target_frame(fb, area, cfg, end=end)
    log.info("[%s] building point-in-time features for %d quarters", area, len(t))
    Xt, extras = design(fb, area, t, cfg)
    ic = cfg["intraday"]
    first = (t["quarter_utc"].min() + pd.Timedelta(days=ic["min_train_days"])).normalize()
    p0 = max(first, pd.Timestamp(start)).normalize() if start else first
    last = t["quarter_utc"].max()
    step = pd.Timedelta(days=ic["retrain_every_days"])
    results, periods, m = [], [], None
    while p0 <= last:
        p1 = p0 + step
        rows = t.index[(t["quarter_utc"] >= p0) & (t["quarter_utc"] < p1)]
        if len(rows) == 0:
            p0 = p1
            continue
        cut = t.loc[rows, "as_of_trade"].min()
        m, params, n_train = train_with_validation(Xt, extras, t, cfg, cut, family=family, area=area)
        pt = m.predict(Xt.loc[rows])
        d = dec.decide(pt, t.loc[rows, "quarter_utc"], m.stress, cfg, params)
        d = apply_guards(pt, d, params.get("guards"))
        d = apply_fallback_buy(d, pt, params, area, cfg)
        r = t.loc[rows, ["quarter_utc", "batch_start_utc", "deadline_utc", "as_of_trade", "label_known_at",
                         "spread"]].copy()
        r = r.join(pt[[c for c in ("exp_spread", "p_up", "p_down", "p_flat", "q10", "q50", "q90",
                                   "p_spike_up", "p_spike_down") if c in pt.columns]])
        for col in ("last_spread", "spread_qod_mean7", "fast_satisfied_demand_mw", "fast_dominating_direction"):
            r[col] = Xt.loc[rows, col].values if col in Xt.columns else np.nan
        r[["action", "mwh", "edge", "reason"]] = d[["action", "mwh", "edge", "reason"]]
        results.append(r)
        lr = params.get("logreg") or {}
        periods.append({"period_start": p0, "cut": cut, "n_train": n_train,
                        **{k: v for k, v in params.items() if k not in ("note", "logreg")},
                        **({"logreg_C": lr.get("C"), "logreg_class_weight": lr.get("class_weight")} if lr else {})})
        log.info("[%s] %s train=%d buy=%s sell=%s trades=%d", area, p0.date(), n_train, params.get("buy"),
                 params.get("sell"), int((d["action"] != dec.HOLD).sum()))
        p0 = p1
    res = apply_risk_overlays(pd.concat(results, ignore_index=True), cfg)
    res["area"] = area
    imp = m.feature_importance() if m is not None else pd.Series(dtype=float)
    return {"area": area, "results": res, "periods": pd.DataFrame(periods), "feature_importance": imp,
            "book": f"v41id-replay:{area}" + ("" if family == "lgbm" else f":{family}"), "family": family}


# ----------------------------------------------------------------------------- live model
@dataclass
class IntradayBundle:
    area: str
    model: SpreadModel
    decision_params: dict
    trained_until: pd.Timestamp
    n_train: int
    version: str = VERSION
    config: dict = field(default_factory=dict)
    family: str = "lgbm"            # 2026-10-04; bundles saved before then are LightGBM

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "IntradayBundle":
        with open(path, "rb") as f:
            return pickle.load(f)


def model_path(cfg, area: str, family: str | None = None) -> Path:
    """LightGBM keeps its original file name (v4_1_batch_DK1.pkl); other families get a suffix
    (v4_1_batch_logreg_DK1.pkl). family=None -> the zone's recommended model."""
    family = family or recommended_family(cfg, area)
    name = "v4_1_batch" if cfg.batch_mode else "v4_1_intraday"
    if family != "lgbm":
        name = f"{name}_{family}"
    return cfg.path("models_dir") / f"{name}_{area}.pkl"


def train_final(store: Store, cfg, area: str, fb=None, now=None, family: str | None = None,
                design_cache: dict | None = None) -> IntradayBundle:
    """Train one model family for the live system. `design_cache` (a dict) lets several
    families share one point-in-time feature build."""
    family = family or recommended_family(cfg, area)
    fb = fb or IntradayFeatureBuilder(store, cfg)
    if design_cache is not None and design_cache.get("area") == area:
        t, Xt, extras = design_cache["t"], design_cache["Xt"], design_cache["extras"]
    else:
        t = target_frame(fb, area, cfg)
        Xt, extras = design(fb, area, t, cfg)
        if design_cache is not None:
            design_cache.clear()
            design_cache.update({"area": area, "t": t, "Xt": Xt, "extras": extras})
    cut = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    m, params, n = train_with_validation(Xt, extras, t, cfg, cut, family=family, area=area)
    return IntradayBundle(area=area, model=m, decision_params=params,
                          trained_until=t.loc[t["label_known_at"] <= cut, "quarter_utc"].max(),
                          n_train=n, version=family_version(cfg, family), config=cfg.raw, family=family)


def predict_quarters(store: Store, cfg, bundle: IntradayBundle, quarters, now=None, fb=None,
                     fallback_buy_ok: bool = True) -> pd.DataFrame:
    """Forecast + decision for the given delivery quarters (UTC-naive).

    Each quarter is evaluated as of min(now, its decision time): past quarters are replayed
    exactly as they would have been decided; future quarters use the information available now
    (provisional; lead_min > decision lead, the model was trained with longer leads too).

    Columns: gate_utc = the deadline (gate closure in gate mode, batch submission deadline in batch
    mode), decision_utc = information cut-off of the final decision, decision_final = that cut-off
    has passed, so the decision can no longer change.
    """
    fb = fb or IntradayFeatureBuilder(store, cfg)
    q = pd.Series(pd.to_datetime(quarters)).reset_index(drop=True)
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    gate = cfg.deadlines(q)
    dtime = cfg.decision_times(q)
    as_of = dtime.where(dtime <= now, now)
    X = build_X(fb, bundle.area, q, as_of)
    pr = bundle.model.predict(X)
    d = dec.decide(pr, q, bundle.model.stress, cfg, bundle.decision_params)
    d = apply_guards(pr, d, (bundle.decision_params or {}).get("guards"))
    out = pd.DataFrame({"quarter_utc": q, "as_of_utc": as_of, "gate_utc": gate,
                        "batch_start_utc": cfg.batch_starts(q), "decision_utc": dtime,
                        "decision_final": (dtime <= now).values})
    out = pd.concat([out, pr.reset_index(drop=True), d[["action", "mwh", "edge", "reason"]].reset_index(drop=True)],
                    axis=1)
    imb = fb.imb[bundle.area]
    known = (q + Q + fb.lag_imb <= now).values
    out["spread_actual"] = np.where(known, imb["spread"].reindex(pd.DatetimeIndex(q)).values, np.nan)
    # Data age AT DECISION TIME: minutes from as_of back to the start of the latest quarter whose
    # imbalance price is actually in the store and was published by as_of. (The feature
    # age_last_min is measured from the delivery quarter, so with a 60-min gate lead it is always
    # >= 105 min and a 90-min guard on it forced every quarter to HOLD.)
    sp = imb["spread"].dropna()
    pub = sp.index + Q + fb.lag_imb                      # publication time of each quarter
    ts = pd.DatetimeIndex(as_of)
    pos = np.searchsorted(pub.values, ts.values, side="right") - 1
    last_q = np.where(pos >= 0, sp.index.values[np.clip(pos, 0, None)], np.datetime64("NaT", "ns"))
    out["data_age_min"] = (ts.values - last_q) / np.timedelta64(1, "m")
    # --- stale-data guard: HOLD when imbalance data is too old to trade reliably ----------
    guard_min = cfg["intraday"].get("stale_data_guard_minutes")
    if guard_min is not None:
        guard_min = float(guard_min)
        stale = out["decision_final"] & (
            out["data_age_min"].isna() | (out["data_age_min"] > guard_min)
        )
        if stale.any():
            log.warning("[%s] stale-data guard: %d quarter(s) forced to HOLD "
                        "(data_age_min > %.0f)", bundle.area, int(stale.sum()), guard_min)
            out.loc[stale, "action"] = "HOLD"
            out.loc[stale, "mwh"] = 0.0
            out.loc[stale, "reason"] = "stale data"

    # --- BUY-FIX [Change 3]: crash guard — suppress BUY when q10 signals deep downside risk
    buy_mask = out["action"] == "BUY"
    if buy_mask.any() and "q10" in out.columns:
        crash_risk = out["q10"] < -30.0
        suppressed = buy_mask & crash_risk
        if suppressed.any():
            log.warning("[%s] BUY crash guard: %d quarter(s) suppressed to HOLD "
                        "(q10 < -30 EUR/MWh)", bundle.area, int(suppressed.sum()))
            out.loc[suppressed, "action"] = "HOLD"
            out.loc[suppressed, "mwh"] = 0.0
            out.loc[suppressed, "reason"] = "buy suppressed: q10 crash risk"

    # --- fallback BUY rule: only while the trained model has no validated BUY rule -------------
    before = out["action"].copy()
    out = apply_fallback_buy(out, out, bundle.decision_params, bundle.area, cfg, ok=fallback_buy_ok)
    fbuy = (out["action"] == "BUY") & (before != "BUY")
    # stale data still forces HOLD
    if fbuy.any() and guard_min is not None:
        st = fbuy & out["decision_final"] & (out["data_age_min"].isna() | (out["data_age_min"] > guard_min))
        out.loc[st, ["action", "mwh", "reason"]] = ["HOLD", 0.0, "stale data"]

    return out
