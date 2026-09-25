"""Nurex V4.1 Intraday: design matrix, walk-forward replay, final training and prediction.

Trading model (simulation):
  For every delivery quarter q the decision is taken at a = q - gate_lead (60 min).
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
from nurex42.storage import Store

log = logging.getLogger("nurex41id")
Q = pd.Timedelta(minutes=15)
VERSION = "4.1-intraday-2026-09-17"


# ----------------------------------------------------------------------------- data
def target_frame(fb: IntradayFeatureBuilder, area: str, cfg, start=None, end=None) -> pd.DataFrame:
    t = fb.targets_available(area)
    t = t[t["quarter_utc"] >= pd.Timestamp(cfg["history_start"])]
    if start is not None:
        t = t[t["quarter_utc"] >= pd.Timestamp(start)]
    if end is not None:
        t = t[t["quarter_utc"] < pd.Timestamp(end)]
    t = t.reset_index(drop=True)
    t["as_of_trade"] = t["quarter_utc"] - cfg.lead
    t["label_known_at"] = t["quarter_utc"] + Q + fb.lag_imb
    t["batch_start_utc"] = t["quarter_utc"]          # intraday: every quarter is its own decision
    return t


def build_X(fb, area, quarters: pd.Series, as_of: pd.Series) -> pd.DataFrame:
    return fb.build(area, pd.DataFrame({"quarter_utc": quarters.values, "as_of_utc": as_of.values}))


def design(fb, area, t, cfg):
    """X at the trading lead, plus extra-lead copies used only for training."""
    Xt = build_X(fb, area, t["quarter_utc"], t["as_of_trade"])
    extras = []
    for m in cfg["intraday"].get("train_extra_leads_minutes", []) or []:
        extras.append(build_X(fb, area, t["quarter_utc"], t["quarter_utc"] - pd.Timedelta(minutes=int(m))))
    cols = sorted(set(Xt.columns).union(*[set(e.columns) for e in extras]))
    return Xt.reindex(columns=cols), [e.reindex(columns=cols) for e in extras]


def usable_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if X[c].notna().mean() > 0.02 and X[c].nunique(dropna=True) > 1]


def fit_model(Xt, extras, y, rows, cfg) -> SpreadModel:
    X = pd.concat([Xt.loc[rows]] + [e.loc[rows] for e in extras], ignore_index=True)
    yy = pd.concat([y.loc[rows]] * (1 + len(extras)), ignore_index=True)
    cols = usable_columns(X)
    m = SpreadModel(params=cfg["model"]["lgbm"], deadband=cfg["model"]["direction_deadband_eur"],
                    quantiles=tuple(cfg["model"]["quantiles"]))
    return m.fit(X[cols], yy, stress_q=cfg["risk"]["stress_quantile"])


def train_with_validation(Xt, extras, t, cfg, cut: pd.Timestamp):
    """Fit on labels known before `cut`; tune BUY/SELL thresholds on the last validation_days
    (profitable in both halves, >= 1x costs per MWh, else that side is switched off)."""
    known = t["label_known_at"] <= cut
    val_start = cut - pd.Timedelta(days=cfg["intraday"]["validation_days"])
    fit_rows = t.index[known & (t["label_known_at"] <= val_start)]
    val_rows = t.index[known & (t["as_of_trade"] >= val_start)]
    all_rows = t.index[known]
    if len(fit_rows) < 3000 or len(val_rows) < 1000:
        params = {"no_trade": True, "val_net_eur": 0.0, "val_trades": 0,
                  "note": f"insufficient history (fit={len(fit_rows)}, val={len(val_rows)})"}
        val_pred = None
    else:
        mv = fit_model(Xt, extras, t["spread"], fit_rows, cfg)
        val_pred = mv.predict(Xt.loc[val_rows])
        params = dec.tune(val_pred, t.loc[val_rows, "spread"], t.loc[val_rows, "batch_start_utc"], mv.stress, cfg)
    if len(all_rows) < 3000:
        raise RuntimeError(f"only {len(all_rows)} labelled quarters before {cut}")
    m = fit_model(Xt, extras, t["spread"], all_rows, cfg)
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
def walk_forward(store: Store, cfg, area: str, start=None, end=None, fb=None) -> dict:
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
        m, params, n_train = train_with_validation(Xt, extras, t, cfg, cut)
        pt = m.predict(Xt.loc[rows])
        d = dec.decide(pt, t.loc[rows, "batch_start_utc"], m.stress, cfg, params)
        r = t.loc[rows, ["quarter_utc", "batch_start_utc", "as_of_trade", "label_known_at", "spread"]].copy()
        r = r.join(pt[["exp_spread", "p_up", "p_down", "p_flat", "q10", "q50", "q90"]])
        for col in ("last_spread", "spread_qod_mean7", "fast_satisfied_demand_mw", "fast_dominating_direction"):
            r[col] = Xt.loc[rows, col].values if col in Xt.columns else np.nan
        r[["action", "mwh", "edge", "reason"]] = d[["action", "mwh", "edge", "reason"]]
        results.append(r)
        periods.append({"period_start": p0, "cut": cut, "n_train": n_train,
                        **{k: v for k, v in params.items() if k != "note"}})
        log.info("[%s] %s train=%d buy=%s sell=%s trades=%d", area, p0.date(), n_train, params.get("buy"),
                 params.get("sell"), int((d["action"] != dec.HOLD).sum()))
        p0 = p1
    res = apply_risk_overlays(pd.concat(results, ignore_index=True), cfg)
    res["area"] = area
    imp = m.feature_importance() if m is not None else pd.Series(dtype=float)
    return {"area": area, "results": res, "periods": pd.DataFrame(periods), "feature_importance": imp,
            "book": f"v41id-replay:{area}"}


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

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "IntradayBundle":
        with open(path, "rb") as f:
            return pickle.load(f)


def model_path(cfg, area: str) -> Path:
    return cfg.path("models_dir") / f"v4_1_intraday_{area}.pkl"


def train_final(store: Store, cfg, area: str, fb=None, now=None) -> IntradayBundle:
    fb = fb or IntradayFeatureBuilder(store, cfg)
    t = target_frame(fb, area, cfg)
    Xt, extras = design(fb, area, t, cfg)
    cut = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    m, params, n = train_with_validation(Xt, extras, t, cfg, cut)
    return IntradayBundle(area=area, model=m, decision_params=params,
                          trained_until=t.loc[t["label_known_at"] <= cut, "quarter_utc"].max(),
                          n_train=n, config=cfg.raw)


def predict_quarters(store: Store, cfg, bundle: IntradayBundle, quarters, now=None, fb=None) -> pd.DataFrame:
    """Forecast + decision for the given delivery quarters (UTC-naive).

    Each quarter is evaluated as of min(now, quarter - gate_lead): past quarters are replayed
    exactly as they would have been decided; future quarters use the information available now
    (lead_min > gate lead, the model was trained with longer leads too).
    """
    fb = fb or IntradayFeatureBuilder(store, cfg)
    q = pd.Series(pd.to_datetime(quarters)).reset_index(drop=True)
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    gate = q - cfg.lead
    as_of = gate.where(gate <= now, now)
    X = build_X(fb, bundle.area, q, as_of)
    pr = bundle.model.predict(X)
    d = dec.decide(pr, q, bundle.model.stress, cfg, bundle.decision_params)
    out = pd.DataFrame({"quarter_utc": q, "as_of_utc": as_of, "gate_utc": gate,
                        "decision_final": (gate <= now).values})
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

    return out
