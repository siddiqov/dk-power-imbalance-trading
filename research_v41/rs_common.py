"""Nurex V4.1 research harness (2026-09-30). READ-ONLY use of the production code.

Nothing here changes the live system: it imports the production modules, reads a private
copy of the data store, and writes only to results/research_v41/.
The walk-forward loop mirrors pipeline_id.walk_forward exactly (30-day retrains, thresholds
tuned on the last 90 validation days with dec.tune, dec.decide, fallback BUY, risk overlay),
so every variant is scored like the official replay, on the same quarters.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from v4_1_intraday import settings as S                     # noqa: E402
from v4_1_intraday import pipeline_id as P                  # noqa: E402
from nurex42 import decision as dec                         # noqa: E402

OUT = BASE / "results" / "research_v41"
CACHE = OUT / "cache"
COPY = BASE / "Nurex_V4_2" / "data" / "nurex42.research_copy.duckdb"
Q = pd.Timedelta(minutes=15)
DOWN, FLAT, UP = 0, 1, 2
N_JOBS = 4


def cfg():
    return S.load(db_path=str(COPY), mode="batch")


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "research.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------- data
def load_area(area):
    t = pd.read_parquet(CACHE / f"t_{area}.parquet")
    Xt = pd.read_parquet(CACHE / f"Xt_{area}.parquet")
    Xe = pd.read_parquet(CACHE / f"Xe_{area}.parquet")
    fd = CACHE / f"drift_{area}.parquet"
    if fd.exists():
        d = pd.read_parquet(fd)
        Xt = Xt.join(d[[c for c in d.columns if c.startswith("t_")]].rename(columns=lambda c: c[2:]))
        Xe = Xe.join(d[[c for c in d.columns if c.startswith("e_")]].rename(columns=lambda c: c[2:]))
    return t, Xt, Xe


DRIFT_PREFIX = "fd_"

CAL = {"dow", "holiday", "hour", "hour_cos", "hour_sin", "is_evening_peak", "lead_min", "minute_of_day",
       "month", "pre_holiday", "q_in_hour", "qod", "weekend"}


def group_of(c: str) -> str:
    if c.startswith(DRIFT_PREFIX):
        return "NEW_forecast_drift"
    if c.startswith(("mfrr_", "oz_mfrr_")):
        return "mfrr_volumes"
    if c.startswith("oz_"):
        return "other_zone_state"
    if c.startswith(("da", "coupled_")) or c in ("res_share_da", "nrv_x_da"):
        return "dayahead_prices"
    if c.startswith(("last_", "spread_")) or c == "pos_qod_frac7":
        return "imbalance_history"
    if c.startswith(("fast_", "age_")):
        return "system_state"
    if c.startswith("live_"):
        return "live_grid"
    if c.startswith(("f_", "rev_", "rev1h_")):
        return "wind_solar_forecasts"
    if c.startswith(("e_", "sched_", "loadfc_")):
        return "entsoe_grid"
    if c.startswith("wx_"):
        return "weather"
    if c.startswith("umm_"):
        return "outages_umm"
    if c.startswith("freq_"):
        return "frequency"
    if c in CAL:
        return "calendar"
    return "other"


# ----------------------------------------------------------------------------- learners
def _impute(X, med=None):
    Xv = X.astype(float)
    if med is None:
        med = Xv.median()
    return Xv.fillna(med).fillna(0.0), med


class TwoPart:
    """Same structure as nurex42.models.SpreadModel (P(down/flat/up) + E[s|UP], E[s|DOWN] +
    flat mean), with the learner type swappable. No quantile models (not used by decisions)."""

    def __init__(self, kind, lgbm_params, deadband):
        self.kind, self.p, self.deadband = kind, dict(lgbm_params), deadband

    # -- factories
    def _clf(self):
        k, p = self.kind, self.p
        if k == "lgbm":
            import lightgbm as lgb
            q = dict(p); q["n_jobs"] = N_JOBS
            return lgb.LGBMClassifier(objective="multiclass", **q)
        if k == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, subsample=0.8,
                                     colsample_bytree=0.7, reg_lambda=2.0, min_child_weight=20,
                                     tree_method="hist", n_jobs=N_JOBS, objective="multi:softprob")
        if k == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                                      loss_function="MultiClass", verbose=0, thread_count=N_JOBS)
        if k == "histgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                  min_samples_leaf=100, l2_regularization=2.0)
        if k == "extratrees":
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=50, max_features=0.3, n_jobs=N_JOBS)
        if k == "randomforest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_features=0.3,
                                          n_jobs=N_JOBS, max_samples=0.5)
        if k == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(C=0.1, max_iter=500)
        if k == "mlp":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True,
                                 max_iter=60, random_state=0)
        raise ValueError(k)

    def _reg(self):
        k, p = self.kind, self.p
        if k == "lgbm":
            import lightgbm as lgb
            q = dict(p); q["n_jobs"] = N_JOBS
            return lgb.LGBMRegressor(objective="regression", **q)
        if k == "xgboost":
            import xgboost as xgb
            return xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, subsample=0.8,
                                    colsample_bytree=0.7, reg_lambda=2.0, min_child_weight=20,
                                    tree_method="hist", n_jobs=N_JOBS)
        if k == "catboost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(iterations=400, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=0,
                                     thread_count=N_JOBS)
        if k == "histgb":
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                 min_samples_leaf=100, l2_regularization=2.0)
        if k == "extratrees":
            from sklearn.ensemble import ExtraTreesRegressor
            return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=50, max_features=0.3, n_jobs=N_JOBS)
        if k == "randomforest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=300, min_samples_leaf=50, max_features=0.3,
                                         n_jobs=N_JOBS, max_samples=0.5)
        if k == "logistic":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=10.0)
        if k == "mlp":
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True,
                                max_iter=60, random_state=0)
        raise ValueError(k)

    @property
    def needs_scaling(self):
        return self.kind in ("logistic", "mlp", "extratrees", "randomforest")

    def _prep(self, X, fit=False):
        if not self.needs_scaling:
            return X.astype(float)
        if fit:
            Xi, self._med = _impute(X)
            self._mu, self._sd = Xi.mean(), Xi.std().replace(0, 1.0).fillna(1.0)
        else:
            Xi, _ = _impute(X, self._med)
        if self.kind in ("logistic", "mlp"):
            return ((Xi - self._mu) / self._sd).clip(-8, 8)
        return Xi

    def fit(self, X, y, stress_q=0.99):
        self.features = list(X.columns)
        yv = y.values.astype(float)
        cls = np.where(yv > self.deadband, UP, np.where(yv < -self.deadband, DOWN, FLAT))
        Xp = self._prep(X, fit=True)
        self.present = sorted(set(cls.tolist()))
        self.clf = self._clf()
        self.clf.fit(Xp, cls)
        self.reg = {}
        for c in (UP, DOWN):
            m = cls == c
            if m.sum() >= 200:
                r = self._reg(); r.fit(Xp[m], yv[m]); self.reg[c] = r
            else:
                self.reg[c] = float(np.mean(yv[m])) if m.any() else 0.0
        self.flat_mean = float(np.mean(yv[cls == FLAT])) if (cls == FLAT).any() else 0.0
        self.stress = {"low": float(np.quantile(yv, 1 - stress_q)), "high": float(np.quantile(yv, stress_q))}
        return self

    def predict(self, X):
        X = X.reindex(columns=self.features)
        Xp = self._prep(X)
        pr = self.clf.predict_proba(Xp)
        probs = np.zeros((len(X), 3))
        for i, c in enumerate(getattr(self.clf, "classes_", self.present)):
            probs[:, int(c)] = pr[:, i]
        f = lambda c: self.reg[c].predict(Xp) if hasattr(self.reg[c], "predict") else np.full(len(X), self.reg[c])
        mu_up, mu_dn = np.maximum(f(UP), 0.0), np.minimum(f(DOWN), 0.0)
        exp = probs[:, UP] * mu_up + probs[:, DOWN] * mu_dn + probs[:, FLAT] * self.flat_mean
        return pd.DataFrame({"p_down": probs[:, DOWN], "p_flat": probs[:, FLAT], "p_up": probs[:, UP],
                             "mu_up": mu_up, "mu_down": mu_dn, "exp_spread": exp}, index=X.index)

    def gain(self):
        if self.kind != "lgbm":
            return None
        imp = pd.Series(0.0, index=self.features)
        for m in [self.clf] + [r for r in self.reg.values() if hasattr(r, "booster_")]:
            g = pd.Series(m.booster_.feature_importance("gain"), index=self.features)
            imp += g / max(g.sum(), 1e-9)
        return imp


# ----------------------------------------------------------------------------- walk-forward
def _usable(X):
    return [c for c in X.columns if X[c].notna().mean() > 0.02 and X[c].nunique(dropna=True) > 1]


def walk(area, t, Xt, Xe, c, cols=None, kind="lgbm", seq=None, perm_groups=None, tag=""):
    """Mirror of pipeline_id.walk_forward with a column subset / learner. Returns (results, extras)."""
    ic = c["intraday"]
    cols = list(cols) if cols is not None else list(Xt.columns)
    first = (t["quarter_utc"].min() + pd.Timedelta(days=ic["min_train_days"])).normalize()
    last = t["quarter_utc"].max()
    step = pd.Timedelta(days=ic["retrain_every_days"])
    mc = c["model"]
    make = (lambda: TwoPart(kind, mc["lgbm"], mc["direction_deadband_eur"])) if seq is None else seq
    results, periods, gains, perm = [], [], [], []
    p0 = first
    while p0 <= last:
        p1 = p0 + step
        rows = t.index[(t["quarter_utc"] >= p0) & (t["quarter_utc"] < p1)]
        if len(rows) == 0:
            p0 = p1; continue
        cut = t.loc[rows, "as_of_trade"].min()
        known = t["label_known_at"] <= cut
        val_start = cut - pd.Timedelta(days=ic["validation_days"])
        fit_rows = t.index[known & (t["label_known_at"] <= val_start)]
        val_rows = t.index[known & (t["as_of_trade"] >= val_start)]
        all_rows = t.index[known]

        def fit(r):
            X = pd.concat([Xt.loc[r, cols], Xe.loc[r, cols]], ignore_index=True)
            y = pd.concat([t.loc[r, "spread"]] * 2, ignore_index=True)
            u = _usable(X)
            return make().fit(X[u], y, stress_q=c["risk"]["stress_quantile"])

        if len(fit_rows) < 3000 or len(val_rows) < 1000:
            params = {"no_trade": True}
        else:
            mv = fit(fit_rows)
            vp = mv.predict(Xt.loc[val_rows, cols])
            params = dec.tune(vp, t.loc[val_rows, "spread"], t.loc[val_rows, "quarter_utc"], mv.stress, c)
        m = fit(all_rows)
        pt = m.predict(Xt.loc[rows, cols])
        d = dec.decide(pt, t.loc[rows, "quarter_utc"], m.stress, c, params)
        d = P.apply_fallback_buy(d, pt, params, area, c)
        r = t.loc[rows, ["quarter_utc", "as_of_trade", "label_known_at", "spread"]].copy()
        r = r.join(pt[["exp_spread", "p_up", "p_down", "p_flat"]])
        r[["action", "mwh", "edge", "reason"]] = d[["action", "mwh", "edge", "reason"]]
        results.append(r)
        periods.append({"p0": p0, "buy": json.dumps(params.get("buy")), "sell": json.dumps(params.get("sell"))})
        if hasattr(m, "gain"):
            g = m.gain()
            if g is not None:
                gains.append(g)
        if perm_groups:
            perm.append(_perm(m, pt, Xt.loc[rows, cols], t.loc[rows], params, area, c, perm_groups))
        log(f"  [{tag}{area}] {p0.date()} n={len(all_rows)} buy={params.get('buy')} sell={params.get('sell')} "
            f"trades={int((d['action'] != dec.HOLD).sum())}")
        p0 = p1
    res = P.apply_risk_overlays(pd.concat(results, ignore_index=True), c)
    extra = {"periods": pd.DataFrame(periods)}
    if gains:
        extra["gain"] = pd.concat(gains, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
    if perm:
        extra["perm"] = pd.DataFrame(perm)
    return res, extra


def _logloss(pt, spread, deadband):
    y = np.where(spread > deadband, UP, np.where(spread < -deadband, DOWN, FLAT))
    p = np.clip(pt[["p_down", "p_flat", "p_up"]].to_numpy(), 1e-6, 1)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def _perm(m, pt, X, trows, params, area, c, groups):
    """Group permutation importance on the out-of-sample month: shuffle one group's columns,
    re-predict, change in 3-class log loss and in raw decision PnL (same thresholds)."""
    rng = np.random.default_rng(0)
    db = c["model"]["direction_deadband_eur"]
    base_ll = _logloss(pt, trows["spread"].values, db)
    base_pnl = _raw_pnl(pt, trows, params, area, c, m.stress)
    out = {}
    for g, gcols in groups.items():
        gc = [x for x in gcols if x in X.columns]
        if not gc:
            continue
        Xp = X.copy()
        idx = rng.permutation(len(Xp))
        Xp[gc] = Xp[gc].to_numpy()[idx]
        pp = m.predict(Xp)
        out[f"ll_{g}"] = _logloss(pp, trows["spread"].values, db) - base_ll
        out[f"pnl_{g}"] = base_pnl - _raw_pnl(pp, trows, params, area, c, m.stress)
    return out


def _raw_pnl(pt, trows, params, area, c, stress):
    d = dec.decide(pt, trows["quarter_utc"], stress, c, params)
    d = P.apply_fallback_buy(d, pt, params, area, c)
    s = np.where(d["action"] == dec.BUY, 1.0, np.where(d["action"] == dec.SELL, -1.0, 0.0))
    return float(np.sum(s * d["mwh"].to_numpy(float) * trows["spread"].to_numpy(float)
                        - np.where(s != 0, d["mwh"].to_numpy(float) * c.cost_per_mwh, 0.0)))


def score(res, c):
    cost = c.cost_per_mwh
    tr = res[res["action"] != dec.HOLD]
    net = res["net_eur"].sum()
    mwh = tr["mwh"].sum()
    day = res.assign(day=res["quarter_utc"].dt.date).groupby("day")["net_eur"].sum()
    eq = day.cumsum()
    dd = float((eq - eq.cummax()).min()) if len(eq) else 0.0
    mon = res.assign(m=res["quarter_utc"].dt.to_period("M")).groupby("m")["net_eur"].sum()
    half = res["quarter_utc"] < res["quarter_utc"].quantile(0.5)
    db = c["model"]["direction_deadband_eur"]
    y = res["spread"].to_numpy(float)
    nf = np.abs(y) > db
    pdir = np.where(res["p_up"] >= res["p_down"], 1, -1)
    from sklearn.metrics import roc_auc_score
    try:
        auc_up = roc_auc_score((y > db).astype(int), res["p_up"])
        auc_dn = roc_auc_score((y < -db).astype(int), res["p_down"])
    except Exception:
        auc_up = auc_dn = np.nan
    return {"net_eur": round(net, 0), "trades": int(len(tr)), "mwh": round(mwh, 1),
            "eur_per_mwh": round(net / mwh, 2) if mwh else 0.0, "max_dd_eur": round(dd, 0),
            "sharpe_daily": round(float(day.mean() / day.std() * np.sqrt(365)) if day.std() > 0 else 0.0, 2),
            "pos_months": f"{int((mon > 0).sum())}/{len(mon)}",
            "net_h1": round(res.loc[half, "net_eur"].sum(), 0), "net_h2": round(res.loc[~half, "net_eur"].sum(), 0),
            "net_2x_cost": round(net - tr["mwh"].sum() * cost, 0),
            "logloss": round(_logloss(res, y, db), 4), "auc_up": round(auc_up, 4), "auc_down": round(auc_dn, 4),
            "dir_hit_nonflat": round(float((pdir[nf] == np.sign(y[nf])).mean()), 4),
            "mae_exp": round(float(np.mean(np.abs(res["exp_spread"] - y))), 2)}


def save_row(suite, area, variant, sc, extra_info=""):
    OUT.mkdir(parents=True, exist_ok=True)
    row = {"time": time.strftime("%Y-%m-%d %H:%M"), "suite": suite, "area": area, "variant": variant,
           **sc, "info": extra_info}
    f = OUT / "summary.csv"
    pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)
    log(f"RESULT {suite} {area} {variant}: net={sc['net_eur']:,.0f} trades={sc['trades']} "
        f"eur/MWh={sc['eur_per_mwh']} dd={sc['max_dd_eur']:,.0f} h1={sc['net_h1']:,.0f} h2={sc['net_h2']:,.0f} "
        f"ll={sc['logloss']} aucU={sc['auc_up']} aucD={sc['auc_down']}")
