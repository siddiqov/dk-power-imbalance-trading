"""Stage 4 (2026-10-02): thorough grid search, seed checks and blends for Logistic regression and
the Transformer versus the production LightGBM. RESEARCH ONLY: imports the production modules
read-only, uses the research copy of the data, writes only to results/research_v41/stage4/.

Every run uses the same walk-forward as the official replay (30-day retrains, thresholds tuned by
dec.tune on the last 90 validation days, dec.decide, fallback BUY, risk overlays). In addition each
run SAVES its per-period validation and test predictions, so that afterwards (seconds, no retrain):
  * blends (averaged probabilities) are replayed with re-tuned thresholds, exactly like a real model;
  * an HONEST grid search is done: in every period the configuration is chosen only by its
    profit on that period's validation window (data known before trading), never by test profit.

Usage:  python rs_grid.py all            (whole plan; runs two streams in parallel if >= 8 CPUs)
        python rs_grid.py <list> [...]   lists: lgbm logit_DK1 logit_DK2 tf_DK1 tf_DK2 analyse
Re-running skips every run whose results file already exists (safe to restart after a stop).
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from rs_common import OUT, P, TwoPart, _raw_pnl, _usable, cfg, dec, group_of, load_area, log, save_row, score, N_JOBS
from rs_run import base_cols

S4 = OUT / "stage4"
S4.mkdir(parents=True, exist_ok=True)
PRED_COLS = ["p_down", "p_flat", "p_up", "mu_up", "mu_down", "exp_spread"]

# ----------------------------------------------------------------------------- the plan
LOGIT_GRID = [(C, cw, fs) for fs in ("all", "nolg") for cw in (None, "balanced")
              for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)]
TF_BASE = dict(L=32, d=32, layers=2, drop=0.1, lr=1e-3, epochs=6, seed=0)
TF_DK1 = [dict(seed=0), dict(seed=1), dict(seed=2), dict(fs="nolg"), dict(L=64), dict(L=16),
          dict(d=64), dict(drop=0.3, epochs=10)]
TF_DK2 = [dict(seed=0), dict(seed=1), dict(seed=2), dict(fs="nolg")]


def jobs(name):
    if name == "lgbm":
        return [("lgbm", "DK1", {"fs": "all"}), ("lgbm", "DK1", {"fs": "nolg"}),
                ("lgbm", "DK2", {"fs": "all"}), ("lgbm", "DK2", {"fs": "nolg"})]
    if name.startswith("logit_"):
        a = name.split("_")[1]
        return [("logit", a, {"C": C, "cw": cw, "fs": fs}) for C, cw, fs in LOGIT_GRID]
    if name.startswith("tf_"):
        a = name.split("_")[1]
        return [("tf", a, {**TF_BASE, "fs": "all", **v}) for v in (TF_DK1 if a == "DK1" else TF_DK2)]
    raise ValueError(name)


def run_name(kind, p):
    if kind == "lgbm":
        return f"lgbm_{p['fs']}"
    if kind == "logit":
        return f"logit_{p['fs']}_C{p['C']}_{'bal' if p['cw'] else 'nobal'}"
    return f"tf_{p['fs']}_L{p['L']}_d{p['d']}_l{p['layers']}_dr{p['drop']}_ep{p['epochs']}_s{p['seed']}"


# ----------------------------------------------------------------------------- learners
class LogReg(TwoPart):
    def __init__(self, C, cw, deadband):
        super().__init__("logistic", {}, deadband)
        self.C, self.cw = C, cw

    def _clf(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=self.C, class_weight=self.cw, max_iter=1000)


class TFModel:
    """Transformer over the last L published spreads + MLP over the static features
    (same design as the stage-3 SeqModel, with the size / sequence length / seed configurable)."""

    def __init__(self, deadband, L, d, layers, drop, lr, epochs, seed, **_):
        self.db, self.L, self.d, self.layers, self.drop = deadband, L, d, layers, drop
        self.lr, self.epochs, self.seed = lr, epochs, seed

    def _net(self, n_static):
        import torch
        import torch.nn as nn
        L, d, layers, drop = self.L, self.d, self.layers, self.drop

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.inp = nn.Linear(2, d)
                layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=2 * d, dropout=drop,
                                                   batch_first=True)
                s.enc = nn.TransformerEncoder(layer, num_layers=layers)
                s.pos = nn.Parameter(torch.zeros(1, L, d))
                s.st = nn.Sequential(nn.Linear(n_static, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
                s.cls = nn.Linear(d + 64, 3); s.up = nn.Linear(d + 64, 1); s.dn = nn.Linear(d + 64, 1)

            def forward(s, seq, st):
                z = s.enc(s.inp(seq) + s.pos).mean(dim=1)
                z = torch.cat([z, s.st(st)], dim=1)
                return s.cls(z), s.up(z).squeeze(1), s.dn(z).squeeze(1)
        return Net()

    def _arrays(self, X, fit=False):
        sq = [c for c in X.columns if c.startswith("sq_")]
        stc = [c for c in X.columns if not c.startswith("sq_")]
        if fit:
            self.sq, self.stc = sorted(sq), stc
            self.med = X[stc].astype(float).median()
            Z = X[stc].astype(float).fillna(self.med).fillna(0.0)
            self.mu, self.sd = Z.mean(), Z.std().replace(0, 1.0).fillna(1.0)
        X = X.reindex(columns=self.sq + self.stc)
        Z = ((X[self.stc].astype(float).fillna(self.med).fillna(0.0) - self.mu) / self.sd).clip(-8, 8)
        S = X[self.sq].astype(float).to_numpy()
        seq = np.stack([np.nan_to_num(S / 100.0), np.isnan(S).astype(float)], axis=2)
        return seq.astype(np.float32), Z.to_numpy(np.float32)

    def fit(self, X, y, stress_q=0.99):
        import torch
        torch.manual_seed(self.seed); torch.set_num_threads(N_JOBS)
        yv = y.values.astype(float)
        cls = np.where(yv > self.db, 2, np.where(yv < -self.db, 0, 1))
        seq, st = self._arrays(X, fit=True)
        self.net = self._net(st.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        ce, hub = torch.nn.CrossEntropyLoss(), torch.nn.HuberLoss(reduction="none", delta=1.0)
        T = torch.from_numpy
        seq_t, st_t, c_t, y_t = T(seq), T(st), T(cls.astype(np.int64)), T((yv / 100.0).astype(np.float32))
        n = len(yv)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 1024):
                b = perm[i:i + 1024]
                lo, u, dn = self.net(seq_t[b], st_t[b])
                cb, yb = c_t[b], y_t[b]
                mu, md = (cb == 2).float(), (cb == 0).float()
                loss = ce(lo, cb) + 0.5 * ((hub(u, yb) * mu).sum() / mu.sum().clamp(min=1)
                                           + (hub(dn, yb) * md).sum() / md.sum().clamp(min=1))
                opt.zero_grad(); loss.backward(); opt.step()
        self.flat_mean = float(np.mean(yv[cls == 1])) if (cls == 1).any() else 0.0
        self.stress = {"low": float(np.quantile(yv, 1 - stress_q)), "high": float(np.quantile(yv, stress_q))}
        return self

    def predict(self, X):
        import torch
        seq, st = self._arrays(X)
        self.net.eval()
        with torch.no_grad():
            lo, u, dn = self.net(torch.from_numpy(seq), torch.from_numpy(st))
            pr = torch.softmax(lo, dim=1).numpy()
        mu_up, mu_dn = np.maximum(u.numpy() * 100, 0), np.minimum(dn.numpy() * 100, 0)
        exp = pr[:, 2] * mu_up + pr[:, 0] * mu_dn + pr[:, 1] * self.flat_mean
        return pd.DataFrame({"p_down": pr[:, 0], "p_flat": pr[:, 1], "p_up": pr[:, 2], "mu_up": mu_up,
                             "mu_down": mu_dn, "exp_spread": exp}, index=X.index)


def add_sequences(t, Xt, Xe, L):
    """Last L imbalance spreads already PUBLISHED at each decision time (point-in-time)."""
    s = t.set_index("quarter_utc")["spread"].sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="15min"))
    lag = (t["label_known_at"] - t["quarter_utc"]).iloc[0]

    def seqs(asof):
        k = (asof - lag).dt.floor("15min")
        return pd.DataFrame({f"sq_{i:02d}": s.reindex((k - pd.Timedelta(minutes=15 * (L - 1 - i))).values).values
                             for i in range(L)}, index=asof.index)
    return Xt.join(seqs(t["as_of_trade"])), Xe.join(seqs(t["quarter_utc"] - pd.Timedelta(minutes=240)))


# ----------------------------------------------------------------------------- walk-forward with saved predictions
def walk_cache(area, t, Xt, Xe, c, cols, make, name):
    """Identical to rs_common.walk, but stores per period: validation predictions, tuned params,
    validation PnL with those params, and test predictions."""
    ic = c["intraday"]
    first = (t["quarter_utc"].min() + pd.Timedelta(days=ic["min_train_days"])).normalize()
    last, step = t["quarter_utc"].max(), pd.Timedelta(days=ic["retrain_every_days"])
    periods, results = [], []
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
            return make().fit(X[_usable(X)], y, stress_q=c["risk"]["stress_quantile"])

        vp, vstress, vpnl = None, None, 0.0
        if len(fit_rows) < 3000 or len(val_rows) < 1000:
            params = {"no_trade": True}
        else:
            mv = fit(fit_rows)
            vp = mv.predict(Xt.loc[val_rows, cols])[PRED_COLS]
            vstress = mv.stress
            params = dec.tune(vp, t.loc[val_rows, "spread"], t.loc[val_rows, "quarter_utc"], mv.stress, c)
            vpnl = _raw_pnl(vp, t.loc[val_rows], params, area, c, mv.stress)
        m = fit(all_rows)
        pt = m.predict(Xt.loc[rows, cols])[PRED_COLS]
        r = _decide(area, t, rows, pt, m.stress, params, c)
        results.append(r)
        periods.append({"p0": p0, "rows": np.asarray(rows), "val_rows": np.asarray(val_rows), "vp": vp,
                        "vstress": vstress, "params": params, "val_pnl": vpnl, "pt": pt, "stress": m.stress})
        log(f"  [s4 {name} {area}] {p0.date()} n={len(all_rows)} buy={params.get('buy')} sell={params.get('sell')} "
            f"val_pnl={vpnl:,.0f} trades={int((r['action'] != dec.HOLD).sum())}")
        p0 = p1
    res = P.apply_risk_overlays(pd.concat(results, ignore_index=True), c)
    return res, periods


def _decide(area, t, rows, pt, stress, params, c):
    d = dec.decide(pt, t.loc[rows, "quarter_utc"], stress, c, params)
    d = P.apply_fallback_buy(d, pt, params, area, c)
    r = t.loc[rows, ["quarter_utc", "as_of_trade", "label_known_at", "spread"]].copy()
    r = r.join(pt[["exp_spread", "p_up", "p_down", "p_flat"]])
    r[["action", "mwh", "edge", "reason"]] = d[["action", "mwh", "edge", "reason"]]
    return r


_DATA = {}


def data(area, L=None):
    if area not in _DATA:
        _DATA[area] = load_area(area)
    t, Xt, Xe = _DATA[area]
    if L is None:
        return t, Xt, Xe
    key = (area, L)
    if key not in _DATA:
        _DATA[key] = add_sequences(t, Xt, Xe, L)
    return (t,) + _DATA[key]


def feature_cols(Xt, fs):
    bc = base_cols(Xt)
    return bc if fs == "all" else [x for x in bc if group_of(x) != "live_grid"]


def do_job(kind, area, p, c):
    name = run_name(kind, p)
    f = S4 / f"{name}_{area}.pkl"
    if f.exists():
        log(f"[s4] skip {name} {area} (done)")
        return
    db = c["model"]["direction_deadband_eur"]
    t0 = time.time()
    if kind == "tf":
        t, Xt, Xe = data(area, p["L"])
        cols = feature_cols(Xt, p["fs"]) + [x for x in Xt.columns if x.startswith("sq_")]
        make = lambda: TFModel(db, **{k: v for k, v in p.items() if k != "fs"})
    else:
        t, Xt, Xe = data(area)
        cols = feature_cols(Xt, p["fs"])
        if kind == "lgbm":
            make = lambda: TwoPart("lgbm", c["model"]["lgbm"], db)
        else:
            make = lambda: LogReg(p["C"], p["cw"], db)
    log(f"[s4] start {name} {area} ({len(cols)} inputs)")
    res, periods = walk_cache(area, t, Xt, Xe, c, cols, make, name)
    sc = score(res, c)
    save_row("stage4", area, name, sc, f"{len(cols)} inputs; {time.time() - t0:,.0f}s")
    tmp = f.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"kind": kind, "area": area, "params": p, "name": name, "score": sc, "periods": periods}, fh)
    os.replace(tmp, f)


# ----------------------------------------------------------------------------- offline analysis
def load_runs(area):
    runs = {}
    for f in sorted(S4.glob(f"*_{area}.pkl")):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        runs[r["name"]] = r
    return runs


def _avg(frames, w=None):
    w = np.ones(len(frames)) / len(frames) if w is None else np.asarray(w, float) / np.sum(w)
    out = sum(wi * f[PRED_COLS].to_numpy(float) for wi, f in zip(w, frames))
    return pd.DataFrame(out, columns=PRED_COLS, index=frames[0].index)


def replay_blend(area, t, c, members, w=None):
    """Average the members' probabilities per period, re-tune thresholds on the averaged validation
    predictions (exactly as a single model would), decide, overlay, score."""
    n = len(members[0]["periods"])
    results = []
    for i in range(n):
        ps = [m["periods"][i] for m in members]
        p0 = ps[0]
        rows = p0["rows"]
        pt = _avg([p["pt"] for p in ps], w)
        if p0["vp"] is None or any(p["vp"] is None for p in ps):
            params = {"no_trade": True}
        else:
            vp = _avg([p["vp"] for p in ps], w)
            params = dec.tune(vp, t.loc[p0["val_rows"], "spread"], t.loc[p0["val_rows"], "quarter_utc"],
                              p0["vstress"], c)
        results.append(_decide(area, t, rows, pt, p0["stress"], params, c))
    return score(P.apply_risk_overlays(pd.concat(results, ignore_index=True), c), c)


def replay_select(area, t, c, family):
    """Honest walk-forward grid search: in each period use the configuration with the best
    VALIDATION profit (known before trading). Returns score and how often each config was chosen."""
    n = len(family[0]["periods"])
    results, chosen = [], []
    for i in range(n):
        best = max(family, key=lambda m: m["periods"][i]["val_pnl"])
        p = best["periods"][i]
        chosen.append(best["name"])
        results.append(_decide(area, t, p["rows"], p["pt"], p["stress"], p["params"], c))
    sc = score(P.apply_risk_overlays(pd.concat(results, ignore_index=True), c), c)
    return sc, pd.Series(chosen).value_counts().to_dict()


def replay_single(area, t, c, m):
    res = [_decide(area, t, p["rows"], p["pt"], p["stress"], p["params"], c) for p in m["periods"]]
    return score(P.apply_risk_overlays(pd.concat(res, ignore_index=True), c), c)


def analyse():
    c = cfg()
    rows = []
    for area in ("DK1", "DK2"):
        runs = load_runs(area)
        if not runs:
            continue
        t = data(area)[0]
        for nm, r in runs.items():
            rows.append({"area": area, "type": "single", "variant": nm, **r["score"]})
        chk = next(iter(runs.values()))
        rep = replay_single(area, t, c, chk)
        log(f"[s4 analyse] replay check {area} {chk['name']}: run={chk['score']['net_eur']:,.0f} "
            f"replay={rep['net_eur']:,.0f}")
        R = lambda k: runs.get(k)
        tf_seeds = [R(f"tf_all_L32_d32_l2_dr0.1_ep6_s{s}") for s in (0, 1, 2)]
        tf_seeds = [x for x in tf_seeds if x]
        tf_all = [r for k, r in runs.items() if k.startswith("tf_")]
        logit = {k: r for k, r in runs.items() if k.startswith("logit_")}
        lg_def = R("logit_all_C0.1_nobal")
        lgbm = R("lgbm_all")
        blends = {}
        if len(tf_seeds) >= 2:
            blends["tf_seed_ensemble"] = tf_seeds
        if lgbm and lg_def:
            blends["lgbm+logit"] = [lgbm, lg_def]
        if tf_seeds and lg_def:
            blends["tf_ens+logit"] = tf_seeds + [lg_def] * len(tf_seeds)
        if tf_seeds and lgbm:
            blends["tf_ens+lgbm"] = tf_seeds + [lgbm] * len(tf_seeds)
        if tf_seeds and lgbm and lg_def:
            blends["tf_ens+lgbm+logit"] = tf_seeds + [lgbm] * len(tf_seeds) + [lg_def] * len(tf_seeds)
        if R("lgbm_nolg") and R("logit_nolg_C0.1_nobal"):
            blends["lgbm_nolg+logit_nolg"] = [R("lgbm_nolg"), R("logit_nolg_C0.1_nobal")]
        for nm, mem in blends.items():
            try:
                rows.append({"area": area, "type": "blend", "variant": nm, **replay_blend(area, t, c, mem)})
            except Exception as e:
                log(f"[s4 analyse] blend {nm} {area} failed: {type(e).__name__}: {e}")
        fams = {"logit_grid_select": list(logit.values()), "tf_grid_select": tf_all}
        if lgbm:
            fams["lgbm|logit|tf_select"] = [lgbm] + list(logit.values()) + tf_all
        for nm, fam in fams.items():
            if len(fam) >= 2:
                sc, ch = replay_select(area, t, c, fam)
                rows.append({"area": area, "type": "honest_grid", "variant": nm, **sc,
                             "info": "; ".join(f"{k} x{v}" for k, v in ch.items())})
    out = pd.DataFrame(rows)
    out.to_csv(S4 / "stage4_results.csv", index=False)
    log(f"[s4 analyse] wrote {len(out)} rows to stage4_results.csv")


# ----------------------------------------------------------------------------- main
def run_lists(names):
    c = cfg()
    for nm in names:
        if nm == "analyse":
            analyse(); continue
        for kind, area, p in jobs(nm):
            try:
                do_job(kind, area, p, c)
            except Exception as e:
                log(f"[s4] FAILED {kind} {area} {p}: {type(e).__name__}: {e}")
        try:
            analyse()
        except Exception as e:
            log(f"[s4 analyse] FAILED {type(e).__name__}: {e}")


def main():
    args = sys.argv[1:]
    if args == ["all"]:
        cpus = os.cpu_count() or 1
        log(f"==== stage 4 start, {cpus} CPUs ====")
        if cpus >= 8:
            child = subprocess.Popen([sys.executable, __file__, "tf_DK1"],
                                     stdout=open(OUT / "stage4_tf_DK1.log", "a"), stderr=subprocess.STDOUT)
            run_lists(["lgbm", "logit_DK1", "logit_DK2", "tf_DK2"])
            child.wait()
        else:
            run_lists(["lgbm", "logit_DK1", "logit_DK2", "tf_DK1", "tf_DK2"])
        analyse()
        log("==== stage 4 done ====")
    else:
        run_lists(args)


if __name__ == "__main__":
    main()
