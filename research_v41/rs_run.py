"""Step 2: experiments on the cached dataset. Usage: python rs_run.py <suite> [<suite> ...] [--areas DK1,DK2]
Suites: baseline | drift | ablation | models | seq | combo
Every variant is scored by the same walk-forward + decision rules as the official replay."""
import sys
import numpy as np
import pandas as pd
from rs_common import OUT, cfg, group_of, load_area, log, save_row, score, walk, DRIFT_PREFIX, N_JOBS

SEQ_LEN = 32


def base_cols(X):
    return [c for c in X.columns if not c.startswith((DRIFT_PREFIX, "sq_", "id15_", "id60_", "book15_", "trd15_"))]


def groups(cols):
    g = {}
    for c in cols:
        g.setdefault(group_of(c), []).append(c)
    return g


# ----------------------------------------------------------------------------- sequences
def add_sequences(t, Xt, Xe):
    """Last SEQ_LEN imbalance spreads that were already published at each decision time."""
    s = t.set_index("quarter_utc")["spread"].sort_index()
    grid = pd.date_range(s.index.min(), s.index.max(), freq="15min")
    s = s.reindex(grid)
    lag = (t["label_known_at"] - t["quarter_utc"]).iloc[0]          # quarter + 15 min + publication lag
    def seqs(asof):
        k = (asof - lag).dt.floor("15min")                          # newest quarter with a known price
        cols = {}
        for i in range(SEQ_LEN):
            cols[f"sq_{i:02d}"] = s.reindex((k - pd.Timedelta(minutes=15 * (SEQ_LEN - 1 - i))).values).values
        return pd.DataFrame(cols, index=asof.index)
    Xt = Xt.join(seqs(t["as_of_trade"]))
    Xe = Xe.join(seqs(t["quarter_utc"] - pd.Timedelta(minutes=240)))
    return Xt, Xe


class SeqModel:
    """Neural model: LSTM or Transformer over the last 32 known spreads + MLP over the static
    features; heads = P(down/flat/up), E[s|UP], E[s|DOWN] (same output contract as SpreadModel)."""

    def __init__(self, arch, deadband, epochs=6):
        self.arch, self.deadband, self.epochs = arch, deadband, epochs

    def _net(self, n_static):
        import torch.nn as nn
        import torch
        arch = self.arch
        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.inp = nn.Linear(2, 32)
                if arch == "lstm":
                    s.enc = nn.LSTM(32, 32, batch_first=True)
                else:
                    layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.1,
                                                       batch_first=True)
                    s.enc = nn.TransformerEncoder(layer, num_layers=2)
                    s.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, 32))
                s.st = nn.Sequential(nn.Linear(n_static, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
                s.cls = nn.Linear(96, 3); s.up = nn.Linear(96, 1); s.dn = nn.Linear(96, 1)
            def forward(s, seq, st):
                h = s.inp(seq)
                if arch == "lstm":
                    _, (hn, _) = s.enc(h); z = hn[-1]
                else:
                    z = s.enc(h + s.pos).mean(dim=1)
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
        torch.manual_seed(0); torch.set_num_threads(N_JOBS)
        yv = y.values.astype(float)
        cls = np.where(yv > self.deadband, 2, np.where(yv < -self.deadband, 0, 1))
        seq, st = self._arrays(X, fit=True)
        self.net = self._net(st.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)
        ce, hub = torch.nn.CrossEntropyLoss(), torch.nn.HuberLoss(reduction="none", delta=1.0)
        T = lambda a: torch.from_numpy(a)
        seq_t, st_t, c_t, y_t = T(seq), T(st), T(cls.astype(np.int64)), T((yv / 100.0).astype(np.float32))
        n = len(yv)
        self.net.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 1024):
                b = perm[i:i + 1024]
                lo, u, d = self.net(seq_t[b], st_t[b])
                cb, yb = c_t[b], y_t[b]
                loss = ce(lo, cb)
                mu, md = (cb == 2).float(), (cb == 0).float()
                loss = loss + 0.5 * ((hub(u, yb) * mu).sum() / mu.sum().clamp(min=1)
                                     + (hub(d, yb) * md).sum() / md.sum().clamp(min=1))
                opt.zero_grad(); loss.backward(); opt.step()
        self.flat_mean = float(np.mean(yv[cls == 1])) if (cls == 1).any() else 0.0
        self.stress = {"low": float(np.quantile(yv, 1 - stress_q)), "high": float(np.quantile(yv, stress_q))}
        return self

    def predict(self, X):
        import torch
        seq, st = self._arrays(X)
        self.net.eval()
        with torch.no_grad():
            lo, u, d = self.net(torch.from_numpy(seq), torch.from_numpy(st))
            pr = torch.softmax(lo, dim=1).numpy()
        mu_up, mu_dn = np.maximum(u.numpy() * 100, 0), np.minimum(d.numpy() * 100, 0)
        exp = pr[:, 2] * mu_up + pr[:, 0] * mu_dn + pr[:, 1] * self.flat_mean
        return pd.DataFrame({"p_down": pr[:, 0], "p_flat": pr[:, 1], "p_up": pr[:, 2], "mu_up": mu_up,
                             "mu_down": mu_dn, "exp_spread": exp}, index=X.index)


# ----------------------------------------------------------------------------- suites
def run(suite, area, c, t, Xt, Xe, **kw):
    bc = base_cols(Xt)
    G = groups(bc)
    if suite == "baseline":
        res, ex = walk(area, t, Xt, Xe, c, cols=bc, perm_groups=G, tag="baseline ")
        save_row(suite, area, "production_features_lgbm", score(res, c), f"{len(bc)} features")
        res.to_parquet(OUT / f"oos_baseline_{area}.parquet")
        if "gain" in ex:
            gg = ex["gain"]
            gg.to_frame("gain").assign(group=[group_of(x) for x in gg.index]).to_csv(OUT / f"gain_{area}.csv")
        if "perm" in ex:
            pm = ex["perm"]
            pm.to_csv(OUT / f"perm_by_fold_{area}.csv", index=False)
            rows = []
            for g in G:
                if f"ll_{g}" in pm:
                    rows.append({"group": g, "n_features": len(G[g]), "logloss_increase": pm[f"ll_{g}"].mean(),
                                 "pnl_loss_eur": pm[f"pnl_{g}"].sum(),
                                 "folds_pnl_helps": int((pm[f"pnl_{g}"] > 0).sum()), "folds": len(pm)})
            pd.DataFrame(rows).sort_values("pnl_loss_eur", ascending=False).to_csv(OUT / f"perm_{area}.csv", index=False)
    elif suite == "drift":
        own = [x for x in Xt.columns if x.startswith(DRIFT_PREFIX) and not x.startswith("fd_oz_")]
        allfd = [x for x in Xt.columns if x.startswith(DRIFT_PREFIX)]
        for name, cols in (("plus_drift_own_zone", bc + own), ("plus_drift_both_zones", bc + allfd)):
            res, _ = walk(area, t, Xt, Xe, c, cols=cols, tag=f"{name} ")
            save_row(suite, area, name, score(res, c), f"{len(cols)} features")
    elif suite == "ablation":
        for g, gc in sorted(G.items()):
            if kw.get("groups") and g not in kw["groups"]:
                continue
            cols = [x for x in bc if x not in set(gc)]
            res, _ = walk(area, t, Xt, Xe, c, cols=cols, tag=f"drop {g} ")
            save_row(suite, area, f"drop_{g}", score(res, c), f"-{len(gc)} features")
    elif suite == "models":
        for kind in kw.get("kinds") or ("xgboost", "catboost", "histgb", "extratrees", "logistic", "mlp"):
            try:
                res, _ = walk(area, t, Xt, Xe, c, cols=bc, kind=kind, tag=f"{kind} ")
                save_row(suite, area, kind, score(res, c), f"{len(bc)} features")
            except Exception as e:
                log(f"  [{kind} {area}] FAILED {type(e).__name__}: {e}")
    elif suite == "seq":
        Xt2, Xe2 = add_sequences(t, Xt, Xe)
        sq = [x for x in Xt2.columns if x.startswith("sq_")]
        db = c["model"]["direction_deadband_eur"]
        for arch in ("lstm", "transformer"):
            try:
                res, _ = walk(area, t, Xt2, Xe2, c, cols=bc + sq, seq=lambda a=arch: SeqModel(a, db), tag=f"{arch} ")
                save_row(suite, area, arch, score(res, c), f"{len(bc)} static + {len(sq)} sequence")
            except Exception as e:
                log(f"  [{arch} {area}] FAILED {type(e).__name__}: {e}")
    elif suite == "combo":
        drop = kw.get("drop", [])
        add = [x for x in Xt.columns if x.startswith(tuple(kw.get("add", [])))] if kw.get("add") else []
        cols = [x for x in bc if group_of(x) not in set(drop)] + add
        kind = kw.get("kind", "lgbm")
        res, _ = walk(area, t, Xt, Xe, c, cols=cols, kind=kind, tag="combo ")
        save_row(suite, area, kw.get("name", "combo"), score(res, c), f"{len(cols)} features, drop={drop}, add={kw.get('add')}")
        res.to_parquet(OUT / f"oos_{kw.get('name', 'combo')}_{area}.parquet")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    areas = opts.get("areas", "DK1,DK2").split(",")
    kw = {}
    if "kinds" in opts: kw["kinds"] = opts["kinds"].split(",")
    if "drop" in opts: kw["drop"] = [x for x in opts["drop"].split(",") if x]
    if "add" in opts: kw["add"] = [x for x in opts["add"].split(",") if x]
    if "kind" in opts: kw["kind"] = opts["kind"]
    if "groups" in opts: kw["groups"] = opts["groups"].split(",")
    if "name" in opts: kw["name"] = opts["name"]
    c = cfg()
    for area in areas:
        t, Xt, Xe = load_area(area)
        for suite in args:
            log(f"=== suite {suite} {area} start")
            run(suite, area, c, t, Xt, Xe, **kw)
            log(f"=== suite {suite} {area} done")


if __name__ == "__main__":
    main()
