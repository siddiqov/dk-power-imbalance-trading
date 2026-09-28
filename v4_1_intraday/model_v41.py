"""V4.1 spread model (2026-09-28 upgrade). Extends the V4.2 two-part SpreadModel with:

  * robust size regressors  - Huber loss (config model.size_objective / huber_delta) instead of
                              squared error, so rare +/-1000 EUR spikes do not dominate the fit
  * recency weights         - every training row weighted 0.5 ** (age_days / recency_half_life_days)
                              (the Nordic mFRR market and the flat-price share changed during 2025-26)
  * spike classifiers       - P(spread > +spike_level) and P(spread < -spike_level), used by the
                              decision guards: no SELL into a likely up-spike, no BUY into a
                              likely down-spike
The flat class (|spread| <= deadband, imbalance price == spot) already exists in the base model
(p_flat); the flat guard in pipeline_id uses it.

Old pickles (plain SpreadModel) keep working: they simply have no p_spike_* columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from nurex42.models import DOWN, FLAT, UP, SpreadModel, direction_labels


def recency_weights(times, half_life_days: float | None) -> np.ndarray | None:
    if not half_life_days:
        return None
    t = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
    age = (t.max() - t).dt.total_seconds().to_numpy() / 86400.0
    w = 0.5 ** (age / float(half_life_days))
    return w / w.mean()


@dataclass
class SpreadModelV41(SpreadModel):
    size_objective: str = "huber"
    huber_delta: float = 40.0
    spike_level: float = 150.0
    clf_spike: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series, stress_q: float = 0.99,
            sample_weight: np.ndarray | None = None) -> "SpreadModelV41":
        self.features = list(X.columns)
        yv = y.values.astype(float)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        cls = direction_labels(yv, self.deadband)
        self.class_present = sorted(set(cls.tolist()))
        p = dict(self.params)
        if len(self.class_present) >= 2:
            self._cmap = {c: i for i, c in enumerate(self.class_present)}
            self.clf = lgb.LGBMClassifier(objective="multiclass" if len(self.class_present) > 2 else "binary", **p)
            self.clf.fit(X, np.vectorize(self._cmap.get)(cls), sample_weight=w)
        reg_kw = {"objective": self.size_objective}
        if self.size_objective == "huber":
            reg_kw["alpha"] = float(self.huber_delta)          # LightGBM: Huber delta is `alpha`
        for c in (UP, DOWN):
            m = cls == c
            if m.sum() >= 200:
                r = lgb.LGBMRegressor(**reg_kw, **p)
                r.fit(X[m], yv[m], sample_weight=None if w is None else w[m])
                self.reg[c] = r
            else:
                self.reg[c] = float(np.mean(yv[m])) if m.any() else 0.0
        self.flat_mean = float(np.mean(yv[cls == FLAT])) if (cls == FLAT).any() else 0.0
        for a in self.quantiles:
            r = lgb.LGBMRegressor(objective="quantile", alpha=a, **p)
            r.fit(X, yv, sample_weight=w)
            self.qreg[a] = r
        # spike classifiers (skipped when there are too few spikes to learn from)
        self.clf_spike = {}
        for side, lab in (("up", yv > self.spike_level), ("down", yv < -self.spike_level)):
            if lab.sum() >= 100 and (~lab).sum() >= 100:
                c = lgb.LGBMClassifier(objective="binary", **p)
                c.fit(X, lab.astype(int), sample_weight=w)
                self.clf_spike[side] = c
            else:
                self.clf_spike[side] = float(lab.mean())
        self.stress = {"low": float(np.quantile(yv, 1 - stress_q)), "high": float(np.quantile(yv, stress_q))}
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = super().predict(X)
        Xf = X.reindex(columns=self.features)
        for side in ("up", "down"):
            c = self.clf_spike.get(side)
            out[f"p_spike_{side}"] = c.predict_proba(Xf)[:, 1] if hasattr(c, "predict_proba") \
                else np.full(len(Xf), float(c or 0.0))
        return out

    def feature_importance(self) -> pd.Series:
        return super().feature_importance()
