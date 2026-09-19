"""Two-part spread model.

spread = imbalance price - day-ahead price (EUR/MWh) for one quarter.

Part 1  P(DOWN), P(FLAT), P(UP)          LightGBM multiclass
Part 2  E[spread | UP], E[spread | DOWN]  LightGBM regressors trained on each regime
Part 3  q10 / q50 / q90                    LightGBM quantile regressors (unconditional)

Expected spread = P(UP)*E[s|UP] + P(DOWN)*E[s|DOWN] + P(FLAT)*E[s|FLAT].
The mixture follows how the Danish single imbalance price is formed: the
dominating regulation direction decides whether the price moves up or down
from the day-ahead price, and the size of the move depends on the activated
balancing bids.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DOWN, FLAT, UP = 0, 1, 2


def direction_labels(y: np.ndarray, deadband: float) -> np.ndarray:
    return np.where(y > deadband, UP, np.where(y < -deadband, DOWN, FLAT))


@dataclass
class SpreadModel:
    params: dict
    deadband: float = 0.5
    quantiles: tuple = (0.1, 0.5, 0.9)
    features: list = field(default_factory=list)
    clf: object = None
    reg: dict = field(default_factory=dict)
    qreg: dict = field(default_factory=dict)
    flat_mean: float = 0.0
    stress: dict = field(default_factory=dict)
    class_present: list = field(default_factory=list)

    def fit(self, X: pd.DataFrame, y: pd.Series, stress_q: float = 0.99) -> "SpreadModel":
        self.features = list(X.columns)
        yv = y.values.astype(float)
        cls = direction_labels(yv, self.deadband)
        self.class_present = sorted(set(cls.tolist()))
        p = dict(self.params)
        if len(self.class_present) >= 2:
            # map present classes to 0..k-1 for LightGBM
            self._cmap = {c: i for i, c in enumerate(self.class_present)}
            self.clf = lgb.LGBMClassifier(objective="multiclass" if len(self.class_present) > 2 else "binary", **p)
            self.clf.fit(X, np.vectorize(self._cmap.get)(cls))
        for c in (UP, DOWN):
            m = cls == c
            if m.sum() >= 200:
                r = lgb.LGBMRegressor(objective="regression", **p)
                r.fit(X[m], yv[m])
                self.reg[c] = r
            else:
                self.reg[c] = float(np.mean(yv[m])) if m.any() else 0.0
        self.flat_mean = float(np.mean(yv[cls == FLAT])) if (cls == FLAT).any() else 0.0
        for a in self.quantiles:
            r = lgb.LGBMRegressor(objective="quantile", alpha=a, **p)
            r.fit(X, yv)
            self.qreg[a] = r
        # empirical stress levels used for position sizing
        self.stress = {"low": float(np.quantile(yv, 1 - stress_q)), "high": float(np.quantile(yv, stress_q))}
        return self

    def _reg_pred(self, c, X):
        r = self.reg[c]
        return r.predict(X) if hasattr(r, "predict") else np.full(len(X), r)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reindex(columns=self.features)
        n = len(X)
        probs = np.zeros((n, 3))
        if self.clf is not None:
            pr = self.clf.predict_proba(X)
            for c, i in self._cmap.items():
                probs[:, c] = pr[:, i]
        else:
            probs[:, self.class_present[0]] = 1.0
        mu_up = self._reg_pred(UP, X)
        mu_dn = self._reg_pred(DOWN, X)
        # regime means must have the regime's sign
        mu_up = np.maximum(mu_up, 0.0)
        mu_dn = np.minimum(mu_dn, 0.0)
        exp = probs[:, UP] * mu_up + probs[:, DOWN] * mu_dn + probs[:, FLAT] * self.flat_mean
        out = pd.DataFrame({"p_down": probs[:, DOWN], "p_flat": probs[:, FLAT], "p_up": probs[:, UP],
                            "mu_up": mu_up, "mu_down": mu_dn, "exp_spread": exp}, index=X.index)
        qs = np.column_stack([self.qreg[a].predict(X) for a in self.quantiles])
        qs.sort(axis=1)  # no quantile crossing
        for i, a in enumerate(self.quantiles):
            out[f"q{int(round(a * 100))}"] = qs[:, i]
        return out

    def feature_importance(self) -> pd.Series:
        imp = pd.Series(0.0, index=self.features)
        models = [self.clf] + [r for r in self.reg.values() if hasattr(r, "booster_")]
        for m in models:
            if m is not None:
                g = pd.Series(m.booster_.feature_importance("gain"), index=self.features)
                imp += g / max(g.sum(), 1e-9)
        return (100 * imp / max(imp.sum(), 1e-9)).sort_values(ascending=False)


@dataclass
class ModelBundle:
    area: str
    model: SpreadModel
    decision_params: dict
    trained_until: pd.Timestamp
    n_train: int
    validation: dict

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "ModelBundle":
        with open(path, "rb") as f:
            return pickle.load(f)
