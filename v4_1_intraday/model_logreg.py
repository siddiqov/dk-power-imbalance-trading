"""V4.1 logistic-regression spread model (2026-10-04, recommended model for DK1).

Same output contract as nurex42.models.SpreadModel, so decisions, guards, journal and dashboard
work unchanged:
  p_down / p_flat / p_up   multinomial logistic regression (lbfgs) on standardised features
  mu_up / mu_down          ridge regression on the up-quarters / down-quarters
  exp_spread               P(UP)*mu_up + P(DOWN)*mu_down + P(FLAT)*mean(flat spread)
  q10 / q50 / q90          LightGBM quantile regressors (kept for the live BUY crash guard)

Inputs: every usable feature (the same columns LightGBM gets). Missing values are filled with
the training median, features are standardised with the training mean / std and clipped at +-8.

Research basis (Stage 4, walk-forward Jul 2025 - Sep 2026, settings chosen per 30-day period on
the 90-day validation window only): DK1 all features EUR 69.1k vs LightGBM 25.2k.
See project doc claude/nurex_v4_1_research_results.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DOWN, FLAT, UP = 0, 1, 2


def _labels(y: np.ndarray, deadband: float) -> np.ndarray:
    return np.where(y > deadband, UP, np.where(y < -deadband, DOWN, FLAT))


@dataclass
class SpreadModelLogReg:
    C: float = 0.1
    class_weight: str | None = None          # None | "balanced"
    deadband: float = 0.5
    lgbm_params: dict = field(default_factory=dict)   # only for the quantile regressors
    quantiles: tuple = (0.1, 0.5, 0.9)
    with_quantiles: bool = True
    ridge_alpha: float = 10.0
    max_iter: int = 1000
    features: list = field(default_factory=list)
    clf: object = None
    reg: dict = field(default_factory=dict)
    qreg: dict = field(default_factory=dict)
    flat_mean: float = 0.0
    stress: dict = field(default_factory=dict)
    med: object = None
    mu: object = None
    sd: object = None

    family = "logreg"

    # ------------------------------------------------------------------ preprocessing
    def _prep(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        Xv = X.astype(float)
        if fit:
            self.med = Xv.median()
            Xi = Xv.fillna(self.med).fillna(0.0)
            self.mu = Xi.mean()
            self.sd = Xi.std().replace(0, 1.0).fillna(1.0)
        else:
            Xi = Xv.fillna(self.med).fillna(0.0)
        return ((Xi - self.mu) / self.sd).clip(-8, 8)

    # ------------------------------------------------------------------ fit / predict
    def fit(self, X: pd.DataFrame, y: pd.Series, stress_q: float = 0.99) -> "SpreadModelLogReg":
        from sklearn.linear_model import LogisticRegression, Ridge
        self.features = list(X.columns)
        yv = y.values.astype(float)
        cls = _labels(yv, self.deadband)
        Xp = self._prep(X, fit=True)
        self.clf = LogisticRegression(C=self.C, class_weight=self.class_weight, max_iter=self.max_iter)
        self.clf.fit(Xp, cls)
        self.reg = {}
        for c in (UP, DOWN):
            m = cls == c
            if m.sum() >= 200:
                r = Ridge(alpha=self.ridge_alpha)
                r.fit(Xp[m], yv[m])
                self.reg[c] = r
            else:
                self.reg[c] = float(np.mean(yv[m])) if m.any() else 0.0
        self.flat_mean = float(np.mean(yv[cls == FLAT])) if (cls == FLAT).any() else 0.0
        self.qreg = {}
        if self.with_quantiles:
            import lightgbm as lgb
            for a in self.quantiles:
                r = lgb.LGBMRegressor(objective="quantile", alpha=a, **self.lgbm_params)
                r.fit(X, yv)
                self.qreg[a] = r
        self.stress = {"low": float(np.quantile(yv, 1 - stress_q)), "high": float(np.quantile(yv, stress_q))}
        return self

    def _reg_pred(self, c, Xp):
        r = self.reg[c]
        return r.predict(Xp) if hasattr(r, "predict") else np.full(len(Xp), r)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reindex(columns=self.features)
        Xp = self._prep(X)
        pr = self.clf.predict_proba(Xp)
        probs = np.zeros((len(X), 3))
        for i, c in enumerate(self.clf.classes_):
            probs[:, int(c)] = pr[:, i]
        mu_up = np.maximum(self._reg_pred(UP, Xp), 0.0)
        mu_dn = np.minimum(self._reg_pred(DOWN, Xp), 0.0)
        exp = probs[:, UP] * mu_up + probs[:, DOWN] * mu_dn + probs[:, FLAT] * self.flat_mean
        out = pd.DataFrame({"p_down": probs[:, DOWN], "p_flat": probs[:, FLAT], "p_up": probs[:, UP],
                            "mu_up": mu_up, "mu_down": mu_dn, "exp_spread": exp}, index=X.index)
        if self.qreg:
            qs = np.column_stack([self.qreg[a].predict(X) for a in self.quantiles])
            qs.sort(axis=1)
            for i, a in enumerate(self.quantiles):
                out[f"q{int(round(a * 100))}"] = qs[:, i]
        else:
            for a in self.quantiles:
                out[f"q{int(round(a * 100))}"] = np.nan
        return out

    # ------------------------------------------------------------------ explanation
    def coefficients(self) -> pd.DataFrame:
        """Standardised coefficients per class (features x down/flat/up)."""
        names = {DOWN: "down", FLAT: "flat", UP: "up"}
        return pd.DataFrame(self.clf.coef_.T, index=self.features,
                            columns=[names[int(c)] for c in self.clf.classes_])

    def feature_importance(self) -> pd.Series:
        """Share (%) of the summed absolute standardised coefficients, largest first."""
        imp = self.coefficients().abs().sum(axis=1)
        return (100 * imp / max(imp.sum(), 1e-9)).sort_values(ascending=False)
