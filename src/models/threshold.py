"""Threshold Autoregressive (TAR) model for regime switching.

The TAR model defines regimes by a threshold τ
applied to a *threshold variable* z_t (commonly y_{t-d} for delay d):

    y_t = (mu_0 + phi_0 * y_{t-1} + X_t' beta_0) * 1[z_t <= τ]
          + (mu_1 + phi_1 * y_{t-1} + X_t' beta_1) * 1[z_t >  τ]  + eps_t

The threshold τ is selected by grid search over candidate quantiles of z,
minimising the total in-sample sum of squared residuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

_THRESHOLD_VARIABLE = "y_lag1"       # default threshold variable
_REG_FEATURE_COLS = ["y_lag1", "roll_mean_5", "roll_std_5", "exog_0", "exog_1"]
_QUANTILE_GRID = np.linspace(0.15, 0.85, 50)  # candidate threshold quantiles


class ThresholdModel:
    """Two-regime Threshold Autoregressive model with grid-search tau selection.

    Parameters
    ----------
    threshold_variable : str
        Column name in the DataFrame used to determine the regime (z_t).
    quantile_grid : array-like
        Quantile values (in [0, 1]) to search over when selecting tau.
    alpha : float
        Ridge regularisation for per-regime regressors.
    """

    def __init__(
        self,
        threshold_variable: str = _THRESHOLD_VARIABLE,
        quantile_grid: np.ndarray = _QUANTILE_GRID,
        alpha: float = 0.5,
    ) -> None:
        self.threshold_variable = threshold_variable
        self.quantile_grid = np.asarray(quantile_grid)
        self.alpha = alpha

        self.tau_: float | None = None
        self._regressors: dict[int, Ridge] = {}
        self._feature_cols: list[str] = []

    # Internal helpers

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in _REG_FEATURE_COLS if c in df.columns]
        self._feature_cols = cols
        return df[cols].to_numpy(dtype=float)

    def _regime_mask(self, z: np.ndarray) -> np.ndarray:
        """Return 0 for z <= tau, 1 for z > tau."""
        return (z > self.tau_).astype(int)

    # Threshold search

    def _find_threshold(self, z: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Grid-search tau by minimising total SSR over candidate quantiles."""
        best_ssr = np.inf
        best_tau = np.quantile(z, 0.5)

        for q in self.quantile_grid:
            tau_cand = np.quantile(z, q)
            mask0 = z <= tau_cand
            mask1 = z > tau_cand

            # Require minimum observations in each regime
            if mask0.sum() < 10 or mask1.sum() < 10:
                continue

            ssr = 0.0
            for mask in [mask0, mask1]:
                X_sub, y_sub = X[mask], y[mask]
                model = Ridge(alpha=self.alpha)
                model.fit(X_sub, y_sub)
                resid = y_sub - model.predict(X_sub)
                ssr += float(resid @ resid)

            if ssr < best_ssr:
                best_ssr = ssr
                best_tau = tau_cand

        return best_tau

    # Fit
  
    def fit(self, df: pd.DataFrame) -> "ThresholdModel":
        """Estimate the threshold tau and per-regime regressions.

        Parameters
        ----------
        df : pd.DataFrame
            Training data; must contain the threshold variable and feature columns.

        Returns
        -------
        self
        """
        z = df[self.threshold_variable].to_numpy(dtype=float)
        X = self._build_features(df)
        y = df["y"].to_numpy(dtype=float)

        self.tau_ = self._find_threshold(z, X, y)

        # Fit per-regime regressors at the chosen threshold
        regimes = self._regime_mask(z)
        for reg in [0, 1]:
            mask = regimes == reg
            self._regressors[reg] = Ridge(alpha=self.alpha)
            self._regressors[reg].fit(X[mask], y[mask])

        return self

    # Predict

    def predict_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """Return 0 (below threshold) or 1 (above threshold) for each row."""
        if self.tau_ is None:
            raise RuntimeError("Call fit() before predict_regimes().")
        z = df[self.threshold_variable].to_numpy(dtype=float)
        return self._regime_mask(z)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict ``y`` using the regime determined by the threshold.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        np.ndarray, shape (n,)
        """
        if self.tau_ is None:
            raise RuntimeError("Call fit() before predict().")
        cols = [c for c in _REG_FEATURE_COLS if c in df.columns]
        X = df[cols].to_numpy(dtype=float)
        regimes = self.predict_regimes(df)
        preds = np.empty(len(df))

        for reg in [0, 1]:
            mask = regimes == reg
            if mask.any():
                preds[mask] = self._regressors[reg].predict(X[mask])

        return preds

    @property
    def threshold(self) -> float | None:
        """The estimated threshold value τ."""
        return self.tau_
