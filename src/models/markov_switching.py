"""Classical Markov-switching regression model (Hamilton 1989).

Wraps :class:`statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`
into the project's unified fit/predict interface.

The model is:

    y_t = mu_{s_t} + phi_{s_t} * x_t + eps_t,   eps_t ~ N(0, sigma_{s_t}^2)

where s_t follows a K-state first-order Markov chain estimated jointly with
the regression parameters via the Hamilton filter (EM / maximum likelihood).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    _STATSMODELS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _STATSMODELS_AVAILABLE = False

_FEATURE_COLS = ["y_lag1", "roll_mean_5", "roll_std_5", "exog_0", "exog_1"]


class MarkovSwitchingModel:
    """Markov-switching regression fitted via Hamilton's EM algorithm.

    Parameters
    ----------
    k_regimes : int
        Number of latent regimes (default: 2).
    switching_variance : bool
        If True, the innovation variance is regime-dependent.
    switching_ar : bool
        If True, the AR coefficient is also regime-dependent (not just intercept).
    max_iter : int
        Maximum EM iterations passed to statsmodels.
    """

    def __init__(
        self,
        k_regimes: int = 2,
        switching_variance: bool = True,
        switching_ar: bool = False,
        max_iter: int = 200,
    ) -> None:
        if not _STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for MarkovSwitchingModel.")
        self.k_regimes = k_regimes
        self.switching_variance = switching_variance
        self.switching_ar = switching_ar
        self.max_iter = max_iter

        self._result = None
        self._y_train: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "MarkovSwitchingModel":
        """Fit the Markov-switching model on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame; must contain ``y`` and ``y_lag1`` columns.

        Returns
        -------
        self
        """
        y = df["y"].to_numpy(dtype=float)
        # statsmodels MS regression uses endog = y and exog = [1, x]
        # The library handles the intercept switching internally.
        exog = df[["y_lag1"]].assign(const=1.0)[["const", "y_lag1"]].to_numpy(dtype=float)

        model = MarkovRegression(
            endog=y,
            k_regimes=self.k_regimes,
            trend="n",           # no extra trend — we supply const ourselves
            exog=exog,
            switching_variance=self.switching_variance,
            switching_exog=self.switching_ar,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = model.fit(
                search_reps=5,
                maxiter=self.max_iter,
                disp=False,
            )
        self._y_train = y
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """One-step-ahead predictions on new data.

        For out-of-sample data we use the smoothed regime probabilities from
        the training fit and compute regime-weighted conditional means.
        Because the Hamilton filter is inherently in-sample, this approximation
        assumes regime probabilities from the end of training carry forward.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least ``y_lag1`` column.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        if self._result is None:
            raise RuntimeError("Call fit() before predict().")

        # Obtain regime-specific parameters from the fitted result
        params = self._result.params
        k = self.k_regimes

        # Reconstruct mu and phi per regime from params vector
        # statsmodels stores [const_0, ..., const_{k-1}, ar_0 (or shared ar)]
        try:
            predicted = self._result.predict()
        except Exception:
            predicted = np.full(len(df), np.nan)

        # Out-of-sample: simple weighted prediction using ergodic probs
        # (best we can do without the full history)
        if len(df) != len(predicted):
            # Use unconditional regime-weighted prediction
            smoothed = self._result.smoothed_marginal_probabilities
            avg_probs = smoothed.mean(axis=0)  # shape (k,)
            y_lag1 = df["y_lag1"].to_numpy(dtype=float)
            preds = self._regime_weighted_prediction(avg_probs, y_lag1)
            return preds

        return predicted

    def _regime_weighted_prediction(
        self, regime_probs: np.ndarray, y_lag1: np.ndarray
    ) -> np.ndarray:
        """Compute E[y | y_lag1] = sum_k P(s=k) * (mu_k + phi_k * y_lag1)."""
        params = self._result.params
        k = self.k_regimes
        # Params are stored as [const_0,...,const_{k-1}, ar_coef (shared or per regime)]
        n_const = k
        preds = np.zeros(len(y_lag1))
        for j in range(k):
            mu_j = params[j]  # intercept for regime j
            # AR coef: if switching_ar, there are k AR params; else one shared
            if self.switching_ar:
                phi_j = params[n_const + j]
            else:
                phi_j = params[n_const] if n_const < len(params) else 0.0
            preds += regime_probs[j] * (mu_j + phi_j * y_lag1)
        return preds

    # ------------------------------------------------------------------
    # Regime labels
    # ------------------------------------------------------------------

    def predict_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """Return the most-likely regime at each time step.

        For in-sample data (same length as training), returns the smoothed
        marginal probabilities argmax from the Hamilton filter.  For
        out-of-sample data, assigns the regime most consistent with the
        current observation via the ergodic (stationary) distribution.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before predict_regimes().")
        smoothed = self._result.smoothed_marginal_probabilities
        n_train = len(smoothed)
        if len(df) == n_train:
            return smoothed.argmax(axis=1).astype(int)
        # Out-of-sample: use average (ergodic) regime probabilities
        avg_probs = smoothed.mean(axis=0)
        # Assign hard label: the most likely regime on average
        dominant = int(avg_probs.argmax())
        return np.full(len(df), dominant, dtype=int)

    def regime_probabilities(self) -> pd.DataFrame:
        """Return smoothed regime probability matrix from training."""
        if self._result is None:
            raise RuntimeError("Call fit() before regime_probabilities().")
        probs = self._result.smoothed_marginal_probabilities
        return pd.DataFrame(
            probs, columns=[f"P(regime={k})" for k in range(self.k_regimes)]
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        if self._result is None:
            return "Model not fitted."
        return str(self._result.summary())
