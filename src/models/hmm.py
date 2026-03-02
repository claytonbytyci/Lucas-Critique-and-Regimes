"""Hidden Markov Model regime detector with per-regime linear regression.

Architecture (two-stage):
    1. A Gaussian HMM (``hmmlearn.hmm.GaussianHMM``) is fitted on an
       observation matrix built from ``y`` and its lags / rolling stats.
       This recovers the latent regime sequence.
    2. A separate :class:`sklearn.linear_model.Ridge` regression is fitted
       for each inferred regime on the training set.

At prediction time:
    - The Viterbi algorithm decodes the most likely regime sequence given the
      new observations.
    - The corresponding per-regime regressor produces the forecast.

Note: HMMs require the *full sequence* to decode regimes, so the ``predict``
and ``predict_regimes`` methods accept sequences rather than individual rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

try:
    from hmmlearn.hmm import GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HMMLEARN_AVAILABLE = False

# Features used for HMM observation matrix
_HMM_OBS_COLS = ["y", "y_lag1", "roll_mean_5", "roll_std_5", "roll_mean_20"]

# Features used for per-regime linear regression
_REG_FEATURE_COLS = ["y_lag1", "roll_mean_5", "roll_std_5", "exog_0", "exog_1"]


class HMMRegimeModel:
    """Gaussian HMM for regime detection + per-regime Ridge regression.

    Parameters
    ----------
    n_components : int
        Number of hidden states (regimes).
    n_iter : int
        Maximum Baum-Welch (EM) iterations for the HMM.
    covariance_type : str
        HMM covariance structure — "diag", "full", "tied", or "spherical".
    alpha : float
        Ridge regularisation strength for per-regime regressors.
    random_state : int or None
        Random seed for HMM initialisation.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_iter: int = 200,
        covariance_type: str = "diag",
        alpha: float = 1.0,
        random_state: int | None = 42,
    ) -> None:
        if not _HMMLEARN_AVAILABLE:
            raise ImportError("hmmlearn is required for HMMRegimeModel.")
        self.n_components = n_components
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.alpha = alpha
        self.random_state = random_state

        self._hmm: GaussianHMM | None = None
        self._regressors: dict[int, Ridge] = {}
        self._n_regimes_found: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_obs_matrix(df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in _HMM_OBS_COLS if c in df.columns]
        return df[cols].to_numpy(dtype=float)

    @staticmethod
    def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in _REG_FEATURE_COLS if c in df.columns]
        return df[cols].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "HMMRegimeModel":
        """Fit HMM on the full training sequence, then fit per-regime regressors.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame produced by :class:`simulation.dgp.MarkovSwitchingDGP`.

        Returns
        -------
        self
        """
        obs = self._build_obs_matrix(df)
        X = self._build_feature_matrix(df)
        y = df["y"].to_numpy(dtype=float)

        # Fit HMM
        self._hmm = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self._hmm.fit(obs)

        # Decode training regimes via Viterbi
        train_regimes = self._hmm.predict(obs)

        # Fit per-regime regressors
        unique_regimes = np.unique(train_regimes)
        self._n_regimes_found = len(unique_regimes)
        self._regressors = {}
        for reg in unique_regimes:
            mask = train_regimes == reg
            if mask.sum() < 5:
                # Too few samples: fall back to global mean
                self._regressors[reg] = None
                continue
            reg_model = Ridge(alpha=self.alpha)
            reg_model.fit(X[mask], y[mask])
            self._regressors[reg] = reg_model

        # Global fallback regressor
        self._global_regressor = Ridge(alpha=self.alpha)
        self._global_regressor.fit(X, y)

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """Viterbi-decode the most likely regime sequence for ``df``."""
        if self._hmm is None:
            raise RuntimeError("Call fit() before predict_regimes().")
        obs = self._build_obs_matrix(df)
        return self._hmm.predict(obs).astype(int)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict ``y`` using decoded regime + per-regime regressor.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        np.ndarray, shape (n,)
        """
        if self._hmm is None:
            raise RuntimeError("Call fit() before predict().")
        regimes = self.predict_regimes(df)
        X = self._build_feature_matrix(df)
        preds = np.empty(len(df))

        for i, reg in enumerate(regimes):
            regressor = self._regressors.get(reg)
            if regressor is None:
                preds[i] = self._global_regressor.predict(X[i : i + 1])[0]
            else:
                preds[i] = regressor.predict(X[i : i + 1])[0]

        return preds

    def regime_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return posterior regime probabilities for each row in ``df``."""
        if self._hmm is None:
            raise RuntimeError("Call fit() before regime_probabilities().")
        obs = self._build_obs_matrix(df)
        _, posteriors = self._hmm.score_samples(obs)
        return pd.DataFrame(
            posteriors,
            columns=[f"P(regime={k})" for k in range(self.n_components)],
        )
