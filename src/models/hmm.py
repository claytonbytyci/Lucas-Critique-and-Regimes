"""Hidden Markov Model regime detector with per-regime linear regression.

Architecture (two-stage):
    1. A Gaussian HMM (``hmmlearn.hmm.GaussianHMM``) is fitted on an
       observation matrix built from ``y`` and its lags / rolling stats.
       This recovers the latent regime sequence.
    2. A separate :class:`sklearn.linear_model.Ridge` regression is fitted
       for each inferred regime on the training set.

At prediction time:
    - ``predict`` uses a **causal forward filter**: at each step t the regime is
      inferred from P(s_t | O_{1:t-1}) — the predicted probability that uses
      only *past* observations, avoiding any look-ahead bias.
    - ``predict_regimes`` uses the Viterbi algorithm (full sequence) for regime
      analysis where look-ahead is acceptable.

Note: HMMs require the *full sequence* for Viterbi decoding, so
``predict_regimes`` accepts full sequences.
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

# Features used for HMM observation matrix from which we attempt to infer regimes
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

    # Internal helpers

    @staticmethod
    def _build_obs_matrix(df: pd.DataFrame) -> np.ndarray: # this is giving us the matrix that we use to fit the HMM and decode regimes
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
        """
        obs = self._build_obs_matrix(df)
        X = self._build_feature_matrix(df)
        y = df["y"].to_numpy(dtype=float)

        # Fit HMM to recover the latent regimes
        self._hmm = GaussianHMM( # this is from the library which we pass the observation matrix into, uses the Viterbi algorithm to decode regimes, and then we fit the per-regime regressors based on the decoded regimes 
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self._hmm.fit(obs)

        # Decode training regimes via Viterbi algorithm
        train_regimes = self._hmm.predict(obs)

        # Fit per-regime regressors by splitting the training to their regimes
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

    # Predict

    def predict_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """Viterbi-decode the most likely regime sequence for ``df``."""
        if self._hmm is None:
            raise RuntimeError("Call fit() before predict_regimes().")
        obs = self._build_obs_matrix(df)
        return self._hmm.predict(obs).astype(int)

    def _causal_log_emission(self, obs: np.ndarray) -> np.ndarray:
        """Log Gaussian emission probabilities for each (t, k) pair.

        Uses the fitted HMM parameters directly so we can run our own
        forward filter without calling hmmlearn's Viterbi.

        Returns
        -------
        log_emit : (T, K) is log P(O_t | s_t=k) for each time t and regime k which is done in the log space for numerical stability traiditionally apparently online.
        """
        K = self.n_components
        T = len(obs)
        log_emit = np.empty((T, K))
        for k in range(K):
            mu = self._hmm.means_[k]
            cov = self._hmm.covars_[k]
            # hmmlearn may store diag covars as 1-D (n_features,) or 2-D
            # (n_features, n_features); extract diagonal in either case.
            var = np.diag(cov) if cov.ndim == 2 else np.asarray(cov).ravel()
            var = np.maximum(var, 1e-6)
            resid = obs - mu                      # (T, n_features)
            log_emit[:, k] = -0.5 * np.sum(
                np.log(2 * np.pi * var) + resid ** 2 / var, axis=1
            )
        return log_emit

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict ``y`` using a causal forward filter — no look-ahead.

        At each step t the regime is chosen as the argmax of
        P(s_t | O_{1:t-1}), the *predicted* (pre-observation) probability
        propagated from the previous filtered state.  This avoids using
        future observations to determine the current regime.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        np.ndarray, shape (n,)
        """
        if self._hmm is None:
            raise RuntimeError("Call fit() before predict().")

        obs = self._build_obs_matrix(df)
        X = self._build_feature_matrix(df)
        T, K = len(obs), self.n_components

        log_A = np.log(self._hmm.transmat_ + 1e-300)
        log_emit = self._causal_log_emission(obs)

        # Initialise with the start distribution
        log_filtered = np.log(self._hmm.startprob_ + 1e-300)
        preds = np.empty(T)

        for t in range(T):
            # Predicted probability P(s_t | O_{1:t-1}) — causal, no y_t yet
            if t == 0:
                log_pred = log_filtered
            else:
                log_pred = np.logaddexp.reduce(
                    log_filtered[:, None] + log_A, axis=0
                )
            log_pred = log_pred - np.logaddexp.reduce(log_pred)

            # Regime assignment from predicted (causal) probability
            regime = int(log_pred.argmax())

            regressor = self._regressors.get(regime)
            if regressor is None:
                preds[t] = self._global_regressor.predict(X[t : t + 1])[0]
            else:
                preds[t] = regressor.predict(X[t : t + 1])[0]

            # Update filter with the observed emission at time t
            log_filtered = log_pred + log_emit[t]

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
