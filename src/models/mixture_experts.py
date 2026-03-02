"""Mixture of Experts (MoE) regime-switching model.

Architecture:
    - A *gating network* (logistic regression / softmax) outputs soft weights
      w_k(x) = P(regime=k | x) for each expert k.
    - K *expert networks* (Ridge regressors) each produce a prediction
      y_hat_k = f_k(x).
    - The final prediction is the weighted mixture:
          y_hat = sum_k w_k(x) * y_hat_k

Training uses an EM-style iterative algorithm:
    E-step: compute posterior responsibilities r_{ik} = w_k(x_i) * N(y_i | mu_k, sigma_k^2)
    M-step: refit gating and experts using weighted samples.

This soft-assignment scheme is the closest ML analogue to the Hamilton filter
and provides a useful intermediate comparison between classical and hard-
assignment ML models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

_FEATURE_COLS = [
    "y_lag1", "y_lag2",
    "roll_mean_5", "roll_std_5",
    "roll_mean_20",
    "exog_0", "exog_1",
]


class MixtureOfExpertsModel:
    """Mixture of Experts with EM training.

    Parameters
    ----------
    n_experts : int
        Number of expert regressors (= number of soft regimes).
    n_iter : int
        EM iterations.
    alpha_expert : float
        Ridge regularisation for each expert.
    C_gate : float
        Logistic regression regularisation (inverse of strength).
    tol : float
        Convergence tolerance on the log-likelihood.
    random_state : int or None
        Seed for reproducible initialisation.
    """

    def __init__(
        self,
        n_experts: int = 2,
        n_iter: int = 100,
        alpha_expert: float = 1.0,
        C_gate: float = 1.0,
        tol: float = 1e-4,
        random_state: int | None = 42,
    ) -> None:
        self.n_experts = n_experts
        self.n_iter = n_iter
        self.alpha_expert = alpha_expert
        self.C_gate = C_gate
        self.tol = tol
        self.random_state = random_state

        self._experts: list[Ridge] = []
        self._gate: LogisticRegression | None = None
        self._scaler = StandardScaler()
        self._expert_sigmas: np.ndarray | None = None  # per-expert residual std

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in _FEATURE_COLS if c in df.columns]
        return df[cols].to_numpy(dtype=float)

    @staticmethod
    def _gaussian_log_likelihood(
        y: np.ndarray, mu: np.ndarray, sigma: float
    ) -> np.ndarray:
        """Log N(y | mu, sigma^2) for each element."""
        sigma = max(sigma, 1e-6)
        return -0.5 * ((y - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "MixtureOfExpertsModel":
        """EM training of gating + experts.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        self
        """
        X_raw = self._build_features(df)
        X = self._scaler.fit_transform(X_raw)
        y = df["y"].to_numpy(dtype=float)
        n = len(y)
        K = self.n_experts

        rng = np.random.default_rng(self.random_state)

        # --- Initialise responsibilities uniformly with jitter ---
        R = rng.dirichlet(np.ones(K), size=n)  # (n, K)

        # Initialise experts and sigmas
        self._experts = [Ridge(alpha=self.alpha_expert) for _ in range(K)]
        self._expert_sigmas = np.ones(K)

        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            # --- M-step: fit each expert with sample weights ---
            for k, expert in enumerate(self._experts):
                w = R[:, k]
                if w.sum() < 1e-6:
                    continue
                expert.fit(X, y, sample_weight=w)
                y_hat_k = expert.predict(X)
                resid = y - y_hat_k
                self._expert_sigmas[k] = max(
                    np.sqrt(np.average(resid ** 2, weights=w)), 1e-6
                )

            # --- M-step: fit gating network (multinomial logistic) ---
            hard_labels = R.argmax(axis=1)
            unique_labels = np.unique(hard_labels)
            if len(unique_labels) < K:
                # Degenerate: some experts unused — break early
                break
            self._gate = LogisticRegression(
                C=self.C_gate,
                solver="lbfgs",
                max_iter=500,
                random_state=self.random_state,
            )
            self._gate.fit(X, hard_labels)

            # --- E-step: recompute responsibilities ---
            gate_probs = self._gate.predict_proba(X)  # (n, K)
            log_R = np.zeros((n, K))
            for k, expert in enumerate(self._experts):
                y_hat_k = expert.predict(X)
                log_R[:, k] = (
                    np.log(gate_probs[:, k] + 1e-12)
                    + self._gaussian_log_likelihood(y, y_hat_k, self._expert_sigmas[k])
                )

            # Log-sum-exp normalisation
            log_R_max = log_R.max(axis=1, keepdims=True)
            log_R_norm = log_R - log_R_max - np.log(
                np.exp(log_R - log_R_max).sum(axis=1, keepdims=True) + 1e-12
            )
            R = np.exp(log_R_norm)

            # Convergence check
            ll = float(np.log(np.exp(log_R).sum(axis=1) + 1e-12).mean())
            if abs(ll - prev_ll) < self.tol and iteration > 5:
                break
            prev_ll = ll

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """Return the highest-probability expert for each row."""
        if self._gate is None:
            raise RuntimeError("Call fit() before predict_regimes().")
        X_raw = self._build_features(df)
        X = self._scaler.transform(X_raw)
        return self._gate.predict(X).astype(int)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Mixture-weighted prediction: y_hat = sum_k w_k * y_hat_k.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        np.ndarray, shape (n,)
        """
        if self._gate is None:
            raise RuntimeError("Call fit() before predict().")
        X_raw = self._build_features(df)
        X = self._scaler.transform(X_raw)

        gate_probs = self._gate.predict_proba(X)  # (n, K)
        preds = np.zeros(len(df))
        for k, expert in enumerate(self._experts):
            preds += gate_probs[:, k] * expert.predict(X)

        return preds

    def regime_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gate probabilities (soft regime memberships)."""
        if self._gate is None:
            raise RuntimeError("Call fit() before regime_probabilities().")
        X_raw = self._build_features(df)
        X = self._scaler.transform(X_raw)
        proba = self._gate.predict_proba(X)
        return pd.DataFrame(
            proba,
            columns=[f"P(regime={k})" for k in range(self.n_experts)],
        )
