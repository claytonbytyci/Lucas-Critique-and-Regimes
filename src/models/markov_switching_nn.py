"""Markov Switching Neural Network (MSNN).

Each regime has an MLP regressor; regime dynamics follow a K-state hidden
Markov chain.  Parameters are estimated jointly via EM:

  E-step  — Hamilton forward filter + backward smoother (exact smoothed
             posteriors computed in log-space for numerical stability).
  M-step  — Transition matrix and initial distribution updated analytically;
             MLP experts updated via Adam gradient descent weighted by the
             smoothed regime posteriors gamma_t(k) = P(s_t=k | y_{1:T}).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

__all__ = ["MarkovSwitchingNeuralNetwork"]

_FEATURE_COLS = ["y_lag1", "y_lag2", "roll_mean_5", "roll_std_5", "exog_0", "exog_1"]

# Tiny MLP — numpy-only, Adam optimiser, supports per-sample weights

class _MLP:
    """Small feedforward MLP (ReLU activations, linear output, Adam optimiser).

    Per-sample ``sample_weight`` support in ``fit`` allows the EM algorithm
    to drive each expert with the smoothed regime responsibilities.
    """

    def __init__(
        self,
        n_in: int,
        hidden: tuple,
        lr: float = 3e-3,
        n_epochs: int = 200,
        l2: float = 1e-3,
        random_state: int = 0,
    ) -> None:
        rng = np.random.default_rng(random_state)
        dims = [n_in, *hidden, 1]
        # He / Xavier initialisation (scale by sqrt(2/fan_in) for ReLU layers) which is standard for MLPs and aids convergence
        self.W = [
            rng.normal(0.0, np.sqrt(2.0 / dims[i]), (dims[i + 1], dims[i]))
            for i in range(len(dims) - 1)
        ]
        self.b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2 = l2

    def _forward(self, X: np.ndarray) -> list:
        h = X
        acts = [h]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W.T + b
            h = np.maximum(0.0, z) if i < len(self.W) - 1 else z  # ReLU / linear
            acts.append(h)
        return acts

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "_MLP":
        n = len(y)
        w = (
            np.ones(n) / n
            if sample_weight is None
            else sample_weight / (sample_weight.sum() + 1e-300)
        )

        # Adam moment estimates
        ms_W = [np.zeros_like(W) for W in self.W]
        vs_W = [np.zeros_like(W) for W in self.W]
        ms_b = [np.zeros_like(b) for b in self.b]
        vs_b = [np.zeros_like(b) for b in self.b]
        b1, b2, eps = 0.9, 0.999, 1e-8

        for step in range(1, self.n_epochs + 1):
            acts = self._forward(X)
            resid = acts[-1].flatten() - y          # (n,)
            delta = (w * resid)[:, None]            # (n, 1) — weighted MSE grad

            gW_list: list = []
            gb_list: list = []
            for i in reversed(range(len(self.W))):
                gW = delta.T @ acts[i] + self.l2 * self.W[i]   # L2 reg
                gb = delta.sum(axis=0)
                gW_list.insert(0, gW)
                gb_list.insert(0, gb)
                if i > 0:
                    delta = delta @ self.W[i]
                    delta = delta * (acts[i] > 0.0)             # ReLU subgradient

            for i in range(len(self.W)):
                ms_W[i] = b1 * ms_W[i] + (1 - b1) * gW_list[i]
                vs_W[i] = b2 * vs_W[i] + (1 - b2) * gW_list[i] ** 2
                ms_b[i] = b1 * ms_b[i] + (1 - b1) * gb_list[i]
                vs_b[i] = b2 * vs_b[i] + (1 - b2) * gb_list[i] ** 2

                m_hat = ms_W[i] / (1 - b1 ** step)
                v_hat = vs_W[i] / (1 - b2 ** step)
                self.W[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

                mb_hat = ms_b[i] / (1 - b1 ** step)
                vb_hat = vs_b[i] / (1 - b2 ** step)
                self.b[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[-1].flatten()

# Markov Switching Neural Network

class MarkovSwitchingNeuralNetwork:
    """Markov Switching Neural Network (MSNN).

    Combines a K-state hidden Markov chain for regime dynamics with
    per-regime MLP regressors.  Parameters are estimated via EM:

    * **E-step** — Hamilton filter (forward) + backward smoother to compute
      smoothed posteriors gamma_t(k) = P(s_t=k | y_{1:T}) and joint
      posteriors xi_t(j,k) = P(s_t=j, s_{t+1}=k | y_{1:T}).
    * **M-step** — Transition matrix A and initial distribution pi_0 are
      updated analytically from the sufficient statistics; each MLP expert
      is updated via Adam gradient descent with sample weights = gamma_t(k).

    Prediction uses Viterbi decoding to find the most-likely regime sequence,
    then applies the corresponding expert MLP at each time step.

    Parameters
    ----------
    k_regimes : int
        Number of latent regimes (default 2).
    hidden_layer_sizes : tuple[int, ...]
        Hidden-layer widths shared by all expert MLPs (default (32, 16)).
    n_iter : int
        Maximum EM iterations (default 50).
    mlp_epochs : int
        Adam gradient-descent epochs per MLP M-step (default 200).
    mlp_lr : float
        Adam learning rate for expert MLPs (default 3e-3).
    tol : float
        EM convergence tolerance on log-likelihood change (default 1e-4).
    random_state : int
        Global reproducibility seed (default 42).
    """

    def __init__(
        self,
        k_regimes: int = 2,
        hidden_layer_sizes: tuple = (32, 16),
        n_iter: int = 50,
        mlp_epochs: int = 200,
        mlp_lr: float = 3e-3,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.k_regimes = k_regimes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_iter = n_iter
        self.mlp_epochs = mlp_epochs
        self.mlp_lr = mlp_lr
        self.tol = tol
        self.random_state = random_state

        # Fitted state (initialised in fit)
        self._experts: list = []
        self._sigmas: np.ndarray = np.ones(k_regimes)
        self._A: np.ndarray = np.full((k_regimes, k_regimes), 1.0 / k_regimes)
        self._pi0: np.ndarray = np.full(k_regimes, 1.0 / k_regimes)
        self._scaler: StandardScaler = StandardScaler()

    # Feature helpers

    @staticmethod
    def _get_features(df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in _FEATURE_COLS if c in df.columns]
        return df[cols].to_numpy(dtype=float)

    @staticmethod
    def _log_obs_prob(
        y: np.ndarray, mu: np.ndarray, sigma: np.ndarray
    ) -> np.ndarray:
        """Log Gaussian likelihood for each (obs, regime) pair.

        Parameters
        ----------
        y     : (T,)
        mu    : (T, K) expert predictions per regime
        sigma : (K,)  residual std per regime

        Returns
        -------
        log_p : (T, K)
        """
        K = mu.shape[1]
        log_p = np.empty((len(y), K))
        for k in range(K):
            resid = y - mu[:, k]
            log_p[:, k] = (
                -0.5 * np.log(2.0 * np.pi * sigma[k] ** 2)
                - 0.5 * (resid / sigma[k]) ** 2
            )
        return log_p

    # E-step: Hamilton filter + backward smoother (log-space)

    def _forward_backward(
        self, log_obs: np.ndarray
    ) -> tuple:
        """Exact smoothed posteriors via forward-backward algorithm.

        Parameters
        ----------
        log_obs : (T, K) log-likelihoods log p(y_t | s_t=k)

        Returns
        -------
        gamma : (T, K)   smoothed posteriors P(s_t=k | y_{1:T})
        xi    : (T-1, K, K) joint smoothed P(s_t=j, s_{t+1}=k | y_{1:T})
        ll    : float    log P(y_{1:T})
        """
        T, K = log_obs.shape
        log_A = np.log(self._A + 1e-300)

        # Forward pass
        log_alpha = np.empty((T, K))
        log_alpha[0] = np.log(self._pi0 + 1e-300) + log_obs[0]
        for t in range(1, T):
            # log_alpha[t-1, :, None] shape (K,1); log_A shape (K,K)
            # element [j,k] = log_alpha_{t-1}(j) + log A(j,k)
            # logsumexp over j (axis=0) → shape (K,)
            log_alpha[t] = (
                np.logaddexp.reduce(log_alpha[t - 1, :, None] + log_A, axis=0)
                + log_obs[t]
            )

        ll = float(np.logaddexp.reduce(log_alpha[-1]))

        # Backward pass 
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            # log_A shape (K,K); log_obs[t+1] shape (K,) bcast over rows
            # element [k,k'] = log_A[k,k'] + log_obs[t+1,k'] + log_beta[t+1,k']
            # logsumexp over k' (axis=1) → shape (K,)
            log_beta[t] = np.logaddexp.reduce(
                log_A + log_obs[t + 1] + log_beta[t + 1], axis=1
            )

        # Smoothed posteriors
        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(np.clip(log_gamma, -500.0, 0.0))

        # Joint smoothed (xi) for transition update
        xi = np.empty((T - 1, K, K))
        for t in range(T - 1):
            # log_alpha[t,:,None] (K,1) + log_A (K,K) + log_obs[t+1] (K,) + log_beta[t+1] (K,)
            log_xi = (
                log_alpha[t, :, None]
                + log_A
                + log_obs[t + 1]
                + log_beta[t + 1]
            )  # (K, K)
            log_xi -= float(np.logaddexp.reduce(log_xi.ravel()))
            xi[t] = np.exp(np.clip(log_xi, -500.0, 0.0))

        return gamma, xi, ll

    # M-step helpers
  
    def _update_transition(self, gamma: np.ndarray, xi: np.ndarray) -> None:
        num = xi.sum(axis=0)                          # (K, K)
        denom = gamma[:-1].sum(axis=0)[:, None]       # (K, 1)
        self._A = num / (denom + 1e-300)
        row_sums = self._A.sum(axis=1, keepdims=True)
        self._A /= np.where(row_sums > 0, row_sums, 1.0)

    # Public interface
    
    def fit(self, df: pd.DataFrame) -> "MarkovSwitchingNeuralNetwork":
        rng = np.random.default_rng(self.random_state)
        K = self.k_regimes

        X_raw = self._get_features(df)
        y = df["y"].to_numpy(dtype=float)
        T, n_in = X_raw.shape

        X = self._scaler.fit_transform(X_raw)

        # Initialise expert MLPs
        self._experts = [
            _MLP(
                n_in,
                tuple(self.hidden_layer_sizes),
                lr=self.mlp_lr,
                n_epochs=self.mlp_epochs,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            for _ in range(K)
        ]

        # Diagonal-dominant transition matrix (regimes tend to persist) which means the EM algorithm starts with a regime structure more likely to capture consistent patterns/regimes
        stay = 0.9
        off = (1.0 - stay) / max(K - 1, 1)
        self._A = np.full((K, K), off)
        np.fill_diagonal(self._A, stay)
        self._A /= self._A.sum(axis=1, keepdims=True)
        self._pi0 = np.full(K, 1.0 / K)
        self._sigmas = np.ones(K)

        # Bootstrap: pre-train each expert on a disjoint random slice of data to break symmetry and give EM algorithm a better starting point
        for k in range(K):
            idx = rng.choice(T, size=max(T // K, 20), replace=False)
            self._experts[k].fit(X[idx], y[idx])
            preds_k = self._experts[k].predict(X)
            self._sigmas[k] = float(np.std(y - preds_k)) + 1e-4

        # EM loop which alternates between the E-step forward backward for computing smoothed posteriors and the M-step for updating the transition matrix
        prev_ll = -np.inf
        for _ in range(self.n_iter):
            # Expert predictions — (T, K)
            mu = np.column_stack([e.predict(X) for e in self._experts])

            # Log observation probabilities — (T, K)
            log_obs = self._log_obs_prob(y, mu, self._sigmas)

            # E-step
            gamma, xi, ll = self._forward_backward(log_obs)

            # M-step: Markov chain parameters
            self._pi0 = gamma[0]
            self._update_transition(gamma, xi)

            # M-step: expert MLPs + residual sigmas
            for k in range(K):
                w = gamma[:, k]
                if w.sum() < 1e-6:
                    continue
                self._experts[k].fit(X, y, sample_weight=w)
                preds_k = self._experts[k].predict(X)
                self._sigmas[k] = float(
                    np.sqrt(np.average((y - preds_k) ** 2, weights=w)) + 1e-6
                )

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def _viterbi(self, log_obs: np.ndarray) -> np.ndarray:
        """Viterbi decoding — most-likely regime sequence."""
        T, K = log_obs.shape
        log_A = np.log(self._A + 1e-300)

        delta = np.log(self._pi0 + 1e-300) + log_obs[0]
        psi = np.zeros((T, K), dtype=int)
        for t in range(1, T):
            scores = delta[:, None] + log_A     # (K, K)
            psi[t] = scores.argmax(axis=0)
            delta = scores.max(axis=0) + log_obs[t]

        states = np.empty(T, dtype=int)
        states[-1] = int(delta.argmax())
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def _compute_log_obs(
        self, df: pd.DataFrame
    ) -> tuple:
        """Scaled features, y, and log-obs matrix for df."""
        X_raw = self._get_features(df)
        X = self._scaler.transform(X_raw)
        y = df["y"].to_numpy(dtype=float)
        mu = np.column_stack([e.predict(X) for e in self._experts])
        return self._log_obs_prob(y, mu, self._sigmas), X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """One-step-ahead predictions using a causal Hamilton forward filter.

        At each step t the regime weight is P(s_t | y_{1:t-1}) — the
        *predicted* probability before observing y_t — so no future
        information leaks into the forecast.  The prediction is a
        soft-mixture over expert outputs weighted by these causal
        probabilities.

        (predict_regimes still uses Viterbi over the full sequence
        for regime-analysis purposes where look-ahead is acceptable.)
        """
        X_raw = self._get_features(df)
        X = self._scaler.transform(X_raw)
        y = df["y"].to_numpy(dtype=float)
        T, K = len(df), self.k_regimes
        log_A = np.log(self._A + 1e-300)

        # Start from the fitted initial distribution
        log_filtered = np.log(self._pi0 + 1e-300)
        preds = np.empty(T)

        for t in range(T):
            # Causal predicted probability P(s_t | y_{1:t-1})
            if t == 0:
                log_pred = log_filtered          # = log π_0
            else:
                log_pred = np.logaddexp.reduce(
                    log_filtered[:, None] + log_A, axis=0
                )
            log_pred = log_pred - np.logaddexp.reduce(log_pred)
            pred_probs = np.exp(log_pred)        # (K,)

            # Forecast: soft mixture of expert outputs (causal)
            expert_preds = np.array(
                [e.predict(X[t : t + 1])[0] for e in self._experts]
            )
            preds[t] = pred_probs @ expert_preds

            # Update filter with observed y_t
            log_obs_t = np.array([
                -0.5 * np.log(2.0 * np.pi * self._sigmas[k] ** 2)
                - 0.5 * ((y[t] - expert_preds[k]) / self._sigmas[k]) ** 2
                for k in range(K)
            ])
            log_filtered = log_pred + log_obs_t

        return preds

    def predict_regimes(self, df: pd.DataFrame) -> np.ndarray:
        log_obs, _ = self._compute_log_obs(df)
        return self._viterbi(log_obs)

    def regime_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        log_obs, _ = self._compute_log_obs(df)
        gamma, _, _ = self._forward_backward(log_obs)
        cols = [f"P(regime={k})" for k in range(self.k_regimes)]
        return pd.DataFrame(gamma, columns=cols, index=df.index)

    @property
    def transition_matrix(self) -> pd.DataFrame:
        """Estimated Markov transition matrix A (K × K)."""
        labels = [f"regime_{k}" for k in range(self.k_regimes)]
        return pd.DataFrame(self._A, index=labels, columns=labels)
