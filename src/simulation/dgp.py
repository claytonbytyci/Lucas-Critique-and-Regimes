"""Data-Generating Processes for regime-switching time series.

The canonical DGP is a Markov-switching AR(1) model:

    y_t = mu_{s_t} + phi_{s_t} * y_{t-1} + eps_t,   eps_t ~ N(0, sigma_{s_t}^2)

where s_t in {0, 1, ..., K-1} follows a first-order Markov chain with
transition matrix P.  Additional observable covariates X_t are generated
jointly so that the models under study have features to work with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMULATED_DATA_DIR = _PROJECT_ROOT / "data" / "simulated"

@dataclass
class RegimeParams:
    """Parameters that define a single regime in the MS-AR(1) DGP.

    Parameters
    ----------
    mu : float
        Intercept / unconditional mean shift for this regime.
    phi : float
        AR(1) coefficient (persistence).  Must satisfy |phi| < 1.
    sigma : float
        Standard deviation of the innovation noise.
    label : str
        Human-readable name (e.g. "expansion", "recession").
    """

    mu: float
    phi: float
    sigma: float
    label: str = "regime"


# Default two-regime setup: expansion vs recession
_DEFAULT_REGIMES: list[RegimeParams] = [
    RegimeParams(mu=-0.4, phi=0.70, sigma=1.8, label="recession"),
    RegimeParams(mu=0.8, phi=0.35, sigma=0.7, label="expansion"),
]

# Default transition matrix (row = from, col = to)
_DEFAULT_TRANSITION: np.ndarray = np.array(
    [[0.90, 0.10],
     [0.15, 0.85]]
)

# Main DGP class

class MarkovSwitchingDGP:
    """Markov-switching AR(1) data-generating process with observable features.

    The DGP generates:
    - ``y``: the target time series.
    - ``s``: the latent regime sequence (integers 0..K-1).
    - ``X``: a matrix of observable features, including lagged ``y``,
      rolling statistics, and independent exogenous noise signals.

    Parameters
    ----------
    regimes : list[RegimeParams]
        Per-regime parameter sets.
    transition : np.ndarray, shape (K, K)
        Row-stochastic Markov transition matrix.
    n_exog : int
        Number of additional exogenous (noise) features to generate.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        regimes: list[RegimeParams] | None = None,
        transition: np.ndarray | None = None,
        n_exog: int = 3,
        seed: int | None = 42,
    ) -> None:
        self.regimes = regimes if regimes is not None else _DEFAULT_REGIMES
        self.transition = (
            transition if transition is not None else _DEFAULT_TRANSITION.copy()
        )
        self.n_exog = n_exog
        self.seed = seed

        self._validate()

    # Validation

    def _validate(self) -> None:
        K = len(self.regimes)
        if self.transition.shape != (K, K):
            raise ValueError(
                f"transition must be ({K}, {K}) for {K} regimes, "
                f"got {self.transition.shape}"
            )
        row_sums = self.transition.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError(f"Transition matrix rows must sum to 1, got {row_sums}")
        for r in self.regimes:
            if abs(r.phi) >= 1.0:
                raise ValueError(
                    f"Regime '{r.label}' has |phi|={abs(r.phi):.3f} >= 1 (non-stationary)"
                )

    # Stationary distribution

    @property
    def stationary_distribution(self) -> np.ndarray:
        """Ergodic (stationary) probabilities of each regime."""
        K = len(self.regimes)
        A = (self.transition.T - np.eye(K))
        A[-1, :] = 1.0
        b = np.zeros(K)
        b[-1] = 1.0
        return np.linalg.solve(A, b)

    # Simulation

    def simulate(self, n_obs: int = 600) -> pd.DataFrame:
        """Simulate ``n_obs`` observations from the Markov-switching AR(1).

        Parameters
        ----------
        n_obs : int
            Number of time-steps to simulate.

        Returns
        -------
        pd.DataFrame
            Columns: ``y``, ``regime`` (int), ``regime_label`` (str),
            ``y_lag1``, ``roll_mean_5``, ``roll_std_5``, ``roll_mean_20``,
            plus ``exog_0`` ... ``exog_{n_exog-1}``.
        """
        rng = np.random.default_rng(self.seed)
        K = len(self.regimes)

        # Simulate hidden Markov chain
        pi = self.stationary_distribution
        s = np.empty(n_obs, dtype=int)
        s[0] = rng.choice(K, p=pi)
        for t in range(1, n_obs):
            s[t] = rng.choice(K, p=self.transition[s[t - 1]])

        # Simulate AR(1) series conditional on regime 
        y = np.empty(n_obs)
        y[0] = self.regimes[s[0]].mu / (1 - self.regimes[s[0]].phi + 1e-9)
        for t in range(1, n_obs):
            r = self.regimes[s[t]]
            y[t] = r.mu + r.phi * y[t - 1] + rng.normal(0.0, r.sigma)

        # Observable features
        df = pd.DataFrame({"y": y, "regime": s})
        df["regime_label"] = df["regime"].map(
            {i: r.label for i, r in enumerate(self.regimes)}
        )

        # Lagged y (primary predictor for AR models)
        df["y_lag1"] = df["y"].shift(1)
        df["y_lag2"] = df["y"].shift(2)

        # Rolling statistics (carry information about regime)
        df["roll_mean_5"] = df["y"].rolling(5).mean()
        df["roll_std_5"] = df["y"].rolling(5).std()
        df["roll_mean_20"] = df["y"].rolling(20).mean()
        df["roll_std_20"] = df["y"].rolling(20).std()

        # Independent exogenous signals (partially correlated with regime)
        for k in range(self.n_exog):
            # Regime-correlated noise
            regime_signal = np.array(
                [self.regimes[si].mu for si in s]
            ) + rng.normal(0, 1.0, n_obs)
            df[f"exog_{k}"] = regime_signal

        # Drop NaN rows from lags / rolling windows (there will only be some at the start)
        df = df.dropna().reset_index(drop=True)
        return df

    # Persistence

    def save(self, df: pd.DataFrame, name: str = "base") -> Path:
        """Save simulated DataFrame to parquet in the data/simulated directory."""
        SIMULATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = SIMULATED_DATA_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return path

    # Repr

    def __repr__(self) -> str:
        regime_str = ", ".join(
            f"{r.label}(mu={r.mu}, phi={r.phi}, sigma={r.sigma})"
            for r in self.regimes
        )
        return f"MarkovSwitchingDGP([{regime_str}], seed={self.seed})"
