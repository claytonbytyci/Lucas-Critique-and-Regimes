"""ML-based regime-switching model.

Architecture (two-stage):
    Stage 1 — Regime detection:
        Unsupervised KMeans clustering on a rich feature space to discover
        latent regimes, followed by an XGBoost classifier trained to
        reproduce those cluster labels from observable features.

    Stage 2 — Regime-conditional regression:
        A separate XGBoost regressor is trained for each discovered regime.
        At prediction time the Stage-1 classifier assigns a regime, and the
        corresponding Stage-2 regressor produces the forecast.

This design tests whether ML's greater expressiveness provides insulation from
the Lucas critique, or whether over-fitting to the pre-break regime structure
still causes severe degradation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGB_AVAILABLE = False

# Features for the clustering / regime classification stage
_CLUSTER_FEATURE_COLS = [
    "y_lag1", "y_lag2",
    "roll_mean_5", "roll_std_5",
    "roll_mean_20", "roll_std_20",
    "exog_0", "exog_1", "exog_2",
]

# Features for per-regime regressors
_REG_FEATURE_COLS = [
    "y_lag1", "y_lag2",
    "roll_mean_5", "roll_std_5",
    "exog_0", "exog_1",
]


class MLRegimeModel:
    """Two-stage ML regime switcher (KMeans → XGBoost classifier + regressors).

    Parameters
    ----------
    n_regimes : int
        Number of clusters (pseudo-regimes) to discover.
    classifier_params : dict or None
        Keyword arguments forwarded to :class:`xgboost.XGBClassifier`.
    regressor_params : dict or None
        Keyword arguments forwarded to each :class:`xgboost.XGBRegressor`.
    cluster_seed : int
        Random seed for KMeans.
    model_seed : int
        Random seed for XGBoost.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        classifier_params: dict | None = None,
        regressor_params: dict | None = None,
        cluster_seed: int = 0,
        model_seed: int = 42,
    ) -> None:
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is required for MLRegimeModel.")
        self.n_regimes = n_regimes
        self.cluster_seed = cluster_seed
        self.model_seed = model_seed

        _clf_defaults = dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="mlogloss",
            random_state=model_seed,
            verbosity=0,
        )
        _reg_defaults = dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=model_seed,
            verbosity=0,
        )
        self._clf_params = {**_clf_defaults, **(classifier_params or {})}
        self._reg_params = {**_reg_defaults, **(regressor_params or {})}

        self._scaler = StandardScaler()
        self._kmeans: KMeans | None = None
        self._classifier: XGBClassifier | None = None
        self._regressors: dict[int, XGBRegressor] = {}
        self._global_regressor: XGBRegressor | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cluster_features(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in _CLUSTER_FEATURE_COLS if c in df.columns]
        return df[cols].to_numpy(dtype=float)

    def _reg_features(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in _REG_FEATURE_COLS if c in df.columns]
        return df[cols].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "MLRegimeModel":
        """Two-stage training: cluster to find regimes, then fit regressors.

        Parameters
        ----------
        df : pd.DataFrame
            Training data from the DGP.

        Returns
        -------
        self
        """
        y = df["y"].to_numpy(dtype=float)
        X_cluster = self._cluster_features(df)
        X_reg = self._reg_features(df)

        # --- Stage 1a: KMeans clustering ---
        X_scaled = self._scaler.fit_transform(X_cluster)
        self._kmeans = KMeans(
            n_clusters=self.n_regimes, random_state=self.cluster_seed, n_init=10
        )
        cluster_labels = self._kmeans.fit_predict(X_scaled)

        # --- Stage 1b: XGBoost classifier to generalise cluster labels ---
        self._classifier = XGBClassifier(**self._clf_params)
        self._classifier.fit(X_cluster, cluster_labels)

        # --- Stage 2: Per-regime XGBoost regressors ---
        self._regressors = {}
        for reg in range(self.n_regimes):
            mask = cluster_labels == reg
            if mask.sum() < 10:
                continue
            regressor = XGBRegressor(**self._reg_params)
            regressor.fit(X_reg[mask], y[mask])
            self._regressors[reg] = regressor

        # Global fallback
        self._global_regressor = XGBRegressor(**self._reg_params)
        self._global_regressor.fit(X_reg, y)

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """Classify each row into a regime using the XGBoost classifier."""
        if self._classifier is None:
            raise RuntimeError("Call fit() before predict_regimes().")
        X_cluster = self._cluster_features(df)
        return self._classifier.predict(X_cluster).astype(int)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict ``y`` using the regime-classified XGBoost regressor.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        np.ndarray, shape (n,)
        """
        if self._classifier is None:
            raise RuntimeError("Call fit() before predict().")
        regimes = self.predict_regimes(df)
        X_reg = self._reg_features(df)
        preds = np.empty(len(df))

        for i, reg in enumerate(regimes):
            regressor = self._regressors.get(reg)
            if regressor is None:
                preds[i] = self._global_regressor.predict(X_reg[i : i + 1])[0]
            else:
                preds[i] = regressor.predict(X_reg[i : i + 1])[0]

        return preds

    def regime_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return XGBoost's soft regime probabilities."""
        if self._classifier is None:
            raise RuntimeError("Call fit() before regime_probabilities().")
        X_cluster = self._cluster_features(df)
        proba = self._classifier.predict_proba(X_cluster)
        return pd.DataFrame(
            proba,
            columns=[f"P(regime={k})" for k in range(self.n_regimes)],
        )
