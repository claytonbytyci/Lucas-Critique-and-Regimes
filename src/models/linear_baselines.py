"""Classical linear time-series baselines (no regime switching).

These models serve as the parametric null hypothesis against which
regime-switching models are compared.  Both implement the project's
standard fit/predict interface.

Models
------
ARModel
    OLS autoregression of order p.  Estimates

        y_t = alpha + beta_1 * y_{t-1} + ... + beta_p * y_{t-p} + gamma * X_t + eps_t

    by OLS on pre-break data and applies frozen coefficients out-of-sample.
    No regime structure is assumed.  This is the parametric baseline that
    all regime-switching models must beat to justify their added complexity.

ARMAModel
    Autoregressive moving-average model estimated via statsmodels ARIMA(p, 0, q).
    The MA component captures autocorrelation in residuals that the AR terms
    miss.  Out-of-sample predictions are produced by applying the fitted model
    to new observations via the statsmodels `apply()` method, which uses a
    Kalman-filter one-step-ahead recursion: at each t the prediction is formed
    from the information set {y_1, ..., y_{t-1}} only, respecting causality.

References
----------
- Box, G. E. P. & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting
  and Control*. Holden-Day.
- Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1–48.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature columns shared by both models
# ---------------------------------------------------------------------------

_LAG_COLS = ["y_lag1", "y_lag2"]
_EXOG_COLS = ["exog_0", "exog_1", "exog_2"]


# ---------------------------------------------------------------------------
# AR(p) — OLS autoregression
# ---------------------------------------------------------------------------


class ARModel:
    """OLS autoregression of order p with optional exogenous regressors.

    The model estimated on pre-break training data is:

        y_t = alpha + sum_{j=1}^{p} beta_j * y_{t-j} + gamma' * X_t + eps_t

    Coefficients are frozen at the pre-break fit and applied to post-break
    data unchanged, making parameter instability fully visible in RMSE.

    Parameters
    ----------
    order : int
        AR order p.  Order 1 uses y_lag1 only; order 2 adds y_lag2.
        Maximum supported: 2 (matching the project feature schema).
    include_exog : bool
        If True, includes exog_0..2 as additional regressors.
    alpha_ridge : float
        Ridge penalty (L2 regularisation) on the OLS coefficients.
        Default 0.0 = pure OLS.  Set > 0 for light regularisation.
    """

    def __init__(
        self,
        order: int = 2,
        include_exog: bool = True,
        alpha_ridge: float = 0.0,
    ) -> None:
        if order not in (1, 2):
            raise ValueError("ARModel supports order 1 or 2.")
        self.order = order
        self.include_exog = include_exog
        self.alpha_ridge = alpha_ridge

        self._coef: np.ndarray | None = None  # [intercept, beta_1, ..., gammas]
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _feature_cols(self) -> list[str]:
        cols = _LAG_COLS[: self.order]
        if self.include_exog:
            cols = cols + _EXOG_COLS
        return cols

    def _build_X(self, df: pd.DataFrame) -> np.ndarray:
        cols = self._feature_cols()
        X = df[cols].to_numpy(dtype=float)
        return np.column_stack([np.ones(len(X)), X])  # prepend intercept

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "ARModel":
        """Estimate AR coefficients by OLS (or Ridge) on pre-break data.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame; must contain ``y`` and lag/exog columns.

        Returns
        -------
        self
        """
        y = df["y"].to_numpy(dtype=float)
        X = self._build_X(df)
        self._feature_names = ["intercept"] + self._feature_cols()

        if self.alpha_ridge == 0.0:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        else:
            # Ridge: (X'X + alpha*I)^{-1} X'y  (no penalty on intercept)
            n_feat = X.shape[1]
            penalty = self.alpha_ridge * np.eye(n_feat)
            penalty[0, 0] = 0.0  # don't penalise intercept
            coef = np.linalg.solve(X.T @ X + penalty, X.T @ y)

        self._coef = coef
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """One-step-ahead forecasts using frozen pre-break coefficients.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the same feature columns used during training.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        if self._coef is None:
            raise RuntimeError("Call fit() before predict().")
        X = self._build_X(df)
        return X @ self._coef

    def predict_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """Returns all zeros — the AR model assumes a single homogeneous regime."""
        return np.zeros(len(df), dtype=int)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        if self._coef is None:
            return "ARModel: not fitted."
        lines = [f"ARModel(order={self.order}, include_exog={self.include_exog})"]
        for name, val in zip(self._feature_names, self._coef):
            lines.append(f"  {name:20s}: {val:+.6f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ARMA(p, q) — statsmodels ARIMA
# ---------------------------------------------------------------------------


class ARMAModel:
    """ARMA(p, q) model estimated via statsmodels ARIMA(p, 0, q).

    The ARMA model enriches the pure AR specification by adding q moving-
    average terms that capture autocorrelation in the residuals:

        y_t = alpha + phi_1 y_{t-1} + ... + phi_p y_{t-p}
                    + theta_1 eps_{t-1} + ... + theta_q eps_{t-q} + eps_t

    The model is estimated by maximum likelihood (Gaussian innovations) on
    the pre-break sample.  Out-of-sample forecasts for the post-break period
    are produced by applying the frozen parameters to the new y observations
    via a Kalman-filter recursion: at each step t the MA innovations are
    updated using the actual y_t, but the ARMA coefficients (phi, theta) and
    sigma are never re-estimated.  This mirrors the Lucas Critique experiment
    design — parameters are frozen at the structural break.

    Parameters
    ----------
    p : int
        AR order.
    q : int
        MA order.
    trend : str
        Trend specification: 'c' (constant), 'n' (none), 'ct' (const+trend).
    """

    def __init__(
        self,
        p: int = 2,
        q: int = 1,
        trend: str = "c",
    ) -> None:
        self.p = p
        self.q = q
        self.trend = trend

        self._result = None
        self._y_train: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "ARMAModel":
        """Fit ARMA(p, q) by MLE on pre-break training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame; must contain ``y``.

        Returns
        -------
        self
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:
            raise ImportError("statsmodels is required for ARMAModel.") from exc

        y = df["y"].to_numpy(dtype=float)
        self._y_train = y

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(y, order=(self.p, 0, self.q), trend=self.trend)
            self._result = model.fit()

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """One-step-ahead ARMA predictions with frozen coefficients.

        For **in-sample** data (same length as training): returns the fitted
        values from the MLE optimisation.

        For **out-of-sample** data: applies the frozen ARMA parameters to the
        new y observations via ``statsmodels.apply()``, which runs the Kalman
        filter forward using the pre-break parameter estimates.  Each
        prediction at time t uses only {y_1, ..., y_{t-1}}, preserving the
        causal (non-look-ahead) nature of the evaluation.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with ``y`` column.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        if self._result is None:
            raise RuntimeError("Call fit() before predict().")

        y_new = df["y"].to_numpy(dtype=float)
        n_train = len(self._y_train)

        # In-sample: use statsmodels fitted values
        if len(y_new) == n_train:
            return self._result.fittedvalues

        # Out-of-sample: apply frozen parameters to new observations
        try:
            from statsmodels.tsa.arima.model import ARIMA
            new_model = ARIMA(y_new, order=(self.p, 0, self.q), trend=self.trend)
            # apply() fixes the parameters and re-filters on new data
            new_result = new_model.apply(self._result.params)
            return new_result.fittedvalues
        except Exception:
            # Fallback: use frozen AR coefficients applied to lag features
            return self._ar_fallback_predict(df)

    def _ar_fallback_predict(self, df: pd.DataFrame) -> np.ndarray:
        """Pure AR fallback if statsmodels apply() fails."""
        try:
            # Extract AR coefficients from the fitted result
            params = self._result.params
            # params layout: [intercept?, ar.L1, ar.L2, ..., ma.L1, ..., sigma2]
            # Use y_lag1 / y_lag2 if available in df
            if "y_lag1" in df.columns:
                y_lag1 = df["y_lag1"].to_numpy(dtype=float)
            else:
                y_lag1 = np.roll(df["y"].to_numpy(dtype=float), 1)
            # Simple AR(1) approximation as last resort
            intercept = float(self._result.params.get("const", 0.0))
            ar_coef = (
                float(self._result.params.get("ar.L1", 0.0))
                if hasattr(self._result.params, "get")
                else 0.0
            )
            return intercept + ar_coef * y_lag1
        except Exception:
            return np.full(len(df), float(self._y_train.mean()))

    def predict_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """Returns all zeros — ARMA assumes a single homogeneous regime."""
        return np.zeros(len(df), dtype=int)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        if self._result is None:
            return "ARMAModel: not fitted."
        return str(self._result.summary())


# ---------------------------------------------------------------------------
# Model Average Ensemble
# ---------------------------------------------------------------------------


class ModelAverageEnsemble:
    """Equal-weighted average of predictions from a collection of fitted models.

    The ensemble produces a forecast combination by averaging the one-step-ahead
    predictions of all constituent models:

        \\hat{y}_t^{\\text{avg}} = \\frac{1}{M} \\sum_{m=1}^{M} \\hat{y}_t^{(m)}

    Forecast combination is a classical tool for improving robustness: by
    averaging across models with different parametric structures and different
    ways of detecting regimes, idiosyncratic errors tend to cancel (Bates &
    Granger, 1969; Timmermann, 2006).  Under the Lucas Critique, forecast
    combination is particularly valuable because the regime-switching models
    may each misclassify the post-break regime *in different ways*, so their
    errors are partially uncorrelated and the average is more stable.

    This class has two usage patterns:

    1. **Pass pre-fitted models** (no fit step needed):
       ``ensemble = ModelAverageEnsemble(fitted_models); preds = ensemble.predict(df)``

    2. **Fit from scratch**:
       ``ensemble = ModelAverageEnsemble(unfitted_models); ensemble.fit(df_train)``

    Parameters
    ----------
    models : dict[str, model]
        Dictionary of ``{name: model}`` instances.  Each model must implement
        ``.predict(df) -> np.ndarray`` and optionally
        ``.predict_regimes(df) -> np.ndarray``.
    weights : array-like of float, optional
        Weights for each model.  If None, equal weights (1/M each) are used.
        Weights are normalised to sum to 1.

    References
    ----------
    - Bates, J. M. & Granger, C. W. J. (1969). The combination of forecasts.
      *Operational Research Quarterly*, 20(4), 451–468.
    - Timmermann, A. (2006). Forecast combinations. In Elliott, G., Granger,
      C. W. J. & Timmermann, A. (Eds.), *Handbook of Economic Forecasting*.
    """

    def __init__(
        self,
        models: dict,
        weights: "list[float] | np.ndarray | None" = None,
    ) -> None:
        self.models = models
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            self._weights = w / w.sum()
        else:
            self._weights = None  # equal weights computed at predict time

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: "pd.DataFrame") -> "ModelAverageEnsemble":
        """Fit all constituent models on the training DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Training data; same format required by all constituent models.

        Returns
        -------
        self
        """
        import warnings
        for name, model in self.models.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(df)
            except Exception as exc:
                print(f"  [ModelAverageEnsemble] {name} fit failed: {exc}")
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, df: "pd.DataFrame") -> "np.ndarray":
        """Weighted average of all model one-step-ahead forecasts.

        Models that fail to predict are silently excluded from the average.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        np.ndarray, shape (n,)
        """
        import warnings
        preds = []
        for name, model in self.models.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = model.predict(df)
                preds.append(np.asarray(p, dtype=float))
            except Exception:
                pass

        if not preds:
            return np.zeros(len(df))

        stack = np.vstack(preds)  # shape (M, n)

        if self._weights is not None and len(self._weights) == len(preds):
            w = self._weights[:, None]
        else:
            w = np.ones((len(preds), 1)) / len(preds)

        return (w * stack).sum(axis=0)

    def predict_regimes(self, df: "pd.DataFrame") -> "np.ndarray":
        """Majority-vote regime assignment across all constituent models.

        For each time step, returns the mode of the regime predictions.
        Models that do not implement ``predict_regimes`` (or fail) contribute 0.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        np.ndarray of int, shape (n,)
        """
        import warnings
        from scipy import stats as _stats

        regime_preds = []
        for model in self.models.values():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = model.predict_regimes(df)
                regime_preds.append(np.asarray(r, dtype=int))
            except Exception:
                regime_preds.append(np.zeros(len(df), dtype=int))

        if not regime_preds:
            return np.zeros(len(df), dtype=int)

        stack = np.vstack(regime_preds)  # shape (M, n)
        mode_result = _stats.mode(stack, axis=0, keepdims=False)
        return mode_result.mode.astype(int)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        members = ", ".join(self.models.keys())
        n = len(self.models)
        w = f"equal (1/{n})" if self._weights is None else str(self._weights.round(3))
        return f"ModelAverageEnsemble: {n} models [{members}]  weights={w}"
