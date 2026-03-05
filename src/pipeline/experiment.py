"""Full Lucas-critique experiment pipeline.

Workflow
--------
1. Simulate a pre-break sample from a Markov-switching DGP.
2. Apply a Lucas shift to get a post-break DGP.
3. Simulate a post-break sample.
4. Train all regime models on pre-break data only.
5. Evaluate each model on both periods.
6. Compute regime accuracy, forecast RMSE, and Lucas Sensitivity Ratios.
7. Optionally run a Chow test and CUSUM on the concatenated series.

This script is also directly executable::

    python -m pipeline.experiment

which saves results to ``data/simulated/`` and figures to ``analyses/figures/``.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure src is on path when run directly
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from simulation import MarkovSwitchingDGP, simulate_pre_post_break, MILD_SHIFT, LucasShift
from models import (
    ARMAModel,
    ARModel,
    HMMRegimeModel,
    MLRegimeModel,
    MixtureOfExpertsModel,
    ThresholdModel,
    MarkovSwitchingNeuralNetwork,
)
from evaluation.metrics import (
    forecast_rmse,
    forecast_mae,
    directional_accuracy,
    regime_accuracy,
    adjusted_rand_regime,
    regime_conditional_rmse,
    lucas_sensitivity_ratio,
    summarise_model_performance,
)
from evaluation.lucas_critique import (
    chow_test,
    compute_rolling_performance,
    compare_pre_post_performance,
)

try:
    from models import MarkovSwitchingModel
    _MSM_AVAILABLE = True
except ImportError:
    _MSM_AVAILABLE = False

_RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "simulated"
_FIGURES_DIR = Path(__file__).resolve().parents[2] / "analyses" / "figures"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ModelEvaluation:
    """Per-model evaluation results."""
    name: str
    pre_rmse: float
    post_rmse: float
    pre_mae: float
    post_mae: float
    pre_dir_acc: float
    post_dir_acc: float
    pre_regime_acc: float
    post_regime_acc: float
    pre_ari: float
    post_ari: float
    lsr: float = field(init=False)

    def __post_init__(self) -> None:
        self.lsr = lucas_sensitivity_ratio(self.pre_rmse, self.post_rmse)

    def to_dict(self) -> dict:
        return {
            "model": self.name,
            "pre_rmse": self.pre_rmse,
            "post_rmse": self.post_rmse,
            "pre_mae": self.pre_mae,
            "post_mae": self.post_mae,
            "pre_dir_acc": self.pre_dir_acc,
            "post_dir_acc": self.post_dir_acc,
            "pre_regime_acc": self.pre_regime_acc,
            "post_regime_acc": self.post_regime_acc,
            "pre_ari": self.pre_ari,
            "post_ari": self.post_ari,
            "LSR": self.lsr,
        }


@dataclass
class ExperimentResult:
    """Full experiment output."""
    df_pre: pd.DataFrame
    df_post: pd.DataFrame
    evaluations: list[ModelEvaluation]
    chow: dict | None = None
    summary: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.summary = pd.DataFrame([e.to_dict() for e in self.evaluations])

    def best_model(self, metric: str = "LSR") -> str:
        """Return the model name with the lowest LSR (or highest, for accuracies)."""
        ascending = metric != "pre_regime_acc"
        return str(self.summary.sort_values(metric, ascending=ascending).iloc[0]["model"])


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------


class LucasCritiqueExperiment:
    """Orchestrates the full Lucas-critique simulation experiment.

    Parameters
    ----------
    dgp : MarkovSwitchingDGP or None
        Pre-break DGP.  If None, uses the default two-regime specification.
    shift : LucasShift or None
        Parameter shift to apply.  If None, uses MILD_SHIFT.
    n_pre : int
        Pre-break sample size.
    n_post : int
        Post-break sample size.
    include_msm : bool
        Whether to include the statsmodels Markov-switching model.
        Requires statsmodels.
    run_chow : bool
        Whether to run the Chow structural break test.
    """

    def __init__(
        self,
        dgp: MarkovSwitchingDGP | None = None,
        shift: LucasShift | None = None,
        n_pre: int = 400,
        n_post: int = 200,
        include_msm: bool = True,
        run_chow: bool = True,
    ) -> None:
        self.dgp = dgp if dgp is not None else MarkovSwitchingDGP()
        self.shift = shift if shift is not None else MILD_SHIFT
        self.n_pre = n_pre
        self.n_post = n_post
        self.include_msm = include_msm and _MSM_AVAILABLE
        self.run_chow = run_chow

    # ------------------------------------------------------------------
    # Build model registry
    # ------------------------------------------------------------------

    def _build_models(self) -> dict:
        models: dict = {}

        if self.include_msm:
            try:
                from models import MarkovSwitchingModel
                models["Markov Switching (MSM)"] = MarkovSwitchingModel(
                    k_regimes=2, switching_variance=True
                )
            except Exception:
                pass

        # --- Classical parametric baselines (no regime switching) ---
        models["AR(2) Baseline"] = ARModel(order=2, include_exog=True)
        models["ARMA(2,1) Baseline"] = ARMAModel(p=2, q=1, trend="c")

        # --- Regime-switching models ---
        models["HMM"] = HMMRegimeModel(n_components=2, random_state=42)
        models["Threshold (TAR)"] = ThresholdModel()
        models["ML Regime (XGB)"] = MLRegimeModel(n_regimes=2)
        models["Mixture of Experts"] = MixtureOfExpertsModel(n_experts=2)
        models["MSNN"] = MarkovSwitchingNeuralNetwork(
            k_regimes=2,
            hidden_layer_sizes=(32, 16),
            n_iter=50,
            mlp_epochs=200,
            random_state=42,
        )

        return models

    # ------------------------------------------------------------------
    # Evaluate one model
    # ------------------------------------------------------------------

    def _evaluate_model(
        self,
        name: str,
        model: object,
        df_train: pd.DataFrame,
        df_pre: pd.DataFrame,
        df_post: pd.DataFrame,
        pred_store: dict | None = None,
    ) -> ModelEvaluation | None:
        """Fit on train, evaluate on pre and post.

        Parameters
        ----------
        pred_store : dict, optional
            If provided, raw predictions are stored as
            ``pred_store[name] = (pred_pre, pred_post)`` for downstream
            ensemble computation.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(df_train)
        except Exception as exc:
            print(f"  [WARN] {name} fit failed: {exc}")
            return None

        def _safe_predict(df: pd.DataFrame) -> np.ndarray | None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return model.predict(df)
            except Exception:
                return None

        def _safe_regimes(df: pd.DataFrame) -> np.ndarray | None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return model.predict_regimes(df)
            except Exception:
                return None

        # Forecast evaluation
        pred_pre = _safe_predict(df_pre)
        pred_post = _safe_predict(df_post)

        if pred_pre is None or pred_post is None:
            print(f"  [WARN] {name} prediction failed.")
            return None

        # Store raw predictions for ensemble computation
        if pred_store is not None:
            pred_store[name] = (pred_pre, pred_post)

        # Regime evaluation
        reg_pre = _safe_regimes(df_pre)
        reg_post = _safe_regimes(df_post)
        true_reg_pre = df_pre["regime"].to_numpy()
        true_reg_post = df_post["regime"].to_numpy()

        pre_racc = regime_accuracy(true_reg_pre, reg_pre) if reg_pre is not None else np.nan
        post_racc = regime_accuracy(true_reg_post, reg_post) if reg_post is not None else np.nan
        pre_ari = adjusted_rand_regime(true_reg_pre, reg_pre) if reg_pre is not None else np.nan
        post_ari = adjusted_rand_regime(true_reg_post, reg_post) if reg_post is not None else np.nan

        y_pre = df_pre["y"].to_numpy()
        y_post = df_post["y"].to_numpy()

        return ModelEvaluation(
            name=name,
            pre_rmse=forecast_rmse(y_pre, pred_pre),
            post_rmse=forecast_rmse(y_post, pred_post),
            pre_mae=forecast_mae(y_pre, pred_pre),
            post_mae=forecast_mae(y_post, pred_post),
            pre_dir_acc=directional_accuracy(y_pre, pred_pre),
            post_dir_acc=directional_accuracy(y_post, pred_post),
            pre_regime_acc=float(pre_racc),
            post_regime_acc=float(post_racc),
            pre_ari=float(pre_ari),
            post_ari=float(post_ari),
        )

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> ExperimentResult:
        """Execute the full experiment.

        Parameters
        ----------
        verbose : bool
            If True, print progress to stdout.

        Returns
        -------
        ExperimentResult
        """
        if verbose:
            print("=" * 60)
            print("Lucas Critique Experiment")
            print("=" * 60)
            print(f"  DGP: {self.dgp}")
            print(f"  n_pre={self.n_pre}, n_post={self.n_post}")
            print()

        # --- Step 1: Simulate data ---
        if verbose:
            print("[1] Simulating pre-break and post-break samples ...")
        df_pre, df_post, post_dgp = simulate_pre_post_break(
            self.dgp, self.shift, n_pre=self.n_pre, n_post=self.n_post
        )

        if verbose:
            pre_regime_counts = df_pre["regime"].value_counts().to_dict()
            post_regime_counts = df_post["regime"].value_counts().to_dict()
            print(f"    Pre-break regime counts : {pre_regime_counts}")
            print(f"    Post-break regime counts: {post_regime_counts}")
            print()

        # --- Step 2: Build models ---
        models = self._build_models()

        # --- Step 3: Fit and evaluate ---
        if verbose:
            print("[2] Training and evaluating models on pre-break data ...")
        evaluations: list[ModelEvaluation] = []
        pred_store: dict = {}  # stores (pred_pre, pred_post) per model for ensemble
        for name, model in models.items():
            if verbose:
                print(f"  Fitting {name} ...")
            result = self._evaluate_model(name, model, df_pre, df_pre, df_post,
                                          pred_store=pred_store)
            if result is not None:
                evaluations.append(result)
                if verbose:
                    print(
                        f"    pre_rmse={result.pre_rmse:.4f}  "
                        f"post_rmse={result.post_rmse:.4f}  "
                        f"LSR={result.lsr:.3f}"
                    )

        # --- Step 3b: Model Average Ensemble ---
        if len(pred_store) >= 2:
            y_pre_arr = df_pre["y"].to_numpy()
            y_post_arr = df_post["y"].to_numpy()
            avg_pred_pre = np.mean(
                np.vstack([v[0] for v in pred_store.values()]), axis=0
            )
            avg_pred_post = np.mean(
                np.vstack([v[1] for v in pred_store.values()]), axis=0
            )
            # Regime: majority-vote across all stored regime preds (approximate as 0s
            # since pred_store doesn't hold regimes — use NaN for regime metrics)
            ens_eval = ModelEvaluation(
                name="Model Average",
                pre_rmse=forecast_rmse(y_pre_arr, avg_pred_pre),
                post_rmse=forecast_rmse(y_post_arr, avg_pred_post),
                pre_mae=forecast_mae(y_pre_arr, avg_pred_pre),
                post_mae=forecast_mae(y_post_arr, avg_pred_post),
                pre_dir_acc=directional_accuracy(y_pre_arr, avg_pred_pre),
                post_dir_acc=directional_accuracy(y_post_arr, avg_pred_post),
                pre_regime_acc=float(np.nan),
                post_regime_acc=float(np.nan),
                pre_ari=float(np.nan),
                post_ari=float(np.nan),
            )
            evaluations.append(ens_eval)
            if verbose:
                print(
                    f"  Model Average: pre_rmse={ens_eval.pre_rmse:.4f}  "
                    f"post_rmse={ens_eval.post_rmse:.4f}  LSR={ens_eval.lsr:.3f}"
                )

        # --- Step 4: Chow test ---
        chow_result = None
        if self.run_chow and len(df_pre) > 20:
            if verbose:
                print()
                print("[3] Running Chow structural break test ...")
            from simulation.lucas_shift import concatenate_periods
            df_full = concatenate_periods(df_pre, df_post)
            y_full = df_full["y"].to_numpy()
            X_full = df_full[["y_lag1"]].assign(const=1.0)[["const", "y_lag1"]].to_numpy()
            try:
                chow_result = chow_test(y_full, X_full, break_index=len(df_pre))
                if verbose:
                    print(
                        f"    F={chow_result['F_stat']:.3f}, "
                        f"p={chow_result['p_value']:.4f}, "
                        f"reject H0={chow_result['reject_H0']}"
                    )
            except Exception as exc:
                if verbose:
                    print(f"    Chow test failed: {exc}")

        # --- Step 5: Save results ---
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df_pre.to_parquet(_RESULTS_DIR / "pre_break.parquet", index=False)
        df_post.to_parquet(_RESULTS_DIR / "post_break.parquet", index=False)

        result = ExperimentResult(
            df_pre=df_pre,
            df_post=df_post,
            evaluations=evaluations,
            chow=chow_result,
        )

        if verbose:
            print()
            print("[4] Summary")
            print(result.summary.to_string(index=False))
            print()
            print(f"    Best model (lowest LSR): {result.best_model('LSR')}")

        return result


# ---------------------------------------------------------------------------
# Executable entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exp = LucasCritiqueExperiment(n_pre=500, n_post=200)
    results = exp.run(verbose=True)

    # Save summary CSV
    out_path = _RESULTS_DIR / "experiment_summary.csv"
    results.summary.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
