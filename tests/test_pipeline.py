"""Tests for pipeline.experiment dataclasses: ModelEvaluation and ExperimentResult."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from simulation.dgp import MarkovSwitchingDGP
from pipeline.experiment import ModelEvaluation, ExperimentResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    return MarkovSwitchingDGP(seed=0).simulate(n_obs=100)


def _make_evaluation(name: str = "M", pre_rmse: float = 1.0, post_rmse: float = 2.0) -> ModelEvaluation:
    return ModelEvaluation(
        name=name,
        pre_rmse=pre_rmse,
        post_rmse=post_rmse,
        pre_mae=0.8,
        post_mae=1.5,
        pre_dir_acc=0.6,
        post_dir_acc=0.5,
        pre_regime_acc=0.7,
        post_regime_acc=0.6,
        pre_ari=0.3,
        post_ari=0.2,
    )


# ---------------------------------------------------------------------------
# ModelEvaluation
# ---------------------------------------------------------------------------


class TestModelEvaluation:
    def test_lsr_computed_on_init(self) -> None:
        ev = _make_evaluation(pre_rmse=1.0, post_rmse=2.0)
        assert ev.lsr == pytest.approx(2.0)

    def test_lsr_no_change(self) -> None:
        ev = _make_evaluation(pre_rmse=1.5, post_rmse=1.5)
        assert ev.lsr == pytest.approx(1.0)

    def test_lsr_improvement(self) -> None:
        ev = _make_evaluation(pre_rmse=2.0, post_rmse=1.0)
        assert ev.lsr == pytest.approx(0.5)

    def test_to_dict_has_required_keys(self) -> None:
        d = _make_evaluation().to_dict()
        for key in (
            "model", "pre_rmse", "post_rmse", "pre_mae", "post_mae",
            "pre_dir_acc", "post_dir_acc", "pre_regime_acc", "post_regime_acc",
            "pre_ari", "post_ari", "LSR",
        ):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_model_name(self) -> None:
        d = _make_evaluation(name="TestModel").to_dict()
        assert d["model"] == "TestModel"

    def test_to_dict_lsr_matches_property(self) -> None:
        ev = _make_evaluation(pre_rmse=1.0, post_rmse=3.0)
        assert ev.to_dict()["LSR"] == pytest.approx(ev.lsr)


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


class TestExperimentResult:
    @pytest.fixture
    def result(self, small_df: pd.DataFrame) -> ExperimentResult:
        evals = [
            _make_evaluation("ModelA", pre_rmse=1.0, post_rmse=2.0),
            _make_evaluation("ModelB", pre_rmse=1.2, post_rmse=1.3),
        ]
        return ExperimentResult(df_pre=small_df, df_post=small_df, evaluations=evals)

    def test_summary_is_dataframe(self, result: ExperimentResult) -> None:
        assert isinstance(result.summary, pd.DataFrame)

    def test_summary_has_expected_columns(self, result: ExperimentResult) -> None:
        for col in ("model", "LSR", "pre_rmse", "post_rmse"):
            assert col in result.summary.columns

    def test_summary_has_correct_row_count(self, result: ExperimentResult) -> None:
        assert len(result.summary) == 2

    def test_best_model_returns_string(self, result: ExperimentResult) -> None:
        best = result.best_model("LSR")
        assert isinstance(best, str)

    def test_best_model_by_lsr(self, result: ExperimentResult) -> None:
        # ModelA LSR=2.0, ModelB LSR≈1.083 — ModelB should win (lowest LSR)
        assert result.best_model("LSR") == "ModelB"

    def test_chow_defaults_to_none(self, small_df: pd.DataFrame) -> None:
        result = ExperimentResult(
            df_pre=small_df,
            df_post=small_df,
            evaluations=[_make_evaluation()],
        )
        assert result.chow is None

    def test_chow_stored_when_provided(self, small_df: pd.DataFrame) -> None:
        chow = {"F_stat": 5.0, "p_value": 0.01, "reject_H0": True}
        result = ExperimentResult(
            df_pre=small_df,
            df_post=small_df,
            evaluations=[_make_evaluation()],
            chow=chow,
        )
        assert result.chow is chow
