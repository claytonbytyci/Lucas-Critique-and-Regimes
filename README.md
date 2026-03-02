# Lucas Critique and Regime-Switching Models

## Overview

This project investigates a fundamental tension in macroeconometrics: the **Lucas Critique** (Lucas, 1976) applied to **regime-switching models**.

The Lucas Critique argues that econometric models estimated on historical data become unreliable when policies change, because rational agents alter their behaviour in response to new regimes. This is particularly acute for regime-switching models, which explicitly learn regime parameters from data — parameters that may shift after a structural break.

### Research Question

> *Do machine-learning regime-switching models exhibit greater structural stability than classical Markov-switching regression models in the presence of Lucas-critique-type parameter shifts?*

## Key Models

| Model | Family | Library |
|---|---|---|
| Markov Switching Regression | Classical | `statsmodels` |
| Hidden Markov Model (HMM) | Semi-classical | `hmmlearn` |
| Threshold Autoregression (TAR) | Classical nonlinear | `scipy` / `numpy` |
| ML Regime Switcher (XGBoost + Clustering) | ML | `scikit-learn`, `xgboost` |
| Mixture of Experts | ML | `scikit-learn` |

## Experimental Design

1. **Simulate** a Markov-switching AR(1) process with known regimes and parameters.
2. **Apply a Lucas shift**: after a structural break, alter the DGP parameters (simulating a policy change that rational agents respond to).
3. **Train** all models on pre-break data.
4. **Evaluate** all models on post-break data.
5. **Compare** performance degradation — the Lucas Sensitivity Ratio (LSR).

## Project Structure

```
src/
├── simulation/   # DGP and structural break utilities
├── models/       # Classical and ML regime-switching models
├── evaluation/   # Metrics, Lucas-critique tests, visualisation
└── pipeline/     # Experiment orchestration

analyses/
├── 01_simulation_eda.ipynb        # DGP exploration
├── 02_model_comparison.ipynb      # In-sample regime recovery
└── 03_lucas_critique_analysis.ipynb  # Core Lucas critique experiment

tests/            # Unit tests for all modules
data/simulated/   # Parquet outputs from DGP
```

## Setup

```bash
conda env create -f environment.yml
conda activate lucas-regimes
pip install -e .[dev]
```

## Usage

```python
from simulation import MarkovSwitchingDGP, apply_lucas_shift
from models import MarkovSwitchingModel, HMMRegimeModel, ThresholdModel, MLRegimeModel
from pipeline import LucasCritiqueExperiment

dgp = MarkovSwitchingDGP(n_regimes=2, seed=42)
experiment = LucasCritiqueExperiment(dgp=dgp, break_fraction=0.6)
results = experiment.run()
```

## References

- Lucas, R. E. (1976). Econometric policy evaluation: A critique. *Carnegie-Rochester Conference Series on Public Policy*, 1, 19–46.
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series. *Econometrica*, 57(2), 357–384.
- Teräsvirta, T. (1994). Specification, estimation, and evaluation of smooth transition autoregressive models. *Journal of the American Statistical Association*, 89(425), 208–218.
