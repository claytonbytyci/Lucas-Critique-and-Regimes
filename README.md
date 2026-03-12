# Lucas Critique and Regime-Switching Models

This project examines whether machine-learning regime-switching models are more robust to structural breaks than classical econometric approaches. It operationalises the Lucas Critique (Lucas, 1976) by simulating Markov-switching time series, applying a known parameter shift at a break point, and comparing how eight different models degrade in out-of-sample performance.

The key metric is the **Lucas Sensitivity Ratio (LSR)**: `(post-break RMSE − pre-break RMSE) / pre-break RMSE`. Models are compared across mild and severe structural shifts, and validated on real macroeconomic data via the FRED API.

---

## Setup

### Requirements

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- Python ≥ 3.9

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate lucas-regimes
```

### 2. Install the package in editable mode

```bash
pip install -e ".[dev]"
```

This installs the `src/` layout as an importable package alongside dev tools (`pytest`, `pre-commit`).

## Running

### Jupyter notebooks

```bash
jupyter lab
```

Open notebooks in `analyses/` in order:

| Notebook | Content |
|---|---|
| `01_simulation_eda.ipynb` | DGP properties and regime diagnostics |
| `02_model_comparison.ipynb` | In-sample regime recovery across models |
| `03_lucas_critique_analysis.ipynb` | Core experiment: mild and severe shifts |
| `05_real_world_analysis.ipynb` | Application to CPI and Industrial Production |

### Run tests

```bash
pytest
```

---

## Project structure

```
src/
├── simulation/   # Markov-switching DGP and structural break utilities
├── models/       # Eight regime-switching models (classical, ML, hybrid)
├── evaluation/   # Forecast metrics, Chow test, CUSUM, LSR
└── pipeline/     # Experiment orchestration

analyses/         # Jupyter notebooks and generated figures
data/
├── simulated/    # Parquet and CSV outputs from experiments
└── real_world/   # FRED macroeconomic series
tests/            # Unit tests for all modules
```

