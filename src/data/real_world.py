"""
Real-world data loading utilities for the Lucas Critique project.

Downloads FRED macroeconomic series and formats them to match the
DataFrame schema expected by the regime-switching models:

    Columns: y, y_lag1, y_lag2, roll_mean_5, roll_std_5,
             roll_mean_20, roll_std_20, exog_0..2, regime

Two canonical datasets are provided, each with a well-documented
structural break point (cited in my project work and which I test 
for in the notebooks with a CUSUM test, for example).

  1. US Industrial Production growth — Great Moderation break (Jan 1984)
     Pre-break:  1960-01 to 1983-12  (~288 monthly obs)
     Post-break: 1984-01 to 2007-07  (~283 monthly obs, stops before GFC)

  2. US CPI Inflation — Volcker disinflation break (Oct 1979)
     Pre-break:  1960-01 to 1979-09  (~237 monthly obs)
     Post-break: 1979-10 to 1999-09  (~240 monthly obs)
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "real_world"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

# FRED download helper

def fetch_fred(series_id: str, start: str = "1948-01-01", cache: bool = True) -> pd.Series:
    """Download a FRED series, optionally using a local parquet cache.

    Returns a pd.Series with a DatetimeIndex.
    """
    cache_path = _DATA_DIR / f"{series_id}.parquet"
    if cache and cache_path.exists():
        s = pd.read_parquet(cache_path).iloc[:, 0]
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(
            io.StringIO(r.text),
            parse_dates=["observation_date"],
            index_col="observation_date",
        )
        df.columns = [series_id]
        df.to_parquet(cache_path)
        s = df.iloc[:, 0]

    s.index = pd.DatetimeIndex(s.index)
    return s[s.index >= start].rename(series_id)

# Feature builder (preserves DatetimeIndex)

def build_features(
    y: pd.Series,
    exog_series: list[pd.Series] | None = None,
    regime_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the model-ready DataFrame from a target series and covariates.
    The covariates are reindexed to match y's index.

    The returned DataFrame preserves the DatetimeIndex of ``y`` so that
    callers can slice on dates before calling reset_index.
    Parameters:
    y : pd.Series
        Stationary target series (e.g. log-differenced GDP, % CPI change).
        Index must be a DatetimeIndex.
    exog_series : list of pd.Series, optional
        Additional covariates, aligned to y's index.
    regime_series : pd.Series, optional
        Binary 0/1 series (e.g. NBER USREC).

    Returns:
    pd.DataFrame  (DatetimeIndex, NaN rows from lags/rolling dropped)
    """
    df = pd.DataFrame({"y": y})

    df["y_lag1"] = df["y"].shift(1) # these form my vector for use in regime identification, explained in the report
    df["y_lag2"] = df["y"].shift(2)
    df["roll_mean_5"]  = df["y"].rolling(5).mean()
    df["roll_std_5"]   = df["y"].rolling(5).std()
    df["roll_mean_20"] = df["y"].rolling(20).mean()
    df["roll_std_20"]  = df["y"].rolling(20).std()

    if exog_series:
        for i, s in enumerate(exog_series[:3]):
            df[f"exog_{i}"] = s.reindex(df.index).ffill()
        for i in range(len(exog_series), 3):
            df[f"exog_{i}"] = 0.0
    else:
        for i in range(3):
            df[f"exog_{i}"] = 0.0

    if regime_series is not None:
        df["regime"] = regime_series.reindex(df.index).ffill().fillna(0).astype(int)
    else:
        df["regime"] = 0

    return df.dropna()

# Dataset 1: Industrial Production — Great Moderation break

def load_industrial_production() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Industrial Production (monthly log-growth), Great Moderation break.

    Structural break: January 1984.
    Pre-break sample:  1960-01 to 1983-12  (~288 obs)
    Post-break sample: 1984-01 to 2007-07  (~283 obs, stops before GFC)
    Regime labels:     NBER USREC (0=expansion, 1=recession)

    Returns a dataframe with columns of the generated covariates and metadata for the break.
    """
    ip     = fetch_fred("INDPRO",    "1960-01-01")
    usrec  = fetch_fred("USREC",     "1960-01-01")
    unrate = fetch_fred("UNRATE",    "1960-01-01")
    cpi    = fetch_fred("CPIAUCSL",  "1960-01-01")

    y        = np.log(ip).diff() * 100
    d_unrate = unrate.diff()
    cpi_inf  = np.log(cpi).diff() * 100

    df_all = build_features(y, exog_series=[d_unrate, cpi_inf, usrec], regime_series=usrec)

    df_pre  = df_all[df_all.index < "1984-01-01"].copy().reset_index(drop=True)
    df_post = df_all[
        (df_all.index >= "1984-01-01") & (df_all.index <= "2007-07-01")
    ].copy().reset_index(drop=True)

    meta = {
        "name": "Industrial Production (Great Moderation break)",
        "series": "INDPRO",
        "break": "1984-01 (Great Moderation)",
        "pre_period": "1960-01 to 1983-12",
        "post_period": "1984-01 to 2007-07",
        "n_pre": len(df_pre),
        "n_post": len(df_post),
        "regime_label": "NBER USREC",
        "y_description": "Monthly log-growth of Industrial Production (%)",
    }
    return df_pre, df_post, meta


# ---------------------------------------------------------------------------
# Dataset 2: CPI Inflation — Volcker break
# ---------------------------------------------------------------------------

def load_cpi_volcker() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """CPI Inflation (monthly, annualised), Volcker disinflation break.

    Structural break: October 1979 (Volcker changes Fed operating procedure).
    Pre-break sample:  1960-01 to 1979-09  (~237 obs)
    Post-break sample: 1979-10 to 1999-09  (~240 obs)
    Regime labels:     NBER USREC

    Returns:
    df_pre, df_post : pd.DataFrame  (integer-indexed)
    meta : dict
    """
    cpi      = fetch_fred("CPIAUCSL",  "1960-01-01")
    fedfunds = fetch_fred("FEDFUNDS",  "1960-01-01")
    usrec    = fetch_fred("USREC",     "1960-01-01")
    unrate   = fetch_fred("UNRATE",    "1960-01-01")

    y          = np.log(cpi).diff() * 100 * 12   # annualised monthly log-change
    d_fedfunds = fedfunds.diff()
    d_unrate   = unrate.diff()

    df_all = build_features(y, exog_series=[d_fedfunds, d_unrate, usrec], regime_series=usrec)

    df_pre  = df_all[df_all.index < "1979-10-01"].copy().reset_index(drop=True)
    df_post = df_all[
        (df_all.index >= "1979-10-01") & (df_all.index <= "1999-09-01")
    ].copy().reset_index(drop=True)

    meta = {
        "name": "CPI Inflation (Volcker break)",
        "series": "CPIAUCSL",
        "break": "1979-10 (Volcker disinflation)",
        "pre_period": "1960-01 to 1979-09",
        "post_period": "1979-10 to 1999-09",
        "n_pre": len(df_pre),
        "n_post": len(df_post),
        "regime_label": "NBER USREC",
        "y_description": "Annualised monthly CPI log-growth rate (%)",
    }
    return df_pre, df_post, meta

# Combined loader

def load_all_datasets() -> list[tuple[pd.DataFrame, pd.DataFrame, dict]]:
    """Return both real-world datasets as a list of (df_pre, df_post, meta)."""
    return [
        load_industrial_production(),
        load_cpi_volcker(),
    ]
