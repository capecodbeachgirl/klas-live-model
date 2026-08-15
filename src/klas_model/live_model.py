from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .conservative_predictive import (
    _feature_frame,
    _fit_ridge,
    _prepare,
    apply_conservative_adjustment,
    year_split_backtest_checkpoint,
)


def fit_live_bundle(daily: pd.DataFrame, hour: int) -> dict[str, Any]:
    """Fit a live model using fixed validation-selected conservatism.

    Hyperparameters are selected without using the 2026 test year. After that selection,
    coefficients are refit on every completed historical row available in `daily`, which is
    appropriate for a live model because those outcomes are already known.
    """
    backtest = year_split_backtest_checkpoint(daily, hour)
    params = backtest.params
    work, X, actual, nws = _prepare(daily, hour)
    target = actual - nws
    pipe, medians = _fit_ridge(X, target, params.alpha)
    metrics = dict(backtest.metrics)
    errors = backtest.predictions["model_error_f"].dropna().astype(float).tolist()
    trained_through = pd.to_datetime(work["date"], errors="coerce").max()
    return {
        "checkpoint_hour": int(hour),
        "pipeline": pipe,
        "medians": medians,
        "params": asdict(params),
        "test_metrics": metrics,
        "calibration_model_errors_f": errors,
        "trained_through": trained_through.date().isoformat() if pd.notna(trained_through) else None,
    }


def save_live_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_live_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def predict_from_bundle(bundle: dict[str, Any], current_daily: pd.DataFrame) -> dict[str, float]:
    hour = int(bundle["checkpoint_hour"])
    X = _feature_frame(current_daily.copy(), hour)
    if X.empty or pd.to_numeric(X["temp_now"], errors="coerce").isna().all():
        raise ValueError(f"current data is missing a usable {hour}:00 checkpoint")
    medians: pd.Series = bundle["medians"]
    raw_residual = float(bundle["pipeline"].predict(X.fillna(medians).fillna(0.0))[0])
    nws = float(pd.to_numeric(current_daily["nws_am_forecast_high_f"], errors="coerce").iloc[0])
    p = bundle["params"]
    pred = float(
        apply_conservative_adjustment(
            np.asarray([nws]),
            np.asarray([raw_residual]),
            shrink=float(p["shrink"]),
            gate_f=float(p["gate_f"]),
            cap_f=float(p["cap_f"]),
        )[0]
    )
    return {
        "nws_high_f": nws,
        "raw_model_residual_f": raw_residual,
        "applied_correction_f": pred - nws,
        "model_predicted_high_f": pred,
    }
