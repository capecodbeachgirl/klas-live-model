from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_MODEL_CHECKPOINTS = tuple(range(8, 19))


@dataclass(frozen=True)
class CheckpointBacktestResult:
    checkpoint_hour: int
    predictions: pd.DataFrame
    metrics: dict[str, float]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _feature_frame(daily: pd.DataFrame, hour: int) -> pd.DataFrame:
    """Create only features that would be knowable at `hour` local time.

    Important: realized postmortem fields (actual CLI, peak time, cause code, any
    before/after-peak features) are never used as predictors.
    """
    h = f"h{hour:02d}"
    data: dict[str, pd.Series | np.ndarray] = {
        "nws_high": _numeric(daily["nws_am_forecast_high_f"]),
        "temp_now": _numeric(daily[f"{h}_temp_f"]),
        "dewpoint_now": _numeric(daily.get(f"{h}_dewpoint_f", pd.Series(index=daily.index, dtype=float))),
        "cloud_now": _numeric(daily.get(f"{h}_cloud_fraction", pd.Series(index=daily.index, dtype=float))),
        "wind_now": _numeric(daily.get(f"{h}_wind_speed_kt", pd.Series(index=daily.index, dtype=float))),
    }

    # Recent heating rates using only already-observed checkpoints.
    for lag in (1, 2, 3):
        prev = hour - lag
        if prev >= 8 and f"h{prev:02d}_temp_f" in daily.columns:
            data[f"temp_change_{lag}h"] = (
                _numeric(daily[f"{h}_temp_f"]) - _numeric(daily[f"h{prev:02d}_temp_f"])
            )

    dates = pd.to_datetime(daily["date"], errors="coerce")
    doy = dates.dt.dayofyear.astype(float)
    data["season_sin"] = np.sin(2 * pi * doy / 365.25)
    data["season_cos"] = np.cos(2 * pi * doy / 365.25)

    return pd.DataFrame(data, index=daily.index)


def _metrics(pred: pd.Series, actual: pd.Series) -> dict[str, float]:
    err = pred - actual
    ae = err.abs()
    return {
        "n": float(len(err)),
        "mae_f": float(ae.mean()),
        "bias_f": float(err.mean()),
        "exact_pct": float((ae < 0.5).mean()),
        "within_1f_pct": float((ae <= 1.0).mean()),
        "within_2f_pct": float((ae <= 2.0).mean()),
        "within_3f_pct": float((ae <= 3.0).mean()),
        "rmse_f": float(np.sqrt(np.mean(np.square(err)))),
    }


def _expanding_splits(n: int, min_train: int = 45, test_block: int = 15) -> list[tuple[np.ndarray, np.ndarray]]:
    """Chronological expanding-window splits; no future day trains an earlier prediction."""
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    train_end = min_train
    while train_end < n:
        test_end = min(n, train_end + test_block)
        splits.append((np.arange(0, train_end), np.arange(train_end, test_end)))
        train_end = test_end
    return splits


def backtest_checkpoint(
    daily: pd.DataFrame,
    hour: int,
    *,
    min_train: int = 45,
    test_block: int = 15,
    alpha: float = 8.0,
) -> CheckpointBacktestResult:
    """Backtest a transparent Ridge model that predicts the NWS residual.

    Target is `actual_cli_high_f - nws_am_forecast_high_f`. Each held-out block is
    predicted by a model trained only on earlier dates.
    """
    if hour not in DEFAULT_MODEL_CHECKPOINTS:
        raise ValueError(f"unsupported checkpoint hour: {hour}")

    required = {"date", "nws_am_forecast_high_f", "actual_cli_high_f", f"h{hour:02d}_temp_f"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily table missing required columns: {sorted(missing)}")

    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.sort_values("date").reset_index(drop=True)
    X = _feature_frame(work, hour)
    actual = _numeric(work["actual_cli_high_f"])
    nws = _numeric(work["nws_am_forecast_high_f"])
    target = actual - nws

    valid = work["date"].notna() & actual.notna() & nws.notna() & X["temp_now"].notna()
    work = work.loc[valid].reset_index(drop=True)
    X = X.loc[valid].reset_index(drop=True)
    actual = actual.loc[valid].reset_index(drop=True)
    nws = nws.loc[valid].reset_index(drop=True)
    target = target.loc[valid].reset_index(drop=True)

    if len(work) <= min_train:
        raise ValueError(f"need more than {min_train} complete days; found {len(work)}")

    oof = pd.Series(np.nan, index=work.index, dtype=float)
    for train_idx, test_idx in _expanding_splits(len(work), min_train=min_train, test_block=test_block):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        # Median imputation using TRAINING data only.
        train_medians = X.iloc[train_idx].median(numeric_only=True)
        X_train = X.iloc[train_idx].fillna(train_medians).fillna(0.0)
        X_test = X.iloc[test_idx].fillna(train_medians).fillna(0.0)
        pipe.fit(X_train, target.iloc[train_idx])
        residual_pred = pipe.predict(X_test)
        oof.iloc[test_idx] = nws.iloc[test_idx].to_numpy() + residual_pred

    tested = oof.notna()
    pred = oof.loc[tested]
    act = actual.loc[tested]
    nws_test = nws.loc[tested]

    out = pd.DataFrame({
        "date": work.loc[tested, "date"].dt.date.astype(str).to_numpy(),
        "checkpoint_hour": hour,
        "nws_am_forecast_high_f": nws_test.to_numpy(),
        "actual_cli_high_f": act.to_numpy(),
        "model_predicted_high_f": pred.to_numpy(),
    })
    out["nws_error_f"] = out["nws_am_forecast_high_f"] - out["actual_cli_high_f"]
    out["model_error_f"] = out["model_predicted_high_f"] - out["actual_cli_high_f"]
    out["model_abs_error_f"] = out["model_error_f"].abs()

    metrics = _metrics(pred, act)
    nws_metrics = _metrics(nws_test, act)
    metrics.update({f"nws_{k}": v for k, v in nws_metrics.items() if k != "n"})
    metrics["mae_improvement_f"] = nws_metrics["mae_f"] - metrics["mae_f"]
    metrics["mae_improvement_pct"] = (
        100.0 * metrics["mae_improvement_f"] / nws_metrics["mae_f"] if nws_metrics["mae_f"] else 0.0
    )
    return CheckpointBacktestResult(hour, out, metrics)


def backtest_all_checkpoints(
    daily: pd.DataFrame,
    checkpoints: Iterable[int] = DEFAULT_MODEL_CHECKPOINTS,
    *,
    min_train: int = 45,
    test_block: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    rows: list[dict[str, float]] = []
    for hour in checkpoints:
        result = backtest_checkpoint(daily, hour, min_train=min_train, test_block=test_block)
        predictions.append(result.predictions)
        rows.append({"checkpoint_hour": hour, **result.metrics})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(rows)


def fit_checkpoint_model(daily: pd.DataFrame, hour: int, alpha: float = 8.0) -> tuple[Pipeline, pd.Series]:
    """Fit a model on all complete historical rows for future/live use."""
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    X = _feature_frame(work, hour)
    actual = _numeric(work["actual_cli_high_f"])
    nws = _numeric(work["nws_am_forecast_high_f"])
    target = actual - nws
    valid = work["date"].notna() & actual.notna() & nws.notna() & X["temp_now"].notna()
    X = X.loc[valid]
    target = target.loc[valid]
    medians = X.median(numeric_only=True)
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    pipe.fit(X.fillna(medians).fillna(0.0), target)
    return pipe, medians
