from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .predictive import DEFAULT_MODEL_CHECKPOINTS


@dataclass(frozen=True)
class ConservativeParams:
    alpha: float
    shrink: float
    gate_f: float
    cap_f: float


@dataclass(frozen=True)
class YearSplitResult:
    checkpoint_hour: int
    predictions: pd.DataFrame
    metrics: dict[str, float | str]
    params: ConservativeParams


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _feature_frame(daily: pd.DataFrame, hour: int) -> pd.DataFrame:
    """Features knowable by the checkpoint, designed around an NWS correction."""
    h = f"h{hour:02d}"
    temp = _num(daily[f"{h}_temp_f"])
    nws = _num(daily["nws_am_forecast_high_f"])
    dew = _num(daily.get(f"{h}_dewpoint_f", pd.Series(index=daily.index, dtype=float)))
    cloud = _num(daily.get(f"{h}_cloud_fraction", pd.Series(index=daily.index, dtype=float)))
    wind = _num(daily.get(f"{h}_wind_speed_kt", pd.Series(index=daily.index, dtype=float)))

    data: dict[str, pd.Series | np.ndarray] = {
        "nws_high": nws,
        "temp_now": temp,
        "nws_remaining": nws - temp,
        "dewpoint_now": dew,
        "dewpoint_depression": temp - dew,
        "cloud_now": cloud,
        "wind_now": wind,
    }

    for lag in (1, 2, 3):
        prev = hour - lag
        prev_temp_col = f"h{prev:02d}_temp_f"
        if prev >= 8 and prev_temp_col in daily.columns:
            prev_temp = _num(daily[prev_temp_col])
            data[f"temp_change_{lag}h"] = temp - prev_temp
            prev_cloud_col = f"h{prev:02d}_cloud_fraction"
            if prev_cloud_col in daily.columns:
                data[f"cloud_change_{lag}h"] = cloud - _num(daily[prev_cloud_col])
            prev_wind_col = f"h{prev:02d}_wind_speed_kt"
            if prev_wind_col in daily.columns:
                data[f"wind_change_{lag}h"] = wind - _num(daily[prev_wind_col])

    dates = pd.to_datetime(daily["date"], errors="coerce")
    doy = dates.dt.dayofyear.astype(float)
    data["season_sin"] = np.sin(2 * pi * doy / 365.25)
    data["season_cos"] = np.cos(2 * pi * doy / 365.25)
    return pd.DataFrame(data, index=daily.index)


def _fit_ridge(X: pd.DataFrame, y: pd.Series, alpha: float) -> tuple[Pipeline, pd.Series]:
    medians = X.median(numeric_only=True)
    filled = X.fillna(medians).fillna(0.0)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    pipe.fit(filled, y)
    return pipe, medians


def apply_conservative_adjustment(
    nws: np.ndarray | pd.Series,
    raw_residual: np.ndarray | pd.Series,
    *,
    shrink: float,
    gate_f: float,
    cap_f: float,
) -> np.ndarray:
    raw = np.asarray(raw_residual, dtype=float)
    adjusted = raw * float(shrink)
    adjusted[np.abs(adjusted) < float(gate_f)] = 0.0
    adjusted = np.clip(adjusted, -float(cap_f), float(cap_f))
    return np.asarray(nws, dtype=float) + adjusted


def _mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))


def _metrics(pred: np.ndarray, actual: np.ndarray, nws: np.ndarray) -> dict[str, float]:
    err = pred - actual
    nws_err = nws - actual
    ae = np.abs(err)
    nws_ae = np.abs(nws_err)
    return {
        "n": float(len(actual)),
        "mae_f": float(ae.mean()),
        "nws_mae_f": float(nws_ae.mean()),
        "mae_improvement_f": float(nws_ae.mean() - ae.mean()),
        "bias_f": float(err.mean()),
        "nws_bias_f": float(nws_err.mean()),
        "exact_pct": float((ae < 0.5).mean()),
        "within_1f_pct": float((ae <= 1.0).mean()),
        "within_2f_pct": float((ae <= 2.0).mean()),
        "within_3f_pct": float((ae <= 3.0).mean()),
        "nws_within_1f_pct": float((nws_ae <= 1.0).mean()),
        "nws_within_2f_pct": float((nws_ae <= 2.0).mean()),
    }


def _prepare(daily: pd.DataFrame, hour: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    X = _feature_frame(work, hour)
    actual = _num(work["actual_cli_high_f"])
    nws = _num(work["nws_am_forecast_high_f"])
    valid = work["date"].notna() & actual.notna() & nws.notna() & X["temp_now"].notna()
    return (
        work.loc[valid].reset_index(drop=True),
        X.loc[valid].reset_index(drop=True),
        actual.loc[valid].reset_index(drop=True),
        nws.loc[valid].reset_index(drop=True),
    )


def year_split_backtest_checkpoint(
    daily: pd.DataFrame,
    hour: int,
    *,
    train_years: Iterable[int] = (2022, 2023, 2024),
    validation_year: int = 2025,
    test_year: int = 2026,
) -> YearSplitResult:
    if hour not in DEFAULT_MODEL_CHECKPOINTS:
        raise ValueError(f"unsupported checkpoint hour: {hour}")

    work, X, actual, nws = _prepare(daily, hour)
    years = work["date"].dt.year
    train_mask = years.isin(list(train_years))
    val_mask = years.eq(validation_year)
    test_mask = years.eq(test_year)
    if train_mask.sum() < 60:
        raise ValueError(f"not enough training rows at hour {hour}: {int(train_mask.sum())}")
    if val_mask.sum() < 20:
        raise ValueError(f"not enough validation rows at hour {hour}: {int(val_mask.sum())}")
    if test_mask.sum() < 10:
        raise ValueError(f"not enough test rows at hour {hour}: {int(test_mask.sum())}")

    target = actual - nws
    alphas = (2.0, 8.0, 20.0, 50.0)
    shrinks = (0.0, 0.25, 0.5, 0.75, 1.0)
    gates = (0.0, 0.25, 0.5, 0.75, 1.0)
    caps = (1.0, 1.5, 2.0, 3.0)

    best_score = float("inf")
    best_params = ConservativeParams(alpha=8.0, shrink=0.0, gate_f=0.0, cap_f=1.0)

    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train = target.loc[train_mask].reset_index(drop=True)
    X_val = X.loc[val_mask].reset_index(drop=True)
    actual_val = actual.loc[val_mask].to_numpy(dtype=float)
    nws_val = nws.loc[val_mask].to_numpy(dtype=float)

    for alpha in alphas:
        pipe, medians = _fit_ridge(X_train, y_train, alpha)
        raw_val = pipe.predict(X_val.fillna(medians).fillna(0.0))
        for shrink in shrinks:
            for gate in gates:
                for cap in caps:
                    pred_val = apply_conservative_adjustment(
                        nws_val, raw_val, shrink=shrink, gate_f=gate, cap_f=cap
                    )
                    score = _mae(pred_val, actual_val)
                    # Tiny complexity penalty breaks near-ties toward smaller adjustments.
                    score += 1e-5 * (shrink + cap / 10.0 - gate / 10.0)
                    if score < best_score:
                        best_score = score
                        best_params = ConservativeParams(alpha, shrink, gate, cap)

    nws_val_mae = _mae(nws_val, actual_val)
    # Conservative safety rule: if validation tuning cannot beat NWS by at least 0.02F,
    # use zero correction rather than force a marginal adjustment.
    if best_score >= nws_val_mae - 0.02:
        best_params = ConservativeParams(alpha=best_params.alpha, shrink=0.0, gate_f=0.0, cap_f=1.0)

    trainval_mask = train_mask | val_mask
    pipe, medians = _fit_ridge(
        X.loc[trainval_mask].reset_index(drop=True),
        target.loc[trainval_mask].reset_index(drop=True),
        best_params.alpha,
    )
    X_test = X.loc[test_mask].reset_index(drop=True)
    actual_test = actual.loc[test_mask].to_numpy(dtype=float)
    nws_test = nws.loc[test_mask].to_numpy(dtype=float)
    raw_test = pipe.predict(X_test.fillna(medians).fillna(0.0))
    pred_test = apply_conservative_adjustment(
        nws_test,
        raw_test,
        shrink=best_params.shrink,
        gate_f=best_params.gate_f,
        cap_f=best_params.cap_f,
    )
    correction = pred_test - nws_test

    metrics = _metrics(pred_test, actual_test, nws_test)
    metrics.update({
        "validation_model_mae_f": float(best_score),
        "validation_nws_mae_f": float(nws_val_mae),
        "validation_improvement_f": float(nws_val_mae - best_score),
        "train_years": ",".join(map(str, train_years)),
        "validation_year": str(validation_year),
        "test_year": str(test_year),
    })

    test_dates = work.loc[test_mask, "date"].dt.date.astype(str).to_numpy()
    predictions = pd.DataFrame({
        "date": test_dates,
        "checkpoint_hour": hour,
        "nws_am_forecast_high_f": nws_test,
        "actual_cli_high_f": actual_test,
        "raw_model_residual_f": raw_test,
        "applied_correction_f": correction,
        "model_predicted_high_f": pred_test,
    })
    predictions["nws_error_f"] = predictions["nws_am_forecast_high_f"] - predictions["actual_cli_high_f"]
    predictions["model_error_f"] = predictions["model_predicted_high_f"] - predictions["actual_cli_high_f"]
    predictions["model_abs_error_f"] = predictions["model_error_f"].abs()

    return YearSplitResult(hour, predictions, metrics, best_params)


def year_split_backtest_all(
    daily: pd.DataFrame,
    checkpoints: Iterable[int] = DEFAULT_MODEL_CHECKPOINTS,
    *,
    train_years: Iterable[int] = (2022, 2023, 2024),
    validation_year: int = 2025,
    test_year: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for hour in checkpoints:
        result = year_split_backtest_checkpoint(
            daily,
            hour,
            train_years=train_years,
            validation_year=validation_year,
            test_year=test_year,
        )
        preds.append(result.predictions)
        p = result.params
        rows.append({
            "checkpoint_hour": hour,
            **result.metrics,
            "alpha": p.alpha,
            "shrink": p.shrink,
            "gate_f": p.gate_f,
            "cap_f": p.cap_f,
        })
    return pd.concat(preds, ignore_index=True), pd.DataFrame(rows)
