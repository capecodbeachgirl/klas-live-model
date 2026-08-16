from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def remaining_heating_feature_frame(
    df: pd.DataFrame,
    hour: int,
) -> pd.DataFrame:
    if hour not in {16, 17, 18}:
        raise ValueError("remaining-heating model is only validated for 16:00-18:00")

    out = pd.DataFrame(index=df.index)
    current = f"h{hour:02d}"
    temp_cols = [f"h{h:02d}_temp_f" for h in range(8, hour + 1)]

    out["pre_peak_f"] = df[temp_cols].apply(
        pd.to_numeric, errors="coerce"
    ).max(axis=1)
    out["temp_now_f"] = _num(df[f"{current}_temp_f"])

    for lag in (1, 2, 3):
        prev = hour - lag
        out[f"temp_change_{lag}h"] = (
            _num(df[f"{current}_temp_f"])
            - _num(df[f"h{prev:02d}_temp_f"])
        )

    recent_hours = list(range(hour - 3, hour + 1))
    recent_temp_cols = [f"h{h:02d}_temp_f" for h in recent_hours]
    recent_temps = df[recent_temp_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    out["plateau_range_last4h_f"] = (
        recent_temps.max(axis=1) - recent_temps.min(axis=1)
    )
    out["below_peak_now_f"] = out["pre_peak_f"] - out["temp_now_f"]

    recent_cloud_cols = [
        f"h{h:02d}_cloud_fraction" for h in recent_hours
    ]
    clouds = df[recent_cloud_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    out["cloud_mean_last4h"] = clouds.mean(axis=1)
    out["cloud_now"] = _num(df[f"{current}_cloud_fraction"])
    out["cloud_change_3h"] = (
        _num(df[f"{current}_cloud_fraction"])
        - _num(df[f"h{hour - 3:02d}_cloud_fraction"])
    )

    out["dewpoint_now_f"] = _num(df[f"{current}_dewpoint_f"])
    out["wind_now_kt"] = _num(df[f"{current}_wind_speed_kt"])

    out["nws_high_f"] = _num(df["nws_am_forecast_high_f"])
    out["nws_gap_vs_peak_f"] = (
        out["nws_high_f"] - out["pre_peak_f"]
    )

    dates = pd.to_datetime(df["date"], errors="coerce")
    doy = dates.dt.dayofyear.astype(float)
    out["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["season_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return out


def predict_remaining_heating_bundle(
    bundle: dict[str, Any],
    current_daily: pd.DataFrame,
) -> dict[str, float]:
    hour = int(bundle["checkpoint_hour"])
    X = remaining_heating_feature_frame(current_daily.copy(), hour)

    if X.empty or pd.to_numeric(
        X["temp_now_f"], errors="coerce"
    ).isna().all():
        raise ValueError(
            f"current data is missing a usable {hour}:00 checkpoint"
        )

    feature_columns = list(bundle["feature_columns"])
    X = X.reindex(columns=feature_columns)

    medians: pd.Series = bundle["medians"]
    predicted_remaining = float(
        np.maximum(
            0.0,
            bundle["model"].predict(
                X.fillna(medians).fillna(0.0)
            )[0],
        )
    )

    pre_peak = float(
        pd.to_numeric(X["pre_peak_f"], errors="coerce").iloc[0]
    )
    predicted_high = pre_peak + predicted_remaining
    nws = float(
        pd.to_numeric(
            current_daily["nws_am_forecast_high_f"],
            errors="coerce",
        ).iloc[0]
    )

    return {
        "nws_high_f": nws,
        "pre_checkpoint_peak_f": pre_peak,
        "predicted_remaining_heating_f": predicted_remaining,
        "model_predicted_high_f": predicted_high,
        "applied_correction_f": predicted_high - nws,
    }
