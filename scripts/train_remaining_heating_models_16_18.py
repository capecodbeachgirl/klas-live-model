from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/processed/klas_daily_heating.csv")
MODEL_DIR = Path("data/model")
CHECKPOINTS = (16, 17, 18)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))


def feature_frame(df: pd.DataFrame, hour: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    current = f"h{hour:02d}"
    temp_cols = [f"h{h:02d}_temp_f" for h in range(8, hour + 1)]

    out["pre_peak_f"] = df[temp_cols].apply(
        pd.to_numeric, errors="coerce"
    ).max(axis=1)
    out["temp_now_f"] = num(df[f"{current}_temp_f"])

    for lag in (1, 2, 3):
        prev = hour - lag
        out[f"temp_change_{lag}h"] = (
            num(df[f"{current}_temp_f"]) - num(df[f"h{prev:02d}_temp_f"])
        )

    recent_hours = list(range(hour - 3, hour + 1))
    recent_temp_cols = [f"h{h:02d}_temp_f" for h in recent_hours]
    recent_temps = df[recent_temp_cols].apply(pd.to_numeric, errors="coerce")
    out["plateau_range_last4h_f"] = (
        recent_temps.max(axis=1) - recent_temps.min(axis=1)
    )
    out["below_peak_now_f"] = out["pre_peak_f"] - out["temp_now_f"]

    recent_cloud_cols = [f"h{h:02d}_cloud_fraction" for h in recent_hours]
    clouds = df[recent_cloud_cols].apply(pd.to_numeric, errors="coerce")
    out["cloud_mean_last4h"] = clouds.mean(axis=1)
    out["cloud_now"] = num(df[f"{current}_cloud_fraction"])
    out["cloud_change_3h"] = (
        num(df[f"{current}_cloud_fraction"])
        - num(df[f"h{hour - 3:02d}_cloud_fraction"])
    )

    out["dewpoint_now_f"] = num(df[f"{current}_dewpoint_f"])
    out["wind_now_kt"] = num(df[f"{current}_wind_speed_kt"])

    out["nws_high_f"] = num(df["nws_am_forecast_high_f"])
    out["nws_gap_vs_peak_f"] = out["nws_high_f"] - out["pre_peak_f"]

    dates = pd.to_datetime(df["date"], errors="coerce")
    doy = dates.dt.dayofyear.astype(float)
    out["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["season_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return out


def make_model(name: str):
    if name.startswith("ridge_alpha_"):
        alpha = float(name.split("_")[-1])
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )

    if name.startswith("rf_leaf_"):
        leaf = int(name.split("_")[-1])
        return RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=leaf,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(name)


def train_one(df: pd.DataFrame, hour: int) -> None:
    X = feature_frame(df, hour)
    actual = num(df["actual_cli_high_f"])
    remaining = (actual - X["pre_peak_f"]).clip(lower=0.0)

    valid = (
        df["date"].notna()
        & actual.notna()
        & X["pre_peak_f"].notna()
        & X["temp_now_f"].notna()
    )

    work = df.loc[valid].copy()
    X = X.loc[valid].copy()
    actual = actual.loc[valid].copy()
    remaining = remaining.loc[valid].copy()

    years = work["date"].dt.year

    train_mask = years.isin([2022, 2023, 2024])
    val_mask = years.eq(2025)
    test_mask = years.eq(2026)

    candidates = [
        "ridge_alpha_1",
        "ridge_alpha_5",
        "ridge_alpha_20",
        "ridge_alpha_50",
        "rf_leaf_3",
        "rf_leaf_5",
        "rf_leaf_10",
        "rf_leaf_15",
    ]

    med_train = X.loc[train_mask].median(numeric_only=True)
    Xtr = X.loc[train_mask].fillna(med_train).fillna(0.0)
    Xv = X.loc[val_mask].fillna(med_train).fillna(0.0)

    best_name = None
    best_val_mae = float("inf")

    print()
    print("=" * 78)
    print(f"{hour:02d}:00 REMAINING-HEATING MODEL")
    print("=" * 78)

    for name in candidates:
        model = make_model(name)
        model.fit(Xtr, remaining.loc[train_mask])

        pred = np.maximum(0.0, model.predict(Xv))
        score = mae(pred, remaining.loc[val_mask].to_numpy(float))

        print(f"{name:>16} validation MAE: {score:.3f} F")

        if score < best_val_mae:
            best_val_mae = score
            best_name = name

    assert best_name is not None

    # Untouched 2026 test.
    trainval_mask = train_mask | val_mask
    med_trainval = X.loc[trainval_mask].median(numeric_only=True)

    test_model = make_model(best_name)
    test_model.fit(
        X.loc[trainval_mask].fillna(med_trainval).fillna(0.0),
        remaining.loc[trainval_mask],
    )

    test_remaining = np.maximum(
        0.0,
        test_model.predict(
            X.loc[test_mask].fillna(med_trainval).fillna(0.0)
        ),
    )

    test_high = (
        X.loc[test_mask, "pre_peak_f"].to_numpy(float)
        + test_remaining
    )

    test_actual = actual.loc[test_mask].to_numpy(float)
    test_errors = test_high - test_actual
    test_mae = mae(test_high, test_actual)

    print()
    print(f"SELECTED MODEL: {best_name}")
    print(f"2025 validation remaining-heating MAE: {best_val_mae:.3f} F")
    print(f"2026 untouched final-high MAE: {test_mae:.3f} F")
    print(f"2026 test rows: {len(test_errors)}")

    # Refit selected architecture on all completed historical rows.
    med_all = X.median(numeric_only=True)
    final_model = make_model(best_name)
    final_model.fit(
        X.fillna(med_all).fillna(0.0),
        remaining,
    )

    trained_through = work["date"].max()

    bundle = {
        "model_type": "remaining_heating",
        "checkpoint_hour": int(hour),
        "selected_model": best_name,
        "model": final_model,
        "medians": med_all,
        "feature_columns": list(X.columns),
        "validation_remaining_mae_f": float(best_val_mae),
        "test_final_high_mae_f": float(test_mae),
        "calibration_model_errors_f": [float(x) for x in test_errors],
        "trained_through": (
            trained_through.date().isoformat()
            if pd.notna(trained_through)
            else None
        ),
        "training_rows": int(len(X)),
    }

    out_path = MODEL_DIR / f"h{hour:02d}_remaining.joblib"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)

    print(f"saved: {out_path}")
    print(f"trained through: {bundle['trained_through']}")
    print(f"training rows: {bundle['training_rows']}")


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    if "day_complete" in df.columns:
        complete = df["day_complete"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
        df = df[complete].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for hour in CHECKPOINTS:
        train_one(df, hour)


if __name__ == "__main__":
    main()
