from __future__ import annotations

import json
import math
import re
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from klas_model.collectors.asos import fetch_asos


TZ = ZoneInfo("America/Los_Angeles")

PARAMS_PATH = (
    ROOT
    / "data"
    / "model"
    / "klas_h10_50_shadow_params.json"
)

MODEL_HISTORY_PATH = (
    ROOT
    / "data"
    / "live"
    / "klas_model_history.csv"
)

SHADOW_LOG_PATH = (
    ROOT
    / "data"
    / "processed"
    / "klas_h10_50_shadow_log.csv"
)


def parse_tgroup(metar: object) -> float:
    if not isinstance(metar, str):
        return np.nan

    match = re.search(
        r"\bT([01])(\d{3})([01])(\d{3})\b",
        metar,
    )

    if not match:
        return np.nan

    sign, digits, _, _ = match.groups()

    temp_c = int(digits) / 10.0

    if sign == "1":
        temp_c = -temp_c

    return temp_c * 9.0 / 5.0 + 32.0


def fetch_hfmetar(
    date_text: str,
) -> pd.DataFrame:

    date = pd.Timestamp(date_text)
    next_date = date + pd.Timedelta(days=1)

    url = (
        "https://mesonet.agron.iastate.edu/"
        "cgi-bin/request/asos.py"
    )

    params = [
        ("station", "LAS"),
        ("data", "all"),

        ("year1", str(date.year)),
        ("month1", str(date.month)),
        ("day1", str(date.day)),

        ("year2", str(next_date.year)),
        ("month2", str(next_date.month)),
        ("day2", str(next_date.day)),

        ("tz", "America/Los_Angeles"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("elev", "no"),
        ("missing", "empty"),
        ("trace", "empty"),
        ("direct", "no"),
        ("report_type", "1"),
    ]

    headers = {
        "User-Agent":
            "KLAS-Kalshi-Shadow-Research/1.0"
    }

    response = None

    for attempt, delay in enumerate(
        [0, 20, 45],
        start=1,
    ):
        if delay:
            print(
                f"IEM retry in {delay} seconds..."
            )
            time.sleep(delay)

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=180,
        )

        print(
            "IEM attempt",
            attempt,
            "status",
            response.status_code,
        )

        if response.status_code == 200:
            break

        if response.status_code != 429:
            response.raise_for_status()

    if (
        response is None
        or response.status_code != 200
    ):
        raise RuntimeError(
            "IEM HFMETAR unavailable after retries."
        )

    hf = pd.read_csv(
        StringIO(response.text)
    )

    hf["valid"] = pd.to_datetime(
        hf["valid"],
        errors="coerce",
    )

    hf["hf_temp_f"] = hf["metar"].map(
        parse_tgroup
    )

    return hf


def main() -> None:

    now = datetime.now(TZ)

    date_text = now.date().isoformat()

    target_time = now.replace(
        hour=10,
        minute=50,
        second=0,
        microsecond=0,
    )

    print(
        "KLAS H10+50 SHADOW GENERATOR"
    )
    print("date:", date_text)
    print("now:", now.isoformat())

    if now < target_time:
        print(
            "Too early. Shadow signal becomes "
            "eligible at 10:50 AM."
        )
        return

    if not PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Missing frozen parameters: "
            f"{PARAMS_PATH}"
        )

    if not MODEL_HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Missing model history: "
            f"{MODEL_HISTORY_PATH}"
        )

    params = json.loads(
        PARAMS_PATH.read_text(
            encoding="utf-8",
        )
    )

    # --------------------------------------------
    # Idempotency: one prospective row per date.
    # --------------------------------------------

    if SHADOW_LOG_PATH.exists():
        existing = pd.read_csv(
            SHADOW_LOG_PATH
        )

        if "date" in existing.columns:
            existing_dates = set(
                pd.to_datetime(
                    existing["date"],
                    errors="coerce",
                )
                .dt.strftime("%Y-%m-%d")
                .dropna()
            )

            if date_text in existing_dates:
                print(
                    "Shadow row already exists "
                    f"for {date_text}; no changes."
                )
                return
    else:
        existing = pd.DataFrame()

    # --------------------------------------------
    # Freeze base forecast at or before 10:50.
    # --------------------------------------------

    history = pd.read_csv(
        MODEL_HISTORY_PATH
    )

    history["date"] = pd.to_datetime(
        history["date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    history["checkpoint_hour"] = (
        pd.to_numeric(
            history["checkpoint_hour"],
            errors="coerce",
        )
    )

    history["updated_at_local"] = (
        pd.to_datetime(
            history["updated_at_local"],
            errors="coerce",
            utc=True,
        )
    )

    cutoff_utc = pd.Timestamp(
        target_time
    ).tz_convert("UTC")

    base_rows = history[
        (history["date"] == date_text)
        & history["checkpoint_hour"].eq(10)
        & (
            history["model_name"]
            .astype(str)
            .eq("KLAS_MODEL")
        )
        & (
            history["updated_at_local"]
            <= cutoff_utc
        )
    ].copy()

    base_rows["predicted_high_f"] = (
        pd.to_numeric(
            base_rows["predicted_high_f"],
            errors="coerce",
        )
    )

    base_rows = base_rows.dropna(
        subset=[
            "updated_at_local",
            "predicted_high_f",
        ]
    )

    if base_rows.empty:
        print(
            "No eligible h10 KLAS_MODEL forecast "
            "exists at or before 10:50. "
            "Today is not scored."
        )
        return

    base_row = (
        base_rows
        .sort_values("updated_at_local")
        .iloc[-1]
    )

    base_forecast = float(
        base_row["predicted_high_f"]
    )

    base_time = pd.Timestamp(
        base_row["updated_at_local"]
    ).tz_convert(TZ)

    # --------------------------------------------
    # Recover exact h10 checkpoint observation.
    # --------------------------------------------

    asos = fetch_asos(
        date_text,
        date_text,
    )

    asos["timestamp"] = pd.to_datetime(
        asos["timestamp"],
        errors="coerce",
    )

    checkpoint_target = pd.Timestamp(
        f"{date_text} 10:00:00",
        tz=TZ,
    )

    asos["checkpoint_delta"] = (
        asos["timestamp"]
        - checkpoint_target
    ).abs()

    checkpoint_rows = asos[
        asos["checkpoint_delta"]
        <= pd.Timedelta(minutes=40)
    ].copy()

    if checkpoint_rows.empty:
        print(
            "No routine ASOS observation within "
            "40 minutes of h10. Today is not scored."
        )
        return

    checkpoint = (
        checkpoint_rows
        .sort_values("checkpoint_delta")
        .iloc[0]
    )

    checkpoint_temp = pd.to_numeric(
        pd.Series(
            [
                checkpoint.get(
                    "metar_precise_temp_f"
                )
            ]
        ),
        errors="coerce",
    ).iloc[0]

    if pd.isna(checkpoint_temp):
        checkpoint_temp = pd.to_numeric(
            pd.Series(
                [checkpoint.get("temp_f")]
            ),
            errors="coerce",
        ).iloc[0]

    if pd.isna(checkpoint_temp):
        print(
            "Checkpoint observation has no "
            "usable temperature. Today is not scored."
        )
        return

    checkpoint_temp = float(
        checkpoint_temp
    )

    checkpoint_time = pd.Timestamp(
        checkpoint["timestamp"]
    )

    # --------------------------------------------
    # Get fresh 10:50 HFMETAR temperature.
    # --------------------------------------------

    hf = fetch_hfmetar(
        date_text
    )

    hf_target = pd.Timestamp(
        f"{date_text} 10:50:00"
    )

    hf["target_delta"] = (
        hf["valid"] - hf_target
    ).abs()

    hf_rows = (
        hf.dropna(
            subset=[
                "valid",
                "hf_temp_f",
            ]
        )
        .sort_values("target_delta")
    )

    if hf_rows.empty:
        print(
            "No usable HFMETAR temperature. "
            "Today is not scored."
        )
        return

    hf_row = hf_rows.iloc[0]

    hf_delta_min = (
        hf_row["target_delta"]
        .total_seconds()
        / 60.0
    )

    if hf_delta_min > 3.0:
        print(
            "Nearest HFMETAR is more than "
            "3 minutes from 10:50. "
            "Today is not scored."
        )
        return

    hf_temp = float(
        hf_row["hf_temp_f"]
    )

    # --------------------------------------------
    # Apply frozen expected-trajectory model.
    # --------------------------------------------

    date = pd.Timestamp(
        date_text
    )

    doy = date.dayofyear

    feature_values = [
        checkpoint_temp,
        base_forecast,
        math.sin(
            2 * math.pi * doy / 365.25
        ),
        math.cos(
            2 * math.pi * doy / 365.25
        ),
    ]

    scaled = [
        (value - mean) / scale
        for value, mean, scale in zip(
            feature_values,
            params["scaler_mean"],
            params["scaler_scale"],
        )
    ]

    expected_hf_temp = (
        float(params["ridge_intercept"])
        + sum(
            float(coef) * value
            for coef, value in zip(
                params["ridge_coef"],
                scaled,
            )
        )
    )

    trajectory_deviation = (
        hf_temp
        - expected_hf_temp
    )

    shadow_adjustment = (
        float(
            params[
                "trajectory_intercept_f"
            ]
        )
        + float(
            params["trajectory_coef"]
        )
        * trajectory_deviation
    )

    shadow_forecast = (
        base_forecast
        + shadow_adjustment
    )

    # --------------------------------------------
    # Save prospective row before final CLI.
    # --------------------------------------------

    row = pd.DataFrame([{
        "date": date_text,

        "checkpoint_hour":
            int(params["checkpoint_hour"]),

        "elapsed_min":
            int(params["elapsed_min"]),

        "base_forecast_time_local":
            base_time.isoformat(),

        "base_forecast_f":
            base_forecast,

        "checkpoint_obs_time_local":
            checkpoint_time.isoformat(),

        "checkpoint_temp_f":
            checkpoint_temp,

        "hf_obs_time_local":
            pd.Timestamp(
                hf_row["valid"]
            ).isoformat(),

        "hf_temp_f":
            hf_temp,

        "expected_hf_temp_f":
            expected_hf_temp,

        "trajectory_deviation_f":
            trajectory_deviation,

        "frozen_coef":
            float(
                params["trajectory_coef"]
            ),

        "frozen_intercept_f":
            float(
                params[
                    "trajectory_intercept_f"
                ]
            ),

        "shadow_adjustment_f":
            shadow_adjustment,

        "shadow_forecast_f":
            shadow_forecast,

        "production_enabled":
            False,

        "actual_cli_high_f":
            pd.NA,

        "base_abs_error_f":
            pd.NA,

        "shadow_abs_error_f":
            pd.NA,

        "shadow_gain_f":
            pd.NA,

        "scored":
            False,
    }])

    combined = pd.concat(
        [existing, row],
        ignore_index=True,
    )

    combined = (
        combined
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    SHADOW_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        SHADOW_LOG_PATH,
        index=False,
    )

    print()
    print("SHADOW FORECAST CREATED")
    print(
        "base forecast time:",
        base_time.isoformat(),
    )
    print(
        "base forecast:",
        round(base_forecast, 3),
    )
    print(
        "checkpoint temp:",
        round(checkpoint_temp, 3),
    )
    print(
        "10:50 HFMETAR:",
        round(hf_temp, 3),
    )
    print(
        "expected 10:50:",
        round(expected_hf_temp, 3),
    )
    print(
        "trajectory deviation:",
        round(
            trajectory_deviation,
            3,
        ),
    )
    print(
        "shadow adjustment:",
        round(shadow_adjustment, 3),
    )
    print(
        "shadow forecast:",
        round(shadow_forecast, 3),
    )
    print(
        "production enabled: False"
    )
    print(
        "saved:",
        SHADOW_LOG_PATH,
    )


if __name__ == "__main__":
    main()
