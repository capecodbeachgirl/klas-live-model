from __future__ import annotations

import os
from datetime import datetime, time, timedelta, timezone
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import requests


FORECASTS_API_URL = "https://wethr.net/api/v2/forecasts.php"
OBSERVATIONS_API_URL = "https://wethr.net/api/v2/observations.php"

LAS_TZ = ZoneInfo("America/Los_Angeles")

# NWS/Kalshi temperature days use Pacific Standard Time year-round.
PST_FIXED = timezone(timedelta(hours=-8))

DEFAULT_MODELS = (
    "HRRR",
    "HRRR-EXT",
    "NBM",
    "RAP",
    "GFS-MOS",
    "LAV-MOS",
)


def _api_key() -> str:
    key = os.environ.get("WETHR_API_KEY")
    if not key:
        raise RuntimeError(
            "WETHR_API_KEY environment variable is not set"
        )
    return key

def fetch_wethr_high(
    *,
    timeout: int = 30,
) -> dict[str, Any]:
    """Fetch Wethr's current NWS/Kalshi trading-day high for KLAS."""

    response = requests.get(
        OBSERVATIONS_API_URL,
        params={
            "station_code": "KLAS",
            "mode": "wethr_high",
            "logic": "nws",
        },
        headers={
            "Authorization": f"Bearer {_api_key()}",
        },
        timeout=timeout,
    )

    response.raise_for_status()
    data = response.json()

    high = data.get("wethr_high")
    sources = data.get("wethr_high_sources") or []

    return {
        "available": high is not None,
        "wethr_high_f": (
            float(high)
            if high is not None
            else None
        ),
        "time_of_high_utc": data.get("time_of_high_utc"),
        "sources": sources,
        "source_detail": (
            data.get("wethr_high_source_detail") or {}
        ),
        "omo_informed": "omo" in sources,
        "data_quality": data.get("data_quality") or {},
        "source": "Wethr.net Observations API v2",
    }

def _contract_window(
    now_local: datetime,
) -> tuple[datetime, datetime]:
    """Return the current NWS/Kalshi temperature day."""

    now_standard = now_local.astimezone(PST_FIXED)

    start_standard = datetime.combine(
        now_standard.date(),
        time.min,
        tzinfo=PST_FIXED,
    )

    end_standard = start_standard + timedelta(days=1)

    return (
        start_standard.astimezone(LAS_TZ),
        end_standard.astimezone(LAS_TZ),
    )


def fetch_latest_model_run(
    model: str,
    *,
    timeout: int = 30,
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch every available forecast hour from Wethr's latest run."""

    response = requests.get(
        FORECASTS_API_URL,
        params={
            "location_name": "KLAS",
            "model": model,
            "run": "latest",
        },
        headers={
            "Authorization": f"Bearer {_api_key()}",
        },
        timeout=timeout,
    )

    response.raise_for_status()

    rows = response.json()

    if not isinstance(rows, list):
        raise RuntimeError(
            f"Unexpected Wethr response for {model}"
        )

    return rows, response.headers.get("X-Run-Time")


def summarize_latest_run(
    model: str,
    rows: list[dict[str, Any]],
    run_time: str | None,
    now_local: datetime,
) -> dict[str, Any]:
    """Summarize coverage and remaining temperature forecast."""

    if not rows:
        return {
            "available": False,
            "model": model,
            "run_time_utc": run_time,
        }

    df = pd.DataFrame(rows)

    if "valid_time" not in df or "temperature_f" not in df:
        return {
            "available": False,
            "model": model,
            "run_time_utc": run_time,
            "reason": "Missing forecast fields",
        }

    df["valid_time_utc"] = pd.to_datetime(
        df["valid_time"],
        errors="coerce",
        utc=True,
    )

    df["valid_time_local"] = (
        df["valid_time_utc"].dt.tz_convert(LAS_TZ)
    )

    df["temperature_f"] = pd.to_numeric(
        df["temperature_f"],
        errors="coerce",
    )

    df["forecast_hour"] = pd.to_numeric(
        df.get("forecast_hour"),
        errors="coerce",
    )

    df = df.dropna(
        subset=["valid_time_local", "temperature_f"]
    ).sort_values("valid_time_local")

    if df.empty:
        return {
            "available": False,
            "model": model,
            "run_time_utc": run_time,
            "reason": "No usable forecast rows",
        }

    contract_start, contract_end = _contract_window(now_local)

    contract_rows = df[
        (df["valid_time_local"] >= pd.Timestamp(contract_start))
        & (df["valid_time_local"] < pd.Timestamp(contract_end))
    ].copy()

    remaining = contract_rows[
        contract_rows["valid_time_local"]
        >= pd.Timestamp(now_local)
    ].copy()

    coverage_start = df["valid_time_local"].min()
    coverage_end = df["valid_time_local"].max()

    # A newly appearing Wethr run can still be partially ingested.
    # Do not treat a partial run as a full daily-high forecast.
    covers_rest_of_contract = bool(
        coverage_end >= pd.Timestamp(contract_end)
    )

    remaining_high = None
    remaining_high_time = None

    if not remaining.empty:
        hottest_index = remaining["temperature_f"].idxmax()
        hottest = remaining.loc[hottest_index]

        remaining_high = float(hottest["temperature_f"])
        remaining_high_time = (
            hottest["valid_time_local"].isoformat()
        )

    max_forecast_hour = None

    if df["forecast_hour"].notna().any():
        max_forecast_hour = int(
            df["forecast_hour"].max()
        )

    return {
        "available": True,
        "model": model,
        "run_time_utc": run_time,
        "rows_received": int(len(df)),
        "max_forecast_hour": max_forecast_hour,
        "coverage_start_local": coverage_start.isoformat(),
        "coverage_end_local": coverage_end.isoformat(),
        "contract_start_local": contract_start.isoformat(),
        "contract_end_local": contract_end.isoformat(),
        "covers_rest_of_contract": covers_rest_of_contract,
        "remaining_high_f": remaining_high,
        "remaining_high_time_local": remaining_high_time,
    }


def fetch_wethr_snapshot(
    models: Iterable[str] = DEFAULT_MODELS,
    *,
    now_local: datetime | None = None,
) -> dict[str, Any]:
    """Collect live Wethr research data without altering the KLAS model."""

    now_local = now_local or datetime.now(LAS_TZ)

    results: dict[str, Any] = {}

    for model in models:
        try:
            rows, run_time = fetch_latest_model_run(model)

            results[model] = summarize_latest_run(
                model,
                rows,
                run_time,
                now_local,
            )

        except Exception as exc:
            results[model] = {
                "available": False,
                "model": model,
                "error": str(exc),
            }
    usable = {
        name: result
        for name, result in results.items()
        if (
            result.get("available")
            and result.get("covers_rest_of_contract")
            and result.get("remaining_high_f") is not None
        )
    }

    highs = [
        float(result["remaining_high_f"])
        for result in usable.values()
    ]

    if highs:
        consensus = {
            "available": True,
            "model_count": len(highs),
            "models_used": list(usable.keys()),
            "median_high_f": round(
                float(pd.Series(highs).median()), 2
            ),
            "mean_high_f": round(
                float(pd.Series(highs).mean()), 2
            ),
            "min_high_f": round(min(highs), 2),
            "max_high_f": round(max(highs), 2),
            "spread_f": round(max(highs) - min(highs), 2),
        }
    else:
        consensus = {
            "available": False,
            "model_count": 0,
            "models_used": [],
        }
    return {
        "available": any(
            result.get("available")
            for result in results.values()
        ),
        "collected_at_local": now_local.isoformat(),
        "models": results,
        "consensus": consensus,
        "source": "Wethr.net Forecasts API v2",
        "research_only": True,
    }

def apply_observed_floor(
    snapshot: dict[str, Any],
    observed_peak_f: float | None,
) -> dict[str, Any]:
    """Prevent projected Wethr highs from falling below the observed KLAS peak."""

    if observed_peak_f is None:
        return snapshot

    floor = float(observed_peak_f)
    snapshot["observed_floor_f"] = floor

    projected = {}

    for name, result in snapshot.get("models", {}).items():
        if (
            result.get("available")
            and result.get("covers_rest_of_contract")
            and result.get("remaining_high_f") is not None
        ):
            projected_high = max(
                floor,
                float(result["remaining_high_f"]),
            )

            result["projected_high_f"] = projected_high
            projected[name] = projected_high

    if projected:
        highs = list(projected.values())

        snapshot["consensus"] = {
            "available": True,
            "model_count": len(highs),
            "models_used": list(projected.keys()),
            "median_high_f": round(
                float(pd.Series(highs).median()), 2
            ),
            "mean_high_f": round(
                float(pd.Series(highs).mean()), 2
            ),
            "min_high_f": round(min(highs), 2),
            "max_high_f": round(max(highs), 2),
            "spread_f": round(max(highs) - min(highs), 2),
        }

    return snapshot