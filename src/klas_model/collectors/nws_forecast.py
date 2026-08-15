from __future__ import annotations

from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

API_BASE = "https://api.weather.gov"
LAS_TZ = ZoneInfo("America/Los_Angeles")
DEFAULT_LAT = 36.0801
DEFAULT_LON = -115.1522
DEFAULT_HEADERS = {
    "User-Agent": "klas-kalshi-model/0.13 (KLAS temperature research)",
    "Accept": "application/geo+json",
}


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _period_local_start(period: dict[str, Any]) -> pd.Timestamp | None:
    raw = period.get("startTime")
    if not raw:
        return None
    ts = pd.to_datetime(raw, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.tz_convert(LAS_TZ)


def summarize_hourly_forecast(
    periods: list[dict[str, Any]],
    now_local: datetime,
    through_hour: int = 20,
) -> dict[str, Any]:
    """Summarize today's remaining NWS hourly forecast for live convection risk.

    This intentionally does not alter the validated temperature model. It supplies
    forward-looking weather-risk information used for confidence / wait flags.
    """
    now_ts = pd.Timestamp(now_local).tz_convert(LAS_TZ) if pd.Timestamp(now_local).tzinfo else pd.Timestamp(now_local, tz=LAS_TZ)
    cutoff = pd.Timestamp(datetime.combine(now_ts.date(), time(hour=through_hour), tzinfo=LAS_TZ))

    rows: list[dict[str, Any]] = []
    for p in periods:
        start = _period_local_start(p)
        if start is None or start < now_ts.floor("h") or start > cutoff:
            continue
        pop = p.get("probabilityOfPrecipitation") or {}
        pop_value = _as_float(pop.get("value")) if isinstance(pop, dict) else None
        short = str(p.get("shortForecast") or "")
        rows.append({
            "time": start.isoformat(),
            "temp_f": _as_float(p.get("temperature")),
            "pop_pct": pop_value,
            "short_forecast": short,
            "wind_speed": p.get("windSpeed"),
            "wind_direction": p.get("windDirection"),
        })

    if not rows:
        return {
            "available": False,
            "hours": [],
            "max_pop_pct": None,
            "thunder_possible": False,
            "rain_possible": False,
            "summary": "NWS hourly forecast unavailable",
        }

    descriptions = " ".join(r["short_forecast"].lower() for r in rows)
    thunder = any(term in descriptions for term in ("thunder", "t-storm", "tstorm"))
    rain = thunder or any(term in descriptions for term in ("rain", "shower", "sprinkle"))
    pops = [r["pop_pct"] for r in rows if r["pop_pct"] is not None]
    max_pop = max(pops) if pops else None
    if thunder:
        summary = "NWS hourly forecast includes thunderstorms"
    elif rain:
        summary = "NWS hourly forecast includes showers/rain"
    elif max_pop is not None and max_pop >= 20:
        summary = f"NWS precipitation chance reaches {max_pop:.0f}%"
    else:
        summary = "No meaningful rain/thunder signal in the remaining NWS hourly forecast"

    return {
        "available": True,
        "hours": rows,
        "max_pop_pct": max_pop,
        "thunder_possible": thunder,
        "rain_possible": rain,
        "summary": summary,
    }


def _value_for_time(values: list[dict[str, Any]], target: pd.Timestamp) -> float | None:
    """Return a grid-data value whose validTime interval contains target.

    NWS grid data validTime strings are ISO8601 interval strings such as
    2026-08-15T18:00:00+00:00/PT3H. Pandas handles the start; we parse simple
    PT#H durations because that is sufficient for sky-cover periods used here.
    """
    for item in values or []:
        valid = str(item.get("validTime") or "")
        if "/" not in valid:
            continue
        start_raw, duration = valid.split("/", 1)
        start = pd.to_datetime(start_raw, errors="coerce", utc=True)
        if pd.isna(start):
            continue
        hours = 1.0
        if duration.startswith("PT") and duration.endswith("H"):
            try:
                hours = float(duration[2:-1])
            except ValueError:
                hours = 1.0
        end = start + pd.Timedelta(hours=hours)
        target_utc = target.tz_convert("UTC")
        if start <= target_utc < end:
            return _as_float(item.get("value"))
    return None


def summarize_grid_forecast(grid: dict[str, Any], hourly_summary: dict[str, Any]) -> dict[str, Any]:
    props = grid.get("properties", {}) if isinstance(grid, dict) else {}
    sky_values = (props.get("skyCover") or {}).get("values", []) if isinstance(props.get("skyCover"), dict) else []
    sky_rows: list[dict[str, Any]] = []
    for hour in hourly_summary.get("hours", []):
        ts = pd.to_datetime(hour["time"], errors="coerce")
        if pd.isna(ts):
            continue
        sky = _value_for_time(sky_values, ts)
        sky_rows.append({"time": hour["time"], "sky_cover_pct": sky})
    sky_vals = [x["sky_cover_pct"] for x in sky_rows if x["sky_cover_pct"] is not None]
    return {
        "max_sky_cover_pct": max(sky_vals) if sky_vals else None,
        "sky_hours": sky_rows,
    }


def fetch_nws_live_forecast(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    now_local: datetime | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    now_local = now_local or datetime.now(LAS_TZ)
    points = requests.get(
        f"{API_BASE}/points/{lat:.4f},{lon:.4f}", headers=DEFAULT_HEADERS, timeout=timeout
    )
    points.raise_for_status()
    p = points.json().get("properties", {})
    hourly_url = p.get("forecastHourly")
    grid_url = p.get("forecastGridData")
    if not hourly_url:
        raise RuntimeError("NWS points response did not contain forecastHourly")

    hourly_resp = requests.get(hourly_url, headers=DEFAULT_HEADERS, timeout=timeout)
    hourly_resp.raise_for_status()
    periods = hourly_resp.json().get("properties", {}).get("periods", [])
    hourly = summarize_hourly_forecast(periods, now_local)

    grid_summary = {"max_sky_cover_pct": None, "sky_hours": []}
    if grid_url:
        try:
            grid_resp = requests.get(grid_url, headers=DEFAULT_HEADERS, timeout=timeout)
            grid_resp.raise_for_status()
            grid_summary = summarize_grid_forecast(grid_resp.json(), hourly)
        except Exception:
            pass

    return {
        **hourly,
        **grid_summary,
        "office": p.get("gridId"),
        "grid_x": p.get("gridX"),
        "grid_y": p.get("gridY"),
        "source": "NWS api.weather.gov",
    }
