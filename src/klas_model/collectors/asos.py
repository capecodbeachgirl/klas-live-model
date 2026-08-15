from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable
import time

import pandas as pd
import requests

from ..metar_extremes import parse_temperature_extremes

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
AWC_METAR_URL = "https://aviationweather.gov/api/data/metar"
USER_AGENT = "KLAS-Kalshi-Research/0.13.1 (weather research dashboard)"


@dataclass(frozen=True)
class AsosRequest:
    station: str = "LAS"
    network: str = "NV_ASOS"
    tz: str = "America/Los_Angeles"


def _coverage_value(code: object) -> float | None:
    mapping = {
        "CLR": 0.0,
        "SKC": 0.0,
        "FEW": 0.25,
        "SCT": 0.5,
        "BKN": 0.875,
        "OVC": 1.0,
        "VV": 1.0,
    }
    if not isinstance(code, str):
        return None
    return mapping.get(code.strip().upper())


def _cloud_fraction_from_codes(row: pd.Series) -> float | None:
    """Return maximum reported sky-cover fraction across all layers."""
    vals: list[float] = []
    for col in ("skyc1", "skyc2", "skyc3", "skyc4"):
        value = _coverage_value(row.get(col))
        if value is not None:
            vals.append(value)
    return max(vals) if vals else None


def _cloud_fraction_below(row: pd.Series, ceiling_ft: float) -> float | None:
    """Maximum reported sky-cover fraction among layers at/below a ceiling."""
    vals: list[float] = []
    for idx in range(1, 5):
        frac = _coverage_value(row.get(f"skyc{idx}"))
        height = pd.to_numeric(pd.Series([row.get(f"skyl{idx}")]), errors="coerce").iloc[0]
        if frac is None or pd.isna(height):
            continue
        if float(height) <= ceiling_ft:
            vals.append(frac)
    return max(vals) if vals else 0.0


def _lowest_bkn_ovc_ft(row: pd.Series) -> float | None:
    heights: list[float] = []
    for idx in range(1, 5):
        code = row.get(f"skyc{idx}")
        if not isinstance(code, str) or code.strip().upper() not in {"BKN", "OVC", "VV"}:
            continue
        height = pd.to_numeric(pd.Series([row.get(f"skyl{idx}")]), errors="coerce").iloc[0]
        if not pd.isna(height):
            heights.append(float(height))
    return min(heights) if heights else None


def normalize_asos_csv(text: str, timezone: str = "America/Los_Angeles") -> pd.DataFrame:
    """Normalize an Iowa Environmental Mesonet ASOS CSV export.

    Returned timestamps are timezone-aware local Las Vegas time. The raw source row
    is intentionally kept compact; all model-facing columns use stable names.
    """
    df = pd.read_csv(StringIO(text), comment="#", na_values=["M", "T", "", "null"])
    if df.empty:
        return df

    if "valid" not in df.columns:
        raise ValueError("ASOS CSV is missing required 'valid' column")

    ts = pd.to_datetime(df["valid"], errors="coerce")
    # IEM returns local timestamps when tz is requested. Localize rather than convert.
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
    else:
        ts = ts.dt.tz_convert(timezone)

    out = pd.DataFrame({"timestamp": ts})
    out["station"] = df.get("station", "LAS")
    out["temp_f"] = pd.to_numeric(df.get("tmpf"), errors="coerce")
    out["dewpoint_f"] = pd.to_numeric(df.get("dwpf"), errors="coerce")
    out["wind_dir_deg"] = pd.to_numeric(df.get("drct"), errors="coerce")
    out["wind_speed_kt"] = pd.to_numeric(df.get("sknt"), errors="coerce")
    out["wind_gust_kt"] = pd.to_numeric(df.get("gust"), errors="coerce")
    out["precip_in"] = pd.to_numeric(df.get("p01i"), errors="coerce")
    out["metar"] = df.get("metar")
    out["wxcodes"] = df.get("wxcodes")

    if any(c in df.columns for c in ("skyc1", "skyc2", "skyc3", "skyc4")):
        out["cloud_fraction"] = df.apply(_cloud_fraction_from_codes, axis=1)
        out["cloud_fraction_below_12000"] = df.apply(lambda row: _cloud_fraction_below(row, 12000.0), axis=1)
        out["cloud_fraction_below_20000"] = df.apply(lambda row: _cloud_fraction_below(row, 20000.0), axis=1)
        out["lowest_bkn_ovc_ft"] = df.apply(_lowest_bkn_ovc_ft, axis=1)
        for idx in range(1, 5):
            out[f"skyc{idx}"] = df.get(f"skyc{idx}")
            out[f"skyl{idx}"] = pd.to_numeric(df.get(f"skyl{idx}"), errors="coerce")
    else:
        out["cloud_fraction"] = pd.NA
        out["cloud_fraction_below_12000"] = pd.NA
        out["cloud_fraction_below_20000"] = pd.NA
        out["lowest_bkn_ovc_ft"] = pd.NA

    # Detect thunder from METAR weather tokens. This is a station-observed indicator.
    out["thunder_observed"] = out["metar"].fillna("").str.contains(r"(?:^|\s)(?:\+|-)?TS", regex=True)

    extremes = out["metar"].apply(parse_temperature_extremes)
    out["metar_precise_temp_f"] = extremes.apply(lambda x: x.precise_temp_f)
    out["metar_precise_dewpoint_f"] = extremes.apply(lambda x: x.precise_dewpoint_f)
    out["six_hour_max_f"] = extremes.apply(lambda x: x.six_hour_max_f)
    out["six_hour_min_f"] = extremes.apply(lambda x: x.six_hour_min_f)
    out["daily_24h_max_f"] = extremes.apply(lambda x: x.daily_24h_max_f)
    out["daily_24h_min_f"] = extremes.apply(lambda x: x.daily_24h_min_f)

    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def fetch_asos(
    start: str,
    end: str,
    request: AsosRequest = AsosRequest(),
    timeout: int = 60,
) -> pd.DataFrame:
    """Download KLAS/LAS ASOS history from IEM and return normalized observations.

    start/end are YYYY-MM-DD inclusive calendar dates in Las Vegas local time.
    """
    start_ts = pd.Timestamp(start)
    # IEM end dates are exclusive; expose an inclusive user-facing --end date.
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    params: list[tuple[str, str]] = [
        ("station", request.station),
        ("data", "tmpf"),
        ("data", "dwpf"),
        ("data", "drct"),
        ("data", "sknt"),
        ("data", "gust"),
        ("data", "p01i"),
        ("data", "skyc1"),
        ("data", "skyc2"),
        ("data", "skyc3"),
        ("data", "skyc4"),
        ("data", "skyl1"),
        ("data", "skyl2"),
        ("data", "skyl3"),
        ("data", "skyl4"),
        ("data", "wxcodes"),
        ("data", "metar"),
        ("year1", str(start_ts.year)),
        ("month1", str(start_ts.month)),
        ("day1", str(start_ts.day)),
        ("year2", str(end_ts.year)),
        ("month2", str(end_ts.month)),
        ("day2", str(end_ts.day)),
        ("tz", request.tz),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("elev", "no"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "no"),
        ("report_type", "3"),  # Routine / once hourly
        ("report_type", "4"),  # Specials
    ]
    last_exc: Exception | None = None
    for attempt, delay in enumerate((0, 2, 5), start=1):
        if delay:
            time.sleep(delay)
        try:
            response = requests.get(
                IEM_ASOS_URL,
                params=params,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            out = normalize_asos_csv(response.text, timezone=request.tz)
            if not out.empty:
                out["data_source"] = "IEM"
            return out
        except requests.RequestException as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            # Retry transient upstream / gateway failures; fail fast on permanent 4xx errors.
            if status is not None and status < 500 and status != 429:
                raise
            if attempt == 3:
                break
    assert last_exc is not None
    raise last_exc


def save_observations(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path



def _awc_obs_timestamp(value: object, timezone: str) -> pd.Timestamp:
    """Parse AviationWeather obsTime whether supplied as epoch seconds or ISO text."""
    if value is None:
        return pd.NaT
    if isinstance(value, (int, float)) and not pd.isna(value):
        ts = pd.to_datetime(value, unit="s", utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert(timezone)


def _c_to_f(value: object) -> float | None:
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(num):
        return None
    return float(num) * 9.0 / 5.0 + 32.0


def normalize_awc_metar_json(payload: list[dict], timezone: str = "America/Los_Angeles") -> pd.DataFrame:
    """Normalize AviationWeather.gov METAR JSON into the same live schema as IEM.

    This is intentionally a live-data fallback. Historical training/backfills continue to use
    IEM so the training dataset remains internally consistent.
    """
    rows: list[dict] = []
    for item in payload or []:
        raw = str(item.get("rawOb") or item.get("raw_text") or "").strip()
        ts = _awc_obs_timestamp(item.get("obsTime") or item.get("reportTime"), timezone)
        if pd.isna(ts):
            continue
        clouds = item.get("clouds") or []
        if not isinstance(clouds, list):
            clouds = []
        row: dict[str, object] = {
            "timestamp": ts,
            "station": item.get("icaoId") or "KLAS",
            "temp_f": _c_to_f(item.get("temp")),
            "dewpoint_f": _c_to_f(item.get("dewp")),
            "wind_dir_deg": item.get("wdir"),
            "wind_speed_kt": item.get("wspd"),
            "wind_gust_kt": item.get("wgst"),
            "precip_in": item.get("precip") if item.get("precip") is not None else item.get("pcp1hr"),
            "metar": raw,
            "wxcodes": item.get("wxString"),
        }
        for idx in range(1, 5):
            layer = clouds[idx - 1] if idx <= len(clouds) and isinstance(clouds[idx - 1], dict) else {}
            row[f"skyc{idx}"] = layer.get("cover")
            row[f"skyl{idx}"] = layer.get("base")
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    raw_df = pd.DataFrame(rows)
    out = pd.DataFrame({"timestamp": raw_df["timestamp"]})
    out["station"] = raw_df.get("station", "KLAS")
    for col in ("temp_f", "dewpoint_f", "wind_dir_deg", "wind_speed_kt", "wind_gust_kt", "precip_in"):
        out[col] = pd.to_numeric(raw_df.get(col), errors="coerce")
    out["metar"] = raw_df.get("metar")
    out["wxcodes"] = raw_df.get("wxcodes")
    out["cloud_fraction"] = raw_df.apply(_cloud_fraction_from_codes, axis=1)
    out["cloud_fraction_below_12000"] = raw_df.apply(lambda row: _cloud_fraction_below(row, 12000.0), axis=1)
    out["cloud_fraction_below_20000"] = raw_df.apply(lambda row: _cloud_fraction_below(row, 20000.0), axis=1)
    out["lowest_bkn_ovc_ft"] = raw_df.apply(_lowest_bkn_ovc_ft, axis=1)
    for idx in range(1, 5):
        out[f"skyc{idx}"] = raw_df.get(f"skyc{idx}")
        out[f"skyl{idx}"] = pd.to_numeric(raw_df.get(f"skyl{idx}"), errors="coerce")
    out["thunder_observed"] = out["metar"].fillna("").str.contains(r"(?:^|\s)(?:\+|-)?TS", regex=True)
    extremes = out["metar"].apply(parse_temperature_extremes)
    out["metar_precise_temp_f"] = extremes.apply(lambda x: x.precise_temp_f)
    out["metar_precise_dewpoint_f"] = extremes.apply(lambda x: x.precise_dewpoint_f)
    out["six_hour_max_f"] = extremes.apply(lambda x: x.six_hour_max_f)
    out["six_hour_min_f"] = extremes.apply(lambda x: x.six_hour_min_f)
    out["daily_24h_max_f"] = extremes.apply(lambda x: x.daily_24h_max_f)
    out["daily_24h_min_f"] = extremes.apply(lambda x: x.daily_24h_min_f)
    out["data_source"] = "AviationWeather.gov"
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def fetch_awc_live_metars(
    hours: int = 24,
    station: str = "KLAS",
    timezone: str = "America/Los_Angeles",
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch recent official Aviation Weather Center METARs for live fallback use."""
    response = requests.get(
        AWC_METAR_URL,
        params={"ids": station, "format": "json", "hours": int(hours)},
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    if response.status_code == 204:
        return pd.DataFrame()
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected AviationWeather METAR response")
    return normalize_awc_metar_json(payload, timezone=timezone)


def fetch_live_asos(
    start: str,
    end: str,
    request: AsosRequest = AsosRequest(),
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch live KLAS observations from both IEM and AviationWeather.gov.

    The two sources may update at slightly different times. Merge both feeds so
    a lagging IEM response cannot hide a newer official AviationWeather METAR.
    For duplicate timestamps, prefer the IEM row because it contains the richer
    historical ASOS fields used by the model.
    """
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    # Source 1: Iowa Environmental Mesonet
    try:
        iem = fetch_asos(start, end, request=request, timeout=timeout)
        if not iem.empty:
            frames.append(iem)
    except Exception as exc:
        errors.append(f"IEM: {exc}")

    # Source 2: official Aviation Weather Center
    try:
        station = "K" + request.station if len(request.station) == 3 else request.station
        awc = fetch_awc_live_metars(
            hours=24,
            station=station,
            timezone=request.tz,
            timeout=min(timeout, 30),
        )
        if not awc.empty:
            frames.append(awc)
    except Exception as exc:
        errors.append(f"AviationWeather.gov: {exc}")

    if not frames:
        detail = "; ".join(errors) if errors else "both sources returned no observations"
        raise RuntimeError(
            f"No live KLAS METAR observations available. {detail}"
        )

    # Combine the feeds. IEM comes first, so on an identical timestamp its row
    # is retained; newer AWC-only timestamps are still added.
    obs = pd.concat(frames, ignore_index=True, sort=False)
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], errors="coerce")

    obs = (
        obs.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="first")
        .reset_index(drop=True)
    )

    # Keep only the requested Las Vegas calendar-day range.
    start_local = pd.Timestamp(start, tz=request.tz)
    end_local = pd.Timestamp(end, tz=request.tz) + pd.Timedelta(days=1)

    obs = obs[
        (obs["timestamp"] >= start_local)
        & (obs["timestamp"] < end_local)
    ].reset_index(drop=True)

    if obs.empty:
        raise RuntimeError(
            "Live METAR sources responded, but no observations matched the requested date."
        )

    return obs
