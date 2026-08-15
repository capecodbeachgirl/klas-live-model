from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

STAR_BASE = "https://www.star.nesdis.noaa.gov/"
STAR_WFO_BAND = "https://www.star.nesdis.noaa.gov/goes/wfo.php"
USER_AGENT = "KLAS-Kalshi-Research/0.14 (weather research dashboard)"


def _latest_star_image(
    band: str,
    wfo: str = "vef",
    timeout: int = 30,
) -> tuple[str | None, str | None]:
    """Return the latest NOAA STAR 600x600 WFO image URL and UTC timestamp."""
    response = requests.get(
        STAR_WFO_BAND,
        params={"wfo": wfo},
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    text = response.text

    band_part = re.escape(str(band))
    wfo_part = re.escape(str(wfo))

    pattern = (
        rf'https://cdn\.star\.nesdis\.noaa\.gov/WFO/'
        rf'{wfo_part}/{band_part}/'
        rf'(\d{{11}})_GOES\d+-ABI-{wfo_part}-{band_part}-600x600\.jpg'
    )

    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    if not matches:
        return None, None

    stamp = matches[-1]

    full_pattern = (
        rf'(https://cdn\.star\.nesdis\.noaa\.gov/WFO/'
        rf'{wfo_part}/{band_part}/'
        rf'{stamp}_GOES\d+-ABI-{wfo_part}-{band_part}-600x600\.jpg)'
    )

    url_matches = re.findall(full_pattern, text, flags=re.IGNORECASE)
    image_url = url_matches[-1] if url_matches else None

    image_time = None
    try:
        dt = datetime.strptime(
            stamp, "%Y%j%H%M"
        ).replace(tzinfo=timezone.utc)
        image_time = dt.isoformat()
    except ValueError:
        pass

    return image_url, image_time


def _numeric_series(obs: pd.DataFrame, column: str) -> pd.Series:
    if column not in obs:
        return pd.Series(dtype=float)
    return pd.to_numeric(obs[column], errors="coerce")


def fetch_satellite_cloud_watch(
    obs: pd.DataFrame,
    timeout: int = 30,
) -> dict[str, Any]:
    """Fetch live Las Vegas GOES imagery and summarize KLAS cloud shading.

    Version 1 keeps the satellite imagery as a visual cross-check.
    The LOW/MEDIUM/HIGH shading label uses actual KLAS METAR sky-cover reports.
    It does not alter the validated high-temperature prediction.
    """
    geocolor_url = None
    geocolor_time = None
    infrared_url = None
    infrared_time = None
    errors: list[str] = []

    try:
        geocolor_url, geocolor_time = _latest_star_image(
            "GEOCOLOR", timeout=timeout
        )
    except Exception as exc:
        errors.append(f"GeoColor: {exc}")

    try:
        infrared_url, infrared_time = _latest_star_image(
            "13", timeout=timeout
        )
    except Exception as exc:
        errors.append(f"Band 13: {exc}")

    if obs.empty:
        return {
            "available": bool(geocolor_url or infrared_url),
            "risk": "UNKNOWN",
            "summary": "No KLAS sky-cover observations available",
            "latest_cloud_fraction": None,
            "mean_cloud_fraction_2h": None,
            "cloud_fraction_trend_2h": None,
            "latest_cloud_fraction_below_20000": None,
            "geocolor_image_url": geocolor_url,
            "geocolor_time_utc": geocolor_time,
            "infrared_image_url": infrared_url,
            "infrared_time_utc": infrared_time,
            "errors": errors,
            "source": "NOAA/NESDIS/STAR GOES imagery",
        }

    work = obs.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp")

    latest_time = work["timestamp"].max()
    recent = work[
        work["timestamp"] >= latest_time - pd.Timedelta(hours=2)
    ].copy()

    cloud = _numeric_series(recent, "cloud_fraction").dropna()
    below_20k = _numeric_series(
        recent, "cloud_fraction_below_20000"
    ).dropna()

    latest_cloud = None if cloud.empty else float(cloud.iloc[-1])
    mean_cloud = None if cloud.empty else float(cloud.mean())
    latest_below_20k = (
        None if below_20k.empty else float(below_20k.iloc[-1])
    )

    trend = None
    if len(cloud) >= 2:
        trend = float(cloud.iloc[-1] - cloud.iloc[0])

    if latest_cloud is None:
        risk = "UNKNOWN"
        summary = "GOES imagery available; KLAS sky-cover fraction unavailable"
    elif latest_cloud >= 0.875 or (
        mean_cloud is not None and mean_cloud >= 0.75
    ):
        risk = "HIGH"
        summary = (
            "Heavy observed cloud cover may be suppressing solar heating at KLAS"
        )
    elif (
        latest_cloud >= 0.50
        or (mean_cloud is not None and mean_cloud >= 0.50)
        or (trend is not None and trend >= 0.25)
    ):
        risk = "MEDIUM"
        summary = "Meaningful cloud shading is present or increasing at KLAS"
    else:
        risk = "LOW"
        summary = "Observed KLAS sky cover suggests limited cloud shading"

    return {
        "available": bool(geocolor_url or infrared_url),
        "risk": risk,
        "summary": summary,
        "latest_cloud_fraction": latest_cloud,
        "mean_cloud_fraction_2h": mean_cloud,
        "cloud_fraction_trend_2h": trend,
        "latest_cloud_fraction_below_20000": latest_below_20k,
        "geocolor_image_url": geocolor_url,
        "geocolor_time_utc": geocolor_time,
        "infrared_image_url": infrared_url,
        "infrared_time_utc": infrared_time,
        "errors": errors,
        "image_note": (
            "GOES images are a visual cross-check in this version. "
            "The shading label uses KLAS-observed sky cover and does not "
            "alter the validated temperature prediction."
        ),
        "source": "NOAA/NESDIS/STAR GOES imagery + KLAS METAR sky cover",
    }
