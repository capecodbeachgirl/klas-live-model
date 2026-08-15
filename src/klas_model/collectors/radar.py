from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import requests

IMAGE_SERVER = (
    "https://mapservices.weather.noaa.gov/eventdriven/rest/services/radar/"
    "radar_base_reflectivity_time/ImageServer"
)
KLAS_LAT = 36.0801
KLAS_LON = -115.1522


def _destination(lat: float, lon: float, bearing_deg: float, miles: float) -> tuple[float, float]:
    radius_miles = 3958.7613
    angular = miles / radius_miles
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(angular) + math.cos(lat1) * math.sin(angular) * math.cos(bearing))
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def radar_sample_points(
    lat: float = KLAS_LAT,
    lon: float = KLAS_LON,
    radii_miles: tuple[int, ...] = (10, 25, 50),
    bearings: int = 16,
) -> list[dict[str, float]]:
    points: list[dict[str, float]] = [{"lat": lat, "lon": lon, "radius_miles": 0.0}]
    for radius in radii_miles:
        for i in range(bearings):
            bearing = i * (360.0 / bearings)
            plat, plon = _destination(lat, lon, bearing, radius)
            points.append({"lat": plat, "lon": plon, "radius_miles": float(radius)})
    return points


def _active_pixel(value: object) -> bool:
    """Detect a non-transparent radar echo in the rendered MRMS RGBA imagery."""
    if value is None:
        return False
    try:
        vals = [float(x) for x in str(value).replace(";", ",").split(",") if str(x).strip()]
    except ValueError:
        return False
    if not vals:
        return False
    if len(vals) >= 4:
        rgb = vals[:3]
        alpha = vals[3]
        return alpha > 0 and max(rgb) > 0
    if len(vals) >= 3:
        return max(vals[:3]) > 0 and not all(v >= 250 for v in vals[:3])
    return vals[0] > 0


def summarize_radar_samples(samples: list[dict[str, Any]], point_meta: list[dict[str, float]]) -> dict[str, Any]:
    active_by_radius: dict[float, int] = {}
    total_by_radius: dict[float, int] = {}
    active_distances: list[float] = []
    for sample, meta in zip(samples, point_meta):
        radius = float(meta["radius_miles"])
        total_by_radius[radius] = total_by_radius.get(radius, 0) + 1
        active = _active_pixel(sample.get("value"))
        if active:
            active_by_radius[radius] = active_by_radius.get(radius, 0) + 1
            active_distances.append(radius)
    fractions = {
        str(int(radius)): active_by_radius.get(radius, 0) / total
        for radius, total in total_by_radius.items() if total
    }
    nearest = min(active_distances) if active_distances else None
    return {"echo_fraction_by_radius": fractions, "nearest_echo_miles": nearest}


def compare_radar_scans(now: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    now_near = now.get("nearest_echo_miles")
    prior_near = prior.get("nearest_echo_miles")
    approaching = False
    if now_near is not None and prior_near is not None and now_near + 5 <= prior_near:
        approaching = True
    elif now_near is not None and prior_near is None and now_near <= 25:
        approaching = True

    if now_near is not None and now_near <= 10:
        risk = "HIGH"
        summary = "Radar echoes detected within about 10 miles of KLAS"
    elif now_near is not None and now_near <= 25:
        risk = "MEDIUM"
        summary = "Radar echoes detected within about 25 miles of KLAS"
    elif approaching:
        risk = "MEDIUM"
        summary = "Radar echoes appear to be moving closer to KLAS"
    else:
        risk = "LOW"
        summary = "No nearby radar echo signal detected by the coarse KLAS ring scan"
    return {"risk": risk, "approaching": approaching, "summary": summary}


def _sample_at(points: list[dict[str, float]], epoch_ms: int, timeout: int) -> list[dict[str, Any]]:
    geometry = {
        "points": [[p["lon"], p["lat"]] for p in points],
        "spatialReference": {"wkid": 4326},
    }
    response = requests.get(
        f"{IMAGE_SERVER}/getSamples",
        params={
            "geometryType": "esriGeometryMultipoint",
            "geometry": json.dumps(geometry, separators=(",", ":")),
            "returnFirstValueOnly": "true",
            "time": str(epoch_ms),
            "f": "json",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("samples", [])


def radar_export_url(lat: float = KLAS_LAT, lon: float = KLAS_LON, span_deg: float = 1.4) -> str:
    # Roughly a 75-mile wide view around KLAS. The image service contains radar only;
    # the dashboard overlays a center marker for the airport.
    bbox = [lon - span_deg / 2, lat - span_deg / 2, lon + span_deg / 2, lat + span_deg / 2]
    query = urlencode({
        "bbox": ",".join(f"{x:.4f}" for x in bbox),
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": "720,520",
        "format": "png32",
        "transparent": "true",
        "f": "image",
    })
    return f"{IMAGE_SERVER}/exportImage?{query}"


def fetch_radar_proximity(timeout: int = 30) -> dict[str, Any]:
    points = radar_sample_points()
    meta = requests.get(IMAGE_SERVER, params={"f": "json"}, timeout=timeout)
    meta.raise_for_status()
    payload = meta.json()
    extent = ((payload.get("timeInfo") or {}).get("timeExtent") or [])
    if len(extent) < 2 or extent[1] is None:
        raise RuntimeError("MRMS radar service did not provide a current time extent")
    latest_ms = int(extent[1])
    prior_ms = latest_ms - 30 * 60 * 1000
    current_samples = _sample_at(points, latest_ms, timeout)
    prior_samples = _sample_at(points, prior_ms, timeout)
    # Some services may return fewer samples when a point is outside a raster footprint.
    current = summarize_radar_samples(current_samples, points[: len(current_samples)])
    prior = summarize_radar_samples(prior_samples, points[: len(prior_samples)])
    trend = compare_radar_scans(current, prior)
    return {
        "available": True,
        "latest_time_utc": datetime.fromtimestamp(latest_ms / 1000, tz=timezone.utc).isoformat(),
        "prior_time_utc": datetime.fromtimestamp(prior_ms / 1000, tz=timezone.utc).isoformat(),
        "current": current,
        "prior": prior,
        **trend,
        "image_url": radar_export_url(),
        "source": "NWS MRMS base reflectivity image service",
    }
