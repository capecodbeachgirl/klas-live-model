from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

import numpy as np


_RANGE_RE = re.compile(r"(?P<lo>-?\d+(?:\.\d+)?)\s*°?\s*(?:to|[-–—])\s*(?P<hi>-?\d+(?:\.\d+)?)\s*°?", re.I)
_BELOW_RE = re.compile(r"(?P<x>-?\d+(?:\.\d+)?)\s*°?\s*(?:or\s*)?(?:below|lower|less)", re.I)
_ABOVE_RE = re.compile(r"(?P<x>-?\d+(?:\.\d+)?)\s*°?\s*(?:or\s*)?(?:above|higher|greater)", re.I)


def empirical_integer_probabilities(
    predicted_high_f: float,
    historical_model_errors_f: Iterable[float],
    *,
    floor_f: int | None = None,
    smoothing: float = 0.35,
    padding_f: int = 3,
) -> dict[int, float]:
    """Turn held-out model errors into a discrete official-high distribution.

    Historical error is prediction - actual, so a plausible live actual is
    ``live_prediction - historical_error``. A small smoothing term avoids false 0%
    probabilities from a modest calibration sample.
    """
    errors = np.asarray([float(x) for x in historical_model_errors_f if x is not None], dtype=float)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        errors = np.asarray([0.0], dtype=float)

    samples = np.rint(float(predicted_high_f) - errors).astype(int)
    center = int(round(float(predicted_high_f)))
    lo = min(int(samples.min()), center - padding_f)
    hi = max(int(samples.max()), center + padding_f)
    if floor_f is not None:
        lo = min(lo, int(floor_f))

    counts = Counter(int(x) for x in samples)
    weights: dict[int, float] = {}
    for temp in range(lo, hi + 1):
        if floor_f is not None and temp < int(floor_f):
            weights[temp] = 0.0
        else:
            weights[temp] = float(counts.get(temp, 0)) + float(smoothing)

    total = sum(weights.values())
    if total <= 0:
        return {center: 1.0}
    return {temp: weight / total for temp, weight in weights.items() if weight > 0}


def probability_for_bounds(
    distribution: dict[int, float],
    floor_strike: float | None,
    cap_strike: float | None,
) -> float | None:
    """Legacy inclusive numeric-bounds helper.

    Use :func:`probability_for_market` for Kalshi markets because the API strike
    boundaries can be threshold boundaries rather than literal inclusive integer values.
    """
    if floor_strike is None and cap_strike is None:
        return None
    total = 0.0
    for temp, probability in distribution.items():
        if floor_strike is not None and temp < floor_strike:
            continue
        if cap_strike is not None and temp > cap_strike:
            continue
        total += probability
    return total


def parse_temperature_bucket_label(label: object) -> tuple[int | None, int | None] | None:
    """Parse human-readable Kalshi temperature subtitles into integer CLI bounds.

    Examples:
      * ``96° to 97°`` -> (96, 97)
      * ``91° or below`` -> (None, 91)
      * ``100° or above`` -> (100, None)

    The official KLAS settlement target is an integer Fahrenheit daily high, so using the
    displayed bucket wording is safer than assuming API floor/cap thresholds are inclusive.
    """
    if not isinstance(label, str):
        return None
    text = label.strip()
    m = _RANGE_RE.search(text)
    if m:
        return int(round(float(m.group("lo")))), int(round(float(m.group("hi"))))
    m = _BELOW_RE.search(text)
    if m:
        return None, int(round(float(m.group("x"))))
    m = _ABOVE_RE.search(text)
    if m:
        return int(round(float(m.group("x")))), None
    return None


def probability_for_market(distribution: dict[int, float], market: dict) -> float | None:
    """Return model probability for a normalized Kalshi temperature market.

    Prefer the displayed bucket wording because it directly describes the integer-Fahrenheit
    outcome the user sees. Fall back to strike-type-aware numeric boundaries when needed.
    """
    parsed = parse_temperature_bucket_label(market.get("subtitle") or market.get("title"))
    if parsed is not None:
        lo, hi = parsed
        return sum(
            p
            for temp, p in distribution.items()
            if (lo is None or temp >= lo) and (hi is None or temp <= hi)
        )

    strike_type = str(market.get("strike_type") or "").lower()
    floor = market.get("floor_strike")
    cap = market.get("cap_strike")

    # Kalshi numeric strike values are often threshold boundaries. For discrete whole-degree
    # highs, treat between as [floor, cap), greater as > floor, and less as < cap.
    if strike_type == "between" and floor is not None and cap is not None:
        return sum(p for temp, p in distribution.items() if temp >= float(floor) and temp < float(cap))
    if strike_type == "greater" and floor is not None:
        return sum(p for temp, p in distribution.items() if temp > float(floor))
    if strike_type == "less" and cap is not None:
        return sum(p for temp, p in distribution.items() if temp < float(cap))

    return probability_for_bounds(distribution, floor, cap)


def central_range(distribution: dict[int, float], mass: float = 0.80) -> tuple[int, int]:
    if not distribution:
        raise ValueError("distribution is empty")
    items = sorted(distribution.items())
    tail = (1.0 - mass) / 2.0
    cumulative = 0.0
    low = items[0][0]
    high = items[-1][0]
    low_set = False
    for temp, prob in items:
        cumulative += prob
        if not low_set and cumulative >= tail:
            low = temp
            low_set = True
        if cumulative >= 1.0 - tail:
            high = temp
            break
    return low, high
