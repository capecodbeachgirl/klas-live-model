from __future__ import annotations

import re
from dataclasses import dataclass


_T_GROUP_RE = re.compile(r"(?:^|\s)T([01])(\d{3})([01])(\d{3})(?=\s|$)")
_SIX_HOUR_MAX_RE = re.compile(r"(?:^|\s)1([01])(\d{3})(?=\s|$)")
_SIX_HOUR_MIN_RE = re.compile(r"(?:^|\s)2([01])(\d{3})(?=\s|$)")
_DAILY_MAX_MIN_RE = re.compile(r"(?:^|\s)4([01])(\d{3})([01])(\d{3})(?=\s|$)")


def _tenths_c(sign_digit: str, digits: str) -> float:
    value = int(digits) / 10.0
    return -value if sign_digit == "1" else value


def c_to_f(value_c: float) -> float:
    return value_c * 9.0 / 5.0 + 32.0


def _near_integer_f(value_c: float) -> int:
    """Reverse the METAR tenths-C encoding back to the nearest whole Fahrenheit."""
    return int(round(c_to_f(value_c)))


@dataclass(frozen=True)
class MetarTemperatureExtremes:
    precise_temp_f: float | None = None
    precise_dewpoint_f: float | None = None
    six_hour_max_f: int | None = None
    six_hour_min_f: int | None = None
    daily_24h_max_f: int | None = None
    daily_24h_min_f: int | None = None


def parse_temperature_extremes(metar: object) -> MetarTemperatureExtremes:
    if not isinstance(metar, str) or not metar.strip():
        return MetarTemperatureExtremes()

    precise_temp_f = precise_dewpoint_f = None
    six_hour_max_f = six_hour_min_f = None
    daily_24h_max_f = daily_24h_min_f = None

    match = _T_GROUP_RE.search(metar)
    if match:
        temp_c = _tenths_c(match.group(1), match.group(2))
        dew_c = _tenths_c(match.group(3), match.group(4))
        precise_temp_f = c_to_f(temp_c)
        precise_dewpoint_f = c_to_f(dew_c)

    match = _SIX_HOUR_MAX_RE.search(metar)
    if match:
        six_hour_max_f = _near_integer_f(_tenths_c(match.group(1), match.group(2)))

    match = _SIX_HOUR_MIN_RE.search(metar)
    if match:
        six_hour_min_f = _near_integer_f(_tenths_c(match.group(1), match.group(2)))

    match = _DAILY_MAX_MIN_RE.search(metar)
    if match:
        daily_24h_max_f = _near_integer_f(_tenths_c(match.group(1), match.group(2)))
        daily_24h_min_f = _near_integer_f(_tenths_c(match.group(3), match.group(4)))

    return MetarTemperatureExtremes(
        precise_temp_f=precise_temp_f,
        precise_dewpoint_f=precise_dewpoint_f,
        six_hour_max_f=six_hour_max_f,
        six_hour_min_f=six_hour_min_f,
        daily_24h_max_f=daily_24h_max_f,
        daily_24h_min_f=daily_24h_min_f,
    )
