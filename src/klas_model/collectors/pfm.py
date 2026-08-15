from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo
from zipfile import ZipFile

import pandas as pd
import requests

IEM_AFOS_RETRIEVE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
LAS_VEGAS_TZ = ZoneInfo("America/Los_Angeles")

_MONTHS = {
    "JAN": 1, "JANUARY": 1,
    "FEB": 2, "FEBRUARY": 2,
    "MAR": 3, "MARCH": 3,
    "APR": 4, "APRIL": 4,
    "MAY": 5,
    "JUN": 6, "JUNE": 6,
    "JUL": 7, "JULY": 7,
    "AUG": 8, "AUGUST": 8,
    "SEP": 9, "SEPT": 9, "SEPTEMBER": 9,
    "OCT": 10, "OCTOBER": 10,
    "NOV": 11, "NOVEMBER": 11,
    "DEC": 12, "DECEMBER": 12,
}

_ISSUED_RE = re.compile(
    r"^\s*(\d{1,4})\s+(AM|PM)\s+(PST|PDT)\s+"
    r"(?:MON|TUE|WED|THU|FRI|SAT|SUN)\s+([A-Z]+)\s+(\d{1,2})\s+(\d{4})\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_DATE_TOKEN_RE = re.compile(r"\b(\d{2})/(\d{2})/(\d{2})\b")
_HOUR_RE = re.compile(r"(?<!\d)(\d{2})(?!\d)")
_VALUE_RE = re.compile(r"MM|-?\d+")


@dataclass(frozen=True)
class PfmDailyMax:
    forecast_date: date
    forecast_high_f: int
    issued_at: datetime
    source_filename: str | None = None


def _parse_issue_time(text: str) -> datetime:
    match = _ISSUED_RE.search(text)
    if not match:
        raise ValueError("Could not find PFM issue time")
    hhmm, ampm, _tz_abbr, month_name, day_s, year_s = match.groups()
    month = _MONTHS.get(month_name.upper())
    if month is None:
        raise ValueError(f"Unknown month in PFM issue time: {month_name}")

    hhmm = hhmm.zfill(4)
    hour = int(hhmm[:-2])
    minute = int(hhmm[-2:])
    if not (1 <= hour <= 12 and 0 <= minute <= 59):
        raise ValueError("Invalid PFM issue clock time")
    if ampm.upper() == "AM":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12

    return datetime(int(year_s), month, int(day_s), hour, minute, tzinfo=LAS_VEGAS_TZ)


def _las_vegas_matrix_lines(text: str) -> tuple[str, str, str]:
    """Return Date, local 3-hourly, and Min/Max lines for Las Vegas-Clark."""
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if "LAS VEGAS-CLARK NV" in line.upper())
    except StopIteration as exc:
        raise ValueError("Could not find Las Vegas-Clark PFM section") from exc

    date_line = None
    time_line = None
    maxmin_line = None
    for line in lines[start + 1 :]:
        if date_line is None and re.match(r"^\s*Date\s+", line, re.IGNORECASE):
            date_line = line
            continue
        if date_line is not None and time_line is None and re.match(
            r"^\s*(?:PST|PDT)\s+3hrly\s+", line, re.IGNORECASE
        ):
            time_line = line
            continue
        if time_line is not None and re.match(r"^\s*(?:Min/Max|Max/Min)\s+", line, re.IGNORECASE):
            maxmin_line = line
            break

    if not (date_line and time_line and maxmin_line):
        raise ValueError("Incomplete Las Vegas PFM matrix")
    return date_line, time_line, maxmin_line


def _first_matrix_date(date_line: str) -> date:
    match = _DATE_TOKEN_RE.search(date_line)
    if not match:
        raise ValueError("Could not find first date in PFM matrix")
    month, day, yy = map(int, match.groups())
    # Historical archive used here is modern; keep the usual 2000-2099 interpretation.
    return date(2000 + yy, month, day)


def _slot_datetimes(date_line: str, time_line: str) -> list[tuple[float, datetime]]:
    first_date = _first_matrix_date(date_line)
    matches = list(_HOUR_RE.finditer(time_line))
    # Exclude the one-digit "3" in "3hrly" automatically; only two-digit tokens match.
    if not matches:
        raise ValueError("Could not find 3-hour slots in PFM matrix")

    current_date = first_date
    previous_hour: int | None = None
    slots: list[tuple[float, datetime]] = []
    for match in matches:
        hour = int(match.group(1))
        if hour > 23:
            continue
        if previous_hour is not None and hour < previous_hour:
            current_date += timedelta(days=1)
        center = (match.start() + match.end()) / 2
        slots.append(
            (
                center,
                datetime.combine(current_date, time(hour=hour), tzinfo=LAS_VEGAS_TZ),
            )
        )
        previous_hour = hour
    return slots


def parse_pfm_daily_maxima(text: str, source_filename: str | None = None) -> list[PfmDailyMax]:
    """Parse daytime maximum forecasts from the first Las Vegas PFM matrix block.

    PFM Max/Min values are fixed-width and right-justified under an approximate
    12-hour ending slot. We map each value to the nearest 3-hour local-time column;
    values aligned to afternoon/evening columns are daytime maxima.
    """
    issued_at = _parse_issue_time(text)
    date_line, time_line, maxmin_line = _las_vegas_matrix_lines(text)
    slots = _slot_datetimes(date_line, time_line)

    label_match = re.match(r"^\s*(?:Min/Max|Max/Min)\s+", maxmin_line, re.IGNORECASE)
    if not label_match:
        raise ValueError("Could not parse PFM Min/Max line")

    maxima: list[PfmDailyMax] = []
    for token in _VALUE_RE.finditer(maxmin_line[label_match.end() :], pos=0):
        value_text = token.group(0)
        if value_text == "MM":
            continue
        # token positions are relative to the sliced line; translate back to full-line position.
        center = label_match.end() + (token.start() + token.end()) / 2
        _, slot_dt = min(slots, key=lambda item: abs(item[0] - center))
        # At KLAS, the daytime max is aligned to the afternoon column (typically 16 PST / 17 PDT).
        if slot_dt.hour < 12:
            continue
        maxima.append(
            PfmDailyMax(
                forecast_date=slot_dt.date(),
                forecast_high_f=int(value_text),
                issued_at=issued_at,
                source_filename=source_filename,
            )
        )
    return maxima


def parse_pfm_zip(
    content: bytes,
    start: str,
    end: str,
    cutoff_hour_local: int = 6,
    lookback_hours: int = 24,
) -> pd.DataFrame:
    """Choose one consistent morning NWS forecast per day from archived PFM products.

    For each target date we select the latest PFM issuance available by 06:00 Las Vegas
    local time (configurable), without using any later update. A maximum 24-hour lookback
    allows the prior-evening product to serve as the morning forecast if an overnight
    issuance is missing.
    """
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    candidates: list[dict[str, object]] = []

    with ZipFile(BytesIO(content)) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            text = archive.read(name).decode("utf-8", errors="replace")
            try:
                maxima = parse_pfm_daily_maxima(text, source_filename=name)
            except ValueError:
                continue
            for item in maxima:
                if not (start_date <= item.forecast_date <= end_date):
                    continue
                cutoff = datetime.combine(
                    item.forecast_date,
                    time(hour=cutoff_hour_local),
                    tzinfo=LAS_VEGAS_TZ,
                )
                earliest = cutoff - timedelta(hours=lookback_hours)
                if earliest <= item.issued_at <= cutoff:
                    candidates.append(
                        {
                            "date": item.forecast_date,
                            "nws_am_forecast_high_f": item.forecast_high_f,
                            "nws_am_issued_at": item.issued_at.isoformat(),
                            "nws_am_cutoff_local": cutoff.isoformat(),
                            "nws_am_lead_hours": (cutoff - item.issued_at).total_seconds() / 3600.0,
                            "pfm_source_filename": item.source_filename,
                        }
                    )

    columns = [
        "date",
        "nws_am_forecast_high_f",
        "nws_am_issued_at",
        "nws_am_cutoff_local",
        "nws_am_lead_hours",
        "pfm_source_filename",
    ]
    if not candidates:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(candidates)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["_issued"] = pd.to_datetime(df["nws_am_issued_at"], utc=True)
    df = (
        df.sort_values(["date", "_issued"])
        .groupby("date", as_index=False, sort=True)
        .tail(1)
        .drop(columns="_issued")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df[columns]


def fetch_pfm_morning_history(
    start: str,
    end: str,
    cutoff_hour_local: int = 6,
    timeout: int = 180,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError("end must be on or after start")

    # Broad UTC window covers the prior-evening fallback and target-day pre-dawn issuance.
    sdate = f"{(start_ts - pd.Timedelta(days=1)).date().isoformat()}T00:00Z"
    edate = f"{(end_ts + pd.Timedelta(days=1)).date().isoformat()}T00:00Z"
    params = {
        "pil": "PFMVEF",
        "fmt": "zip",
        "sdate": sdate,
        "edate": edate,
        "limit": "9999",
        "order": "asc",
    }
    response = requests.get(IEM_AFOS_RETRIEVE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return parse_pfm_zip(
        response.content,
        start=start,
        end=end,
        cutoff_hour_local=cutoff_hour_local,
    )


def save_pfm_history(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def merge_pfm_with_daily(daily: pd.DataFrame, pfm: pd.DataFrame) -> pd.DataFrame:
    left = daily.copy()
    right = pfm.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.date
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.date
    return left.merge(right, on="date", how="left", validate="one_to_one")
