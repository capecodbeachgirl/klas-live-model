from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo
from zipfile import ZipFile

import pandas as pd
import requests

IEM_AFOS_RETRIEVE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
LAS_VEGAS_TZ = ZoneInfo("America/Los_Angeles")


@dataclass(frozen=True)
class CliDailyHigh:
    climate_date: date
    high_f: int
    peak_time_text: str | None
    peak_time_flags: str | None
    period_label: str
    issued_at: datetime | None = None


_DATE_RE = re.compile(
    r"THE LAS VEGAS NV CLIMATE SUMMARY FOR\s+([A-Z]+)\s+(\d{1,2})\s+(\d{4})",
    re.IGNORECASE,
)
_MAX_RE = re.compile(
    r"^\s*MAXIMUM\s+(-?\d+)\s*(.*?)\s+(?:-?\d+|MM)\s+(?:\d{4}|MM)",
    re.IGNORECASE | re.MULTILINE,
)
_PERIOD_RE = re.compile(r"^\s*(TODAY|YESTERDAY)\s*$", re.IGNORECASE | re.MULTILINE)
_ISSUED_RE = re.compile(
    r"^\s*(\d{1,4})\s+(AM|PM)\s+(?:PST|PDT)\s+"
    r"(?:MON|TUE|WED|THU|FRI|SAT|SUN)\s+([A-Z]+)\s+(\d{1,2})\s+(\d{4})\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_MONTH_NAMES = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_MONTHS: dict[str, int] = {}
for number, name in enumerate(_MONTH_NAMES):
    if name:
        _MONTHS[name.upper()] = number
        _MONTHS[name[:3].upper()] = number


def _parse_issue_time(text: str) -> datetime | None:
    match = _ISSUED_RE.search(text)
    if not match:
        return None
    hhmm, ampm, month_name, day_s, year_s = match.groups()
    month = _MONTHS.get(month_name.upper())
    if month is None:
        return None

    # NWS product headers commonly use compact clock strings such as 140 or 1223.
    hhmm = hhmm.zfill(4)
    hour = int(hhmm[:-2])
    minute = int(hhmm[-2:])
    if hour < 1 or hour > 12 or minute > 59:
        return None
    if ampm.upper() == "AM":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12

    return datetime(
        int(year_s), month, int(day_s), hour, minute, tzinfo=LAS_VEGAS_TZ
    )


def _clean_peak_time(raw: str | None) -> tuple[str | None, str | None]:
    if not raw:
        return None, None
    value = " ".join(raw.strip().split())
    # CLI may prefix the max-time field with a record flag such as R. Keep the
    # flag separately so peak-time parsing/display stays clean.
    match = re.match(
        r"^(?:(?P<flags>[A-Z*]+)\s+)?(?P<time>\d{1,2}:\d{2}\s*[AP]M|\d{3,4}\s*[AP]M)$",
        value,
        re.IGNORECASE,
    )
    if not match:
        return value, None
    flags = match.group("flags")
    peak = match.group("time").upper()
    compact = re.match(r"^(\d{1,2})(\d{2})\s*([AP]M)$", peak)
    if compact and ":" not in peak:
        peak = f"{int(compact.group(1))}:{compact.group(2)} {compact.group(3)}"
    else:
        peak = re.sub(r"\s+", " ", peak)
    return peak, flags.upper() if flags else None


def parse_cli_daily_high(text: str) -> CliDailyHigh:
    """Parse the official Las Vegas NWS CLI daily maximum from product text.

    This parser intentionally targets CLILAS wording. It reads the climate-summary
    date from the title, then the first MAXIMUM line in the temperature section.
    """
    date_match = _DATE_RE.search(text)
    if not date_match:
        raise ValueError("Could not find Las Vegas climate-summary date in CLI text")
    month_name, day_s, year_s = date_match.groups()
    month = _MONTHS.get(month_name.upper())
    if month is None:
        raise ValueError(f"Unknown month in CLI text: {month_name}")
    climate_date = date(int(year_s), month, int(day_s))

    max_match = _MAX_RE.search(text)
    if not max_match:
        raise ValueError("Could not find MAXIMUM temperature in CLI text")
    high_f = int(max_match.group(1))
    peak_time, peak_flags = _clean_peak_time(max_match.group(2).strip() or None)

    period_match = _PERIOD_RE.search(text)
    period_label = period_match.group(1).upper() if period_match else "UNKNOWN"

    return CliDailyHigh(
        climate_date=climate_date,
        high_f=high_f,
        peak_time_text=peak_time,
        peak_time_flags=peak_flags,
        period_label=period_label,
        issued_at=_parse_issue_time(text),
    )


def _decode_product(payload: bytes) -> str:
    # NWS text products are ASCII-ish; latin-1 preserves every byte if odd characters occur.
    return payload.decode("utf-8", errors="replace")


def parse_cli_zip(content: bytes, start: str, end: str) -> pd.DataFrame:
    """Parse an IEM AFOS CLILAS ZIP and keep the final report for each climate date.

    Final next-morning products label the temperature period as YESTERDAY. Same-day
    afternoon products are preliminary (TODAY) and are never used as the settlement
    target when a final product is available.
    """
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    rows: list[dict[str, object]] = []

    with ZipFile(BytesIO(content)) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            text = _decode_product(archive.read(name))
            try:
                parsed = parse_cli_daily_high(text)
            except ValueError:
                continue
            if not (start_date <= parsed.climate_date <= end_date):
                continue
            rows.append(
                {
                    "date": parsed.climate_date,
                    "actual_cli_high_f": parsed.high_f,
                    "cli_peak_time_text": parsed.peak_time_text,
                    "cli_peak_time_flags": parsed.peak_time_flags,
                    "cli_period_label": parsed.period_label,
                    "cli_issued_at": parsed.issued_at.isoformat() if parsed.issued_at else None,
                    "cli_source_filename": name,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "actual_cli_high_f",
                "cli_peak_time_text",
                "cli_peak_time_flags",
                "cli_period_label",
                "cli_issued_at",
                "cli_source_filename",
            ]
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    issued = pd.to_datetime(df["cli_issued_at"], errors="coerce", utc=True)
    df["_issued_sort"] = issued
    df["_final_rank"] = df["cli_period_label"].eq("YESTERDAY").astype(int)

    # Prefer a YESTERDAY/final product, then the latest issuance if duplicates exist.
    df = (
        df.sort_values(["date", "_final_rank", "_issued_sort"], na_position="first")
        .groupby("date", as_index=False, sort=True)
        .tail(1)
        .drop(columns=["_issued_sort", "_final_rank"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # A TODAY-only record is preliminary and must not silently become our Kalshi target.
    df["cli_is_final"] = df["cli_period_label"].eq("YESTERDAY")
    return df


def fetch_cli_history(
    start: str,
    end: str,
    timeout: int = 120,
) -> pd.DataFrame:
    """Download archived CLILAS products and return one row per requested climate date.

    The IEM AFOS API uses UTC issuance timestamps and an exclusive `edate`. We request
    through two UTC midnights after the final climate date so the next-morning final
    YESTERDAY product is included.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError("end must be on or after start")

    sdate = f"{start_ts.date().isoformat()}T00:00Z"
    edate = f"{(end_ts + pd.Timedelta(days=2)).date().isoformat()}T00:00Z"
    params = {
        "pil": "CLILAS",
        "fmt": "zip",
        "sdate": sdate,
        "edate": edate,
        "limit": "9999",
        "order": "asc",
    }
    response = requests.get(IEM_AFOS_RETRIEVE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return parse_cli_zip(response.content, start=start, end=end)


def save_cli_history(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def merge_cli_with_heating(daily: pd.DataFrame, cli: pd.DataFrame) -> pd.DataFrame:
    """Merge final CLI highs into a KLAS daily heating table by local climate date."""
    left = daily.copy()
    right = cli.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.date
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.date
    if "cli_is_final" in right.columns:
        right = right[right["cli_is_final"].fillna(False)].copy()
    return left.merge(right, on="date", how="left", validate="one_to_one")
