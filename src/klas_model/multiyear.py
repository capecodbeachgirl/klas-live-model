from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SeasonWindow:
    year: int
    start: date
    end: date


def summer_windows(
    years: Iterable[int],
    *,
    start_month: int = 6,
    start_day: int = 1,
    end_month: int = 9,
    end_day: int = 30,
    through: str | date | None = None,
) -> list[SeasonWindow]:
    """Return one inclusive warm-season window per requested year.

    If ``through`` falls within a requested year, that year's window is capped at
    that date. Years after ``through`` are omitted. This is useful for an incomplete
    current season such as 2026-08-14.
    """
    through_date: date | None
    if through is None:
        through_date = None
    elif isinstance(through, date):
        through_date = through
    else:
        through_date = pd.Timestamp(through).date()

    out: list[SeasonWindow] = []
    for year in sorted(set(int(y) for y in years)):
        start = date(year, start_month, start_day)
        end = date(year, end_month, end_day)
        if through_date is not None:
            if start > through_date:
                continue
            if year == through_date.year:
                end = min(end, through_date)
        if end >= start:
            out.append(SeasonWindow(year=year, start=start, end=end))
    return out


def concat_dedupe(frames: list[pd.DataFrame], key: str) -> pd.DataFrame:
    """Concatenate collector outputs, keeping the latest duplicate by ``key``."""
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        return pd.DataFrame()
    out = pd.concat(usable, ignore_index=True)
    if key in out.columns:
        out = out.drop_duplicates(subset=[key], keep="last")
        out = out.sort_values(key).reset_index(drop=True)
    return out
