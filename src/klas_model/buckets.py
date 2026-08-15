from __future__ import annotations


def two_degree_bucket(temp_f: int, anchor_even: bool = False) -> str:
    """Return a generic 2°F bucket label.

    Kalshi strike sets can move day to day, so the exact market strike mapping should be
    read from the market when available. This helper is for historical grouping only.
    """
    if anchor_even:
        low = temp_f if temp_f % 2 == 0 else temp_f - 1
    else:
        low = temp_f if temp_f % 2 == 1 else temp_f - 1
    return f"{low}-{low + 1}"


def bucket_for_strikes(temp_f: int, strikes: list[tuple[int | None, int | None, str]]) -> str | None:
    """Map an integer settlement high to the day's actual Kalshi strike definitions."""
    for low, high, label in strikes:
        if low is None and high is not None and temp_f <= high:
            return label
        if high is None and low is not None and temp_f >= low:
            return label
        if low is not None and high is not None and low <= temp_f <= high:
            return label
    return None
