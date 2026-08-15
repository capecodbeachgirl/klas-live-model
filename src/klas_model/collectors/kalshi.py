from __future__ import annotations

from datetime import date
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
LAS_VEGAS_TZ = ZoneInfo("America/Los_Angeles")


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_market(market: dict[str, Any]) -> dict[str, Any]:
    bid = _float_or_none(market.get("yes_bid_dollars"))
    ask = _float_or_none(market.get("yes_ask_dollars"))
    last = _float_or_none(market.get("last_price_dollars"))
    if bid is not None and ask is not None:
        midpoint = (bid + ask) / 2.0
    else:
        midpoint = last if last is not None else bid if bid is not None else ask
    return {
        "ticker": market.get("ticker"),
        "event_ticker": market.get("event_ticker"),
        "title": market.get("title"),
        "subtitle": market.get("subtitle") or market.get("yes_sub_title") or market.get("title"),
        "strike_type": market.get("strike_type"),
        "floor_strike": _float_or_none(market.get("floor_strike")),
        "cap_strike": _float_or_none(market.get("cap_strike")),
        "yes_bid": bid,
        "yes_ask": ask,
        "last_price": last,
        "market_mid": midpoint,
        "close_time": market.get("close_time"),
        "occurrence_datetime": market.get("occurrence_datetime"),
        "updated_time": market.get("updated_time"),
    }


def fetch_open_temperature_markets(
    series_ticker: str = "KXHIGHTLV",
    timeout: int = 30,
) -> list[dict[str, Any]]:
    response = requests.get(
        f"{KALSHI_BASE_URL}/markets",
        params={"series_ticker": series_ticker, "status": "open", "limit": 1000},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return [normalize_market(m) for m in payload.get("markets", [])]


def select_event_markets(markets: list[dict[str, Any]], target_date: date) -> list[dict[str, Any]]:
    if not markets:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for market in markets:
        event = str(market.get("event_ticker") or "")
        grouped.setdefault(event, []).append(market)

    def group_date(rows: list[dict[str, Any]]) -> date | None:
        for key in ("occurrence_datetime", "close_time"):
            raw = rows[0].get(key)
            if raw:
                ts = pd.to_datetime(raw, errors="coerce", utc=True)
                if not pd.isna(ts):
                    return ts.tz_convert(LAS_VEGAS_TZ).date()
        return None

    ranked: list[tuple[int, int, str, list[dict[str, Any]]]] = []
    for event, rows in grouped.items():
        d = group_date(rows)
        if d is None:
            distance = 9999
            future_penalty = 1
        else:
            delta = (d - target_date).days
            distance = abs(delta)
            future_penalty = 0 if delta >= 0 else 1
        ranked.append((distance, future_penalty, event, rows))
    ranked.sort(key=lambda x: (x[0], x[1], x[2]))
    return sorted(ranked[0][3], key=lambda r: (r.get("floor_strike") is None, r.get("floor_strike") or -999))
