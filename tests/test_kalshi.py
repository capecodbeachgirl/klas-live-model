from datetime import date
from klas_model.collectors.kalshi import normalize_market, select_event_markets


def test_normalize_market_midpoint_and_strike_type():
    m = normalize_market({"ticker":"X","event_ticker":"E","yes_bid_dollars":"0.30","yes_ask_dollars":"0.40","floor_strike":109,"cap_strike":110,"strike_type":"between"})
    assert m["market_mid"] == .35
    assert m["floor_strike"] == 109
    assert m["strike_type"] == "between"


def test_select_today_event():
    rows = [
        {"event_ticker":"E1","occurrence_datetime":"2026-08-15T20:00:00Z","floor_strike":109},
        {"event_ticker":"E2","occurrence_datetime":"2026-08-16T20:00:00Z","floor_strike":109},
    ]
    picked = select_event_markets(rows, date(2026,8,15))
    assert picked[0]["event_ticker"] == "E1"
