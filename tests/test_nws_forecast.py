from datetime import datetime
from zoneinfo import ZoneInfo

from klas_model.collectors.nws_forecast import summarize_hourly_forecast

TZ = ZoneInfo("America/Los_Angeles")


def test_hourly_forecast_detects_thunder_and_pop():
    periods = [
        {
            "startTime": "2026-08-15T13:00:00-07:00",
            "temperature": 94,
            "probabilityOfPrecipitation": {"value": 20},
            "shortForecast": "Mostly Sunny",
        },
        {
            "startTime": "2026-08-15T14:00:00-07:00",
            "temperature": 95,
            "probabilityOfPrecipitation": {"value": 40},
            "shortForecast": "Chance Thunderstorms",
        },
    ]
    out = summarize_hourly_forecast(periods, datetime(2026, 8, 15, 12, 30, tzinfo=TZ))
    assert out["available"] is True
    assert out["thunder_possible"] is True
    assert out["max_pop_pct"] == 40
