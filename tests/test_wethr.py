from datetime import datetime
from zoneinfo import ZoneInfo

from klas_model.collectors.wethr import (
    apply_observed_floor,
    fetch_wethr_high,
    summarize_latest_run,
)


LAS_TZ = ZoneInfo("America/Los_Angeles")


def test_wethr_full_contract_run():
    now = datetime(
        2026, 8, 16, 10, 0,
        tzinfo=LAS_TZ,
    )

    rows = [
        {
            "valid_time": "2026-08-16T17:00:00Z",
            "temperature_f": 95.0,
            "forecast_hour": 5,
        },
        {
            "valid_time": "2026-08-16T22:00:00Z",
            "temperature_f": 101.5,
            "forecast_hour": 10,
        },
        {
            "valid_time": "2026-08-17T08:00:00Z",
            "temperature_f": 88.0,
            "forecast_hour": 20,
        },
    ]

    result = summarize_latest_run(
        "TEST",
        rows,
        "2026-08-16 12:00:00",
        now,
    )

    assert result["available"] is True
    assert result["covers_rest_of_contract"] is True
    assert result["remaining_high_f"] == 101.5
    assert result["max_forecast_hour"] == 20
    assert (
        result["remaining_high_time_local"]
        == "2026-08-16T15:00:00-07:00"
    )


def test_wethr_partial_run_is_flagged():
    now = datetime(
        2026, 8, 16, 10, 0,
        tzinfo=LAS_TZ,
    )

    rows = [
        {
            "valid_time": "2026-08-16T17:00:00Z",
            "temperature_f": 95.0,
            "forecast_hour": 5,
        },
        {
            "valid_time": "2026-08-16T19:00:00Z",
            "temperature_f": 98.0,
            "forecast_hour": 7,
        },
    ]

    result = summarize_latest_run(
        "TEST",
        rows,
        "2026-08-16 12:00:00",
        now,
    )

    assert result["available"] is True
    assert result["covers_rest_of_contract"] is False
    assert result["remaining_high_f"] == 98.0

def test_wethr_observed_floor_recomputes_consensus():
    snapshot = {
        "models": {
            "RAP": {
                "available": True,
                "covers_rest_of_contract": True,
                "remaining_high_f": 101.0,
            },
            "NBM": {
                "available": True,
                "covers_rest_of_contract": True,
                "remaining_high_f": 99.5,
            },
            "HRRR": {
                "available": True,
                "covers_rest_of_contract": False,
                "remaining_high_f": 104.0,
            },
        },
        "consensus": {
            "available": True,
        },
    }

    result = apply_observed_floor(
        snapshot,
        102.0,
    )

    assert result["observed_floor_f"] == 102.0
    assert result["models"]["RAP"]["projected_high_f"] == 102.0
    assert result["models"]["NBM"]["projected_high_f"] == 102.0
    assert "projected_high_f" not in result["models"]["HRRR"]

    consensus = result["consensus"]

    assert consensus["model_count"] == 2
    assert consensus["median_high_f"] == 102.0
    assert consensus["min_high_f"] == 102.0
    assert consensus["max_high_f"] == 102.0

def test_fetch_wethr_high_parses_omo(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "wethr_high": 89.0,
                "time_of_high_utc": "2026-08-16 16:43:00",
                "wethr_high_sources": ["omo", "hf_metar"],
                "wethr_high_source_detail": {
                    "omo": {
                        "first_confirmed_utc": "2026-08-16 16:29:00",
                        "count": 15,
                    }
                },
                "data_quality": {
                    "nws": {
                        "ok": True,
                        "warnings": [],
                    }
                },
            }

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setenv("WETHR_API_KEY", "test-key")
    monkeypatch.setattr(
        "klas_model.collectors.wethr.requests.get",
        fake_get,
    )

    result = fetch_wethr_high()

    assert result["available"] is True
    assert result["wethr_high_f"] == 89.0
    assert result["omo_informed"] is True
    assert result["sources"] == ["omo", "hf_metar"]
    assert result["data_quality"]["nws"]["ok"] is True