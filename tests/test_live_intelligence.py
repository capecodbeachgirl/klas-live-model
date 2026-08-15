from klas_model.live import combine_weather_intelligence


def test_forecast_thunder_can_raise_risk_without_observed_storm():
    risk, reasons, parts = combine_weather_intelligence(
        "LOW",
        ["No major live weather disruption signal"],
        {"available": True, "thunder_possible": True, "max_pop_pct": 40, "max_sky_cover_pct": 80},
        {"available": True, "risk": "LOW"},
        {"available": True, "risk": "LOW"},
    )
    assert risk == "HIGH"
    assert parts["forecast_risk"] == "HIGH"
    assert any("thunderstorm" in x.lower() for x in reasons)
