from klas_model.dashboard import render_dashboard


def test_dashboard_has_core_cards_and_actionable_edge_label():
    html = render_dashboard({
        "updated_at_local":"2026-08-15T10:52:00-07:00",
        "latest_metar_time":"2026-08-15T09:56:00-07:00",
        "latest_temp_f":101,
        "latest_precise_temp_f":101.3,
        "nws_am_forecast_high_f":109,
        "raw_metar_peak_f":103,
        "precise_metar_peak_f":103.4,
        "six_hour_max_f":104,
        "model_available":True,
        "model_predicted_high_f":108.8,
        "model_correction_f":-0.2,
        "likely_low_f":108,
        "likely_high_f":110,
        "confidence":"HIGH",
        "weather_risk":"LOW",
        "weather_reasons":["clear"],
        "model_mae_f":.7,
        "checkpoint_hour":13,
        "bucket_probability_total":1.0,
        "markets":[],
    })
    assert "KLAS Live High Model" in html
    assert "Latest 6-Hour Max Report" in html
    assert "Precise METAR Peak" in html
    assert "80% Model Range" in html
    assert "Model − ask" in html
