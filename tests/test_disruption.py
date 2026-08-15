import pandas as pd

from klas_model.disruption import add_postmortem_labels, build_disruption_features


def _base_obs():
    times = pd.date_range("2026-08-14 08:00", periods=11, freq="h", tz="America/Los_Angeles")
    return pd.DataFrame(
        {
            "timestamp": times,
            "temp_f": [82, 86, 90, 93, 95, 96, 97, 96, 95, 94, 92],
            "dewpoint_f": [35] * 11,
            "cloud_fraction": [0.0] * 11,
            "precip_in": [0.0] * 11,
            "thunder_observed": [False] * 11,
            "wind_dir_deg": [180] * 11,
            "wind_speed_kt": [6] * 11,
            "wind_gust_kt": [8] * 11,
        }
    )


def test_rain_after_peak_not_marked_before_peak():
    obs = _base_obs()
    obs.loc[9, "precip_in"] = 0.20
    f = build_disruption_features(obs).iloc[0]
    assert bool(f["precip_before_peak"]) is False
    assert bool(f["precip_after_peak"]) is True


def test_time_weighted_cloud_minutes_detect_sustained_cloud():
    obs = _base_obs()
    obs.loc[3:6, "cloud_fraction"] = 1.0  # 11 AM through 2 PM
    f = build_disruption_features(obs).iloc[0]
    assert f["cloudy_minutes_pre_peak"] >= 180.0
    assert f["cloud_burden_timeweighted_pre_peak"] > 0.4
    assert f["cloud_onset_minutes_before_peak"] == 180.0


def test_outflow_candidate_requires_drop_and_wind_signal():
    obs = _base_obs()
    obs.loc[4, "temp_f"] = 94
    obs.loc[5, "temp_f"] = 90
    obs.loc[5, "wind_gust_kt"] = 30
    obs.loc[5, "wind_dir_deg"] = 280
    f = build_disruption_features(obs).iloc[0]
    assert f["largest_pre_peak_temp_drop_90m_f"] >= 4
    assert bool(f["outflow_candidate"]) is True


def test_postmortem_does_not_blame_rain_after_peak_for_cold_forecast():
    daily = pd.DataFrame(
        [{
            "date": "2026-08-14", "nws_am_error_f": -4.0, "asos_minus_cli_f": -1.0,
            "precip_before_peak": False, "precip_after_peak": True,
            "thunder_before_peak": False, "outflow_candidate": False,
            "cloud_burden_timeweighted_pre_peak": 0.1, "cloudy_minutes_pre_peak": 20.0,
            "overcast_minutes_pre_peak": 0.0, "cloud_burden_last3h_pre_peak": 0.1,
            "elevated_moisture_signal": False, "midday_stall_signal": False,
            "raw_peak_hour_local": 12.5, "heat_14_16_f": -1.0,
        }]
    )
    out = add_postmortem_labels(daily)
    assert out.loc[0, "primary_cause"] == "FORECAST_BIAS"
    assert bool(out.loc[0, "settlement_gap_flag"]) is True


def test_postmortem_cloud_when_warm_miss_and_clouds_pre_peak():
    daily = pd.DataFrame(
        [{
            "date": "2026-08-10", "nws_am_error_f": 5.0, "asos_minus_cli_f": 0.0,
            "precip_before_peak": False, "thunder_before_peak": False,
            "outflow_candidate": False, "cloud_burden_timeweighted_pre_peak": 0.55,
            "cloudy_minutes_pre_peak": 180.0, "overcast_minutes_pre_peak": 90.0,
            "cloud_burden_last3h_pre_peak": 0.7, "elevated_moisture_signal": False,
            "midday_stall_signal": True, "raw_peak_hour_local": 13.3, "heat_14_16_f": -2.0,
        }]
    )
    out = add_postmortem_labels(daily)
    assert out.loc[0, "primary_cause"] == "CLOUD"
    assert out.loc[0, "cause_confidence"] == "HIGH"


def test_one_degree_forecast_error_is_normal_range_even_with_cli_gap():
    daily = pd.DataFrame(
        [{
            "date": "2026-08-08", "nws_am_error_f": -1.0, "asos_minus_cli_f": -1.0,
        }]
    )
    out = add_postmortem_labels(daily)
    assert out.loc[0, "primary_cause"] == "NORMAL_RANGE"
    assert bool(out.loc[0, "settlement_gap_flag"]) is True


def test_no_cli_result_is_unknown():
    daily = pd.DataFrame([{"date": "2026-08-15", "nws_am_error_f": None, "asos_minus_cli_f": None}])
    out = add_postmortem_labels(daily)
    assert out.loc[0, "primary_cause"] == "UNKNOWN"
    assert bool(out.loc[0, "settlement_gap_flag"]) is False


def test_merge_disruption_features_does_not_create_suffix_duplicates():
    from klas_model.disruption import merge_disruption_features

    daily = pd.DataFrame([
        {"date": "2026-08-10", "cloud_burden_timeweighted_pre_peak": 0.8, "nws_am_error_f": 5.0}
    ])
    features = pd.DataFrame([
        {
            "date": "2026-08-10",
            "cloud_burden_timeweighted_pre_peak": 0.9,
            "convective_cloud_before_peak": True,
        }
    ])
    out = merge_disruption_features(daily, features)
    assert "cloud_burden_timeweighted_pre_peak_x" not in out.columns
    assert "cloud_burden_timeweighted_pre_peak_y" not in out.columns
    assert out.loc[0, "cloud_burden_timeweighted_pre_peak"] == 0.8
    assert bool(out.loc[0, "convective_cloud_before_peak"]) is True


def test_aug10_like_cloud_stall_gets_cloud_with_convective_secondary():
    daily = pd.DataFrame([
        {
            "date": "2026-08-10",
            "nws_am_error_f": 5.0,
            "asos_minus_cli_f": 0.0,
            "thunder_before_peak": False,
            "precip_before_peak": False,
            "outflow_candidate": False,
            "cloud_burden_timeweighted_pre_peak": 0.875,
            "cloudy_minutes_pre_peak": 300.0,
            "overcast_minutes_pre_peak": 60.0,
            "cloud_burden_last3h_pre_peak": 0.875,
            "cloud_burden_below_12000_pre_peak": 0.0,
            "cloud_burden_below_20000_pre_peak": 0.3,
            "midday_stall_signal": True,
            "convective_cloud_before_peak": True,
            "elevated_moisture_signal": True,
            "raw_peak_hour_local": 14.03,
        }
    ])
    out = add_postmortem_labels(daily)
    assert out.loc[0, "primary_cause"] == "CLOUD"
    assert out.loc[0, "secondary_cause"] == "TS_MONSOON"
    assert out.loc[0, "cause_confidence"] == "HIGH"


def test_aug13_like_thunderstorm_day_gets_ts_monsoon_with_rain_secondary():
    daily = pd.DataFrame([
        {
            "date": "2026-08-13",
            "nws_am_error_f": 4.0,
            "asos_minus_cli_f": 0.0,
            "thunder_before_peak": True,
            "precip_before_peak": True,
            "outflow_candidate": True,
            "convective_cloud_before_peak": True,
        }
    ])
    out = add_postmortem_labels(daily)
    assert out.loc[0, "primary_cause"] == "TS_MONSOON"
    assert out.loc[0, "secondary_cause"] == "RAIN"
    assert out.loc[0, "cause_confidence"] == "HIGH"


def test_aug14_like_storm_after_peak_not_blamed_for_warm_actual():
    daily = pd.DataFrame([
        {
            "date": "2026-08-14",
            "nws_am_error_f": -4.0,
            "asos_minus_cli_f": -1.0,
            "raw_peak_hour_local": 12.93,
            "heat_14_16_f": -5.0,
            "thunder_after_peak": True,
            "precip_after_peak": True,
        }
    ])
    out = add_postmortem_labels(daily)
    assert out.loc[0, "primary_cause"] == "FORECAST_BIAS"
    assert out.loc[0, "cause_confidence"] == "MEDIUM"
    assert "before later thunder/rain arrived" in out.loc[0, "postmortem_notes"]


def test_calm_to_routine_wind_does_not_create_false_outflow():
    obs = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-08-10 10:56", "2026-08-10 11:56", "2026-08-10 12:56",
            "2026-08-10 13:56", "2026-08-10 14:02"
        ]).tz_localize("America/Los_Angeles"),
        "temp_f": [103, 101, 100, 103, 104],
        "dewpoint_f": [51, 53, 58, 57, 57],
        "cloud_fraction": [0.875, 1.0, 0.875, 0.875, 0.875],
        "precip_in": [0, 0, 0, 0, 0],
        "thunder_observed": [False] * 5,
        "wind_dir_deg": [0, 160, 170, 160, 150],
        "wind_speed_kt": [0, 7, 7, 6, 8],
        "wind_gust_kt": [None, None, 16, 15, None],
        "metar": ["", "", "", "CB DSNT S-SW", "CB DSNT S-SW"],
    })
    f = build_disruption_features(obs).iloc[0]
    assert bool(f["outflow_candidate"]) is False


def test_temp_crash_with_strong_gust_is_coherent_outflow():
    obs = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-08-13 13:33", "2026-08-13 13:56", "2026-08-13 14:56",
            "2026-08-13 17:56"
        ]).tz_localize("America/Los_Angeles"),
        "temp_f": [86, 77, 78, 91],
        "dewpoint_f": [63, 62, 61, 58],
        "cloud_fraction": [1.0, 1.0, 0.875, 0.875],
        "precip_in": [0, 0, 0.01, 0],
        "thunder_observed": [True, True, True, False],
        "wind_dir_deg": [180, 200, 130, 180],
        "wind_speed_kt": [10, 21, 5, 9],
        "wind_gust_kt": [16, 36, 18, None],
        "metar": ["TS", "-TSRA", "TS", ""],
    })
    f = build_disruption_features(obs).iloc[0]
    assert bool(f["outflow_candidate"]) is True
    assert f["outflow_event_drop_f"] >= 9.0
    assert f["outflow_event_gust_kt"] >= 36.0


def test_midday_peak_with_afternoon_rebound_is_not_late_surge():
    daily = pd.DataFrame([{
        "date": "2026-08-12",
        "nws_am_error_f": -2.0,
        "asos_minus_cli_f": -1.0,
        "raw_peak_hour_local": 12.9,
        "heat_14_16_f": 3.0,
        "thunder_after_peak": False,
        "precip_after_peak": False,
    }])
    out = add_postmortem_labels(daily)
    assert out.loc[0, "primary_cause"] == "FORECAST_BIAS"


def test_true_late_peak_with_late_heating_is_late_surge():
    daily = pd.DataFrame([{
        "date": "2026-07-20",
        "nws_am_error_f": -3.0,
        "asos_minus_cli_f": 0.0,
        "raw_peak_hour_local": 16.2,
        "heat_14_16_f": 2.5,
    }])
    out = add_postmortem_labels(daily)
    assert out.loc[0, "primary_cause"] == "LATE_SURGE"
