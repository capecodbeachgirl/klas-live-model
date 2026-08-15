import pandas as pd

from klas_model.collectors.asos import normalize_asos_csv
from klas_model.heating_curve import build_daily_heating_table, summarize_heating_by_hour


def test_normalize_asos_csv():
    text = """station,valid,tmpf,dwpf,drct,sknt,gust,p01i,skyc1,skyc2,skyc3,skyc4,metar
LAS,2026-08-01 10:00,100,30,180,5,M,0.00,CLR,M,M,M,KLAS 011700Z 18005KT 10SM CLR 38/M01 A2990
LAS,2026-08-01 11:00,103,31,190,6,M,0.00,SCT,M,M,M,KLAS 011800Z 19006KT 10SM SCT120 39/M01 A2990
"""
    df = normalize_asos_csv(text)
    assert list(df["temp_f"]) == [100, 103]
    assert df.loc[0, "cloud_fraction"] == 0.0
    assert df.loc[1, "cloud_fraction"] == 0.5
    assert str(df["timestamp"].dt.tz) == "America/Los_Angeles"


def test_daily_heating_table():
    times = pd.date_range("2026-08-01 08:00", periods=11, freq="h", tz="America/Los_Angeles")
    temps = [88, 92, 96, 100, 104, 107, 109, 110, 111, 110, 108]
    obs = pd.DataFrame(
        {
            "timestamp": times,
            "temp_f": temps,
            "dewpoint_f": [30] * 11,
            "cloud_fraction": [0.0] * 11,
            "precip_in": [0.0] * 11,
            "thunder_observed": [False] * 11,
            "wind_speed_kt": [5] * 11,
            "wind_gust_kt": [8] * 11,
        }
    )
    daily = build_daily_heating_table(obs)
    assert daily.loc[0, "raw_peak_f"] == 111
    assert daily.loc[0, "h10_temp_f"] == 96
    assert daily.loc[0, "h10_heating_remaining_raw_f"] == 15

    summary = summarize_heating_by_hour(daily)
    h10 = summary.loc[summary["hour_local"] == 10].iloc[0]
    assert h10["median_remaining_f"] == 15
