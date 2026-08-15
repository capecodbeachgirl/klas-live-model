from klas_model.collectors.asos import normalize_asos_csv, normalize_awc_metar_json, fetch_live_asos


def test_cloud_height_features_are_preserved():
    text = """station,valid,tmpf,dwpf,drct,sknt,gust,p01i,skyc1,skyl1,skyc2,skyl2,skyc3,skyl3,skyc4,skyl4,wxcodes,metar
LAS,2026-08-10 11:56,101,53,160,7,M,0.00,FEW,13000,SCT,18000,OVC,23000,M,M,,KLAS 101856Z 16007KT 10SM FEW130 SCT180 OVC230 38/12 A2996
"""
    out = normalize_asos_csv(text)
    row = out.iloc[0]
    assert row["cloud_fraction"] == 1.0
    assert row["cloud_fraction_below_12000"] == 0.0
    assert row["cloud_fraction_below_20000"] == 0.5
    assert row["lowest_bkn_ovc_ft"] == 23000.0


def test_awc_live_json_normalizes_to_iem_shape():
    payload = [{
        "icaoId": "KLAS",
        "obsTime": "2026-08-15T17:56:00Z",
        "temp": 30.0,
        "dewp": 12.0,
        "wdir": 180,
        "wspd": 8,
        "wgst": 18,
        "precip": 0.0,
        "wxString": None,
        "clouds": [{"cover": "SCT", "base": 12000}, {"cover": "BKN", "base": 22000}],
        "rawOb": "KLAS 151756Z 18008G18KT 10SM SCT120 BKN220 30/12 A2998 RMK AO2 T03000120 10300 20200",
    }]
    out = normalize_awc_metar_json(payload)
    row = out.iloc[0]
    assert round(row["temp_f"], 1) == 86.0
    assert row["cloud_fraction"] == 0.875
    assert row["cloud_fraction_below_12000"] == 0.5
    assert row["data_source"] == "AviationWeather.gov"


def test_awc_live_json_detects_thunder_and_six_hour_max():
    payload = [{
        "icaoId": "KLAS",
        "obsTime": "2026-08-15T19:56:00Z",
        "temp": 35.0,
        "dewp": 15.0,
        "clouds": [{"cover": "BKN", "base": 10000}],
        "rawOb": "KLAS 151956Z 18010KT 10SM TS BKN100 35/15 A2990 RMK AO2 T03500150 10350 20250",
    }]
    out = normalize_awc_metar_json(payload)
    assert bool(out.iloc[0]["thunder_observed"]) is True
    # 10350 is the six-hour maximum group: +35.0 C = 95.0 F.
    assert round(out.iloc[0]["six_hour_max_f"], 1) == 95.0
