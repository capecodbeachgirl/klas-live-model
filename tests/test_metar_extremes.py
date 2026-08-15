from klas_model.metar_extremes import parse_temperature_extremes


def test_parse_metar_temperature_groups():
    x = parse_temperature_extremes("KLAS 142356Z 07018KT 10SM BKN120 30/14 A2995 RMK AO2 T03000139 10389 20222 403890183")
    assert round(x.precise_temp_f, 1) == 86.0
    assert x.six_hour_max_f == 102
    assert x.six_hour_min_f == 72
    assert x.daily_24h_max_f == 102
    assert x.daily_24h_min_f == 65


def test_missing_groups_are_none():
    x = parse_temperature_extremes("KLAS 141756Z 09008KT 10SM SCT220 33/13 A2999")
    assert x.six_hour_max_f is None
    assert x.precise_temp_f is None
