from io import BytesIO
from zipfile import ZipFile

from klas_model.collectors.pfm import parse_pfm_daily_maxima, parse_pfm_zip


JAN_SAMPLE = """
FOUS55 KVEF 280825
PFMVEF

Point Forecast Matrices
National Weather Service Las Vegas NV
1225 AM PST Wed Jan 28 2026

NVZ020-291200-
Las Vegas-Clark NV
36.07N 115.16W Elev. (Average Over 2.5KM Grid Box): 2181 ft
1225 AM PST Wed Jan 28 2026

Date          Tue 01/27/26      Wed 01/28/26            Thu 01/29/26         Fri
PST 3hrly     13 16 19 22 01 04 07 10 13 16 19 22 01 04 07 10 13 16 19 22 01 04
UTC 3hrly     21 00 03 06 09 12 15 18 21 00 03 06 09 12 15 18 21 00 03 06 09 12

Min/Max                      41          62          42          65          44
Temp                      43 41 42 53 60 62 56 50 47 44 43 56 62 65 59 52 49 46
"""

MAR_SAMPLE = """
FOUS55 KVEF 041001
PFMVEF

Point Forecast Matrices
National Weather Service Las Vegas NV
201 AM PST Wed Mar 4 2026

NVZ020-050000-
Las Vegas-Clark NV
36.07N 115.16W Elev. (Average Over 2.5KM Grid Box): 2181 ft
201 AM PST Wed Mar 4 2026

Date                Wed 03/04/26            Thu 03/05/26            Fri 03/06/26
PST 3hrly     01 04 07 10 13 16 19 22 01 04 07 10 13 16 19 22 01 04 07 10 13 16
UTC 3hrly     09 12 15 18 21 00 03 06 09 12 15 18 21 00 03 06 09 12 15 18 21 00

Min/Max                      76          51          66          46          65
Temp             53 54 65 73 76 70 63 58 54 54 59 64 65 60 55 51 47 48 57 63 65
"""


def test_parse_pfm_max_with_previous_day_columns():
    rows = parse_pfm_daily_maxima(JAN_SAMPLE)
    by_date = {r.forecast_date.isoformat(): r.forecast_high_f for r in rows}
    assert by_date["2026-01-28"] == 62
    assert by_date["2026-01-29"] == 65


def test_parse_pfm_max_same_day_start():
    rows = parse_pfm_daily_maxima(MAR_SAMPLE)
    by_date = {r.forecast_date.isoformat(): r.forecast_high_f for r in rows}
    assert by_date["2026-03-04"] == 76
    assert by_date["2026-03-05"] == 66


def _synthetic(issue_line: str, high: int) -> str:
    return f"""
FOUS55 KVEF 041001
PFMVEF
Point Forecast Matrices
National Weather Service Las Vegas NV
{issue_line}
NVZ020-050000-
Las Vegas-Clark NV
36.07N 115.16W Elev. (Average Over 2.5KM Grid Box): 2181 ft
{issue_line}
Date                Wed 03/04/26            Thu 03/05/26            Fri 03/06/26
PST 3hrly     01 04 07 10 13 16 19 22 01 04 07 10 13 16 19 22 01 04 07 10 13 16
UTC 3hrly     09 12 15 18 21 00 03 06 09 12 15 18 21 00 03 06 09 12 15 18 21 00
Min/Max                      {high:2d}          51          66          46          65
"""


def test_morning_archive_uses_latest_forecast_before_6am():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as zf:
        zf.writestr("early.txt", _synthetic("101 AM PST Wed Mar 4 2026", 75))
        zf.writestr("morning.txt", _synthetic("401 AM PST Wed Mar 4 2026", 76))
        zf.writestr("late.txt", _synthetic("701 AM PST Wed Mar 4 2026", 79))
    df = parse_pfm_zip(buffer.getvalue(), "2026-03-04", "2026-03-04", cutoff_hour_local=6)
    assert len(df) == 1
    assert int(df.iloc[0]["nws_am_forecast_high_f"]) == 76
    assert "04:01" in df.iloc[0]["nws_am_issued_at"]
