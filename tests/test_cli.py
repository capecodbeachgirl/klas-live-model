from io import BytesIO
from zipfile import ZipFile

import pandas as pd

from klas_model.collectors.cli import (
    merge_cli_with_heating,
    parse_cli_daily_high,
    parse_cli_zip,
)


def _cli_text(date_text: str, period: str, high: int, issued: str, peak: str = "2:46 PM") -> str:
    return f"""
CLIMATE REPORT
NATIONAL WEATHER SERVICE LAS VEGAS, NV
{issued}

...THE LAS VEGAS NV CLIMATE SUMMARY FOR {date_text}...

TEMPERATURE (F)
 {period}
  MAXIMUM        {high}   {peak} 115    1942 105      7       98
  MINIMUM         90   5:09 AM  61    1938  83      7       79
"""


def test_parse_cli_daily_high_yesterday():
    text = _cli_text(
        "JULY 23 2026",
        "YESTERDAY",
        112,
        "123 AM PDT FRI JUL 24 2026",
    )
    parsed = parse_cli_daily_high(text)
    assert parsed.high_f == 112
    assert parsed.climate_date.isoformat() == "2026-07-23"
    assert parsed.peak_time_text == "2:46 PM"
    assert parsed.period_label == "YESTERDAY"
    assert parsed.issued_at is not None
    assert parsed.issued_at.hour == 1
    assert parsed.issued_at.minute == 23


def test_parse_cli_zip_prefers_final_yesterday_product():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as zf:
        zf.writestr(
            "preliminary.txt",
            _cli_text(
                "JULY 23 2026", "TODAY", 111, "524 PM PDT THU JUL 23 2026", peak="3:15 PM"
            ),
        )
        zf.writestr(
            "final.txt",
            _cli_text("JULY 23 2026", "YESTERDAY", 112, "123 AM PDT FRI JUL 24 2026"),
        )

    df = parse_cli_zip(buffer.getvalue(), "2026-07-23", "2026-07-23")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["actual_cli_high_f"] == 112
    assert row["cli_period_label"] == "YESTERDAY"
    assert bool(row["cli_is_final"]) is True


def test_merge_cli_ignores_preliminary_only_rows():
    daily = pd.DataFrame({"date": ["2026-07-23"], "raw_peak_f": [111.2]})
    cli = pd.DataFrame(
        {
            "date": ["2026-07-23"],
            "actual_cli_high_f": [111],
            "cli_period_label": ["TODAY"],
            "cli_is_final": [False],
        }
    )
    merged = merge_cli_with_heating(daily, cli)
    assert pd.isna(merged.loc[0, "actual_cli_high_f"])


def test_parse_cli_cleans_record_flag_from_peak_time():
    text = _cli_text(
        "AUGUST 5 2026",
        "YESTERDAY",
        113,
        "123 AM PDT THU AUG 6 2026",
        peak="R  3:43 PM",
    )
    parsed = parse_cli_daily_high(text)
    assert parsed.peak_time_text == "3:43 PM"
    assert parsed.peak_time_flags == "R"
