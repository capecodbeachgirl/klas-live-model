from datetime import date

import pandas as pd

from klas_model.multiyear import concat_dedupe, summer_windows


def test_summer_windows_caps_current_year():
    windows = summer_windows([2024, 2025, 2026], through="2026-08-14")
    assert windows[0].start == date(2024, 6, 1)
    assert windows[0].end == date(2024, 9, 30)
    assert windows[-1].start == date(2026, 6, 1)
    assert windows[-1].end == date(2026, 8, 14)


def test_concat_dedupe_keeps_latest():
    a = pd.DataFrame({"date": ["2025-06-01"], "x": [1]})
    b = pd.DataFrame({"date": ["2025-06-01", "2025-06-02"], "x": [2, 3]})
    out = concat_dedupe([a, b], "date")
    assert out["x"].tolist() == [2, 3]
