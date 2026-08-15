from __future__ import annotations

import argparse

import pandas as pd


def _fmt(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize KLAS weather-disruption postmortems")
    parser.add_argument("--input", default="data/processed/klas_daily_postmortem.csv")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        print("No postmortem rows found.")
        return

    if "nws_am_abs_error_f" in df.columns:
        df["nws_am_abs_error_f"] = pd.to_numeric(df["nws_am_abs_error_f"], errors="coerce")
        df = df.sort_values("nws_am_abs_error_f", ascending=False, na_position="last")

    cols = [
        "date", "nws_am_forecast_high_f", "actual_cli_high_f", "nws_am_error_f",
        "primary_cause", "secondary_cause", "cause_confidence", "cloud_burden_timeweighted_pre_peak",
        "cloudy_minutes_pre_peak", "precip_before_peak", "precip_after_peak",
        "outflow_candidate", "raw_peak_hour_local", "settlement_gap_flag",
    ]
    cols = [c for c in cols if c in df.columns]
    view = df[cols].head(args.top).copy()
    rename = {
        "nws_am_forecast_high_f": "NWS",
        "actual_cli_high_f": "CLI",
        "nws_am_error_f": "Err",
        "primary_cause": "Cause",
        "secondary_cause": "Secondary",
        "cause_confidence": "Conf",
        "cloud_burden_timeweighted_pre_peak": "CloudTW",
        "cloudy_minutes_pre_peak": "CloudMin",
        "precip_before_peak": "RainPre",
        "precip_after_peak": "RainPost",
        "outflow_candidate": "Outflow",
        "raw_peak_hour_local": "PeakHr",
        "settlement_gap_flag": "CLIgap",
    }
    view = view.rename(columns=rename)
    for c in ("NWS", "CLI", "Err", "CloudTW", "CloudMin", "PeakHr"):
        if c in view.columns:
            view[c] = view[c].map(_fmt)

    print(view.to_string(index=False))
    if "primary_cause" in df.columns:
        print("\nCause counts:")
        print(df["primary_cause"].value_counts(dropna=False).to_string())
    if "settlement_gap_flag" in df.columns:
        print("\nSettlement-gap days:")
        print(int(df["settlement_gap_flag"].fillna(False).astype(bool).sum()))


if __name__ == "__main__":
    main()
