from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from klas_model.collectors.cli import merge_cli_with_heating
from klas_model.collectors.pfm import merge_pfm_with_daily
from klas_model.disruption import build_disruption_features
from klas_model.heating_curve import (
    add_cli_heating_remaining,
    build_daily_heating_table,
    summarize_heating_by_hour,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KLAS daily heating curves")
    parser.add_argument("--input", default="data/raw/asos/klas_asos.csv")
    parser.add_argument(
        "--cli",
        default=None,
        help="Optional official CLI CSV from backfill_cli.py. Final reports are merged by date.",
    )
    parser.add_argument(
        "--pfm",
        default=None,
        help="Optional NWS morning PFM CSV from backfill_pfm.py.",
    )
    parser.add_argument("--daily-output", default="data/processed/klas_daily_heating.csv")
    parser.add_argument("--summary-output", default="data/processed/heating_by_hour.csv")
    args = parser.parse_args()

    obs = pd.read_csv(args.input)
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], errors="coerce")

    daily = build_daily_heating_table(obs)
    disruption = build_disruption_features(obs)
    if not disruption.empty:
        disruption["date"] = disruption["date"].astype(str)
        daily["date"] = daily["date"].astype(str)
        daily = daily.merge(disruption, on="date", how="left")
    if args.cli:
        cli = pd.read_csv(args.cli)
        daily = merge_cli_with_heating(daily, cli)
        if "actual_cli_high_f" in daily.columns:
            daily = add_cli_heating_remaining(daily)
            daily["asos_minus_cli_f"] = daily["raw_peak_f"] - daily["actual_cli_high_f"]
            daily["day_complete"] = daily.get("cli_is_final", False).fillna(False).astype(bool)

    if args.pfm:
        pfm = pd.read_csv(args.pfm)
        daily = merge_pfm_with_daily(daily, pfm)
        if "actual_cli_high_f" in daily.columns and "nws_am_forecast_high_f" in daily.columns:
            daily["nws_am_error_f"] = daily["nws_am_forecast_high_f"] - daily["actual_cli_high_f"]
            daily["nws_am_abs_error_f"] = daily["nws_am_error_f"].abs()

    summary = summarize_heating_by_hour(daily)

    Path(args.daily_output).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.daily_output, index=False)
    summary.to_csv(args.summary_output, index=False)

    print(f"saved {len(daily):,} daily rows to {args.daily_output}")
    if args.cli and "actual_cli_high_f" in daily.columns:
        matched = int(daily["actual_cli_high_f"].notna().sum())
        print(f"matched {matched:,} days to a final official CLI high")
    if args.pfm and "nws_am_forecast_high_f" in daily.columns:
        matched = int(daily["nws_am_forecast_high_f"].notna().sum())
        print(f"matched {matched:,} days to an NWS pre-6AM forecast high")
    print(f"saved {len(summary):,} hourly summary rows to {args.summary_output}")


if __name__ == "__main__":
    main()
