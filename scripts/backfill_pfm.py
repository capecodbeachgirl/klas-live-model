from __future__ import annotations

import argparse

from klas_model.collectors.pfm import fetch_pfm_morning_history, save_pfm_history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill the KLAS-area NWS morning forecast high from archived PFMVEF products"
    )
    parser.add_argument("--start", required=True, help="First target date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Last target date, YYYY-MM-DD")
    parser.add_argument(
        "--cutoff-hour",
        type=int,
        default=6,
        help="Use the latest PFM available by this Las Vegas local hour (default: 6)",
    )
    parser.add_argument("--output", default="data/raw/pfm/klas_nws_morning_forecast.csv")
    args = parser.parse_args()

    df = fetch_pfm_morning_history(args.start, args.end, cutoff_hour_local=args.cutoff_hour)
    path = save_pfm_history(df, args.output)
    print(f"saved {len(df):,} NWS morning forecast rows to {path}")
    if not df.empty:
        missing_expected = (pd.Timestamp(args.end).date() - pd.Timestamp(args.start).date()).days + 1 - len(df)
        if missing_expected > 0:
            print(f"warning: {missing_expected:,} requested dates had no qualifying pre-cutoff PFM")


if __name__ == "__main__":
    import pandas as pd

    main()
