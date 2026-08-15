from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from klas_model.collectors.asos import fetch_asos, save_observations
from klas_model.collectors.cli import fetch_cli_history, save_cli_history
from klas_model.collectors.pfm import fetch_pfm_morning_history, save_pfm_history
from klas_model.multiyear import concat_dedupe, summer_windows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill multiple KLAS warm seasons into one ASOS/CLI/PFM dataset"
    )
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--through",
        default=None,
        help="Optional inclusive cap for the final season, e.g. 2026-08-14",
    )
    parser.add_argument("--cutoff-hour", type=int, default=6)
    parser.add_argument("--asos-output", default="data/raw/asos/klas_asos.csv")
    parser.add_argument("--cli-output", default="data/raw/cli/klas_cli_daily.csv")
    parser.add_argument("--pfm-output", default="data/raw/pfm/klas_nws_morning_forecast.csv")
    args = parser.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    windows = summer_windows(years, through=args.through)
    if not windows:
        raise SystemExit("No valid summer windows requested")

    asos_frames: list[pd.DataFrame] = []
    cli_frames: list[pd.DataFrame] = []
    pfm_frames: list[pd.DataFrame] = []

    print("KLAS multi-year summer backfill")
    for window in windows:
        start = window.start.isoformat()
        end = window.end.isoformat()
        print(f"\n[{window.year}] {start} through {end}")

        print("  ASOS...", flush=True)
        asos = fetch_asos(start, end)
        asos_frames.append(asos)
        print(f"    {len(asos):,} observations")

        print("  CLI...", flush=True)
        cli = fetch_cli_history(start, end)
        cli_frames.append(cli)
        final_count = int(cli.get("cli_is_final", pd.Series(dtype=bool)).fillna(False).sum())
        print(f"    {len(cli):,} rows ({final_count:,} final)")

        print("  PFM morning forecast...", flush=True)
        pfm = fetch_pfm_morning_history(start, end, cutoff_hour_local=args.cutoff_hour)
        pfm_frames.append(pfm)
        print(f"    {len(pfm):,} forecast rows")

    asos_all = concat_dedupe(asos_frames, "timestamp")
    cli_all = concat_dedupe(cli_frames, "date")
    pfm_all = concat_dedupe(pfm_frames, "date")

    save_observations(asos_all, Path(args.asos_output))
    save_cli_history(cli_all, args.cli_output)
    save_pfm_history(pfm_all, args.pfm_output)

    print("\nCombined files")
    print(f"  ASOS: {len(asos_all):,} observations -> {args.asos_output}")
    print(f"  CLI:  {len(cli_all):,} daily rows -> {args.cli_output}")
    print(f"  PFM:  {len(pfm_all):,} daily rows -> {args.pfm_output}")


if __name__ == "__main__":
    main()
