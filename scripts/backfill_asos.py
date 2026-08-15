from __future__ import annotations

import argparse
from pathlib import Path

from klas_model.collectors.asos import fetch_asos, save_observations


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill KLAS ASOS observations")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default="data/raw/asos/klas_asos.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = fetch_asos(args.start, args.end)
    path = save_observations(df, Path(args.output))
    print(f"saved {len(df):,} observations to {path}")


if __name__ == "__main__":
    main()
