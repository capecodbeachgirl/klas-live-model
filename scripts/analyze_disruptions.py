from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from klas_model.disruption import add_postmortem_labels, build_disruption_features, merge_disruption_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze KLAS weather disruption timing")
    parser.add_argument("--asos", default="data/raw/asos/klas_asos.csv")
    parser.add_argument("--daily", default="data/processed/klas_daily_heating.csv")
    parser.add_argument("--output", default="data/processed/klas_daily_postmortem.csv")
    args = parser.parse_args()

    obs = pd.read_csv(args.asos)
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], errors="coerce")
    daily = pd.read_csv(args.daily)
    daily["date"] = daily["date"].astype(str)

    # build_heating_curves.py already includes disruption features in v0.6+.
    # Older versions may not. Merge ONLY missing columns so we never create
    # cloud_burden_x/cloud_burden_y duplicates that silently break classification.
    features = build_disruption_features(obs)
    daily = merge_disruption_features(daily, features)
    merged = add_postmortem_labels(daily)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    print(f"saved {len(merged):,} postmortem rows to {args.output}")
    if "primary_cause" in merged.columns:
        counts = merged["primary_cause"].value_counts(dropna=False)
        print("cause counts:")
        for cause, count in counts.items():
            print(f"  {cause}: {count}")


if __name__ == "__main__":
    main()
