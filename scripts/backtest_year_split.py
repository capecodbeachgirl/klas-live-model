from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from klas_model.conservative_predictive import year_split_backtest_all
from klas_model.predictive import DEFAULT_MODEL_CHECKPOINTS


def _years(value: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 2022-24, tune 2025, and test conservative NWS corrections on 2026"
    )
    parser.add_argument("--input", default="data/processed/klas_daily_heating.csv")
    parser.add_argument("--train-years", default="2022,2023,2024")
    parser.add_argument("--validation-year", type=int, default=2025)
    parser.add_argument("--test-year", type=int, default=2026)
    parser.add_argument("--predictions-output", default="data/processed/year_split_predictions.csv")
    parser.add_argument("--summary-output", default="data/processed/year_split_accuracy.csv")
    args = parser.parse_args()

    daily = pd.read_csv(args.input)
    pred, summary = year_split_backtest_all(
        daily,
        DEFAULT_MODEL_CHECKPOINTS,
        train_years=_years(args.train_years),
        validation_year=args.validation_year,
        test_year=args.test_year,
    )

    p1 = Path(args.predictions_output)
    p2 = Path(args.summary_output)
    p1.parent.mkdir(parents=True, exist_ok=True)
    p2.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(p1, index=False)
    summary.to_csv(p2, index=False)

    print(f"saved {len(pred):,} untouched-test predictions to {p1}")
    print(f"saved {len(summary):,} checkpoint rows to {p2}\n")

    cols = [
        "checkpoint_hour", "n", "mae_f", "nws_mae_f", "mae_improvement_f",
        "within_1f_pct", "within_2f_pct", "shrink", "gate_f", "cap_f",
        "validation_improvement_f",
    ]
    display = summary[cols].copy()
    for c in ["mae_f", "nws_mae_f", "mae_improvement_f", "gate_f", "cap_f", "validation_improvement_f"]:
        display[c] = pd.to_numeric(display[c]).map(lambda x: f"{x:.2f}")
    for c in ["within_1f_pct", "within_2f_pct"]:
        display[c] = pd.to_numeric(display[c]).map(lambda x: f"{100*x:.1f}%")
    display["n"] = display["n"].astype(int)
    display["shrink"] = pd.to_numeric(display["shrink"]).map(lambda x: f"{x:.2f}")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
