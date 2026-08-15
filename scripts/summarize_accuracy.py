from __future__ import annotations

import argparse

import pandas as pd

from klas_model.scoring import accuracy_summary


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize KLAS NWS/model accuracy versus final CLI")
    parser.add_argument("--input", default="data/processed/klas_daily_heating.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "day_complete" in df.columns:
        df = df[df["day_complete"].astype(str).str.lower().isin(["true", "1"])]

    if "nws_am_forecast_high_f" not in df.columns:
        raise SystemExit("Input does not contain nws_am_forecast_high_f; merge PFM data first.")

    stats = accuracy_summary(df, "nws_am_forecast_high_f")
    print("NWS morning forecast vs final KLAS CLI")
    print(f"days: {stats.get('n', 0)}")
    if stats.get("n", 0):
        print(f"MAE: {stats['mae_f']:.2f} F")
        print(f"bias: {stats['bias_f']:+.2f} F (positive = forecast too warm)")
        print(f"exact: {_pct(stats['exact_pct'])}")
        print(f"within 1F: {_pct(stats['within_1f_pct'])}")
        print(f"within 2F: {_pct(stats['within_2f_pct'])}")
        print(f"within 3F: {_pct(stats['within_3f_pct'])}")


if __name__ == "__main__":
    main()
