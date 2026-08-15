from __future__ import annotations

import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Show largest held-out KLAS model misses")
    parser.add_argument("--input", default="data/processed/model_backtest_predictions.csv")
    parser.add_argument("--hour", type=int, default=10)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    work = df[df["checkpoint_hour"] == args.hour].copy()
    work = work.sort_values("model_abs_error_f", ascending=False).head(args.top)
    cols = ["date", "nws_am_forecast_high_f", "model_predicted_high_f", "actual_cli_high_f", "nws_error_f", "model_error_f"]
    print(work[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
