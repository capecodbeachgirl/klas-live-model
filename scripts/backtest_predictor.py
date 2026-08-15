from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from klas_model.predictive import DEFAULT_MODEL_CHECKPOINTS, backtest_all_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronological out-of-sample KLAS predictor backtest")
    parser.add_argument("--input", default="data/processed/klas_daily_heating.csv")
    parser.add_argument("--predictions-output", default="data/processed/model_backtest_predictions.csv")
    parser.add_argument("--summary-output", default="data/processed/model_checkpoint_accuracy.csv")
    parser.add_argument("--min-train", type=int, default=45)
    parser.add_argument("--test-block", type=int, default=15)
    args = parser.parse_args()

    daily = pd.read_csv(args.input)
    pred, summary = backtest_all_checkpoints(
        daily,
        DEFAULT_MODEL_CHECKPOINTS,
        min_train=args.min_train,
        test_block=args.test_block,
    )

    p1 = Path(args.predictions_output)
    p2 = Path(args.summary_output)
    p1.parent.mkdir(parents=True, exist_ok=True)
    p2.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(p1, index=False)
    summary.to_csv(p2, index=False)

    print(f"saved {len(pred)} held-out predictions to {p1}")
    print(f"saved {len(summary)} checkpoint rows to {p2}")
    print()
    cols = [
        "checkpoint_hour", "n", "mae_f", "nws_mae_f", "mae_improvement_f",
        "mae_improvement_pct", "within_1f_pct", "within_2f_pct",
    ]
    display = summary[cols].copy()
    for c in ["mae_f", "nws_mae_f", "mae_improvement_f"]:
        display[c] = display[c].map(lambda x: f"{x:.2f}")
    display["mae_improvement_pct"] = display["mae_improvement_pct"].map(lambda x: f"{x:+.1f}%")
    display["within_1f_pct"] = display["within_1f_pct"].map(lambda x: f"{100*x:.1f}%")
    display["within_2f_pct"] = display["within_2f_pct"].map(lambda x: f"{100*x:.1f}%")
    display["n"] = display["n"].astype(int)
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
