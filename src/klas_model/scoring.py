from __future__ import annotations

import pandas as pd


def accuracy_summary(df: pd.DataFrame, prediction_col: str, actual_col: str = "actual_cli_high_f") -> dict:
    work = df[[prediction_col, actual_col]].dropna().copy()
    if work.empty:
        return {"n": 0}
    error = work[prediction_col] - work[actual_col]
    ae = error.abs()
    return {
        "n": int(len(work)),
        "mae_f": float(ae.mean()),
        "bias_f": float(error.mean()),
        "exact_pct": float((ae < 0.5).mean()),
        "within_1f_pct": float((ae <= 1.0).mean()),
        "within_2f_pct": float((ae <= 2.0).mean()),
        "within_3f_pct": float((ae <= 3.0).mean()),
    }
