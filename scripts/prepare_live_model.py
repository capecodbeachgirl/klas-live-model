from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from klas_model.live_model import fit_live_bundle, save_live_bundle
from klas_model.predictive import DEFAULT_MODEL_CHECKPOINTS


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare compact live KLAS model bundles")
    ap.add_argument("--input", default="data/processed/klas_daily_heating.csv")
    ap.add_argument("--output-dir", default="data/model")
    args = ap.parse_args()

    daily = pd.read_csv(args.input)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    manifest = {"version": "0.12", "checkpoints": {}}
    for hour in DEFAULT_MODEL_CHECKPOINTS:
        bundle = fit_live_bundle(daily, hour)
        path = save_live_bundle(bundle, out / f"h{hour:02d}.joblib")
        m = bundle["test_metrics"]
        manifest["checkpoints"][str(hour)] = {
            "path": str(path),
            "trained_through": bundle.get("trained_through"),
            "mae_f": m.get("mae_f"),
            "nws_mae_f": m.get("nws_mae_f"),
            "improvement_f": m.get("mae_improvement_f"),
            "n_calibration": len(bundle.get("calibration_model_errors_f", [])),
            "params": bundle.get("params"),
        }
        print(f"prepared {hour:02d}:00 model -> {path}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved manifest to {out / 'manifest.json'}")

if __name__ == "__main__": main()
