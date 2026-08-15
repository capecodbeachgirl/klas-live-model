from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    model_dir = Path("data/model")
    expected = [model_dir / f"h{hour:02d}.joblib" for hour in range(8, 19)]
    missing = [p.as_posix() for p in expected if not p.exists()]
    if missing:
        print("DEPLOYMENT NOT READY: trained live-model bundles are missing:")
        for path in missing:
            print(f"  - {path}")
        print("Copy your trained data/model folder into this repository and commit it before enabling the hourly workflow.")
        sys.exit(2)
    manifest = model_dir / "manifest.json"
    if not manifest.exists():
        print("DEPLOYMENT NOT READY: data/model/manifest.json is missing.")
        sys.exit(2)
    print("Deployment preflight passed: 11 hourly model bundles + manifest found.")


if __name__ == "__main__":
    main()
