from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from klas_model.collectors.cli import fetch_cli_history


SHADOW_LOG = (
    ROOT
    / "data"
    / "processed"
    / "klas_h10_50_shadow_log.csv"
)


def main() -> None:

    if not SHADOW_LOG.exists():
        raise FileNotFoundError(
            f"Shadow log not found: {SHADOW_LOG}"
        )

    log = pd.read_csv(SHADOW_LOG)

    if log.empty:
        print("Shadow log is empty.")
        return

    log["date"] = pd.to_datetime(
        log["date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    scored_flag = (
        log["scored"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    pending = log[
        ~scored_flag
        & log["date"].notna()
    ].copy()

    if pending.empty:
        print("No pending shadow forecasts.")
        return

    start = pending["date"].min()
    end = pending["date"].max()

    print(
        f"Checking final CLILAS from {start} through {end}..."
    )

    cli = fetch_cli_history(
        start=start,
        end=end,
    )

    if cli.empty:
        print("No CLI products available yet.")
        return

    if "cli_is_final" in cli.columns:

        final = (
            cli["cli_is_final"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

        cli = cli[final].copy()

    if cli.empty:
        print(
            "No final next-morning CLILAS products "
            "available yet."
        )
        return

    cli["date"] = pd.to_datetime(
        cli["date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    cli["actual_cli_high_f"] = pd.to_numeric(
        cli["actual_cli_high_f"],
        errors="coerce",
    )

    truth = (
        cli[
            [
                "date",
                "actual_cli_high_f",
                "cli_source_filename",
            ]
        ]
        .dropna(
            subset=[
                "date",
                "actual_cli_high_f",
            ]
        )
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .set_index("date")
    )

    scored_now = 0

    for idx, row in log.iterrows():

        already_scored = str(
            row.get("scored", "")
        ).lower() in {
            "true",
            "1",
            "yes",
        }

        if already_scored:
            continue

        date = row["date"]

        if date not in truth.index:
            continue

        actual = float(
            truth.loc[
                date,
                "actual_cli_high_f",
            ]
        )

        base = float(
            row["base_forecast_f"]
        )

        shadow = float(
            row["shadow_forecast_f"]
        )

        base_abs = abs(
            base - actual
        )

        shadow_abs = abs(
            shadow - actual
        )

        gain = (
            base_abs
            - shadow_abs
        )

        log.at[
            idx,
            "actual_cli_high_f",
        ] = actual

        log.at[
            idx,
            "base_abs_error_f",
        ] = base_abs

        log.at[
            idx,
            "shadow_abs_error_f",
        ] = shadow_abs

        log.at[
            idx,
            "shadow_gain_f",
        ] = gain

        log.at[
            idx,
            "scored",
        ] = True

        scored_now += 1

        source = truth.loc[
            date,
            "cli_source_filename",
        ]

        print()
        print(date)
        print("final CLILAS:", actual)
        print("source:", source)
        print("base forecast:", round(base, 3))
        print("shadow forecast:", round(shadow, 3))
        print("base abs error:", round(base_abs, 3))
        print("shadow abs error:", round(shadow_abs, 3))
        print("shadow gain:", round(gain, 3))

    if scored_now:
        log.to_csv(
            SHADOW_LOG,
            index=False,
        )

        print(
            f"\nScored {scored_now} new shadow forecast(s)."
        )
    else:
        print(
            "\nNo pending dates have a final "
            "next-morning CLILAS yet."
        )

    scored_mask = (
        log["scored"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    completed = log[
        scored_mask
    ].copy()

    if not completed.empty:

        completed["base_abs_error_f"] = pd.to_numeric(
            completed["base_abs_error_f"],
            errors="coerce",
        )

        completed["shadow_abs_error_f"] = pd.to_numeric(
            completed["shadow_abs_error_f"],
            errors="coerce",
        )

        print("\nSHADOW SCORECARD")
        print("days:", len(completed))

        print(
            "base MAE:",
            round(
                completed[
                    "base_abs_error_f"
                ].mean(),
                3,
            ),
        )

        print(
            "shadow MAE:",
            round(
                completed[
                    "shadow_abs_error_f"
                ].mean(),
                3,
            ),
        )

        print(
            "net gain:",
            round(
                completed[
                    "base_abs_error_f"
                ].mean()
                - completed[
                    "shadow_abs_error_f"
                ].mean(),
                3,
            ),
        )


if __name__ == "__main__":
    main()
