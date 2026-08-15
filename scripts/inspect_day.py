from __future__ import annotations

import argparse

import pandas as pd


def _fmt_bool(v: object) -> str:
    if pd.isna(v):
        return ""
    return "Y" if bool(v) else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one KLAS ASOS/METAR day and its postmortem features")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--asos", default="data/raw/asos/klas_asos.csv")
    parser.add_argument("--postmortem", default="data/processed/klas_daily_postmortem.csv")
    args = parser.parse_args()

    obs = pd.read_csv(args.asos)
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], errors="coerce")
    target = pd.Timestamp(args.date).date()
    obs = obs[obs["timestamp"].dt.date == target].copy()
    obs = obs[(obs["timestamp"].dt.hour >= 7) & (obs["timestamp"].dt.hour < 20)]

    print(f"\nKLAS timeline for {args.date}")
    if obs.empty:
        print("No ASOS rows found for this date.")
    else:
        cols = [
            "timestamp", "temp_f", "dewpoint_f", "cloud_fraction",
            "cloud_fraction_below_12000", "cloud_fraction_below_20000", "lowest_bkn_ovc_ft",
            "precip_in", "wind_dir_deg", "wind_speed_kt", "wind_gust_kt",
            "thunder_observed", "metar",
        ]
        cols = [c for c in cols if c in obs.columns]
        view = obs[cols].copy()
        view["timestamp"] = view["timestamp"].dt.strftime("%H:%M")
        if "thunder_observed" in view:
            view["thunder_observed"] = view["thunder_observed"].map(_fmt_bool)
        print(view.to_string(index=False, max_colwidth=90))

    try:
        pm = pd.read_csv(args.postmortem)
    except FileNotFoundError:
        return
    row = pm[pm["date"].astype(str) == args.date]
    if row.empty:
        return

    keys = [
        "nws_am_forecast_high_f", "actual_cli_high_f", "nws_am_error_f",
        "raw_peak_f", "raw_peak_time_proxy", "raw_peak_hour_local",
        "cloud_mean_pre_peak", "cloudy_obs_fraction_pre_peak",
        "cloud_burden_timeweighted_pre_peak", "cloudy_minutes_pre_peak",
        "overcast_minutes_pre_peak", "cloud_burden_last3h_pre_peak",
        "cloud_burden_below_12000_pre_peak", "cloud_burden_below_20000_pre_peak",
        "lowest_bkn_ovc_ft_pre_peak",
        "first_cloudy_time", "cloud_onset_minutes_before_peak",
        "precip_pre_peak_dedup_in", "precip_post_peak_dedup_in",
        "precip_before_peak", "precip_after_peak", "thunder_before_peak", "thunder_after_peak",
        "convective_cloud_before_peak", "convective_cloud_after_peak",
        "largest_pre_peak_temp_drop_90m_f", "max_pre_peak_wind_shift_deg",
        "max_pre_peak_gust_kt", "outflow_candidate", "max_pre_peak_dewpoint_f",
        "heat_10_12_f", "heat_12_14_f", "heat_14_16_f", "midday_stall_signal",
        "primary_cause", "secondary_cause", "cause_confidence", "settlement_gap_f", "settlement_gap_flag",
        "postmortem_notes",
    ]
    keys = [k for k in keys if k in row.columns]
    print("\nPostmortem features")
    for key in keys:
        value = row.iloc[0][key]
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
