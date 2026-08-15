from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_CHECKPOINTS = tuple(range(8, 19))  # 08:00 through 18:00 local


@dataclass(frozen=True)
class CheckpointConfig:
    hours: tuple[int, ...] = DEFAULT_CHECKPOINTS
    tolerance_minutes: int = 40


def _nearest_obs(group: pd.DataFrame, hour: int, tolerance_minutes: int) -> pd.Series | None:
    date = pd.Timestamp(group["local_date"].iloc[0])
    # timestamp is already tz-aware; construct a target with matching timezone.
    tz = group["timestamp"].dt.tz
    target = pd.Timestamp.combine(date.date(), time(hour=hour)).tz_localize(tz)
    delta = (group["timestamp"] - target).abs()
    if delta.empty:
        return None
    idx = delta.idxmin()
    if delta.loc[idx] > pd.Timedelta(minutes=tolerance_minutes):
        return None
    return group.loc[idx]


def build_daily_heating_table(
    observations: pd.DataFrame,
    config: CheckpointConfig = CheckpointConfig(),
) -> pd.DataFrame:
    """Create one row per local day with checkpoint temperatures and heating remaining.

    `raw_peak_f` is the maximum reported ASOS temperature in the supplied observations.
    It is *not* the official NWS CLI settlement high. Once CLI highs are merged, the
    same checkpoint columns can be compared to `actual_cli_high_f` instead.
    """
    required = {"timestamp", "temp_f"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"observations missing required columns: {sorted(missing)}")

    obs = observations.copy()
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], errors="coerce")
    obs = obs.dropna(subset=["timestamp", "temp_f"]).sort_values("timestamp")
    obs["local_date"] = obs["timestamp"].dt.date

    rows: list[dict[str, object]] = []
    for local_date, group in obs.groupby("local_date", sort=True):
        peak_idx = group["temp_f"].idxmax()
        peak_row = group.loc[peak_idx]
        row: dict[str, object] = {
            "date": local_date,
            "raw_peak_f": float(peak_row["temp_f"]),
            "raw_peak_time": peak_row["timestamp"],
            "obs_count": int(len(group)),
            "max_cloud_fraction": float(group["cloud_fraction"].max())
            if "cloud_fraction" in group and group["cloud_fraction"].notna().any()
            else np.nan,
            "precip_total_in": float(group["precip_in"].fillna(0).sum())
            if "precip_in" in group
            else np.nan,
            "thunder_observed": bool(group["thunder_observed"].fillna(False).any())
            if "thunder_observed" in group
            else False,
            "max_wind_gust_kt": float(group["wind_gust_kt"].max())
            if "wind_gust_kt" in group and group["wind_gust_kt"].notna().any()
            else np.nan,
        }

        for hour in config.hours:
            found = _nearest_obs(group, hour, config.tolerance_minutes)
            prefix = f"h{hour:02d}"
            if found is None:
                row[f"{prefix}_temp_f"] = np.nan
                row[f"{prefix}_dewpoint_f"] = np.nan
                row[f"{prefix}_cloud_fraction"] = np.nan
                row[f"{prefix}_wind_speed_kt"] = np.nan
                row[f"{prefix}_heating_remaining_raw_f"] = np.nan
                continue

            temp_f = float(found["temp_f"])
            row[f"{prefix}_temp_f"] = temp_f
            row[f"{prefix}_dewpoint_f"] = found.get("dewpoint_f", np.nan)
            row[f"{prefix}_cloud_fraction"] = found.get("cloud_fraction", np.nan)
            row[f"{prefix}_wind_speed_kt"] = found.get("wind_speed_kt", np.nan)
            row[f"{prefix}_heating_remaining_raw_f"] = float(peak_row["temp_f"] - temp_f)

        rows.append(row)

    return pd.DataFrame(rows)


def add_cli_heating_remaining(daily: pd.DataFrame) -> pd.DataFrame:
    """Add settlement-target heating remaining after official CLI highs are merged."""
    if "actual_cli_high_f" not in daily.columns:
        raise ValueError("daily table must contain actual_cli_high_f")
    out = daily.copy()
    for col in [c for c in out.columns if c.endswith("_temp_f") and c.startswith("h")]:
        prefix = col.removesuffix("_temp_f")
        out[f"{prefix}_heating_remaining_cli_f"] = out["actual_cli_high_f"] - out[col]
    return out


def summarize_heating_by_hour(daily: pd.DataFrame) -> pd.DataFrame:
    """Summarize remaining heating distributions at each checkpoint hour."""
    rows = []
    target_suffix = "_heating_remaining_cli_f" if any(
        c.endswith("_heating_remaining_cli_f") for c in daily.columns
    ) else "_heating_remaining_raw_f"

    for col in sorted(c for c in daily.columns if c.endswith(target_suffix)):
        values = pd.to_numeric(daily[col], errors="coerce").dropna()
        if values.empty:
            continue
        hour = int(col[1:3])
        rows.append(
            {
                "hour_local": hour,
                "n_days": int(values.size),
                "mean_remaining_f": float(values.mean()),
                "median_remaining_f": float(values.median()),
                "p10_remaining_f": float(values.quantile(0.10)),
                "p25_remaining_f": float(values.quantile(0.25)),
                "p75_remaining_f": float(values.quantile(0.75)),
                "p90_remaining_f": float(values.quantile(0.90)),
                "std_remaining_f": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)
