from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import CauseCode


@dataclass(frozen=True)
class DisruptionConfig:
    analysis_start_hour: int = 8
    analysis_end_hour: int = 19
    cloudy_threshold: float = 0.5
    overcast_threshold: float = 0.875
    max_carry_minutes: int = 90
    outflow_temp_drop_f: float = 2.0
    outflow_gust_kt: float = 20.0
    outflow_wind_shift_deg: float = 60.0
    elevated_dewpoint_f: float = 50.0
    material_cloud_minutes: float = 60.0
    material_overcast_minutes: float = 35.0
    material_cloud_burden: float = 0.25


def _circular_direction_change(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _iso_or_blank(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _as_bool(value: object) -> bool:
    """Robustly interpret booleans after CSV round-trips."""
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "t", "1", "yes", "y"}


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    return series.map(_as_bool).astype(bool)


def _checkpoint_temp(group: pd.DataFrame, hour: int, tolerance_minutes: int = 40) -> float | None:
    if group.empty:
        return None
    date = pd.Timestamp(group["timestamp"].iloc[0]).date()
    tz = group["timestamp"].dt.tz
    target = pd.Timestamp(date).tz_localize(tz) + pd.Timedelta(hours=hour)
    delta = (group["timestamp"] - target).abs()
    idx = delta.idxmin()
    if delta.loc[idx] > pd.Timedelta(minutes=tolerance_minutes):
        return None
    value = group.loc[idx, "temp_f"]
    return None if pd.isna(value) else float(value)


def _hourly_precip_total(group: pd.DataFrame) -> float:
    """Estimate precipitation without double-counting routine + special METAR rows."""
    if "precip_in" not in group.columns or group["precip_in"].dropna().empty:
        return 0.0
    temp = group[["timestamp", "precip_in"]].copy()
    temp["precip_in"] = pd.to_numeric(temp["precip_in"], errors="coerce").fillna(0.0)
    temp["clock_hour"] = temp["timestamp"].dt.floor("h")
    return float(temp.groupby("clock_hour")["precip_in"].max().sum())


def _largest_temp_drop(group: pd.DataFrame, max_minutes: int = 90) -> tuple[float, pd.Timestamp | None]:
    if len(group) < 2:
        return 0.0, None
    best_drop = 0.0
    best_time: pd.Timestamp | None = None
    rows = list(group[["timestamp", "temp_f"]].dropna().itertuples(index=False, name=None))
    for i, (t0, temp0) in enumerate(rows):
        for t1, temp1 in rows[i + 1 :]:
            elapsed = (t1 - t0).total_seconds() / 60.0
            if elapsed > max_minutes:
                break
            drop = float(temp0 - temp1)
            if drop > best_drop:
                best_drop = drop
                best_time = t1
    return best_drop, best_time


def _max_wind_shift(group: pd.DataFrame) -> float:
    if "wind_dir_deg" not in group.columns:
        return 0.0
    valid = group.dropna(subset=["wind_dir_deg"]).copy()
    if "wind_speed_kt" in valid.columns:
        speed = pd.to_numeric(valid["wind_speed_kt"], errors="coerce")
        # Direction is not meaningful for calm/nearly calm winds; including 00000KT
        # creates huge false shifts when the next report has an ordinary direction.
        valid = valid[speed >= 4.0]
    if len(valid) < 2:
        return 0.0
    directions = valid["wind_dir_deg"].astype(float).tolist()
    return float(max(_circular_direction_change(a, b) for a, b in zip(directions, directions[1:])))


def _detect_outflow_event(
    group: pd.DataFrame,
    config: DisruptionConfig,
) -> dict[str, object]:
    """Detect a coherent pre-peak outflow event rather than mixing unrelated signals.

    A candidate requires a >= configured temperature drop inside 90 minutes and, in
    that same interval, either a strong gust or a large direction shift while winds
    are non-calm. This avoids false positives such as 00000KT followed by a routine
    southerly wind hours before the actual temperature peak.
    """
    if len(group) < 2:
        return {
            "candidate": False, "event_time": None, "drop_f": 0.0,
            "gust_kt": 0.0, "wind_shift_deg": 0.0,
        }

    g = group.sort_values("timestamp").copy()
    rows = list(g[["timestamp", "temp_f"]].dropna().itertuples(index=False, name=None))
    best = {
        "candidate": False, "event_time": None, "drop_f": 0.0,
        "gust_kt": 0.0, "wind_shift_deg": 0.0,
    }
    for i, (t0, temp0) in enumerate(rows):
        for t1, temp1 in rows[i + 1:]:
            elapsed = (t1 - t0).total_seconds() / 60.0
            if elapsed > 90.0:
                break
            drop = float(temp0 - temp1)
            if drop < config.outflow_temp_drop_f:
                continue

            interval = g[(g["timestamp"] >= t0) & (g["timestamp"] <= t1)]
            gust = 0.0
            if "wind_gust_kt" in interval.columns:
                vals = pd.to_numeric(interval["wind_gust_kt"], errors="coerce")
                if vals.notna().any():
                    gust = float(vals.max())

            shift = 0.0
            if {"wind_dir_deg", "wind_speed_kt"}.issubset(interval.columns):
                valid = interval.dropna(subset=["wind_dir_deg", "wind_speed_kt"]).copy()
                valid["wind_speed_kt"] = pd.to_numeric(valid["wind_speed_kt"], errors="coerce")
                valid = valid[valid["wind_speed_kt"] >= 5.0]
                if len(valid) >= 2:
                    dirs = valid["wind_dir_deg"].astype(float).tolist()
                    shift = float(max(_circular_direction_change(a, b) for a, b in zip(dirs, dirs[1:])))

            strong_gust = gust >= config.outflow_gust_kt
            coherent_shift = shift >= config.outflow_wind_shift_deg
            if not (strong_gust or coherent_shift):
                continue

            if (not best["candidate"]) or drop > float(best["drop_f"]):
                best = {
                    "candidate": True, "event_time": t1, "drop_f": drop,
                    "gust_kt": gust, "wind_shift_deg": shift,
                }
    return best


def _time_weighted_cloud_metrics(
    group: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    config: DisruptionConfig,
    column: str = "cloud_fraction",
) -> dict[str, float]:
    """Time-weight cloud observations so METAR specials do not distort cloud burden.

    Each report carries forward until the next report, capped at max_carry_minutes. This
    is still a proxy, but it is materially better than counting reports because stormy
    periods often generate many special METAR observations.
    """
    if group.empty or column not in group.columns or end_time <= start_time:
        return {
            "cloud_burden": np.nan,
            "cloudy_minutes": 0.0,
            "overcast_minutes": 0.0,
            "covered_minutes": 0.0,
        }

    g = group[["timestamp", column]].copy().sort_values("timestamp")
    g[column] = pd.to_numeric(g[column], errors="coerce")
    g = g[(g["timestamp"] <= end_time) & (g["timestamp"] >= start_time - pd.Timedelta(minutes=config.max_carry_minutes))]
    if g.empty:
        return {
            "cloud_burden": np.nan,
            "cloudy_minutes": 0.0,
            "overcast_minutes": 0.0,
            "covered_minutes": 0.0,
        }

    weighted_sum = 0.0
    covered = 0.0
    cloudy = 0.0
    overcast = 0.0
    rows = list(g.itertuples(index=False, name=None))
    for i, (ts, frac) in enumerate(rows):
        if pd.isna(frac):
            continue
        seg_start = max(ts, start_time)
        next_ts = rows[i + 1][0] if i + 1 < len(rows) else end_time
        seg_end = min(next_ts, end_time, ts + pd.Timedelta(minutes=config.max_carry_minutes))
        if seg_end <= seg_start:
            continue
        minutes = (seg_end - seg_start).total_seconds() / 60.0
        frac_f = float(frac)
        weighted_sum += frac_f * minutes
        covered += minutes
        if frac_f >= config.cloudy_threshold:
            cloudy += minutes
        if frac_f >= config.overcast_threshold:
            overcast += minutes

    return {
        "cloud_burden": weighted_sum / covered if covered > 0 else np.nan,
        "cloudy_minutes": cloudy,
        "overcast_minutes": overcast,
        "covered_minutes": covered,
    }


def build_disruption_features(
    observations: pd.DataFrame,
    config: DisruptionConfig = DisruptionConfig(),
) -> pd.DataFrame:
    """Create postmortem timing features for each local KLAS day.

    The public ASOS/METAR peak is used only as a timing proxy. The merged CLI maximum
    remains the settlement target.
    """
    if observations.empty:
        return pd.DataFrame()
    required = {"timestamp", "temp_f"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"observations missing required columns: {sorted(missing)}")

    obs = observations.copy()
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], errors="coerce")
    obs["temp_f"] = pd.to_numeric(obs["temp_f"], errors="coerce")
    obs = obs.dropna(subset=["timestamp", "temp_f"]).sort_values("timestamp")
    obs["local_date"] = obs["timestamp"].dt.date

    rows: list[dict[str, object]] = []
    for local_date, day in obs.groupby("local_date", sort=True):
        peak_idx = day["temp_f"].idxmax()
        peak_time = day.loc[peak_idx, "timestamp"]
        tz = day["timestamp"].dt.tz
        day_start = pd.Timestamp(local_date).tz_localize(tz)
        analysis_start = day_start + pd.Timedelta(hours=config.analysis_start_hour)
        analysis_end = day_start + pd.Timedelta(hours=config.analysis_end_hour)

        window = day[(day["timestamp"] >= analysis_start) & (day["timestamp"] < analysis_end)].copy()
        pre_peak = window[window["timestamp"] <= peak_time].copy()
        post_peak = window[window["timestamp"] > peak_time].copy()

        cloud = pd.to_numeric(pre_peak.get("cloud_fraction"), errors="coerce") if "cloud_fraction" in pre_peak else pd.Series(dtype=float)
        cloudy_mask = cloud >= config.cloudy_threshold
        overcast_mask = cloud >= config.overcast_threshold
        first_cloud_time = pre_peak.loc[cloudy_mask, "timestamp"].min() if cloudy_mask.any() else pd.NaT
        first_overcast_time = pre_peak.loc[overcast_mask, "timestamp"].min() if overcast_mask.any() else pd.NaT

        weighted = _time_weighted_cloud_metrics(day, analysis_start, peak_time, config)
        last3_start = max(analysis_start, peak_time - pd.Timedelta(hours=3))
        weighted_last3 = _time_weighted_cloud_metrics(day, last3_start, peak_time, config)
        weighted_low12 = _time_weighted_cloud_metrics(
            day, analysis_start, peak_time, config, column="cloud_fraction_below_12000"
        ) if "cloud_fraction_below_12000" in day.columns else {"cloud_burden": np.nan, "cloudy_minutes": 0.0, "overcast_minutes": 0.0, "covered_minutes": 0.0}
        weighted_low20 = _time_weighted_cloud_metrics(
            day, analysis_start, peak_time, config, column="cloud_fraction_below_20000"
        ) if "cloud_fraction_below_20000" in day.columns else {"cloud_burden": np.nan, "cloudy_minutes": 0.0, "overcast_minutes": 0.0, "covered_minutes": 0.0}

        precip = pd.to_numeric(day.get("precip_in"), errors="coerce") if "precip_in" in day else pd.Series(dtype=float)
        precip_mask = precip.fillna(0.0) > 0
        first_precip_time = day.loc[precip_mask, "timestamp"].min() if precip_mask.any() else pd.NaT
        precip_before_peak = bool((day.loc[precip_mask, "timestamp"] <= peak_time).any()) if precip_mask.any() else False
        precip_after_peak = bool((day.loc[precip_mask, "timestamp"] > peak_time).any()) if precip_mask.any() else False
        precip_pre = day[day["timestamp"] <= peak_time]
        precip_post = day[day["timestamp"] > peak_time]

        thunder_raw = day.get("thunder_observed", pd.Series(False, index=day.index))
        thunder_mask = _coerce_bool_series(thunder_raw)
        thunder_before_peak = bool((day.loc[thunder_mask, "timestamp"] <= peak_time).any()) if thunder_mask.any() else False
        thunder_after_peak = bool((day.loc[thunder_mask, "timestamp"] > peak_time).any()) if thunder_mask.any() else False

        # Distant/local cumulonimbus is useful context on Las Vegas monsoon days even
        # when thunder is not reported at the field itself. This does not by itself
        # prove a thunderstorm caused the miss, so it is mainly used as secondary evidence.
        if "metar" in day.columns:
            metar_text = day["metar"].fillna("").astype(str).str.upper()
            cb_mask = metar_text.str.contains(r"\bCB\b|\d{3}CB", regex=True)
        else:
            cb_mask = pd.Series(False, index=day.index)
        convective_cloud_before_peak = bool((day.loc[cb_mask, "timestamp"] <= peak_time).any()) if cb_mask.any() else False
        convective_cloud_after_peak = bool((day.loc[cb_mask, "timestamp"] > peak_time).any()) if cb_mask.any() else False

        largest_drop_f, largest_drop_time = _largest_temp_drop(pre_peak, max_minutes=90)
        max_shift = _max_wind_shift(pre_peak)
        max_gust = (
            float(pd.to_numeric(pre_peak["wind_gust_kt"], errors="coerce").max())
            if "wind_gust_kt" in pre_peak and pre_peak["wind_gust_kt"].notna().any()
            else 0.0
        )
        outflow = _detect_outflow_event(pre_peak, config)
        outflow_candidate = bool(outflow["candidate"])

        dew = pd.to_numeric(pre_peak.get("dewpoint_f"), errors="coerce") if "dewpoint_f" in pre_peak else pd.Series(dtype=float)
        max_dewpoint = float(dew.max()) if not dew.dropna().empty else np.nan
        elevated_moisture = bool(not pd.isna(max_dewpoint) and max_dewpoint >= config.elevated_dewpoint_f)
        lowest_bkn_ovc = (
            float(pd.to_numeric(pre_peak["lowest_bkn_ovc_ft"], errors="coerce").min())
            if "lowest_bkn_ovc_ft" in pre_peak.columns and pd.to_numeric(pre_peak["lowest_bkn_ovc_ft"], errors="coerce").notna().any()
            else np.nan
        )

        h08 = _checkpoint_temp(day, 8)
        h10 = _checkpoint_temp(day, 10)
        h12 = _checkpoint_temp(day, 12)
        h14 = _checkpoint_temp(day, 14)
        h16 = _checkpoint_temp(day, 16)

        def delta(a: float | None, b: float | None) -> float:
            return np.nan if a is None or b is None else float(b - a)

        heat_10_12 = delta(h10, h12)
        heat_12_14 = delta(h12, h14)
        midday_stall = bool(
            (not pd.isna(heat_10_12) and heat_10_12 <= 0.5)
            or (not pd.isna(heat_12_14) and heat_12_14 <= 0.5)
        )

        rows.append(
            {
                "date": local_date,
                "raw_peak_time_proxy": peak_time,
                "raw_peak_hour_local": float(peak_time.hour + peak_time.minute / 60.0),
                "cloud_mean_pre_peak": float(cloud.mean()) if not cloud.dropna().empty else np.nan,
                "cloudy_obs_fraction_pre_peak": float(cloudy_mask.mean()) if len(cloud) else np.nan,
                "overcast_obs_fraction_pre_peak": float(overcast_mask.mean()) if len(cloud) else np.nan,
                "cloud_burden_timeweighted_pre_peak": weighted["cloud_burden"],
                "cloudy_minutes_pre_peak": weighted["cloudy_minutes"],
                "overcast_minutes_pre_peak": weighted["overcast_minutes"],
                "cloud_covered_minutes_pre_peak": weighted["covered_minutes"],
                "cloud_burden_last3h_pre_peak": weighted_last3["cloud_burden"],
                "cloudy_minutes_last3h_pre_peak": weighted_last3["cloudy_minutes"],
                "cloud_burden_below_12000_pre_peak": weighted_low12["cloud_burden"],
                "cloudy_minutes_below_12000_pre_peak": weighted_low12["cloudy_minutes"],
                "cloud_burden_below_20000_pre_peak": weighted_low20["cloud_burden"],
                "cloudy_minutes_below_20000_pre_peak": weighted_low20["cloudy_minutes"],
                "lowest_bkn_ovc_ft_pre_peak": lowest_bkn_ovc,
                "first_cloudy_time": _iso_or_blank(first_cloud_time),
                "first_overcast_time": _iso_or_blank(first_overcast_time),
                "cloud_onset_minutes_before_peak": (
                    float((peak_time - first_cloud_time).total_seconds() / 60.0)
                    if not pd.isna(first_cloud_time)
                    else np.nan
                ),
                "precip_total_dedup_in": _hourly_precip_total(day),
                "precip_pre_peak_dedup_in": _hourly_precip_total(precip_pre),
                "precip_post_peak_dedup_in": _hourly_precip_total(precip_post),
                "precip_before_peak": precip_before_peak,
                "precip_after_peak": precip_after_peak,
                "first_precip_time": _iso_or_blank(first_precip_time),
                "precip_onset_minutes_before_peak": (
                    float((peak_time - first_precip_time).total_seconds() / 60.0)
                    if not pd.isna(first_precip_time) and first_precip_time <= peak_time
                    else np.nan
                ),
                "thunder_before_peak": thunder_before_peak,
                "thunder_after_peak": thunder_after_peak,
                "convective_cloud_before_peak": convective_cloud_before_peak,
                "convective_cloud_after_peak": convective_cloud_after_peak,
                "largest_pre_peak_temp_drop_90m_f": largest_drop_f,
                "largest_pre_peak_temp_drop_time": _iso_or_blank(largest_drop_time),
                "max_pre_peak_wind_shift_deg": max_shift,
                "max_pre_peak_gust_kt": max_gust,
                "outflow_candidate": outflow_candidate,
                "outflow_event_time": _iso_or_blank(outflow["event_time"]),
                "outflow_event_drop_f": float(outflow["drop_f"]),
                "outflow_event_gust_kt": float(outflow["gust_kt"]),
                "outflow_event_wind_shift_deg": float(outflow["wind_shift_deg"]),
                "outflow_minutes_before_peak": (
                    float((peak_time - outflow["event_time"]).total_seconds() / 60.0)
                    if outflow["candidate"] and outflow["event_time"] is not None
                    else np.nan
                ),
                "max_pre_peak_dewpoint_f": max_dewpoint,
                "elevated_moisture_signal": elevated_moisture,
                "heat_08_10_f": delta(h08, h10),
                "heat_10_12_f": heat_10_12,
                "heat_12_14_f": heat_12_14,
                "heat_14_16_f": delta(h14, h16),
                "midday_stall_signal": midday_stall,
                "post_peak_obs_count": int(len(post_peak)),
            }
        )
    return pd.DataFrame(rows)



def merge_disruption_features(daily: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Merge only feature columns that are not already present on the daily table.

    This prevents pandas from creating `_x`/`_y` duplicates when
    build_heating_curves.py has already embedded the same disruption features.
    """
    out = daily.copy()
    if out.empty or features.empty:
        return out
    out["date"] = out["date"].astype(str)
    f = features.copy()
    f["date"] = f["date"].astype(str)
    missing = [c for c in f.columns if c != "date" and c not in out.columns]
    if not missing:
        return out
    return out.merge(f[["date", *missing]], on="date", how="left")

def _num(row: pd.Series, key: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def classify_disruption_row(row: pd.Series) -> tuple[CauseCode, str, str]:
    """Return primary meteorological explanation for the NWS morning forecast error.

    Settlement reporting gaps are intentionally *not* used as a primary weather cause.
    They are added in separate settlement-gap columns by add_postmortem_labels().
    """
    error = _num(row, "nws_am_error_f")
    if error is None:
        return CauseCode.UNKNOWN, "LOW", "No final NWS/CLI error is available for postmortem."

    # A one-degree miss is ordinary forecast noise for this use case. Keep it separate
    # from the ASOS-vs-CLI settlement-gap diagnostic.
    if abs(error) <= 1.0:
        return CauseCode.NORMAL_RANGE, "HIGH", "NWS morning forecast finished within 1F of the official CLI high."

    if error > 0:  # NWS too warm: search for pre-peak suppression.
        if _as_bool(row.get("thunder_before_peak", False)):
            return CauseCode.TS_MONSOON, "HIGH", "Thunder was observed before the peak, a strong heating-disruption signal."
        if _as_bool(row.get("precip_before_peak", False)):
            return CauseCode.RAIN, "HIGH", "Precipitation was observed before the peak and likely reduced daytime heating."
        if _as_bool(row.get("outflow_candidate", False)):
            return CauseCode.WIND_OUTFLOW, "HIGH", "A pre-peak temperature drop coincided with a gust or major wind shift."

        cloud_burden = _num(row, "cloud_burden_timeweighted_pre_peak")
        cloudy_minutes = _num(row, "cloudy_minutes_pre_peak") or 0.0
        overcast_minutes = _num(row, "overcast_minutes_pre_peak") or 0.0
        last3_burden = _num(row, "cloud_burden_last3h_pre_peak")
        low12_burden = _num(row, "cloud_burden_below_12000_pre_peak")
        low20_burden = _num(row, "cloud_burden_below_20000_pre_peak")
        stalled = _as_bool(row.get("midday_stall_signal", False))
        # Cloud-height fields are preferred when available. High BKN/OVC alone is not
        # automatically blamed unless the observed heating curve also stalled.
        cloud_signal = (
            (low12_burden is not None and low12_burden >= 0.25)
            or (low20_burden is not None and low20_burden >= 0.35)
            or overcast_minutes >= 60.0
            or (stalled and cloudy_minutes >= 120.0)
            or (stalled and cloud_burden is not None and cloud_burden >= 0.50)
            or (stalled and last3_burden is not None and last3_burden >= 0.50)
        )
        if cloud_signal:
            confidence = "HIGH" if abs(error) >= 3.0 and stalled else "MEDIUM"
            note = "Material cloud cover occurred before the peak and coincided with suppressed/stalled heating."
            if _as_bool(row.get("convective_cloud_before_peak", False)):
                note += " Cumulonimbus was also reported before the peak, supporting a convective/monsoon contribution."
            return CauseCode.CLOUD, confidence, note

        if _as_bool(row.get("elevated_moisture_signal", False)) and _as_bool(row.get("midday_stall_signal", False)):
            return CauseCode.MOISTURE, "MEDIUM", "Elevated moisture coincided with a midday heating stall."

        peak_hour = _num(row, "raw_peak_hour_local")
        if peak_hour is not None and peak_hour < 14.0 and _as_bool(row.get("midday_stall_signal", False)):
            return CauseCode.EARLY_PEAK, "MEDIUM", "KLAS peaked early and the midday heating curve stalled without a stronger weather signal."

        return CauseCode.FORECAST_BIAS, "LOW", "NWS forecast was too warm, but the available ASOS timeline does not identify a dominant disruption yet."

    # NWS too cold. Post-peak rain/clouds are not allowed to explain the miss.
    peak_hour = _num(row, "raw_peak_hour_local")
    heat_14_16 = _num(row, "heat_14_16_f")
    # A rebound between 14-16 is not a late surge if the day's maximum already occurred
    # around midday. Require both meaningful late heating and a genuinely late peak.
    if (
        peak_hour is not None
        and peak_hour >= 15.5
        and heat_14_16 is not None
        and heat_14_16 >= 1.5
    ):
        return CauseCode.LATE_SURGE, "MEDIUM", "KLAS continued meaningful late-afternoon heating and set the daily peak late."
    if (peak_hour is not None and peak_hour < 14.5) and (
        _as_bool(row.get("thunder_after_peak", False))
        or _as_bool(row.get("precip_after_peak", False))
    ):
        return (
            CauseCode.FORECAST_BIAS,
            "MEDIUM",
            "KLAS exceeded the morning forecast before later thunder/rain arrived; the later storm did not cause the warm miss.",
        )
    return CauseCode.FORECAST_BIAS, "LOW", "NWS morning forecast was too cool; no dominant late-heating signal is identified yet."


def _secondary_cause_row(row: pd.Series) -> str:
    """Add supporting weather context without replacing the primary explanation."""
    primary = str(row.get("primary_cause", ""))
    if primary == CauseCode.TS_MONSOON:
        if _as_bool(row.get("precip_before_peak", False)):
            return CauseCode.RAIN
        if _as_bool(row.get("outflow_candidate", False)):
            return CauseCode.WIND_OUTFLOW
        if _as_bool(row.get("convective_cloud_before_peak", False)):
            return CauseCode.CLOUD
    if primary == CauseCode.RAIN and _as_bool(row.get("outflow_candidate", False)):
        return CauseCode.WIND_OUTFLOW
    if primary == CauseCode.WIND_OUTFLOW and _as_bool(row.get("convective_cloud_before_peak", False)):
        return CauseCode.TS_MONSOON
    if primary == CauseCode.CLOUD and _as_bool(row.get("convective_cloud_before_peak", False)):
        return CauseCode.TS_MONSOON
    return ""


def add_postmortem_labels(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    if out.empty:
        return out

    labels = out.apply(classify_disruption_row, axis=1, result_type="expand")
    labels.columns = ["primary_cause", "cause_confidence", "postmortem_notes"]
    out = pd.concat([out, labels], axis=1)
    out["secondary_cause"] = out.apply(_secondary_cause_row, axis=1)

    # Settlement diagnostic: separate from the meteorological cause of the forecast miss.
    gap = pd.to_numeric(out.get("asos_minus_cli_f"), errors="coerce") if "asos_minus_cli_f" in out else pd.Series(np.nan, index=out.index)
    out["settlement_gap_f"] = gap
    out["settlement_gap_abs_f"] = gap.abs()
    out["settlement_gap_flag"] = gap.abs().ge(1.0) & gap.notna()
    out["settlement_gap_note"] = np.where(
        out["settlement_gap_flag"],
        "Public ASOS/METAR peak differed by at least 1F from official CLI high.",
        "",
    )
    return out
