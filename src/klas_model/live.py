from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json

import pandas as pd

from .heating_curve import CheckpointConfig, build_daily_heating_table
from .live_model import load_live_bundle, predict_from_bundle
from .remaining_heating import predict_remaining_heating_bundle
from .probabilities import central_range, empirical_integer_probabilities, probability_for_market

LAS_TZ = ZoneInfo("America/Los_Angeles")


def latest_checkpoint_hour(obs: pd.DataFrame) -> int | None:
    if obs.empty:
        return None
    ts = pd.to_datetime(obs["timestamp"], errors="coerce").dropna()
    if ts.empty:
        return None
    latest = ts.max()
    rounded = latest.round("h")
    hour = int(rounded.hour)
    return hour if 8 <= hour <= 18 else None


def latest_six_hour_max(obs: pd.DataFrame) -> tuple[int | None, str | None]:
    if obs.empty or "six_hour_max_f" not in obs:
        return None, None
    work = obs.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work["six_hour_max_f"] = pd.to_numeric(work["six_hour_max_f"], errors="coerce")
    rows = work.dropna(subset=["timestamp", "six_hour_max_f"]).sort_values("timestamp")
    if rows.empty:
        return None, None
    row = rows.iloc[-1]
    # Only use daytime reports as a floor for the current climate-day maximum.
    usable = 9 <= int(row["timestamp"].hour) <= 20
    return (int(row["six_hour_max_f"]) if usable else None, row["timestamp"].isoformat())


def live_weather_risk(obs: pd.DataFrame) -> tuple[str, list[str]]:
    if obs.empty:
        return "UNKNOWN", ["No KLAS observations available"]
    work = obs.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    latest = work["timestamp"].max()
    recent = work[work["timestamp"] >= latest - pd.Timedelta(hours=3)].copy()
    reasons: list[str] = []
    thunder = bool(recent.get("thunder_observed", pd.Series(False, index=recent.index)).fillna(False).any())
    rain = pd.to_numeric(recent.get("precip_in", 0), errors="coerce").fillna(0).sum() > 0
    cloud = pd.to_numeric(recent.get("cloud_fraction", 0), errors="coerce").fillna(0).mean()
    gust = pd.to_numeric(recent.get("wind_gust_kt", 0), errors="coerce").fillna(0).max()
    if thunder:
        reasons.append("Thunder/convective weather observed recently")
    if rain:
        reasons.append("Recent precipitation at KLAS")
    if cloud >= 0.75:
        reasons.append("Heavy recent cloud cover")
    if gust >= 25:
        reasons.append(f"Strong gusts up to {gust:.0f} kt")
    if thunder or rain or gust >= 30:
        return "HIGH", reasons
    if cloud >= 0.65 or gust >= 20:
        return "MEDIUM", reasons or ["Some weather disruption risk"]
    return "LOW", reasons or ["No major live weather disruption signal"]


def build_current_daily(obs: pd.DataFrame, nws_high_f: float, nws_issued_at: str | None = None) -> pd.DataFrame:
    daily = build_daily_heating_table(obs, CheckpointConfig(hours=tuple(range(8, 19))))
    if daily.empty:
        raise ValueError("No current-day KLAS observations could be summarized")
    daily = daily.tail(1).copy()
    daily["nws_am_forecast_high_f"] = float(nws_high_f)
    daily["nws_am_issued_at"] = nws_issued_at
    return daily


def _risk_rank(value: str | None) -> int:
    return {"UNKNOWN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(str(value or "UNKNOWN").upper(), 0)


def combine_weather_intelligence(
    observed_risk: str,
    observed_reasons: list[str],
    nws_forecast: dict | None = None,
    afd: dict | None = None,
    radar: dict | None = None,
) -> tuple[str, list[str], dict]:
    nws_forecast = nws_forecast or {}
    afd = afd or {}
    radar = radar or {}

    forecast_risk = "LOW"
    max_pop = nws_forecast.get("max_pop_pct")
    thunder = bool(nws_forecast.get("thunder_possible"))
    max_sky = nws_forecast.get("max_sky_cover_pct")
    if thunder and (max_pop is None or max_pop >= 30):
        forecast_risk = "HIGH"
    elif thunder or (max_pop is not None and max_pop >= 30) or (max_sky is not None and max_sky >= 85):
        forecast_risk = "MEDIUM"

    # AFD covers a broad WFO area, so it can raise situational awareness but does not
    # by itself force HIGH risk at the airport. Radar / observed conditions can.
    afd_raw = str(afd.get("risk") or "LOW").upper()
    afd_risk = "MEDIUM" if afd_raw == "HIGH" else afd_raw
    radar_risk = str(radar.get("risk") or "LOW").upper()
    overall = max((observed_risk, forecast_risk, afd_risk, radar_risk), key=_risk_rank)

    future_signal = max(_risk_rank(forecast_risk), _risk_rank(afd_risk), _risk_rank(radar_risk)) >= _risk_rank("MEDIUM")
    reasons = [
        r for r in observed_reasons
        if not (future_signal and "No major live weather disruption signal" in str(r))
    ]
    if nws_forecast.get("available"):
        if thunder:
            reasons.append("NWS hourly forecast includes thunderstorm potential")
        elif max_pop is not None and max_pop >= 20:
            reasons.append(f"NWS rain chance reaches {max_pop:.0f}% through this evening")
        if max_sky is not None and max_sky >= 70:
            reasons.append(f"NWS sky cover forecast reaches {max_sky:.0f}%")
    if afd.get("available") and afd_risk in {"MEDIUM", "HIGH"}:
        reasons.append("Las Vegas NWS discussion contains a convective/monsoon signal")
    if radar.get("available") and radar_risk in {"MEDIUM", "HIGH"}:
        reasons.append(str(radar.get("summary") or "Nearby radar signal"))

    # Keep the card readable.
    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    if not deduped:
        deduped = ["No major observed or forecast weather disruption signal"]

    detail = {
        "observed_risk": observed_risk,
        "forecast_risk": forecast_risk,
        "afd_risk": afd_risk,
        "radar_risk": radar_risk,
    }
    return overall, deduped[:5], detail


def build_live_state(
    obs: pd.DataFrame,
    nws_high_f: float,
    nws_issued_at: str | None,
    model_dir: str | Path,
    kalshi_markets: list[dict],
    nws_live_forecast: dict | None = None,
    afd: dict | None = None,
    radar: dict | None = None,
) -> dict:
    checkpoint = latest_checkpoint_hour(obs)
    latest = obs.sort_values("timestamp").iloc[-1]
    latest_ts = pd.to_datetime(latest["timestamp"])
    six_max, six_time = latest_six_hour_max(obs)
    raw_peak = float(pd.to_numeric(obs["temp_f"], errors="coerce").max())
    precise_series = pd.to_numeric(obs.get("metar_precise_temp_f"), errors="coerce") if "metar_precise_temp_f" in obs else pd.Series(dtype=float)
    precise_peak = None if precise_series.dropna().empty else float(precise_series.max())
    latest_precise = None
    if "metar_precise_temp_f" in latest.index and not pd.isna(latest.get("metar_precise_temp_f")):
        latest_precise = float(latest.get("metar_precise_temp_f"))
    observed_risk, observed_reasons = live_weather_risk(obs)
    risk, risk_reasons, risk_components = combine_weather_intelligence(
        observed_risk, observed_reasons, nws_live_forecast, afd, radar
    )

    state = {
        "date": latest_ts.date().isoformat(),
        "updated_at_local": datetime.now(LAS_TZ).isoformat(),
        "latest_metar_time": latest_ts.isoformat(),
        "latest_temp_f": float(latest["temp_f"]),
        "latest_precise_temp_f": latest_precise,
        "latest_dewpoint_f": None if pd.isna(latest.get("dewpoint_f")) else float(latest.get("dewpoint_f")),
        "raw_metar_peak_f": raw_peak,
        "precise_metar_peak_f": precise_peak,
        "six_hour_max_f": six_max,
        "six_hour_max_report_time": six_time,
        "nws_am_forecast_high_f": float(nws_high_f),
        "nws_am_issued_at": nws_issued_at,
        "checkpoint_hour": checkpoint,
        "weather_risk": risk,
        "weather_reasons": risk_reasons,
        "weather_risk_components": risk_components,
        "nws_live_forecast": nws_live_forecast or {},
        "afd": afd or {},
        "radar": radar or {},
        "model_available": False,
        "markets": [],
    }
    if checkpoint is None:
        return state

    # From 16:00 through 18:00, use the separately validated remaining-heating
    # models. Earlier checkpoints keep the existing NWS-anchored model.
    standard_bundle_path = Path(model_dir) / f"h{checkpoint:02d}.joblib"
    remaining_bundle_path = (
        Path(model_dir) / f"h{checkpoint:02d}_remaining.joblib"
    )
    use_remaining_heating = (
        checkpoint in {16, 17, 18}
        and remaining_bundle_path.exists()
    )

    bundle_path = (
        remaining_bundle_path
        if use_remaining_heating
        else standard_bundle_path
    )
    model_method = (
        "remaining_heating"
        if use_remaining_heating
        else "nws_anchored"
    )

    if not bundle_path.exists():
        state["model_note"] = (
            f"No validated bundle for {checkpoint}:00 yet"
        )
        return state

    current = build_current_daily(obs, nws_high_f, nws_issued_at)
    bundle = load_live_bundle(bundle_path)

    if use_remaining_heating:
        pred = predict_remaining_heating_bundle(bundle, current)
    else:
        pred = predict_from_bundle(bundle, current)

    model_high = float(pred["model_predicted_high_f"])

    # A live forecast can never be below a temperature already observed today.
    observed_floor_candidates = [raw_peak]
    if precise_peak is not None:
        observed_floor_candidates.append(float(precise_peak))
    if six_max is not None:
        observed_floor_candidates.append(float(six_max))
    observed_floor = max(observed_floor_candidates)

    if use_remaining_heating:
        model_high = max(model_high, observed_floor)
        distribution_floor = observed_floor
        mae = float(bundle.get("test_final_high_mae_f", 99))
    else:
        if six_max is not None:
            model_high = max(model_high, float(six_max))
        distribution_floor = six_max
        mae = float(
            bundle.get("test_metrics", {}).get("mae_f", 99)
        )

    distribution = empirical_integer_probabilities(
        model_high,
        bundle.get("calibration_model_errors_f", []),
        floor_f=distribution_floor,
    )
    low, high = central_range(distribution, 0.80)

    state["model_method"] = model_method
    state["model_bundle"] = bundle_path.name
    state["observed_floor_f"] = observed_floor
    if use_remaining_heating:
        state["predicted_remaining_heating_f"] = float(
            pred["predicted_remaining_heating_f"]
        )
        state["pre_checkpoint_peak_f"] = float(
            pred["pre_checkpoint_peak_f"]
        )
    if risk == "HIGH":
        confidence = "LOW"
    elif mae <= 0.8 and risk == "LOW":
        confidence = "HIGH"
    else:
        confidence = "MEDIUM"

    market_rows = []
    for m in kalshi_markets:
        p = probability_for_market(distribution, m)
        ask = m.get("yes_ask")
        mid = m.get("market_mid")
        market_rows.append({
            **m,
            "model_probability": p,
            "edge_vs_ask": (p - ask) if p is not None and ask is not None else None,
            "difference": (p - mid) if p is not None and mid is not None else None,
        })
    market_rows.sort(key=lambda x: (x.get("model_probability") or 0), reverse=True)
    bucket_total = sum(float(x.get("model_probability") or 0.0) for x in market_rows) if market_rows else None
    gaps = [x for x in market_rows if x.get("edge_vs_ask") is not None]
    largest_gap = max(gaps, key=lambda x: x.get("edge_vs_ask")) if gaps else None
    max_edge = None if largest_gap is None else float(largest_gap.get("edge_vs_ask") or 0.0)
    if risk == "HIGH":
        research_status = "WEATHER RISK — WAIT"
    elif confidence == "LOW":
        research_status = "LOW CONFIDENCE"
    elif max_edge is not None and max_edge >= 0.08:
        research_status = "LARGE MODEL/MARKET GAP"
    elif max_edge is not None and max_edge >= 0.05:
        research_status = "WATCH"
    else:
        research_status = "NO STRONG EDGE"

    state.update({
        "model_available": True,
        "model_predicted_high_f": model_high,
        "model_correction_f": model_high - float(nws_high_f),
        "likely_low_f": low,
        "likely_high_f": high,
        "confidence": confidence,
        "model_mae_f": mae,
        "distribution": {str(k): v for k, v in distribution.items()},
        "markets": market_rows,
        "bucket_probability_total": bucket_total,
        "largest_model_ask_gap": largest_gap,
        "research_status": research_status,
    })
    return state


def save_json(state: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    return path
