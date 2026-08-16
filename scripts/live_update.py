from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from klas_model.collectors.afd import fetch_latest_vef_afd
from klas_model.collectors.asos import fetch_live_asos
from klas_model.collectors.kalshi import fetch_open_temperature_markets, select_event_markets
from klas_model.collectors.nws_forecast import fetch_nws_live_forecast
from klas_model.collectors.pfm import fetch_pfm_morning_history
from klas_model.collectors.radar import fetch_radar_proximity, radar_export_url
from klas_model.collectors.satellite import fetch_satellite_cloud_watch
from klas_model.dashboard import save_dashboard
from klas_model.live import build_live_state, save_json

TZ = ZoneInfo("America/Los_Angeles")


def _safe_fetch(label: str, func, fallback: dict) -> dict:
    try:
        return func()
    except Exception as exc:
        print(f"{label} warning: {exc}")
        return {**fallback, "error": str(exc)}


def append_history(state: dict, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "date": state.get("date"),
        "updated_at_local": state.get("updated_at_local"),
        "latest_metar_time": state.get("latest_metar_time"),
        "checkpoint_hour": state.get("checkpoint_hour"),
        "latest_temp_f": state.get("latest_temp_f"),
        "latest_precise_temp_f": state.get("latest_precise_temp_f"),
        "six_hour_max_f": state.get("six_hour_max_f"),
        "nws_am_forecast_high_f": state.get("nws_am_forecast_high_f"),
        "model_predicted_high_f": state.get("model_predicted_high_f"),
        "confidence": state.get("confidence"),
        "weather_risk": state.get("weather_risk"),
        "forecast_max_pop_pct": (state.get("nws_live_forecast") or {}).get("max_pop_pct"),
        "forecast_thunder": (state.get("nws_live_forecast") or {}).get("thunder_possible"),
        "radar_risk": (state.get("radar") or {}).get("risk"),
        "research_status": state.get("research_status"),
    }
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
        new = new.drop_duplicates(subset=["latest_metar_time"], keep="last")
    new.to_csv(path, index=False)
    return new


def progression_rows(history: pd.DataFrame, today: str, limit: int = 8) -> list[dict]:
    if history.empty or "date" not in history:
        return []
    work = history[history["date"].astype(str) == today].copy()
    if work.empty:
        return []
    work["_metar"] = pd.to_datetime(work["latest_metar_time"], errors="coerce")
    work = work.sort_values("_metar").drop_duplicates(subset=["checkpoint_hour"], keep="last").tail(limit)
    fields = [
        "checkpoint_hour", "latest_precise_temp_f", "latest_temp_f", "six_hour_max_f",
        "model_predicted_high_f", "nws_am_forecast_high_f", "weather_risk",
    ]
    rows = []
    for _, r in work.iterrows():
        rows.append({k: (None if pd.isna(r.get(k)) else r.get(k)) for k in fields})
    return rows

def historical_analogs(
    state: dict,
    path: Path = Path("data/processed/klas_daily_heating.csv"),
) -> dict:
    """Find past KLAS days that looked similar at the current checkpoint hour."""
    checkpoint = state.get("checkpoint_hour")
    current_temp = state.get("latest_precise_temp_f")
    if current_temp is None:
        current_temp = state.get("latest_temp_f")

    if checkpoint is None or current_temp is None or not path.exists():
        return {"available": False}

    hour = int(checkpoint)
    if hour < 8 or hour > 18:
        return {"available": False}

    temp_col = f"h{hour:02d}_temp_f"
    remain_col = f"h{hour:02d}_heating_remaining_cli_f"

    hist = pd.read_csv(path)

    required = {temp_col, "actual_cli_high_f"}
    if not required.issubset(hist.columns):
        return {"available": False}

    hist[temp_col] = pd.to_numeric(hist[temp_col], errors="coerce")
    hist["actual_cli_high_f"] = pd.to_numeric(
        hist["actual_cli_high_f"], errors="coerce"
    )

    if remain_col in hist.columns:
        hist[remain_col] = pd.to_numeric(hist[remain_col], errors="coerce")

    if "nws_am_forecast_high_f" in hist.columns:
        hist["nws_am_forecast_high_f"] = pd.to_numeric(
            hist["nws_am_forecast_high_f"], errors="coerce"
        )

    hist = hist.dropna(subset=[temp_col, "actual_cli_high_f"]).copy()

    if "day_complete" in hist.columns:
        complete = hist["day_complete"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
        hist = hist[complete]

    # Never use today itself as one of the historical analogs.
    if "date" in hist.columns and state.get("date"):
        hist = hist[hist["date"].astype(str) != str(state["date"])]

    if hist.empty:
        return {"available": False}

    hist["_temp_diff"] = (hist[temp_col] - float(current_temp)).abs()

    nws_today = state.get("nws_am_forecast_high_f")
    if nws_today is not None and "nws_am_forecast_high_f" in hist.columns:
        hist["_nws_diff"] = (
            hist["nws_am_forecast_high_f"] - float(nws_today)
        ).abs()
    else:
        hist["_nws_diff"] = 0.0

    # Start fairly strict, then widen only if we do not have enough examples.
    analogs = hist[
        (hist["_temp_diff"] <= 2.0)
        & (hist["_nws_diff"] <= 3.0)
    ].copy()

    if len(analogs) < 12:
        analogs = hist[
            (hist["_temp_diff"] <= 3.0)
            & (hist["_nws_diff"] <= 5.0)
        ].copy()

    if len(analogs) < 8:
        analogs = hist[hist["_temp_diff"] <= 4.0].copy()

    if analogs.empty:
        return {"available": False}

    # Favor the closest temperature/NWS matches if the pool is large.
    analogs["_match_score"] = (
        analogs["_temp_diff"] + 0.35 * analogs["_nws_diff"]
    )
    analogs = analogs.sort_values("_match_score").head(40)

    final_highs = analogs["actual_cli_high_f"].dropna()

    result = {
        "available": True,
        "count": int(len(analogs)),
        "checkpoint_hour": hour,
        "current_temp_f": round(float(current_temp), 1),
        "median_final_high_f": round(float(final_highs.median()), 1),
        "range_80_low_f": round(float(final_highs.quantile(0.10)), 1),
        "range_80_high_f": round(float(final_highs.quantile(0.90)), 1),
    }

    if remain_col in analogs.columns:
        remaining = analogs[remain_col].dropna()
        if not remaining.empty:
            result["median_heating_remaining_f"] = round(
                float(remaining.median()), 1
            )

    return result
    
def next_scheduled_update(now: datetime) -> str:
    for hours_ahead in range(25):
        base = now + timedelta(hours=hours_ahead)
        minutes = [5, 20, 35, 50] if 8 <= base.hour <= 19 else [5]

        for minute in minutes:
            candidate = base.replace(minute=minute, second=0, microsecond=0)
            if candidate > now:
                return candidate.isoformat()

    return (now + timedelta(hours=1)).isoformat()


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh the live KLAS/Kalshi dashboard")
    ap.add_argument("--model-dir", default="data/model")
    ap.add_argument("--json", default="data/live/klas_live.json")
    ap.add_argument("--history", default="data/live/klas_live_history.csv")
    ap.add_argument("--dashboard", default="docs/index.html")
    args = ap.parse_args()

    now = datetime.now(TZ)
    today = now.date()
    obs = fetch_live_asos(today.isoformat(), today.isoformat())
    obs_source = str(obs.get("data_source", pd.Series(["unknown"])).iloc[-1]) if not obs.empty else "unknown"
    print(f"KLAS observation source: {obs_source}")
    pfm = fetch_pfm_morning_history(today.isoformat(), today.isoformat())
    if pfm.empty:
        raise RuntimeError("No pre-06:00 NWS PFM high found for today")
    nws = pfm.iloc[-1]

    nws_live = _safe_fetch(
        "NWS hourly forecast",
        lambda: fetch_nws_live_forecast(now_local=now),
        {"available": False, "max_pop_pct": None, "thunder_possible": False, "max_sky_cover_pct": None},
    )
    afd = _safe_fetch(
        "NWS AFD",
        fetch_latest_vef_afd,
        {"available": False, "risk": "LOW", "snippet": "AFD unavailable"},
    )
    radar = _safe_fetch(
        "NWS MRMS radar",
        fetch_radar_proximity,
        {
            "available": False,
            "risk": "LOW",
            "summary": "Automated radar ring scan unavailable",
            "image_url": radar_export_url(),
        },
    )

    satellite = _safe_fetch(
        "GOES satellite",
        lambda: fetch_satellite_cloud_watch(obs),
        {
            "available": False,
            "risk": "UNKNOWN",
            "summary": "Satellite cloud watch unavailable",
        },
    )

    print(
        "satellite cloud: "
        f"{satellite.get('risk')} | "
        f"{satellite.get('summary')}"
    )

    try:
        all_markets = fetch_open_temperature_markets()
        markets = select_event_markets(all_markets, today)
    except Exception as exc:
        print(f"Kalshi market fetch warning: {exc}")
        markets = []

    state = build_live_state(
        obs,
        float(nws["nws_am_forecast_high_f"]),
        nws.get("nws_am_issued_at"),
        args.model_dir,
        markets,
        nws_live_forecast=nws_live,
        afd=afd,
        radar=radar,
    )
    state["satellite"] = satellite
    if state.get("model_available"):
        risk = str(state.get("weather_risk") or "UNKNOWN").upper()
        top_gap = state.get("largest_model_ask_gap") or {}
        edge = top_gap.get("edge_vs_ask")

        if risk == "MEDIUM":
            if edge is not None and float(edge) >= 0.08:
                state["research_status"] = "EDGE WATCH — WEATHER RISK MEDIUM"
            elif edge is not None and float(edge) >= 0.05:
                state["research_status"] = "WATCH — WEATHER RISK MEDIUM"
            else:
                state["research_status"] = "WEATHER WATCH"

    state["historical_analogs"] = historical_analogs(state)
    print(f"historical analogs: {state['historical_analogs']}")
    
    state["next_update_local"] = next_scheduled_update(now)
    history = append_history(state, Path(args.history))
    state["progression"] = progression_rows(history, state["date"])
    save_json(state, args.json)
    save_dashboard(state, args.dashboard)

    print(f"updated {args.dashboard}")
    print(
        f"KLAS {state.get('latest_precise_temp_f') or state.get('latest_temp_f')}F | "
        f"NWS {state.get('nws_am_forecast_high_f')}F | "
        f"model {state.get('model_predicted_high_f','—')}F | "
        f"risk {state.get('weather_risk')} | "
        f"status {state.get('research_status','—')}"
    )
    print(
        "weather intel: "
        f"PoP {(nws_live or {}).get('max_pop_pct')}% | "
        f"thunder {(nws_live or {}).get('thunder_possible')} | "
        f"AFD {(state.get('weather_risk_components') or {}).get('afd_risk')} | radar {(state.get('weather_risk_components') or {}).get('radar_risk')}"
    )


if __name__ == "__main__":
    main()
