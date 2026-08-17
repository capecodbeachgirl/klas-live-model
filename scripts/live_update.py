from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from klas_model.collectors.cli import fetch_cli_history
from klas_model.collectors.afd import fetch_latest_vef_afd
from klas_model.collectors.asos import fetch_live_asos
from klas_model.collectors.kalshi import fetch_open_temperature_markets, select_event_markets
from klas_model.collectors.nws_forecast import fetch_nws_live_forecast
from klas_model.collectors.pfm import fetch_pfm_morning_history
from klas_model.collectors.radar import fetch_radar_proximity, radar_export_url
from klas_model.collectors.satellite import fetch_satellite_cloud_watch
from klas_model.collectors.wethr import (
    apply_observed_floor,
    fetch_wethr_high,
    fetch_wethr_snapshot,
)
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

def append_model_history(state: dict, path: Path) -> pd.DataFrame:
    """Archive every model prediction so it can be scored after the final KLAS high is known."""

    path.parent.mkdir(parents=True, exist_ok=True)

    base = {
        "date": state.get("date"),
        "updated_at_local": state.get("updated_at_local"),
        "checkpoint_hour": state.get("checkpoint_hour"),
        "wethr_observed_high_f": (
            state.get("wethr_observed_high") or {}
        ).get("wethr_high_f"),
    }

    rows = []

    # Our validated KLAS model
    our_prediction = state.get("model_predicted_high_f")
    if our_prediction is not None:
        rows.append({
            **base,
            "model_name": "KLAS_MODEL",
            "predicted_high_f": float(our_prediction),
            "raw_forecast_high_f": float(our_prediction),
            "model_run_time_utc": None,
            "complete_run": True,
            "eligible_for_score": True,
            "model_method": state.get("model_method"),
        })

    # NWS morning forecast
    nws_prediction = state.get("nws_am_forecast_high_f")
    if nws_prediction is not None:
        rows.append({
            **base,
            "model_name": "NWS_MORNING",
            "predicted_high_f": float(nws_prediction),
            "raw_forecast_high_f": float(nws_prediction),
            "model_run_time_utc": state.get("nws_am_issued_at"),
            "complete_run": True,
            "eligible_for_score": True,
            "model_method": "nws_morning",
        })

    wethr = state.get("wethr") or {}
    consensus = wethr.get("consensus") or {}

    # Wethr multi-model consensus
    if consensus.get("available") and consensus.get("median_high_f") is not None:
        rows.append({
            **base,
            "model_name": "WETHR_CONSENSUS",
            "predicted_high_f": float(consensus["median_high_f"]),
            "raw_forecast_high_f": float(consensus["median_high_f"]),
            "model_run_time_utc": None,
            "complete_run": True,
            "eligible_for_score": True,
            "model_method": "median_consensus",
        })

    # Individual Wethr models
    for model_name, result in (wethr.get("models") or {}).items():
        raw_high = result.get("remaining_high_f")
        projected_high = result.get("projected_high_f")

        if raw_high is None and projected_high is None:
            rows.append({
                **base,
                "model_name": model_name,
                "predicted_high_f": None,
                "raw_forecast_high_f": None,
                "model_run_time_utc": result.get("run_time_utc"),
                "complete_run": bool(result.get("covers_rest_of_contract")),
                "eligible_for_score": False,
                "model_method": "wethr",
            })
            continue

        complete = bool(result.get("covers_rest_of_contract"))

        rows.append({
            **base,
            "model_name": model_name,
            "predicted_high_f": (
                float(projected_high)
                if projected_high is not None
                else float(raw_high)
            ),
            "raw_forecast_high_f": (
                float(raw_high)
                if raw_high is not None
                else None
            ),
            "model_run_time_utc": result.get("run_time_utc"),
            "complete_run": complete,
            "eligible_for_score": complete,
            "model_method": "wethr",
        })

    new = pd.DataFrame(rows)

    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)

    if not new.empty:
        new = new.drop_duplicates(
            subset=[
                "date",
                "updated_at_local",
                "model_name",
            ],
            keep="last",
        )

    new.to_csv(path, index=False)

    return new    

def score_model_history(
    history: pd.DataFrame,
    cli: pd.DataFrame,
) -> pd.DataFrame:
    """Score archived model forecasts against the official final KLAS CLI high."""

    scored = history.copy()

    score_columns = [
        "actual_cli_high_f",
        "signed_error_f",
        "abs_error_f",
        "forecast_rounded_f",
        "exact_hit",
        "within_1f",
        "within_2f",
    ]

    if scored.empty:
        for col in score_columns:
            scored[col] = pd.Series(dtype="object")
        return scored

    # Remove old scoring columns if this file has already been scored before.
    scored = scored.drop(
        columns=[col for col in score_columns if col in scored.columns],
        errors="ignore",
    )

    if cli.empty:
        for col in score_columns:
            scored[col] = pd.NA
        return scored

    truth = cli.copy()

    # Only final next-morning CLI reports are valid settlement truth.
    if "cli_is_final" in truth.columns:
        final_mask = (
            truth["cli_is_final"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        truth = truth[final_mask].copy()

    if truth.empty:
        for col in score_columns:
            scored[col] = pd.NA
        return scored

    truth["date"] = pd.to_datetime(
        truth["date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    truth["actual_cli_high_f"] = pd.to_numeric(
        truth["actual_cli_high_f"],
        errors="coerce",
    )

    truth = (
        truth[
            ["date", "actual_cli_high_f"]
        ]
        .dropna()
        .drop_duplicates(subset=["date"], keep="last")
    )

    scored["date"] = pd.to_datetime(
        scored["date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    scored = scored.merge(
        truth,
        on="date",
        how="left",
        validate="many_to_one",
    )

    predicted = pd.to_numeric(
        scored["predicted_high_f"],
        errors="coerce",
    )

    actual = pd.to_numeric(
        scored["actual_cli_high_f"],
        errors="coerce",
    )

    eligible = (
        scored["eligible_for_score"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    scorable = eligible & predicted.notna() & actual.notna()

    scored["signed_error_f"] = pd.NA
    scored["abs_error_f"] = pd.NA
    scored["forecast_rounded_f"] = pd.NA
    scored["exact_hit"] = pd.NA
    scored["within_1f"] = pd.NA
    scored["within_2f"] = pd.NA

    error = predicted - actual
    abs_error = error.abs()

    # Temperatures are positive here, so +0.5 then floor gives normal
    # nearest-integer rounding rather than Python's bankers rounding.
    rounded = ((predicted + 0.5) // 1).astype("Int64")

    scored.loc[scorable, "signed_error_f"] = error[scorable]
    scored.loc[scorable, "abs_error_f"] = abs_error[scorable]
    scored.loc[scorable, "forecast_rounded_f"] = rounded[scorable]

    scored.loc[scorable, "exact_hit"] = (
        rounded[scorable] == actual[scorable]
    )

    scored.loc[scorable, "within_1f"] = (
        abs_error[scorable] <= 1.0
    )

    scored.loc[scorable, "within_2f"] = (
        abs_error[scorable] <= 2.0
    )

    return scored

def fair_checkpoint_scores(history: pd.DataFrame) -> pd.DataFrame:
    """Keep one fair scored forecast per date, checkpoint hour, and model."""

    if history.empty:
        return history.copy()

    work = history.copy()

    required = {
        "date",
        "checkpoint_hour",
        "model_name",
        "updated_at_local",
        "predicted_high_f",
        "actual_cli_high_f",
        "eligible_for_score",
    }

    if not required.issubset(work.columns):
        return pd.DataFrame()

    work["checkpoint_hour"] = pd.to_numeric(
        work["checkpoint_hour"],
        errors="coerce",
    )

    work["predicted_high_f"] = pd.to_numeric(
        work["predicted_high_f"],
        errors="coerce",
    )

    work["actual_cli_high_f"] = pd.to_numeric(
        work["actual_cli_high_f"],
        errors="coerce",
    )

    work["_updated"] = pd.to_datetime(
        work["updated_at_local"],
        errors="coerce",
        utc=True,
    )

    eligible = (
        work["eligible_for_score"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    work = work[
        eligible
        & work["checkpoint_hour"].notna()
        & work["predicted_high_f"].notna()
        & work["actual_cli_high_f"].notna()
        & work["_updated"].notna()
    ].copy()

    if work.empty:
        return work.drop(columns=["_updated"], errors="ignore")

    # There may be several 15-minute refreshes during one checkpoint hour.
    # Use only the latest one so no model gets extra weight from extra refreshes.
    work = (
        work.sort_values("_updated")
        .drop_duplicates(
            subset=[
                "date",
                "checkpoint_hour",
                "model_name",
            ],
            keep="last",
        )
        .drop(columns=["_updated"])
        .reset_index(drop=True)
    )

    return work
def model_accuracy_summary(
    history: pd.DataFrame,
    days: int | None = None,
) -> pd.DataFrame:
    """Summarize model accuracy using one fair forecast per checkpoint."""

    fair = fair_checkpoint_scores(history)

    columns = [
        "model_name",
        "forecasts",
        "mae_f",
        "bias_f",
        "exact_pct",
        "within_1f_pct",
        "within_2f_pct",
        "closest_wins",
        "closest_win_pct",
    ]

    if fair.empty:
        return pd.DataFrame(columns=columns)

    work = fair.copy()

    work["date"] = pd.to_datetime(
        work["date"],
        errors="coerce",
    )

    work = work[work["date"].notna()].copy()

    if work.empty:
        return pd.DataFrame(columns=columns)

    if days is not None:
        latest_date = work["date"].max()
        cutoff = latest_date - pd.Timedelta(days=days - 1)
        work = work[work["date"] >= cutoff].copy()

    if work.empty:
        return pd.DataFrame(columns=columns)

    work["error_f"] = (
        work["predicted_high_f"]
        - work["actual_cli_high_f"]
    )

    work["abs_error_f"] = work["error_f"].abs()

    work["forecast_rounded_f"] = (
        (work["predicted_high_f"] + 0.5) // 1
    ).astype("Int64")

    work["exact_hit"] = (
        work["forecast_rounded_f"]
        == work["actual_cli_high_f"]
    )

    work["within_1f"] = work["abs_error_f"] <= 1.0
    work["within_2f"] = work["abs_error_f"] <= 2.0

    # Find the lowest error among all available models
    # at the same date/checkpoint. Ties each receive a win.
    group_min_error = work.groupby(
        ["date", "checkpoint_hour"]
    )["abs_error_f"].transform("min")

    work["closest_win"] = (
        work["abs_error_f"] == group_min_error
    )

    summary = (
        work.groupby("model_name", as_index=False)
        .agg(
            forecasts=("predicted_high_f", "size"),
            mae_f=("abs_error_f", "mean"),
            bias_f=("error_f", "mean"),
            exact_pct=("exact_hit", "mean"),
            within_1f_pct=("within_1f", "mean"),
            within_2f_pct=("within_2f", "mean"),
            closest_wins=("closest_win", "sum"),
        )
    )

    summary["exact_pct"] *= 100.0
    summary["within_1f_pct"] *= 100.0
    summary["within_2f_pct"] *= 100.0

    summary["closest_win_pct"] = (
        summary["closest_wins"]
        / summary["forecasts"]
        * 100.0
    )

    summary = summary.sort_values(
        ["mae_f", "model_name"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return summary[columns]
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
    ap.add_argument(
    "--model-history",
    default="data/live/klas_model_history.csv",
)
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

    wethr_high = _safe_fetch(
    "Wethr observed high",
    fetch_wethr_high,
    {
        "available": False,
        "wethr_high_f": None,
        "omo_informed": False,
    },
)

    wethr = _safe_fetch(
    "Wethr multi-model forecasts",
    lambda: fetch_wethr_snapshot(now_local=now),
    {
        "available": False,
        "models": {},
        "research_only": True,
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

    observed_peak_candidates = [
    state.get("raw_metar_peak_f"),
    state.get("precise_metar_peak_f"),
    state.get("six_hour_max_f"),
    (
        wethr_high.get("wethr_high_f")
        if wethr_high.get("available")
        else None
    ),
]

    observed_peak_values = [
        float(value)
        for value in observed_peak_candidates
        if value is not None
    ]

    observed_floor = (
        max(observed_peak_values)
        if observed_peak_values
        else None
    )

    state["wethr"] = apply_observed_floor(
        wethr,
        observed_floor,
    )

    state["wethr_observed_high"] = wethr_high

    if not state.get("model_available"):
        if state.get("checkpoint_hour") is None and now.hour < 8:
            state["research_status"] = "WAITING FOR 8:00 AM MODEL"
        else:
            state["research_status"] = "MODEL UNAVAILABLE"

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

    model_history = append_model_history(
    state,
    Path(args.model_history),
)
    # Score archived forecasts only when an official final CLI high exists.
    history_dates = pd.to_datetime(
        model_history["date"],
        errors="coerce",
    ).dropna()

    yesterday = (now.date() - timedelta(days=1))

    if not history_dates.empty:
        first_history_date = history_dates.min().date()

        if first_history_date <= yesterday:
            try:
                cli_history = fetch_cli_history(
                    first_history_date.isoformat(),
                    yesterday.isoformat(),
                )

                model_history = score_model_history(
                    model_history,
                    cli_history,
                )

                model_history.to_csv(
                    Path(args.model_history),
                    index=False,
                )

            except Exception as exc:
                print(f"CLI scoring warning: {exc}")

    state["model_accuracy"] = {
        "7_day": model_accuracy_summary(
            model_history,
            days=7,
        ).to_dict(orient="records"),
        "30_day": model_accuracy_summary(
            model_history,
            days=30,
        ).to_dict(orient="records"),
        "all_time": model_accuracy_summary(
            model_history,
        ).to_dict(orient="records"),
    }
        
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
