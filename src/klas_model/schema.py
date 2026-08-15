from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class CauseCode(StrEnum):
    NORMAL_RANGE = "NORMAL_RANGE"
    NORMAL_CLEAR = "NORMAL_CLEAR"
    CLOUD = "CLOUD"
    RAIN = "RAIN"
    TS_MONSOON = "TS_MONSOON"
    MOISTURE = "MOISTURE"
    WIND_OUTFLOW = "WIND_OUTFLOW"
    FRONT_TROUGH = "FRONT_TROUGH"
    EARLY_PEAK = "EARLY_PEAK"
    LATE_SURGE = "LATE_SURGE"
    FORECAST_BIAS = "FORECAST_BIAS"
    ROUNDING_CLI = "ROUNDING_CLI"
    UNKNOWN = "UNKNOWN"


class DailyRecord(BaseModel):
    date: date
    station: str = "KLAS"

    # Forecasts captured before the outcome is known.
    nws_forecast_high_f: Optional[float] = None
    nws_forecast_issued_at: Optional[datetime] = None
    model_predicted_high_f: Optional[float] = None
    model_predicted_low_f: Optional[float] = None
    model_predicted_high_range_f: Optional[float] = None
    model_run_at: Optional[datetime] = None
    model_version: str = "baseline-v0.1"

    # Outcome / settlement.
    actual_cli_high_f: Optional[int] = None
    cli_issued_at: Optional[datetime] = None
    settlement_bucket: Optional[str] = None

    # Diagnostics.
    observed_peak_time: Optional[datetime] = None
    max_raw_asos_f: Optional[float] = None
    measurable_precip: Optional[bool] = None
    thunder_observed: Optional[bool] = None
    primary_cause: Optional[CauseCode] = None
    secondary_cause: Optional[CauseCode] = None
    postmortem_notes: Optional[str] = None

    # Audit fields.
    nws_source_url: Optional[str] = None
    cli_source_url: Optional[str] = None
    kalshi_market_ticker: Optional[str] = None

    @property
    def model_error_f(self) -> Optional[float]:
        if self.model_predicted_high_f is None or self.actual_cli_high_f is None:
            return None
        return self.model_predicted_high_f - self.actual_cli_high_f

    @property
    def nws_error_f(self) -> Optional[float]:
        if self.nws_forecast_high_f is None or self.actual_cli_high_f is None:
            return None
        return self.nws_forecast_high_f - self.actual_cli_high_f
