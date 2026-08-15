from __future__ import annotations

from .schema import CauseCode


def classify_primary_cause(
    *,
    model_error_f: float,
    cloud_fraction_max: float | None = None,
    precip_in: float | None = None,
    thunder: bool = False,
    wind_shift_or_outflow: bool = False,
    elevated_moisture: bool = False,
    synoptic_front_or_trough: bool = False,
    raw_cli_boundary_issue: bool = False,
) -> CauseCode:
    """Transparent first-pass postmortem classifier.

    Positive error means the model forecast was too warm. The hierarchy intentionally
    prioritizes observed disruptive weather over generic forecast bias.
    """
    if abs(model_error_f) < 1.0:
        return CauseCode.NORMAL_CLEAR
    if raw_cli_boundary_issue:
        return CauseCode.ROUNDING_CLI
    if thunder:
        return CauseCode.TS_MONSOON
    if precip_in is not None and precip_in > 0:
        return CauseCode.RAIN
    if wind_shift_or_outflow:
        return CauseCode.WIND_OUTFLOW
    if cloud_fraction_max is not None and cloud_fraction_max >= 0.5 and model_error_f > 0:
        return CauseCode.CLOUD
    if elevated_moisture and model_error_f > 0:
        return CauseCode.MOISTURE
    if synoptic_front_or_trough:
        return CauseCode.FRONT_TROUGH
    return CauseCode.FORECAST_BIAS
