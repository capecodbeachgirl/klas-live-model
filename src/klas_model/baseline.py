from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaselinePrediction:
    predicted_high_f: float
    uncertainty_f: float
    explanation: list[str]


def predict_from_nws_and_curve(
    nws_high_f: float,
    current_temp_f: float | None = None,
    expected_temp_now_f: float | None = None,
    disruptive_weather: bool = False,
) -> BaselinePrediction:
    """Baseline v0.1: NWS forecast adjusted by KLAS heating-curve deviation.

    This is deliberately simple so historical validation can tell us whether each added
    feature improves the model rather than hiding logic in a black box.
    """
    prediction = float(nws_high_f)
    uncertainty = 1.5
    explanation = [f"Started from NWS high {nws_high_f:.0f}°F"]

    if current_temp_f is not None and expected_temp_now_f is not None:
        deviation = current_temp_f - expected_temp_now_f
        adjustment = max(-2.0, min(2.0, 0.6 * deviation))
        prediction += adjustment
        explanation.append(
            f"Heating curve deviation {deviation:+.1f}°F; adjusted high {adjustment:+.1f}°F"
        )

    if disruptive_weather:
        uncertainty += 2.0
        explanation.append("Disruptive weather flag widened uncertainty")

    return BaselinePrediction(round(prediction, 1), uncertainty, explanation)
