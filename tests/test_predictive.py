import numpy as np
import pandas as pd

from klas_model.predictive import backtest_checkpoint


def _sample(n=90):
    dates = pd.date_range("2024-06-01", periods=n, freq="D")
    nws = 105 + 4 * np.sin(np.arange(n) / 8)
    # deterministic pattern where current temp provides useful residual information
    anomaly = np.sin(np.arange(n) / 5)
    actual = nws + anomaly
    df = pd.DataFrame({
        "date": dates,
        "nws_am_forecast_high_f": nws,
        "actual_cli_high_f": actual,
        "h08_temp_f": nws - 20 + anomaly,
        "h08_dewpoint_f": 35 + anomaly,
        "h08_cloud_fraction": 0.1,
        "h08_wind_speed_kt": 5,
        "h09_temp_f": nws - 17 + anomaly,
        "h09_dewpoint_f": 35 + anomaly,
        "h09_cloud_fraction": 0.1,
        "h09_wind_speed_kt": 5,
        "h10_temp_f": nws - 13 + anomaly,
        "h10_dewpoint_f": 35 + anomaly,
        "h10_cloud_fraction": 0.1,
        "h10_wind_speed_kt": 5,
    })
    return df


def test_checkpoint_backtest_is_chronological_and_returns_predictions():
    result = backtest_checkpoint(_sample(), 10, min_train=45, test_block=15)
    assert len(result.predictions) == 45
    assert result.predictions["date"].min() >= "2024-07-16"
    assert result.metrics["n"] == 45
    assert "mae_improvement_f" in result.metrics


def test_checkpoint_rejects_unsupported_hour():
    try:
        backtest_checkpoint(_sample(), 19)
    except ValueError as exc:
        assert "unsupported checkpoint" in str(exc)
    else:
        raise AssertionError("expected ValueError")
