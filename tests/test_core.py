from klas_model.baseline import predict_from_nws_and_curve
from klas_model.buckets import bucket_for_strikes
from klas_model.postmortem import classify_primary_cause
from klas_model.schema import CauseCode


def test_bucket_mapping():
    strikes = [(None, 108, "108 or below"), (109, 110, "109-110"), (111, 112, "111-112"), (113, None, "113 or above")]
    assert bucket_for_strikes(110, strikes) == "109-110"
    assert bucket_for_strikes(114, strikes) == "113 or above"


def test_cloud_postmortem():
    cause = classify_primary_cause(model_error_f=3.0, cloud_fraction_max=0.8)
    assert cause == CauseCode.CLOUD


def test_baseline_curve_adjustment():
    result = predict_from_nws_and_curve(110, current_temp_f=101, expected_temp_now_f=99)
    assert result.predicted_high_f > 110
