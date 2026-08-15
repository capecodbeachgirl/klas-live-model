import numpy as np

from klas_model.conservative_predictive import apply_conservative_adjustment


def test_conservative_adjustment_gate_and_cap():
    nws = np.array([100.0, 100.0, 100.0])
    raw = np.array([0.4, 2.0, -10.0])
    pred = apply_conservative_adjustment(nws, raw, shrink=0.5, gate_f=0.5, cap_f=2.0)
    assert np.allclose(pred, [100.0, 101.0, 98.0])


def test_zero_shrink_equals_nws():
    nws = np.array([99.0, 101.0])
    raw = np.array([5.0, -5.0])
    pred = apply_conservative_adjustment(nws, raw, shrink=0.0, gate_f=0.0, cap_f=3.0)
    assert np.allclose(pred, nws)
