import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stock_modelv14 import _get_ci_params

def _make_data(n):
    X = np.arange(n * 5, dtype=float).reshape(n, 5)
    y = np.arange(n, dtype=float)
    return X, y


def test_wide_uses_full_history():
    X, y = _make_data(1000)
    X_q, y_q, la, ua = _get_ci_params('wide', X, y)
    assert len(X_q) == 1000
    assert len(y_q) == 1000
    assert la == 0.05
    assert ua == 0.95


def test_narrow_uses_last_756():
    X, y = _make_data(1000)
    X_q, y_q, la, ua = _get_ci_params('narrow', X, y)
    assert len(X_q) == 756
    assert len(y_q) == 756
    np.testing.assert_array_equal(X_q, X[-756:])
    assert la == 0.25
    assert ua == 0.75


def test_narrow_fallback_small_data():
    X, y = _make_data(500)
    X_q, y_q, la, ua = _get_ci_params('narrow', X, y)
    assert len(X_q) == 500
    assert len(y_q) == 500
    assert la == 0.25
    assert ua == 0.75


def test_narrow_exact_756_boundary():
    X, y = _make_data(756)
    X_q, y_q, _, _ = _get_ci_params('narrow', X, y)
    assert len(X_q) == 756
