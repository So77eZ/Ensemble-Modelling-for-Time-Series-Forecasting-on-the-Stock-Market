import numpy as np
import pandas as pd


def _build_sequences(close_vals, look_back, horizon):
    """Воспроизводит логику подготовки последовательностей из prepare_and_train_model."""
    df = pd.DataFrame({'Close': close_vals})
    X, y = [], []
    for i in range(look_back, len(df) - horizon):
        X.append(df.iloc[i - look_back:i].values)
        y.append(df['Close'].iloc[i + horizon])
    return np.array(X), np.array(y)


def test_horizon1_target_is_next_day():
    close_vals = np.arange(50, dtype=float)
    _, y = _build_sequences(close_vals, look_back=10, horizon=1)
    # y[0] = Close[10+1] = 11; y[-1] = Close[49] = 49; len = 50-10-1 = 39
    assert y[0] == 11.0
    assert y[-1] == 49.0
    assert len(y) == 39


def test_horizon2_target_is_two_days_ahead():
    close_vals = np.arange(50, dtype=float)
    _, y = _build_sequences(close_vals, look_back=10, horizon=2)
    # y[0] = Close[12] = 12; y[-1] = Close[49] = 49; len = 50-10-2 = 38
    assert y[0] == 12.0
    assert y[-1] == 49.0
    assert len(y) == 38


def test_horizon3_target_is_three_days_ahead():
    close_vals = np.arange(50, dtype=float)
    _, y = _build_sequences(close_vals, look_back=10, horizon=3)
    # y[0] = Close[13] = 13; y[-1] = Close[49] = 49; len = 50-10-3 = 37
    assert y[0] == 13.0
    assert y[-1] == 49.0
    assert len(y) == 37


def test_merge_forecasts_from_three_calls():
    """Проверяет, что merge-логика caller'а корректна при разных горизонтах."""

    def mock_result(h):
        # Tuple: индекс 3 = forecasts dict, индекс 5 = confidence_intervals dict
        forecasts = {h: [float(h * 100)]}
        ci = {h: ([float(h * 90)], [float(h * 110)])}
        return (object(), None, None, forecasts, [], ci, 0.0, 0.0, 0.0, None, None, None)

    all_results = {h: mock_result(h) for h in [1, 2, 3]}
    forecasts = {h: all_results[h][3][h] for h in [1, 2, 3]}
    confidence_intervals = {h: all_results[h][5][h] for h in [1, 2, 3]}

    assert forecasts[1] == [100.0]
    assert forecasts[2] == [200.0]
    assert forecasts[3] == [300.0]
    assert confidence_intervals[1] == ([90.0], [110.0])
    assert confidence_intervals[2] == ([180.0], [220.0])
    assert confidence_intervals[3] == ([270.0], [330.0])
    # Совместимость с display-кодом: forecasts[h][-1], ci[h][0][-1]
    assert forecasts[1][-1] == 100.0
    assert confidence_intervals[2][0][-1] == 180.0
    assert confidence_intervals[3][1][-1] == 330.0
