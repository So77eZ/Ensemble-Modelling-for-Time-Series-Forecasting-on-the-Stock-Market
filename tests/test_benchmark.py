import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from unittest.mock import patch, MagicMock


def test_benchmark_mode_yes_skips_other_questions():
    """При ответе '2' на вопрос 0 возвращается benchmark_mode=True без других вопросов."""
    with patch('builtins.input', side_effect=['2', 'y']):
        from stock_modelv14 import get_user_inputs
        result = get_user_inputs()
    assert result['benchmark_mode'] is True


def test_benchmark_mode_no_returns_false():
    """При ответе '1' возвращается benchmark_mode=False и продолжаются обычные вопросы."""
    answers = [
        '1',   # вопрос 0: нет бенчмарку
        '',    # тикер — Enter = SBER
        '1',   # режим — обычный прогноз
        'n',   # оптимизация — нет (нет сохранённых параметров)
        'n',   # show_ci
        'n',   # show_plot
        '1',   # ci_mode — wide
        '1',   # presentation — нет
        'y',   # подтверждение
    ]
    # load_hyperparams мокируем → (None, None), чтобы убрать условную ветку
    with patch('builtins.input', side_effect=answers), \
         patch('stock_modelv14.load_hyperparams', return_value=(None, None)):
        from stock_modelv14 import get_user_inputs
        result = get_user_inputs()
    assert result['benchmark_mode'] is False
    assert result['ticker'] == 'SBER'


def _sample_metrics():
    return {
        'version': 'stock_modelv14',
        'run_at': '2026-05-04 14:00',
        'horizons': {
            1: {'forecast': 263.1, 'real': 261.8, 'error_pct': 0.50,
                'in_ci': True, 'rmse': 4.21, 'mae': 3.10, 'r2': 0.91},
            2: {'forecast': 264.0, 'real': 263.2, 'error_pct': 0.30,
                'in_ci': True, 'rmse': 5.03, 'mae': 4.20, 'r2': 0.88},
            3: {'forecast': 262.5, 'real': 260.1, 'error_pct': 0.92,
                'in_ci': False, 'rmse': 6.11, 'mae': 5.30, 'r2': 0.84},
        },
        'avg_rmse': 5.12, 'avg_mae': 4.20, 'avg_r2': 0.876,
        'avg_error_pct': 0.57, 'ci_coverage': 66.7,
    }


def test_append_creates_file_with_header(tmp_path):
    bm_file = str(tmp_path / 'benchmarks.md')
    with patch('stock_modelv14.BENCHMARKS_FILE', bm_file):
        from stock_modelv14 import append_benchmark_result
        append_benchmark_result(_sample_metrics())
    content = Path(bm_file).read_text(encoding='utf-8')
    assert '# Benchmarks' in content
    assert '## Методология' in content
    assert '## Результаты' in content


def test_append_writes_row_with_metrics(tmp_path):
    bm_file = str(tmp_path / 'benchmarks.md')
    with patch('stock_modelv14.BENCHMARKS_FILE', bm_file):
        from stock_modelv14 import append_benchmark_result
        append_benchmark_result(_sample_metrics())
    content = Path(bm_file).read_text(encoding='utf-8')
    assert 'stock_modelv14' in content
    assert '4.21/5.03/6.11' in content
    assert '263.1/264.0/262.5' in content
    assert '261.8/263.2/260.1' in content
    assert '67%' in content


def test_append_twice_adds_two_rows(tmp_path):
    bm_file = str(tmp_path / 'benchmarks.md')
    m1 = _sample_metrics()
    m2 = _sample_metrics()
    m2['version'] = 'stock_modelv15'
    with patch('stock_modelv14.BENCHMARKS_FILE', bm_file):
        from stock_modelv14 import append_benchmark_result
        append_benchmark_result(m1)
        append_benchmark_result(m2)
    content = Path(bm_file).read_text(encoding='utf-8')
    assert content.count('stock_modelv1') == 2


def _fake_backtest_return():
    results_list = [
        {'horizon': 1, 'forecast': 263.1, 'real': 261.8,
         'error': 1.3, 'error_pct': 0.50, 'in_ci': True,
         'lower_ci': 258.0, 'upper_ci': 268.0},
        {'horizon': 2, 'forecast': 264.0, 'real': 263.2,
         'error': 0.8, 'error_pct': 0.30, 'in_ci': True,
         'lower_ci': 259.0, 'upper_ci': 269.0},
        {'horizon': 3, 'forecast': 262.5, 'real': 260.1,
         'error': 2.4, 'error_pct': 0.92, 'in_ci': False,
         'lower_ci': 256.0, 'upper_ci': 265.0},
    ]
    forecasts = {1: [263.1], 2: [264.0], 3: [262.5]}
    forecast_dates = ['2024-10-15', '2024-10-16', '2024-10-17']
    confidence_intervals = {}
    all_results = {
        1: (None, None, None, None, None, None, 4.21, 3.10, 0.91, None, None, None),
        2: (None, None, None, None, None, None, 5.03, 4.20, 0.88, None, None, None),
        3: (None, None, None, None, None, None, 6.11, 5.30, 0.84, None, None, None),
    }
    return results_list, forecasts, forecast_dates, confidence_intervals, all_results


def test_run_benchmark_structure():
    import pandas as pd
    fake_data = MagicMock(spec=pd.DataFrame)

    with patch('stock_modelv14.run_backtest', return_value=_fake_backtest_return()):
        from stock_modelv14 import run_benchmark
        result = run_benchmark(fake_data)

    assert result['version'] is not None
    assert set(result['horizons'].keys()) == {1, 2, 3}
    assert abs(result['avg_rmse'] - (4.21 + 5.03 + 6.11) / 3) < 0.001
    assert abs(result['avg_mae'] - (3.10 + 4.20 + 5.30) / 3) < 0.001
    assert abs(result['avg_r2'] - (0.91 + 0.88 + 0.84) / 3) < 0.001
    assert abs(result['ci_coverage'] - 66.666) < 0.1
    assert result['horizons'][1]['forecast'] == 263.1
    assert result['horizons'][3]['in_ci'] is False
