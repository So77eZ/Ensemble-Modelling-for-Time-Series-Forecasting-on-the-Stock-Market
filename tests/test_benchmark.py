import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch


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
