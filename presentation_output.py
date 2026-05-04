"""
Презентационный режим визуализации прогнозов.

Назначение: подготовка графиков для слайдов защиты ВКР и публичных
демонстраций. Отличия от стандартной визуализации:

* Все подписи на русском языке.
* Скрыта линия предсказаний на тестовой выборке (Predicted/test).
* Только три элемента: фактическая цена, точечный прогноз 1-3 дня
  и заливка 90%-го доверительного интервала.
* Окно отображения настраивается (по умолчанию 90 торговых дней истории).
* Крупный шрифт, светлая сетка, поля под проектор и печать.
* Сохранение в PNG 1920x1080 при 150 DPI.

Интеграция в stock_modelv14.py — три точки:
    1. Импорт функции plot_presentation в начале файла.
    2. В get_user_inputs() добавить вопрос о презентационном режиме
       (см. patch_user_inputs ниже).
    3. В argparse добавить флаги --presentation и --history-window
       (см. patch_argparse ниже).
    4. После основного построения графика — вызвать plot_presentation,
       если установлен соответствующий флаг.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Константы стиля. Менять здесь, если нужно подогнать под фирменный стиль вуза.
# ---------------------------------------------------------------------------

_FIG_SIZE = (16, 9)            # 16:9 для слайда
_DPI = 150                     # достаточно для проектора
_TITLE_FONT_SIZE = 18
_LABEL_FONT_SIZE = 14
_TICK_FONT_SIZE = 12
_LEGEND_FONT_SIZE = 13

_COLOR_REAL = '#1f4e79'        # тёмно-синий — фактическая цена
_COLOR_FORECAST = '#c0392b'    # красно-кирпичный — прогноз
_COLOR_CI_FILL = '#5b9bd5'     # голубой — заливка CI
_CI_ALPHA = 0.22

_MARKER_SIZE = 9
_LINE_WIDTH_REAL = 2.0
_LINE_WIDTH_FORECAST = 2.2


def plot_presentation(
    history_dates: Sequence,
    history_prices: Sequence[float],
    forecast_dates: Sequence,
    forecast_values: Sequence[float],
    ci_lower: Sequence[float],
    ci_upper: Sequence[float],
    ticker: str,
    output_dir: str | Path,
    history_window: int = 90,
    timestamp: str | None = None,
) -> Path:
    """Построить презентационный график и сохранить в PNG.

    Parameters
    ----------
    history_dates : Sequence
        Даты исторических наблюдений (pd.Timestamp/datetime/str — что угодно,
        что matplotlib умеет интерпретировать как дату).
    history_prices : Sequence[float]
        Фактические цены закрытия за историю.
    forecast_dates : Sequence
        Три даты прогноза (горизонты 1, 2, 3 торговых дня).
    forecast_values : Sequence[float]
        Три точечных прогноза, выровненных с forecast_dates.
    ci_lower, ci_upper : Sequence[float]
        Нижние и верхние границы 90%-го доверительного интервала
        для каждого из горизонтов.
    ticker : str
        Тикер инструмента, например 'SBER'.
    output_dir : str | Path
        Директория для сохранения PNG. Создаётся при необходимости.
    history_window : int
        Сколько последних точек истории отрисовать. По умолчанию 90.
    timestamp : str | None
        Метка времени для имени файла; если None — берётся текущая.

    Returns
    -------
    Path
        Полный путь к сохранённому PNG-файлу.
    """
    # ---- Подготовка данных ------------------------------------------------
    history_dates = pd.to_datetime(pd.Series(history_dates))
    history_prices = np.asarray(history_prices, dtype=float)
    forecast_dates = pd.to_datetime(pd.Series(forecast_dates))
    forecast_values = np.asarray(forecast_values, dtype=float)
    ci_lower = np.asarray(ci_lower, dtype=float)
    ci_upper = np.asarray(ci_upper, dtype=float)

    if history_window < len(history_dates):
        history_dates = history_dates.iloc[-history_window:].reset_index(drop=True)
        history_prices = history_prices[-history_window:]

    # ---- Построение фигуры ------------------------------------------------
    fig, ax = plt.subplots(figsize=_FIG_SIZE, dpi=_DPI)

    # Фактическая цена
    ax.plot(
        history_dates, history_prices,
        color=_COLOR_REAL,
        linewidth=_LINE_WIDTH_REAL,
        label='Реальная цена',
    )

    # Соединительная линия от последней реальной точки к прогнозу
    last_real_date = history_dates.iloc[-1]
    last_real_price = history_prices[-1]

    bridge_dates = [last_real_date, forecast_dates.iloc[0]]
    bridge_prices = [last_real_price, forecast_values[0]]
    ax.plot(
        bridge_dates, bridge_prices,
        color=_COLOR_FORECAST,
        linewidth=_LINE_WIDTH_FORECAST,
        linestyle='--',
        alpha=0.8,
    )

    # Прогноз 1-3 дня
    ax.plot(
        forecast_dates, forecast_values,
        color=_COLOR_FORECAST,
        linewidth=_LINE_WIDTH_FORECAST,
        marker='o',
        markersize=_MARKER_SIZE,
        markeredgecolor='white',
        markeredgewidth=1.2,
        label='Прогноз 1–3 дня',
    )

    # Заливка доверительного интервала
    ax.fill_between(
        forecast_dates, ci_lower, ci_upper,
        color=_COLOR_CI_FILL,
        alpha=_CI_ALPHA,
        linewidth=0,
        label='90% доверительный интервал',
    )

    # ---- Оформление осей --------------------------------------------------
    ax.set_title(
        f'Прогноз цены акций {ticker} на 1–3 торговых дня',
        fontsize=_TITLE_FONT_SIZE,
        fontweight='bold',
        pad=16,
    )
    ax.set_xlabel('Дата', fontsize=_LABEL_FONT_SIZE, labelpad=10)
    ax.set_ylabel('Цена закрытия, ₽', fontsize=_LABEL_FONT_SIZE, labelpad=10)

    # Формат дат на оси X — русские месяцы через явный словарь.
    # Системная локаль ru_RU есть не везде, поэтому форматируем сами.
    _RU_MONTHS = {
        1: 'янв', 2: 'фев', 3: 'мар', 4: 'апр', 5: 'май', 6: 'июн',
        7: 'июл', 8: 'авг', 9: 'сен', 10: 'окт', 11: 'ноя', 12: 'дек',
    }

    def _ru_date_fmt(x, pos=None):
        d = mdates.num2date(x)
        return f'{d.day:02d} {_RU_MONTHS[d.month]} {d.year}'

    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_ru_date_fmt))
    fig = ax.figure
    fig.autofmt_xdate(rotation=0, ha='center')

    ax.tick_params(axis='both', labelsize=_TICK_FONT_SIZE)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    # Убираем верхнюю и правую рамки — современный вид
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888')
    ax.spines['bottom'].set_color('#888')

    # Лёгкое расширение по Y, чтобы CI не упирался в границу
    y_min = min(history_prices.min(), ci_lower.min())
    y_max = max(history_prices.max(), ci_upper.max())
    pad = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.legend(
        loc='upper left',
        fontsize=_LEGEND_FONT_SIZE,
        framealpha=0.95,
        edgecolor='#cccccc',
    )

    fig.tight_layout()

    # ---- Сохранение -------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    out_path = output_dir / f'{ticker}_presentation_{timestamp}.png'
    fig.savefig(out_path, dpi=_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# Подсказки для интеграции в stock_modelv14.py.
# Эти объекты — не код, а готовые блоки для копипасты.
# ---------------------------------------------------------------------------

PATCH_USER_INPUTS = '''
# Добавить в get_user_inputs() после блока с ci_mode (вопрос 5):

print("\\n6. Презентационный режим (дополнительный график для слайдов)?")
print("   1 — Нет (по умолчанию)")
print("   2 — Да, построить упрощённый график для презентации")
choice = input("Ваш выбор [1/2, по умолчанию 1]: ").strip() or "1"
presentation_mode = (choice == "2")

if presentation_mode:
    raw = input("   Окно истории на графике, торговых дней [по умолчанию 90]: ").strip()
    history_window = int(raw) if raw.isdigit() else 90
else:
    history_window = 90

return {
    # ... существующие ключи ...
    "presentation_mode": presentation_mode,
    "history_window": history_window,
}
'''


PATCH_ARGPARSE = '''
# Добавить в блок parser.add_argument(...) рядом с --ci-mode:

parser.add_argument(
    "--presentation",
    action="store_true",
    help="Построить дополнительный график в презентационном стиле",
)
parser.add_argument(
    "--history-window",
    type=int,
    default=90,
    help="Окно истории для презентационного графика, торговых дней (по умолчанию 90)",
)
'''


PATCH_PIPELINE_CALL = '''
# Добавить после существующего matplotlib-вывода прогноза, до завершения main:

if user_inputs.get("presentation_mode"):
    from presentation_plot import plot_presentation

    # Подготовить данные для презентационной отрисовки.
    # Имена переменных подгони под актуальный код stock_modelv14.py:
    # data            — DataFrame с историей (содержит Date и Close)
    # forecast_dates  — список из 3 дат прогноза (h=1, 2, 3)
    # point_forecasts — список из 3 точечных прогнозов из meta-learner
    # ci_lower, ci_upper — границы 90% CI для каждого горизонта

    out_path = plot_presentation(
        history_dates=data["Date"],
        history_prices=data["Close"],
        forecast_dates=forecast_dates,
        forecast_values=point_forecasts,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ticker=TICKER,
        output_dir=output_dir,
        history_window=user_inputs["history_window"],
    )
    print(f"\\nПрезентационный график сохранён: {out_path}")
'''


if __name__ == "__main__":
    # Демонстрационный прогон со синтетическими данными.
    # Запуск: python presentation_plot.py
    rng = np.random.default_rng(42)
    n = 90
    dates = pd.date_range(end="2026-04-29", periods=n, freq="B")
    base = 300.0
    walk = np.cumsum(rng.normal(0, 1.2, size=n))
    prices = base + walk

    forecast_dates = pd.bdate_range(start=dates[-1] + pd.Timedelta(days=1), periods=3)
    forecast_values = [prices[-1] + 1.5, prices[-1] + 2.0, prices[-1] + 1.0]
    ci_low = [v - 4 for v in forecast_values]
    ci_high = [v + 4 for v in forecast_values]

    out = plot_presentation(
        history_dates=dates,
        history_prices=prices,
        forecast_dates=forecast_dates,
        forecast_values=forecast_values,
        ci_lower=ci_low,
        ci_upper=ci_high,
        ticker="SBER",
        output_dir="./demo_out",
    )
    print(f"Демо-график сохранён: {out}")