"""Извлекает Feature Importance из training-логов и агрегирует по 108 моделям.

Каждый multi-date backtest содержит:
- 12 дат × 3 горизонта × 3 walk-forward сплита = 108 XGBoost-моделей
- В каждом split'е лог печатает Top-10 Feature Importance

Скрипт парсит блоки 'Top 10 Feature Importances' и считает:
- Среднее значение importance по всем моделям, где фича попала в Top-10
- Частоту попадания в Top-10
- Топ-15 для финального графика

Запуск:
    python experiments/aggregate_feature_importance.py outputs/stock_modelv16/logs/training_detailed_*.log
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path


_TOP_HEADER_RE = re.compile(r"Top 10 Feature Importances")
_FEATURE_LINE_RE = re.compile(r"\[INFO\]\s+([A-Za-z0-9_]+):\s+([0-9.]+)")


def parse_log(log_path: Path) -> dict:
    """Парсит один лог-файл. Возвращает dict feature → list[float]."""
    importances: dict[str, list[float]] = defaultdict(list)
    in_block = False
    block_count = 0
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if _TOP_HEADER_RE.search(line):
                in_block = True
                block_count += 1
                continue
            if in_block:
                m = _FEATURE_LINE_RE.search(line)
                if m:
                    feat = m.group(1)
                    val = float(m.group(2))
                    importances[feat].append(val)
                elif 'Granger causality' in line or 'Generating level' in line:
                    in_block = False
    return importances, block_count


def aggregate(log_paths: list[Path]) -> None:
    total_imp: dict[str, list[float]] = defaultdict(list)
    total_blocks = 0
    for p in log_paths:
        imps, blocks = parse_log(p)
        for feat, vals in imps.items():
            total_imp[feat].extend(vals)
        total_blocks += blocks

    # Статистика
    print(f"\n=== Feature Importance Aggregation ===")
    print(f"Файлов обработано: {len(log_paths)}")
    print(f"Top-10 блоков найдено: {total_blocks}")
    print(f"Уникальных фич в Top-10: {len(total_imp)}\n")

    # Сводные метрики
    summary = []
    for feat, vals in total_imp.items():
        n = len(vals)
        mean_when_present = sum(vals) / n
        # frequency = доля моделей где попал в Top-10
        freq = n / total_blocks if total_blocks else 0.0
        # эффективный вклад: важность × частота
        effective = mean_when_present * freq
        summary.append((feat, mean_when_present, freq, effective, n))

    # Сортируем по effective (среднее × частота)
    summary.sort(key=lambda x: -x[3])

    print(f"{'Признак':<25} {'Mean (Top-10)':>15} {'Частота %':>12} {'Эффективная':>14} {'N':>5}")
    print("-" * 75)
    for feat, mean_val, freq, eff, n in summary[:20]:
        print(f"{feat:<25} {mean_val:>15.4f} {freq*100:>11.1f}% {eff:>14.4f} {n:>5}")


if __name__ == '__main__':
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("Usage: python experiments/aggregate_feature_importance.py <log1> [log2 ...]")
        sys.exit(1)
    aggregate(paths)
