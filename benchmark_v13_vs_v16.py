"""
Benchmark aggregator: stock_modelv13 (recurrent) vs stock_modelv16 (direct MIMO).

Читает JSON-метрики из outputs/stock_modelv13/benchmark/{ticker}_metrics.json и
outputs/stock_modelv16/benchmark/{ticker}_metrics.json для каждого тикера,
строит сравнительные таблицы для статьи.

Usage:
    python benchmark_v13_vs_v16.py
"""

import json
import os
from pathlib import Path

TICKERS = ['SBER', 'GAZP', 'LKOH']
V13_DIR = Path('outputs/stock_modelv13/benchmark')
V16_DIR = Path('outputs/stock_modelv16/benchmark')


def load_metrics(model_dir: Path, ticker: str):
    path = model_dir / f'{ticker}_metrics.json'
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def fmt_v13(m, h):
    """v13: для h=1 — walk-forward (3 folds avg); для h=2/h=3 — rollout per-split averaged.
    Apples-to-apples с v16 walk-forward (тоже per-split averaged)."""
    if h == 1:
        wf = m['walk_forward_h1']
        return wf['rmse'], wf['mae'], wf['r2']
    else:
        per_split = m.get('rollout_per_split', [])
        rmses, maes, r2s = [], [], []
        for split_m in per_split:
            sh = split_m.get(str(h)) or split_m.get(h)
            if sh:
                rmses.append(sh['rmse'])
                maes.append(sh['mae'])
                r2s.append(sh['r2'])
        if not rmses:
            return None, None, None
        import statistics as _st
        return _st.mean(rmses), _st.mean(maes), _st.mean(r2s)


def fmt_v16(m, h):
    """v16: per-horizon walk-forward."""
    ph = m['per_horizon_walk_forward'].get(str(h)) or m['per_horizon_walk_forward'].get(h)
    if ph is None:
        return None, None, None
    return ph['rmse'], ph['mae'], ph['r2']


def print_table():
    print()
    print('=' * 116)
    print(f"{'Ticker':<6} {'H':<3} {'v13 RMSE':<10} {'v13 MAE':<10} {'v13 R²':<10} {'v16 RMSE':<10} {'v16 MAE':<10} {'v16 R²':<10} {'Δ RMSE%':<10} {'Δ R²':<10}")
    print('=' * 116)

    for ticker in TICKERS:
        m13 = load_metrics(V13_DIR, ticker)
        m16 = load_metrics(V16_DIR, ticker)
        for h in (1, 2, 3):
            r13, ma13, r2_13 = fmt_v13(m13, h) if m13 else (None, None, None)
            r16, ma16, r2_16 = fmt_v16(m16, h) if m16 else (None, None, None)

            def fmt(x, p=4): return f"{x:.{p}f}" if x is not None else "—"

            d_rmse = (r13 - r16) / r13 * 100 if (r13 and r16) else None
            d_r2 = (r2_16 - r2_13) if (r2_13 is not None and r2_16 is not None) else None

            print(f"{ticker:<6} h={h:<2} {fmt(r13):<10} {fmt(ma13):<10} {fmt(r2_13):<10} "
                  f"{fmt(r16):<10} {fmt(ma16):<10} {fmt(r2_16):<10} "
                  f"{fmt(d_rmse,1) if d_rmse is not None else '—':<10} "
                  f"{fmt(d_r2,3) if d_r2 is not None else '—':<10}")
    print('=' * 116)


def print_test_dates():
    print()
    print('--- Test period per ticker ---')
    for ticker in TICKERS:
        m13 = load_metrics(V13_DIR, ticker)
        m16 = load_metrics(V16_DIR, ticker)
        if m13:
            dates13 = m13.get('last_split_test_dates') or m13.get('test_dates')
            print(f"  {ticker} v13: {dates13[0]} → {dates13[1]} (data_size={m13.get('data_size')})")
        if m16:
            dates16 = m16.get('test_dates')
            print(f"  {ticker} v16: {dates16[0]} → {dates16[1]} (data_size={m16.get('data_size')})")


def print_markdown_table():
    """Готовая markdown-таблица для статьи."""
    print()
    print('--- Markdown table для статьи ---')
    print()
    print('| Ticker | H | v13 RMSE | v13 MAE | v13 R² | v16 RMSE | v16 MAE | v16 R² | Δ RMSE% | Δ R² |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')

    for ticker in TICKERS:
        m13 = load_metrics(V13_DIR, ticker)
        m16 = load_metrics(V16_DIR, ticker)
        for h in (1, 2, 3):
            r13, ma13, r2_13 = fmt_v13(m13, h) if m13 else (None, None, None)
            r16, ma16, r2_16 = fmt_v16(m16, h) if m16 else (None, None, None)

            def f(x, p=4): return f"{x:.{p}f}" if x is not None else "—"

            d_rmse = (r13 - r16) / r13 * 100 if (r13 and r16) else None
            d_r2 = (r2_16 - r2_13) if (r2_13 is not None and r2_16 is not None) else None

            print(f"| {ticker} | {h} | {f(r13)} | {f(ma13)} | {f(r2_13)} | "
                  f"{f(r16)} | {f(ma16)} | {f(r2_16)} | "
                  f"{f(d_rmse,1) if d_rmse is not None else '—'} | "
                  f"{f(d_r2,3) if d_r2 is not None else '—'} |")


if __name__ == '__main__':
    print_table()
    print_test_dates()
    print_markdown_table()
