"""Постфактум-анализ multi-date backtest: что если инвертировать h=2 предсказания?

IC h=2 = -0.448 на multi-date backtest SBER (14.05.2026, 12 дат, ci_mode=garch)
говорит, что модель систематически анти-предсказывает направление на 2-дневном
горизонте. Если это реальный сигнал, то `pred_inv = 2·last_close - pred`
должен давать DA ~67% и IC ~+0.45.

Запуск:
    python experiments/h2_inversion_analysis.py outputs/stock_modelv16/multidate/SBER_multidate_20260514_004051.json

Скрипт читает JSON-отчёт и пересчитывает метрики для исходных и
инвертированных предсказаний. Не требует повторного прогона модели.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def analyze(json_path: Path) -> None:
    with open(json_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    per_date = report['per_date']
    print(f"Анализ файла: {json_path.name}")
    print(f"Ticker: {report['ticker']}, ci_mode: {report['ci_mode']}, n_dates: {report['n_dates']}")
    print("=" * 88)

    for h in [1, 2, 3]:
        rows = []
        for entry in per_date:
            for r in entry['horizons']:
                if r['horizon'] == h:
                    rows.append({'date':       entry['date'],
                                  'last_close': r['naive_price'],   # base_price на дату бэктеста
                                  'forecast':   r['forecast'],
                                  'real':       r['real']})
                    break

        if not rows:
            continue

        df = pd.DataFrame(rows)

        # Оригинал
        df['err_orig']        = (df['forecast'] - df['real']).abs()
        df['err_orig_pct']    = df['err_orig'] / df['real'] * 100
        df['dir_fc_orig']     = np.sign(df['forecast'] - df['last_close'])
        df['dir_real']        = np.sign(df['real']     - df['last_close'])
        df['dir_ok_orig']     = (df['dir_fc_orig'] == df['dir_real']) & (df['dir_real'] != 0)
        df['ret_model_orig']  = (df['forecast'] - df['last_close']) / df['last_close'] * 100
        df['ret_real']        = (df['real']     - df['last_close']) / df['last_close'] * 100

        # Инвертированные
        df['forecast_inv']   = 2 * df['last_close'] - df['forecast']
        df['err_inv']        = (df['forecast_inv'] - df['real']).abs()
        df['err_inv_pct']    = df['err_inv'] / df['real'] * 100
        df['dir_fc_inv']     = np.sign(df['forecast_inv'] - df['last_close'])
        df['dir_ok_inv']     = (df['dir_fc_inv'] == df['dir_real']) & (df['dir_real'] != 0)
        df['ret_model_inv']  = (df['forecast_inv'] - df['last_close']) / df['last_close'] * 100

        n = len(df)
        # Метрики оригинал
        mean_err_orig = df['err_orig_pct'].mean()
        std_err_orig  = df['err_orig_pct'].std()
        da_orig_k     = int(df['dir_ok_orig'].sum())
        ic_orig       = df['ret_model_orig'].corr(df['ret_real'], method='spearman')
        wilson_orig   = _wilson_ci(da_orig_k, n)

        # Метрики инверт
        mean_err_inv = df['err_inv_pct'].mean()
        std_err_inv  = df['err_inv_pct'].std()
        da_inv_k     = int(df['dir_ok_inv'].sum())
        ic_inv       = df['ret_model_inv'].corr(df['ret_real'], method='spearman')
        wilson_inv   = _wilson_ci(da_inv_k, n)

        print(f"\nHorizon +{h}d  (n={n}):")
        print(f"  {'':22s}  {'orig':>20s}    {'inverted':>20s}")
        print(f"  {'Mean |Error|%:':22s}  {mean_err_orig:>13.2f} ± {std_err_orig:.2f}    "
              f"{mean_err_inv:>13.2f} ± {std_err_inv:.2f}")
        print(f"  {'Direction Accuracy:':22s}  {da_orig_k}/{n} = {da_orig_k/n*100:>5.1f}%  "
              f"[{wilson_orig[0]*100:.1f}-{wilson_orig[1]*100:.1f}%]   "
              f"{da_inv_k}/{n} = {da_inv_k/n*100:>5.1f}%  "
              f"[{wilson_inv[0]*100:.1f}-{wilson_inv[1]*100:.1f}%]")
        print(f"  {'IC (Spearman):':22s}  {ic_orig:>+20.3f}    {ic_inv:>+20.3f}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python experiments/h2_inversion_analysis.py <multidate_json>")
        sys.exit(1)
    analyze(Path(sys.argv[1]))
