# CI Mode Selection — Design Spec
Date: 2026-04-28

## Цель

Дать пользователю выбор режима доверительных интервалов:
- **wide** — 5/95 перцентили, полная история (по умолчанию, академический)
- **narrow** — 25/75 перцентили, последние 3 года (практический, актуальная волатильность)

## Интерактивный UX

Новый вопрос `5.` добавляется в `get_user_inputs()` после секции визуализации:

```
5. Режим доверительных интервалов:
   [1] Широкий  — 5/95 перцентили, полная история обучения
                  (академический: учитывает все кризисы, в т.ч. 2022)
   [2] Узкий    — 25/75 перцентили, последние 3 года
                  (практический: актуальная волатильность)
Режим CI (1/2, по умолчанию 1):
```

В итоговом summary добавляется строка:
```
  Режим CI: Широкий (5/95, вся история)
```

## CLI

Новый аргумент `argparse`:
```
--ci-mode wide|narrow   (default: wide)
```

CLI-блок в `if args.ticker:` дополняется ключом `ci_mode`.

## Изменения в коде

### 1. `get_user_inputs()`
- Добавить вопрос 5 после вопроса `show_plot`
- Вернуть `ci_mode: 'wide' | 'narrow'` в словаре параметров

### 2. `argparse`
- `parser.add_argument('--ci-mode', choices=['wide', 'narrow'], default='wide')`
- Передать в `user_params['ci_mode']`

### 3. `prepare_and_train_model(signature)`
- Новый параметр: `ci_mode: str = 'wide'`
- Все три вызова (main, run_backtest) передают `ci_mode`

### 4. Логика квантильных моделей внутри walk-forward сплита

```python
if ci_mode == 'narrow':
    NARROW_WINDOW = 756  # ~3 торговых года
    q_start = max(0, len(X_train) - NARROW_WINDOW)
    X_q, y_q = X_train[q_start:], y_train[q_start:]
    lower_alpha, upper_alpha = 0.25, 0.75
else:
    X_q, y_q = X_train, y_train
    lower_alpha, upper_alpha = 0.05, 0.95

lower_params  = {**xgb_params, 'objective': 'reg:quantileerror', 'quantile_alpha': lower_alpha}
upper_params  = {**xgb_params, 'objective': 'reg:quantileerror', 'quantile_alpha': upper_alpha}
median_params = {**xgb_params, 'objective': 'reg:quantileerror', 'quantile_alpha': 0.5}

lower_model.fit(X_q.reshape(len(X_q), -1), y_q)
upper_model.fit(X_q.reshape(len(X_q), -1), y_q)
median_model.fit(X_q.reshape(len(X_q), -1), y_q)
```

Основной ансамбль (LSTM, XGBoost, Meta-Learner) не изменяется.

Если данных меньше 756 в `narrow` режиме — берётся весь `X_train` без ошибки.

## Документация

- `CHANGELOG.md` — добавить запись v14.1
- `architecture.md` — добавить описание параметра `ci_mode` и поведения двух режимов
- `improvements.md` — отметить как улучшение (если было записано)

## Тесты

Новые тесты в `tests/test_multihorizon.py` (или отдельный файл `tests/test_ci_mode.py`):
- `test_wide_mode_uses_full_history` — `X_q` совпадает с `X_train` при `wide`
- `test_narrow_mode_uses_last_756` — `X_q` = последние 756 строк при `len(X_train) > 756`
- `test_narrow_mode_fallback_small_data` — если `len(X_train) < 756`, берётся всё
- `test_narrow_alpha_is_25_75` — `lower_alpha=0.25`, `upper_alpha=0.75` при `narrow`
- `test_wide_alpha_is_05_95` — `lower_alpha=0.05`, `upper_alpha=0.95` при `wide`
