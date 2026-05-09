# Архитектура проекта: Stock Price Forecasting Model v15

<!-- markdownlint-disable MD060 -->

## Обзор

Система прогнозирования рыночных цен акций российского рынка (MOEX). Реализует стэкинг-ансамбль LSTM + XGBoost с доверительными интервалами на основе квантильной регрессии. Поддерживает два режима: прогноз на будущее и бэктест на исторических данных. Начиная с v15 включает time-varying макропризнаки (USD/RUB, ставка ЦБ, Brent), Granger pre-screening и статистическую диагностику остатков. С v15.1 — фиксированный seed (`RANDOM_SEED=42`) и метрика Direction Accuracy. С v15.2 — признаки сезонности (`day_of_week`, `month`, `quarter`). С v15.3 — Ridge мета-леарнер с интерпретируемыми весами. С v15.4 — автосклейка YNDX+YDEX и Ljung-Box guard для коротких рядов. С v15.6 — Optuna-оптимизация alpha Ridge: 20-trial поиск на накопленных OOS-данных walk-forward. С v15.7 — IMOEX и RTSI как time-varying признаки рыночного контекста. С v15.8 — LSTM Ensemble по seeds (N=3): три модели с разными инициализациями, усреднение до Ridge. С v15.9 — дивидендные time-varying признаки через MOEX ISS (`fundamentals_loader.py`): `div_days_to_next`, `div_next_amount`, `div_days_since_last`; работает для любого тикера MOEX; look-ahead bias исключён окном 45 дней.

---

## Запуск проекта

### 1. Активация виртуального окружения (PowerShell)

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ;
(& "s:\ISUCT\for diplom Dev\forMasterCourse++\venv312\Scripts\Activate.ps1")
```

### 2. Запуск скрипта

```bash
# Интерактивный режим (рекомендуется)
python .\stock_modelv15.py

# CLI-режим (headless, все параметры через аргументы)
python .\stock_modelv15.py --ticker SBER --no-gui
python .\stock_modelv15.py --ticker LKOH --backtest 2025-03-01
python .\stock_modelv15.py --ticker GAZP --optimize --trials 30 --no-gui
```

### Аргументы CLI

| Аргумент       | Тип            | Описание                                                                                       |
| -------------- | -------------- | ---------------------------------------------------------------------------------------------- |
| `--ticker`           | str           | Тикер акции (SBER, LKOH, GAZP, …)                                                       |
| `--backtest`         | str           | Дата для бэктеста в формате YYYY-MM-DD                                                   |
| `--optimize`         | flag          | Запустить Optuna для поиска гиперпараметров                                              |
| `--trials`           | int           | Количество итераций Optuna (по умолчанию 20)                                             |
| `--no-gui`           | flag          | Режим без графического интерфейса (headless)                                             |
| `--ci-mode`          | wide / narrow | Режим CI: wide=5/95 вся история, narrow=25/75 последние 3 года (default: wide)          |
| `--benchmark`        | flag          | Воспроизводимый бэктест: SBER, 2024-10-14, дефолтные гиперпараметры, wide CI, headless  |
| `--presentation`     | flag          | Сохранить презентационный график (16:9, PNG 1920×1080)                                   |
| `--history-window`   | int           | Количество дней истории для презентационного графика (по умолчанию 90)                  |

---

## Что вводит пользователь

При запуске без CLI-аргументов скрипт запрашивает параметры интерактивно:

| Шаг | Вопрос | Значения |
| --- | --- | --- |
| 1 | Тикер акции | Например: `SBER`, `LKOH`, `GAZP`; по умолчанию `SBER` |
| 2 | Режим работы | `1` — прогноз на будущее; `2` — бэктест |
| 2a | Дата бэктеста (если режим 2) | YYYY-MM-DD; по умолчанию 3 дня назад |
| 3 | Гиперпараметры | Использовать сохранённые / запустить Optuna / использовать дефолтные |
| 3a | Количество итераций Optuna | Целое число; по умолчанию 20 |
| 4 | Доверительные интервалы | `y` — показывать; `n` — нет |
| 4b | Показать график | `y` — открыть окно; `n` — только сохранить |
| 5 | Режим CI | `1` — широкий 5/95 вся история; `2` — узкий 25/75 последние 3 года |
| Подтверждение | Продолжить с выбранными параметрами | `y` / `n` |

---

## Переменные окружения (.env)

### API-токены

| Переменная                       | Назначение                                               |
| -------------------------------- | -------------------------------------------------------- |
| `TINVEST_TOKEN` / `INVEST_TOKEN` | Токен T-Bank Invest API (фундаментальные данные)         |
| `TINVEST_SANDBOX_TOKEN`          | Sandbox-токен T-Bank (используется так же, как основной) |
| `MOEXALGO_TOKEN`                 | JWT-токен для библиотеки `moexalgo`                      |
| `FINANCEMARKER_TOKEN`            | Не используется в v14 (legacy)                           |

### Гиперпараметры LSTM

| Переменная           | По умолчанию | Описание                                |
| -------------------- | ------------ | --------------------------------------- |
| `LSTM_LOOK_BACK`     | `30`         | Длина входной последовательности (дней) |
| `LSTM_EPOCHS`        | `10`         | Максимальное число эпох обучения        |
| `LSTM_PATIENCE`      | `3`          | Терпение для EarlyStopping              |
| `LSTM_BATCH_SIZE`    | `32`         | Размер батча                            |
| `LSTM_LEARNING_RATE` | `0.001`      | Скорость обучения Adam                  |
| `LSTM_DROPOUT_RATE`  | `0.2`        | Доля дропаута                           |
| `LSTM_UNITS`         | `64`         | Количество нейронов в первом LSTM-слое  |

### Гиперпараметры XGBoost

| Переменная                 | По умолчанию | Описание                                |
| -------------------------- | ------------ | --------------------------------------- |
| `XGBOOST_N_ESTIMATORS`     | `100`        | Количество деревьев                     |
| `XGBOOST_MAX_DEPTH`        | `6`          | Максимальная глубина дерева             |
| `XGBOOST_LEARNING_RATE`    | `0.1`        | Скорость обучения                       |
| `XGBOOST_SUBSAMPLE`        | `0.8`        | Доля выборки строк на дерево            |
| `XGBOOST_COLSAMPLE_BYTREE` | `0.8`        | Доля признаков на дерево                |
| `XGBOOST_RANDOM_STATE`     | `42`         | Зерно случайности                       |
| `XGBOOST_VERBOSITY`        | `0`          | Уровень логирования XGBoost (0 = тихий) |
| `XGBOOST_DEVICE`           | `'cpu'`      | Устройство: `'cpu'` или `'cuda'` (GPU)  |

---

## Архитектура компонентов

```text
┌─────────────────────────────────────────────────────────────────┐
│                          ПОЛЬЗОВАТЕЛЬ                           │
│                  (интерактивный ввод / CLI)                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ ticker, mode, dates, optuna, show_ci
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ЗАГРУЗКА ДАННЫХ                           │
│  ┌───────────────────────────┐  ┌──────────────────────────┐    │
│  │  moexalgo (библиотека)    │  │  MOEX ISS REST API       │    │
│  │  (первичный источник)     │  │  (fallback при ошибке)   │    │
│  └──────────────┬────────────┘  └────────────┬─────────────┘    │
│                 └─────────────┬──────────────┘                  │
│                               │ OHLCV с 2014-01-01              │
│  При ticker=YDEX: автосклейка YNDX (до 2024-06-14)              │
│  + YDEX (с 2024-07-24), ratio≈1.019, итого 3053 строки          │
│  ┌────────────────────────────▼──────────────────────────────┐  │
│  │              TinkoffFundamentalLoader                     │  │
│  │  T-Bank Invest REST API → P/E, P/B, ROE, Beta,            │  │
│  │  DivYield, MarketCap                                      │  │
│  └────────────────────────────┬──────────────────────────────┘  │
└───────────────────────────────┼─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ФОРМИРОВАНИЕ ПРИЗНАКОВ                        │
│                                                                 │
│  OHLCV (5) + Технические индикаторы (17) + Returns-фичи (5)     │
│  + Фундаментал (6) + Макро (0–3, динамически) + Сезонность (3)  │
│                                                                 │
│  Технические индикаторы:                                        │
│  SMA(10), EMA(20), RSI(14), MACD+Signal+Histogram,              │
│  BB(Middle/Upper/Lower/Width), ATR(14),                         │
│  Stochastic K/D, ADX(14), Momentum(10),                         │
│  PriceChange(1d, 5d)                                            │
│                                                                 │
│  Returns-фичи (v13.6):                                          │
│  Vol_Return_5/10/20 (скользящая волатильность доходностей)      │
│  Return_MA_5/10 (momentum на доходностях)                       │
│                                                                 │
│  Фундаментальные (из Tinkoff API):                              │
│  market_cap, roe, dividend_yield, pe_ratio, pb_ratio, beta      │
│                                                                 │
│  Макропризнаки time-varying (v15, macro_loader.py):             │
│  usd_rub_hist (ЦБ РФ XML API), cbr_rate (SOAP DailyInfo.asmx)  │
│  brent_price (MOEX ISS BRN фьючерс)                             │
│  imoex, rtsi (MOEX ISS candles, engines/stock/markets/index)    │
│                                                                 │
│  Дивидендные time-varying (v15.9, fundamentals_loader.py):      │
│  div_days_to_next  — обратный отсчёт до реестра (sentinel 999)  │
│  div_next_amount   — объявленный дивиденд ₽ (0 если нет)        │
│  div_days_since_last — дней с последней отсечки (sentinel 999)  │
│  look-ahead bias исключён: окно объявления ≤ 45 дней            │
│  Работает для любого тикера MOEX (SBER, GAZP, LKOH, …)         │
│                                                                 │
│  Granger pre-screening: все macro + div с p ≥ 0.05 исключаются │
│                                                                 │
│  Сезонность (v15.2):                                            │
│  day_of_week (0=пн…4=пт), month (1–12), quarter (1–4)          │
│  вычисляются в update_technical_indicators из Date              │
│                                                                 │
│          Итого: 36–44 признаков (динамически)                   │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              НОРМАЛИЗАЦИЯ (MinMaxScaler)                        │
│  Общий скейлер для всех 36–44 признаков + отдельный для Close   │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│           ФОРМИРОВАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ                      │
│  Скользящее окно LSTM_LOOK_BACK (30 дней)                       │
│  X.shape = (N, 30, 36–39),  y.shape = (N,)                      │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              WALK-FORWARD ВАЛИДАЦИЯ (3 сплита)                  │
│                                                                 │
│  Сплит 1: train=[0..80%]  test=[80%..85%]                       │
│  Сплит 2: train=[0..85%]  test=[85%..90%]                       │
│  Сплит 3: train=[0..90%]  test=[90%..100%]                      │
│                                                                 │
│  Финальные модели берутся из последнего (3-го) сплита           │
│                                                                 │
│  [После цикла] Тест Льюнга–Бокса (20 лагов) на OOS-остатках:   │
│  диагностика автокорреляции → stdout + logger                   │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     АНСАМБЛЕВАЯ МОДЕЛЬ                          │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Уровень 0 — базовые модели                              │   │
│  │                                                          │   │
│  │  ┌─────────────────────┐  ┌─────────────────────────┐    │   │
│  │  │      LSTM (3 слоя)  │  │     XGBoost Regressor   │    │   │
│  │  │                     │  │  (вход: X сплющен       │    │   │
│  │  │  Input(30, 36–39)   │  │  в 30×(36–39) призн.   │    │   │
│  │  │  LSTM(units)        │  │                         │    │   │
│  │  │  Dropout            │  └─────────────────────────┘    │   │
│  │  │  LSTM(units//2)     │           │                     │   │
│  │  │  Dropout×0.5        │           │                     │   │
│  │  │  LSTM(units//2)     │           │                     │   │
│  │  │  Dropout            │           │                     │   │
│  │  │  Dense(32, relu)    │           │                     │   │
│  │  │  Dense(1)           │           │                     │   │
│  │  └──────────┬──────────┘           │                     │   │
│  │             └────────────┬─────────┘                     │   │
│  └──────────────────────────┼───────────────────────────────┘   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Уровень 0.5 — LSTM Ensemble (v15.8)                       │   │
│  │  LSTM×3 seeds [42,7,123] → среднее lstm_avg_pred          │   │
│  └──────────────────────────┼───────────────────────────────┘   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Уровень 1 — Meta-Learner (Ridge, v15.3+v15.6)           │   │
│  │  Вход: [lstm_avg_pred, xgb_pred] (2 признака)            │   │
│  │  alpha: Optuna 20-trial log-uniform [1e-3,100] на OOS    │   │
│  │  Ridge(alpha=best) → точечный прогноз Close              │   │
│  │  coef_[0]=LSTM weight, coef_[1]=XGB weight (в лог)       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  CI-модели (после цикла, на OOS-остатках)                 │   │
│  │  Вход: [lstm_pred, xgb_pred], цель: actual − meta_pred   │   │
│  │  lower_model → q-й перцентиль остатка (α = 0.05 / 0.25)  │   │
│  │  upper_model → q-й перцентиль остатка (α = 0.95 / 0.75)  │   │
│  │  CI = pred + residual_q  (симметрично, без ценового bias) │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ (Optuna)             │
│  (опционально, запускается до обучения финальной модели)        │
│                                                                 │
│  LSTM: units ∈ [32,128], dropout ∈ [0.1,0.5], lr ∈ [1e-4,1e-2] │
│  XGB:  n_estimators, max_depth, lr, subsample, colsample_bytree │
│                                                                 │
│  Результаты сохраняются в:                                      │
│  outputs/{version}/hyperparams/{TICKER}_hyperparams.json        │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ПРОГНОЗИРОВАНИЕ                             │
│                                                                 │
│  Горизонты: 1, 2, 3 дня вперёд                                  │
│  Метод: Direct Multi-Step / MIMO — для каждого горизонта h      │
│  обучается независимый ансамбль, цель сдвинута на h дней:       │
│  y[i] = Close[i + h]                                            │
│  Итого: 3 горизонта × 18 моделей = 54 модели на один тикер      │
│                                                                 │
│  Доверительные интервалы: две квантильные XGBoost-модели,       │
│  обученные на OOS-остатках (actual − meta_pred) тестовых окон   │
│  walk-forward; CI = pred ± residual_quantile; перцентили и      │
│  объём обучающего окна зависят от ci_mode                       │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│            РЕЖИМ ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ (ci_mode)             │
│                                                                 │
│  --ci-mode wide|narrow (default: wide)                          │
│                                                                 │
│  [Wide] (широкий) — 5/95 перцентили на полной истории           │
│    Учитывает всю волатильность (включая кризисы)                │
│    Преимущества: полный диапазон, не обезглавливает историю     │
│    Недостатки: может быть неинформативно при редких событиях    │
│                                                                 │
│  [Narrow] (узкий) — 25/75 перцентили на последних 3 годах       │
│    Фокусируется на актуальной волатильности (756 торговых дней) │
│    Преимущества: практичен, отражает текущий режим рынка        │
│    Недостатки: игнорирует редкие исторические события           │
│                                                                 │
│  Выбор режима: вопрос 5 в интерактивном режиме или флаг CLI     │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   РЕЖИМЫ РАБОТЫ                                 │
│                                                                 │
│  [Прогноз]                    [Бэктест]                         │
│  Обучение на всех данных      Обучение до backtest_date         │
│  → прогноз на 1-3 дня         → прогноз на 1-3 дня              │
│  от текущей даты              → сравнение с реальными ценами    │
│                               → расчёт accuracy, CI coverage    │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ВЫХОДНЫЕ ДАННЫЕ                         │
│                                                                 │
│  outputs/{MODEL_VERSION}/                                       │
│  ├── logs/                                                      │
│  │   ├── training_detailed_{timestamp}.log  (основной лог;      │
│  │   │     включает FORECAST SUMMARY: per-horizon прогноз,      │
│  │   │     RMSE/MAE/R², CI [lower–upper], ширина, флаг ✓/✗)    │
│  │   └── {TICKER}_forecast_v14_{timestamp}.txt  (результаты;    │
│  │         per-horizon метрики для всех 3 горизонтов)           │
│  ├── graphs/                                                    │
│  │   └── {TICKER}_price_forecast_v14_{timestamp}.jpg            │
│  ├── hyperparams/                                               │
│  │   └── {TICKER}_hyperparams.json                              │
│  └── backtest/                                                  │
│      └── {TICKER}_backtest_{date}_{timestamp}.txt               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Стек технологий

| Библиотека | Версия | Роль |
| --- | --- | --- |
| `tensorflow` / `keras` | 2.x+ | LSTM-нейросеть (CPU на Windows; GPU не поддерживается в TF 2.11+) |
| `xgboost` | 3.x+ | Градиентный бустинг + квантильная регрессия; GPU через `device='cuda'` |
| `optuna` | 3.x | Байесовская оптимизация гиперпараметров |
| `scikit-learn` | 1.x | `MinMaxScaler`, метрики (RMSE, MAE, R²) |
| `statsmodels` | 0.14+ | Тест Льюнга–Бокса (`acorr_ljungbox`), тест Грэнджера (`grangercausalitytests`) |
| `pandas` | 2.x | Работа с табличными данными |
| `numpy` | 1.x | Математические операции, массивы |
| `matplotlib` | 3.x | Визуализация прогнозов |
| `moexalgo` | — | Загрузка свечей с MOEX (основной источник) |
| `requests` | — | HTTP-запросы к MOEX ISS, Tinkoff API, ЦБ РФ XML/SOAP |
| `python-dotenv` | — | Загрузка `.env` конфигурации |

---

## Поток данных (Data Flow)

```text
.env → load_dotenv() → константы конфигурации

Пользователь → get_user_inputs() или argparse
    → ticker, mode, dates, optimize, n_trials, show_ci, show_plot

ticker + dates → load_stock_data_moex_test()
    → DataFrame [Date, Open, High, Low, Close, Volume]

DataFrame → update_technical_indicators()
    → +17 колонок технических индикаторов
    → +3 колонки сезонности (day_of_week, month, quarter)

ticker → TinkoffFundamentalLoader.get_fundamentals()
    → +6 колонок фундаментальных данных (market_cap, roe, dividend_yield, pe_ratio, pb_ratio, beta)

load_macro_data(start_date, end_date) → DataFrame[Date, usd_rub_hist, cbr_rate, brent_price, imoex, rtsi]
    → merge по Date → +0–5 макропризнаков (ffill/bfill для выходных)

load_dividend_features(ticker, start_date, end_date)   ← fundamentals_loader.py
    → DataFrame[Date, div_days_to_next, div_next_amount, div_days_since_last]
    → merge по Date → +0–3 дивидендных признаков (sentinel-значения 999/0)
    → Granger pre-screening (macro + div): признаки с p ≥ 0.05 исключаются

DataFrame (36–44 признаков) → MinMaxScaler → нормализованный массив
    → скользящее окно LOOK_BACK=30 → X (N, 30, 36–44), y (N,)

X, y → Optuna (опционально) → best_lstm_params, best_xgb_params
    → сохранение в {TICKER}_hyperparams.json

для каждого горизонта h ∈ {1, 2, 3}:
    y_h[i] = Close[i + h]  (сдвинутая цель)
    X_h, y_h + hyperparams → walk-forward (3 сплита):
        каждый сплит: LSTM → XGBoost → meta_train=[lstm_pred, xgb_pred]
        → Meta-Learner (точечный прогноз)
        → метрики RMSE, MAE, R²; OOS-остатки (actual − pred) накапливаются
    [После walk-forward] acorr_ljungbox(lags=min(20,len//2)) → stdout + logger
    (пропускается если lags < 2 — защита для тикеров с короткой историей)
    Финальные модели (из 3-го сплита) → прямой прогноз на день h
    [После feature importances] grangercausalitytests топ-15 → stdout + logger

[Прогноз] → FORECAST SUMMARY в logger (per-horizon RMSE/MAE/R², CI ✓/✗)
         → график (.jpg) + отчёт (.txt) с per-horizon метриками
[Бэктест] → сравнение с реальными данными → accuracy report (.txt)
           включает: Error%, CI Coverage, Direction Accuracy
```

---

## Метрики качества

| Метрика | Описание |
| --- | --- |
| RMSE | Root Mean Squared Error (среднеквадратичная ошибка) |
| MAE | Mean Absolute Error (средняя абсолютная ошибка) |
| R² | Коэффициент детерминации |
| Error % | Ошибка прогноза в % от реальной цены (в режиме бэктеста) |
| CI Coverage | Доля случаев, когда реальная цена попала в доверительный интервал |
| Direction Accuracy | Доля горизонтов, где модель верно предсказала направление движения (вверх/вниз от последней известной цены) |

---

## Тесты

Тесты расположены в директории `tests/` и запускаются через `pytest`:

```bash
python -m pytest tests/ -v
```

### `tests/test_multihorizon.py`

Проверяет корректность ключевых функций Direct Multi-Step архитектуры.

**Тесты подготовки последовательностей** (`_build_sequences`):

| Тест | Что проверяет |
| --- | --- |
| `test_horizon1_target_is_next_day` | При `horizon=1` цель `y[i] = Close[i+1]`; длина выборки = N−look_back−1 |
| `test_horizon2_target_is_two_days_ahead` | При `horizon=2` цель `y[i] = Close[i+2]`; длина = N−look_back−2 |
| `test_horizon3_target_is_three_days_ahead` | При `horizon=3` цель `y[i] = Close[i+3]`; длина = N−look_back−3 |

**Тесты `merge_horizon_results`** — функция, объединяющая результаты трёх горизонтов:

| Тест | Что проверяет |
| --- | --- |
| `test_merge_forecasts_from_three_calls` | Словари `forecasts[h]` и `confidence_intervals[h]` собираются корректно |
| `test_merge_partial_none_returns_none` | Если один горизонт вернул `None`, функция возвращает `None` |
| `test_merge_data_none_returns_none` | Если поле `data` в результате равно `None`, функция возвращает `None` |
| `test_merge_all_ok_returns_dicts` | При всех успешных горизонтах возвращает `(forecasts, ci)` с ключами 1, 2, 3 |

**Тесты `_forecast_dates_for_horizon`**:

| Тест | Что проверяет |
| --- | --- |
| `test_forecast_dates_horizon1_returns_one_date` | Для `horizon=1` возвращает ровно 1 дату |
| `test_forecast_dates_horizon3_returns_three_dates` | Для `horizon=3` возвращает ровно 3 даты |

### `tests/test_macro_loader.py`

Проверяет корректность загрузки макроэкономических данных (`macro_loader.py`).

| Тест | Что проверяет |
| --- | --- |
| `test_load_macro_data_columns` | DataFrame содержит колонки `Date`, `usd_rub_hist`, `cbr_rate`, `brent_price` |
| `test_fallback_on_failed_source` | При падении загрузчика колонка `NaN`, `warning` залогирован, остальные колонки целы |
| `test_date_range_coverage` | Нет дат вне диапазона; нет NaN после ffill/bfill при частичных пропусках |

### `tests/test_ci_mode.py`

Проверяет функцию `_get_ci_params(ci_mode, X_train, y_train)`, которая выбирает
обучающее окно и перцентили для квантильных моделей CI.

| Тест | Что проверяет |
| --- | --- |
| `test_wide_uses_full_history` | `wide`: возвращает всю историю, `lower_alpha=0.05`, `upper_alpha=0.95` |
| `test_narrow_uses_last_756` | `narrow`: срез последних 756 строк, `lower_alpha=0.25`, `upper_alpha=0.75` |
| `test_narrow_fallback_small_data` | `narrow` при N < 756: возвращает всё, что есть (без усечения) |
| `test_narrow_exact_756_boundary` | `narrow` при N = 756: граничный случай, срез = весь массив |

### Запуск отдельного файла

```bash
python -m pytest tests/test_ci_mode.py -v
python -m pytest tests/test_multihorizon.py -v
python -m pytest tests/test_macro_loader.py -v
```
