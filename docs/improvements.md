# Анализ проекта: плюсы, минусы, рекомендации

<!-- markdownlint-disable MD029 MD036 -->

## Плюсы текущего решения

- **Стэкинг-ансамбль LSTM + XGBoost + Ridge мета-леарнер** — LSTM улавливает временные зависимости; XGBoost устойчив к выбросам; Ridge(alpha=1.0) линейно комбинирует их с интерпретируемыми весами (coef_ в логе).
- **Квантильная регрессия для CI** — две квантильные XGBoost-модели на OOS-остатках; два режима: wide (5/95) и narrow (25/75).
- **Walk-forward валидация** — 3 сплита с расширяющимся окном, предотвращает data leakage.
- **MIMO (Direct Multi-Step Forecasting)** — отдельная модель на каждый горизонт, нет накопления ошибки авторегрессии.
- **Bayesian hyperparameter search (Optuna)** — прогресс-бар в консоли; персистентность результатов в JSON.
- **Time-varying макропризнаки** — USD/RUB (ЦБ РФ XML), ставка ЦБ (SOAP), Brent (MOEX ISS BRN), IMOEX и RTSI (MOEX ISS candles); Granger pre-screening исключает незначимые. Для SBER: RTSI retained (p=0.020), IMOEX excluded.
- **Дивидендные time-varying признаки** — `fundamentals_loader.py`; `load_dividend_features(ticker, start, end)` работает для любого тикера MOEX; три признака: `div_days_to_next` (обратный отсчёт 45→0, sentinel 999), `div_next_amount` (сумма ₽), `div_days_since_last`; look-ahead bias исключён окном `_ANNOUNCE_WINDOW = 45` дней; Granger-скрининг применяется ко всем трём признакам.
- **LSTM Ensemble по seeds (N=3)** — `ENSEMBLE_SEEDS = [42, 7, 123]`; усреднение предсказаний до Ridge; снижает дисперсию avg Error% между прогонами; Ridge по-прежнему получает `[lstm_avg, xgb_pred]`.
- **Раздельный feature set: LSTM vs XGB (v15.11)** — XGB видит полный набор (~39 фич: OHLC + технические + макро + дивы + сезонные + фундаменталии), LSTM получает только сырое OHLCV + Price_Change_1 + Vol_Return_10 (7 фич). Дублирование фич с XGB убивало OOS-полезность LSTM (Ridge обнулял вес в 14/15 случаев); раздельный feature set вернул LSTM комплементарный вклад в 6/15 случаев (SBER h=1/h=3, LKOH все 3 горизонта, GAZP h=1). На SBER R² подрос на h=1/h=2/h=3 (+1.8/+0.6/+1.0pp).
- **Ridge мета-леарнер с positive=True (v15.10)** — запрет отрицательных весов в `optimize_ridge_alpha` и финальной модели; убирает артефакт инверсии LSTM при слабой регуляризации (мультиколлинеарность LSTM/XGB давала alpha≈1e-3 + LSTM=-0.13). Теперь Ridge либо честно использует LSTM (вес >0), либо обнуляет его (вес=0).
- **Признаки сезонности** — day_of_week, month, quarter; автоматически нормализуются MinMaxScaler.
- **Автосклейка YNDX+YDEX** — для реструктурированных тикеров история собирается из двух листингов без ручного вмешательства.
- **Статистическая диагностика** — Ljung-Box на OOS-остатках, Granger causality на топ-15 признаках.
- **Direction Accuracy** — метрика % верно угаданных направлений; дополняет RMSE/MAE в бэктесте.
- **Воспроизводимость** — RANDOM_SEED=42 для Python/NumPy/TensorFlow; benchmark-режим фиксирует тикер, дату и гиперпараметры.
- **Кросс-тикерные бэктесты** — 5 тикеров (SBER, LKOH, GAZP, TCSG, YDEX), 2 даты (2024-10-14 и 2024-03-15).
- **Два источника данных** — `moexalgo` основной, MOEX ISS REST API как fallback.
- **Два режима запуска** — интерактивный и CLI; все артефакты сохраняются (логи, графики, гиперпараметры, бэктесты).

---

## Открытые задачи

### Производительность

- **Медленная валидация** — walk-forward обучает 54 модели (18 на горизонт × 3); с Optuna прогон занимает значительное время на CPU.
- **Переоптимизировать LSTM hyperparams под новый feature set (Optuna)** — с v15.11 LSTM получает подмножество фич (`_LSTM_FEATURES` = OHLCV + return + волатильность, 7 шт), но текущие сохранённые hyperparams в `outputs/.../hyperparams/*.json` оптимизированы под старый набор из 39 фич. Перезапустить Optuna (`--optimize --trials 30`) для каждого тикера, пересохранить параметры — потенциальное улучшение R² и/или LSTM_coef.
- **OHLCV-загрузка на уровне multi-backtest** — сейчас `--multi-backtest N` повторно вызывает `load_stock_data_moex_test()` для каждой из N дат, хотя ряд одного тикера один и тот же; macro/dividend/fund уже шарятся (см. `architecture.md` → in-memory sharing). По аналогии: загрузить максимальный диапазон OHLCV один раз в `run_multidate_backtest`, передавать срез до `backtest_date` в `prepare_and_train_model`. Экономия ~5 сек × N дат × 3 горизонта на тикер. Disk cache не нужен — только in-memory sharing, как для macro.

### Код

- **Дата начала данных захардкожена** — `start_date = '2014-01-01'` не вынесена в `.env` или CLI.
- **Логирование сырого ответа API** — `logger.info(f"RAW API RESPONSE: {resp.text[:500]}...")` пишет первые 500 символов; ответ может содержать чувствительные данные.
- **Фундаментальные признаки — только snapshot** — `market_cap`, `roe`, `pe_ratio` берутся как текущие значения и продублированы на всю историю; модель не видит их динамику. (Дивиденды решены через MOEX ISS в v15.9; P/E/ROE в динамике требуют платного источника.)
- ~~**ARCH-эффект в остатках**~~ — **РЕШЕНО (v16)**: Ljung-Box на `|res|` подтверждал ARCH-эффект; добавлен `ci_mode='garch'` — GARCH(1,1) на OOS-остатках даёт σ_t, CI = pred ± 1.645·σ_t (90%). Адаптивная ширина: уже в спокойные периоды, шире в волатильные. На SBER 13.05.2026 GARCH-σ для h=1 = 4.84 RUB vs эмпирическая std = 5.73 RUB — GARCH видит, что *сейчас* рынок спокойнее средней истории. Старые `wide`/`narrow` режимы остаются как альтернатива.
- **Отрицательный R² на трендовых рынках** — при нисходящем тренде (GAZP 2024) R² падает до −1.4: OOS-окна имеют иной тренд, чем обучающие. Точечные прогнозы при этом адекватны (~2%), CI покрывает 100%.
- **Проверить adjusted vs unadjusted close в `moexalgo`** — потенциальный data quality bug. `Ticker(...).candles(period='1D')` использует MOEX ISS endpoint, который по умолчанию отдаёт **unadjusted** свечи (без коррекции на дивиденды и сплиты). Если это так, дивидендный гэп в данных выглядит как реальное падение цены, и модель может научиться "предсказывать падение перед реестром" — но это **технический артефакт**, не торговый сигнал. Проверка: для SBER на дату закрытия реестра сравнить `Close[ex_date] − Close[ex_date−1]` с `−div_amount`. Если разница ≈ 0, значит unadjusted (баг). Если ≈ `−div_amount`, значит уже adjusted. Решение в случае бага: либо использовать adjusted candles (если `moexalgo` поддерживает), либо вручную корректировать через `div_amount` из `fundamentals_loader`. Затронуты все тикеры с дивидендами (SBER, LKOH, GAZP, MGNT, MTSS).
- ~~**TCSG данные обрываются 2024-11-27**~~ — **РЕШЕНО (v15.13)**: TCS Group → «Т-Технологии» (тикер MOEX: **T**), MOEX уже отдаёт под тикером `T` склеенную историю (1697 строк, 2019-10-28 → сегодня), отдельная склейка не нужна (в отличие от YDEX, где YNDX и YDEX раздельны). В коде: `--ticker TCSG` или `--ticker T` оба возвращают объединённую серию + дамми `is_post_restructure` по дате 2024-11-28. Эффект: R² h=1: 0.014 → **0.659** (+64.5pp), h=2: -0.31 → **0.509** (+82pp), h=3: -1.00 → **0.373** (+137pp). Среднее R²: -0.43 → +0.514 (+95pp). Модель TCSG/T теперь работает на уровне SBER/LKOH/GAZP.
- ~~**YDEX: структурный разрыв в июле 2024**~~ — **РЕШЕНО (v15.12)**: добавлен дамми-признак `is_post_restructure` (0 для YNDX-истории, 1 для YDEX) в `_load_single_ticker`. XGB видит дамми как "переключатель режима". Эффект: R² h=1: 0.493 → 0.604 (+11.1pp), h=3: 0.320 → 0.396 (+7.6pp), h=2: 0.403 → 0.362 (-4.1pp, шум). В среднем +4.9pp R². LSTM остался 0.000 — структурный шок это разовое событие, его может использовать только XGB.

### Что можно убрать

| Что | Где | Причина |
| --- | --- | --- |
| `FINANCEMARKER_TOKEN` | `.env` | Источник данных не подключён |
| Логирование `RAW API RESPONSE` | `stock_modelv16.py` (унаследовано из v15) | Шум + потенциальная утечка данных |

### Что можно добавить

**В работе (запланировано)**

**Ближайшая разработка — Streamlit-дашборд для предзащиты/защиты**

Минимальный интерактивный демо-инструмент на 1–2 вечера для показа комиссии. Цель — иметь интерактив на случай "запустите на другом тикере" вместо лезть в консоль.

Стек:

- `streamlit` (один файл `streamlit_app.py`)
- Переиспользует существующие функции `load_stock_data_moex_test`, `run_backtest`, `load_macro_data` и т.д. из `stock_modelv16.py`
- Опционально: `@st.cache_data` для кэширования загрузки данных между переключениями тикеров
- Pre-cache JSON-результатов `outputs/stock_modelv16/multidate/*.json` — UI читает уже посчитанное, не запускает walk-forward в живую

Структура:

- **Сайдбар:** выпадающий список тикеров (SBER/LKOH/GAZP/MGNT/MTSS), date picker для бэктеста, ползунок `horizon`, тоггл `ci_mode` (wide/narrow/garch), кнопка "Запустить"
- **Главная панель:**
  - график matplotlib через `st.pyplot` (история + прогноз + полоса CI)
  - таблица с per-horizon: forecast / Δ% / direction / CI / confidence
  - плашка "beats Naive" (зелёная/красная) + сравнение Error % vs Naive Error %
  - метрики (если выбрана историческая дата): Direction Accuracy, IC, CI Coverage
- **Multi-date вкладка:** табличка 12 дат × 3 горизонта с агрегированными Wilson 95% CI

Запуск: `streamlit run streamlit_app.py`. Никакого деплоя — локально на ноуте, защита в LAN.

Риски: если запустить полный walk-forward через UI — 5–10 минут ожидания. Решение: дефолт — читать pre-cached JSON, кнопка "Live run" отдельная с предупреждением.

Время: 6–8 часов.

**Долгосрочная разработка (после защиты) — FastAPI + React/Vue SPA**

Полноценный SaaS-вид как продолжение проекта после защиты диплома. Цель — упаковать исследование в продуктовую оболочку и использовать как portfolio piece.

Архитектура:

```text
backend/                         frontend/
  app.py (FastAPI + uvicorn)      src/
  ├── /api/tickers (GET)          ├── pages/
  ├── /api/forecast (POST)        │   ├── Dashboard.tsx
  │     {ticker, date, horizon,   │   ├── MultiDate.tsx
  │      ci_mode}                 │   └── Compare.tsx
  ├── /api/multidate/{ticker}     ├── components/
  │     (читает pre-cached JSON)  │   ├── TickerSelect, ForecastCard
  ├── /api/history/{ticker}       │   ├── CIChart (recharts/plotly.js)
  │     (OHLCV для графика)       │   └── MetricsTable
  ├── models/ (pydantic схемы)    ├── api/ (TanStack Query клиент,
  └── services/ (обёртки над      │     типизация через openapi-typescript)
        prepare_and_train_model   └── styles/ (Tailwind + shadcn/ui)
        run_backtest)
```

Стек:

- **Backend:** Python FastAPI (см. ниже про "почему не Go"), pydantic для схем, uvicorn, переиспользование существующих функций без переписывания
- **Frontend:** Vite + React + TypeScript + TailwindCSS + shadcn/ui + recharts/plotly.js + TanStack Query + Zod
- **Кэширование:** `@functools.lru_cache` на загрузку данных, Redis опционально для prod
- **Async:** FastAPI native async для параллельных загрузок macro/dividend/fund (уже распараллелено в `ThreadPoolExecutor`, можно перевести на `asyncio.gather`)

Ключевые UI-фичи (того, что Streamlit не умеет):

- Анимированные CI-полосы (расширяющийся "конус неопределённости" через d3/recharts)
- Tooltip с детализацией метрик при наведении на точку прогноза
- Side-by-side сравнение тикеров с синхронизированным zoom
- Multi-date heatmap (5 тикеров × 12 дат, цветом — DA или IC)
- Тёмная тема + современный UI (Tailwind + shadcn)
- WebSocket прогресс-бар для Optuna в реальном времени
- Loader/skeleton при загрузке

Деплой (опционально):

- Backend — Railway/Render/Fly.io (env vars для секретов)
- Frontend — Vercel/Netlify (статика)
- Domain — собственный, через Cloudflare
- API key middleware для своего сервиса (rate limiting + auth)

Защитимая формулировка для комиссии: "Прикладной интерфейс реализован как отдельный SPA с FastAPI-бэкендом, чтобы продемонстрировать возможность интеграции исследовательского кода в продуктовую среду. Это **не часть научного вклада**, а демонстрационная оболочка."

Время: 30–40 часов (~1.5–2 недели вечеров для уверенного фронтендера).

**Docker для SaaS-деплоя**

До защиты Docker **не нужен** — добавит сложности (GPU pass-through через WSL2 + nvidia-docker под Windows) без реального выигрыша при локальном запуске. После защиты, при деплое — нужен для backend.

Два пути деплоя:

| Подход | Когда выбрать | Минусы |
| --- | --- | --- |
| **Pre-compute + read JSON** (рекомендуемый) | Если важна простота и дешевизна (~$0 при free tier Railway/Vercel) | Нет live "запустить на новой дате" — UI читает только pre-computed результаты |
| **VPS с GPU + полный Docker** | Если хочется реальных live-прогонов через UI | ~$0.5/час GPU аренда (Selectel/Lambda Labs/RunPod), сложнее настройка |

**Основной путь — pre-compute локально + лёгкий контейнер для бэка:**

1. Локально на RTX 5070 Ti прогоняешь walk-forward для всех тикеров и интересующих дат, сохраняешь `outputs/stock_modelv16/multidate/*.json` в репозиторий или S3-bucket
2. Backend читает эти JSON, не запускает ML
3. Контейнер не содержит `torch`, `xgboost`, `optuna`, `arch` — только `fastapi`, `pandas`, `pydantic`
4. Размер образа ~150MB (вместо ~5GB с PyTorch+CUDA), деплой занимает 30 секунд, попадает в free tier Railway/Render

Минимальный Dockerfile:

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY backend/ ./backend/
COPY outputs/stock_modelv16/multidate/ ./data/multidate/
COPY outputs/stock_modelv16/hyperparams/ ./data/hyperparams/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

`requirements-prod.txt` — подмножество без ML-зависимостей:

```text
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.9.0
pandas==2.2.0
numpy==1.26.0
python-dotenv==1.0.0
```

Время: 2–3 часа на Dockerfile + настройку CI на GitHub Actions для auto-build + deploy в Railway.

**Опциональный путь 2 — VPS с GPU (когда захочется live-прогнозов):**

Если после защиты захочется иметь полноценный live-pipeline без pre-compute:

- VPS с GPU: Selectel (~₽15k/мес за RTX 3060), Lambda Labs (~$0.5/час on-demand), RunPod (~$0.3/час A4000)
- Базовый образ `nvidia/cuda:13.0-runtime-ubuntu22.04`
- Полный `requirements.txt` с PyTorch+cu130, XGBoost, Optuna, arch
- Размер образа ~5GB, build ~10 минут
- Имеет смысл только если будет реальный пользовательский трафик с запросами на новые даты — для портфолио избыточно

Frontend в обоих случаях деплоится без Docker — Vercel/Netlify билдят React/Vue из source автоматически.

**Почему backend на Python, а не Go**

Соблазнительно сделать backend на Go для портфолио (быстрее, single binary, goroutines), но для этого проекта **Python FastAPI — правильный выбор**:

1. **Вся ML-логика на Python.** PyTorch, XGBoost, scikit-learn, pandas, Optuna, statsmodels, arch — всё Python-only. Go придётся либо вызывать Python через subprocess (медленно, уродливо), либо строить gRPC-мост к Python-сервису (двойная сложность), либо ограничиться чтением pre-cached JSON (тогда Go = file server, незачем).

2. **Bottleneck — это ML compute, не API latency.** Walk-forward на 3000 точек × 54 модели занимает минуты, а не миллисекунды. Скорость Go (~10× быстрее Python на routing) тут не имеет значения — пока Go-handler ждёт ответа от Python-воркера, ты теряешь весь выигрыш.

3. **FastAPI достаточно быстр.** На современном Python с `uvicorn[standard]` (uvloop + httptools) FastAPI выдаёт ~30–50k req/sec на простых endpoint'ах. Для дипломного демо/портфолио этого хватает с запасом.

4. **Pydantic ≈ Go structs.** Если нравится type safety Go — Pydantic v2 даёт почти то же с runtime-валидацией. OpenAPI генерится автоматически.

**Когда Go реально оправдан для расширения:**

- **Тонкий proxy-слой** перед Python API (auth, rate limiting, caching) — реальный production-паттерн. Но это overkill для дипломного демо.
- Если хочешь Go-experience в резюме — лучше сделать **отдельный pet-project** (не привязанный к ML), чем городить мост.

**API-ключи и секреты в SaaS-режиме**

Текущее состояние (локально): `.env` файл с `TINVEST_TOKEN`, `MOEXALGO_TOKEN`, `INVEST_TOKEN` загружается через `python-dotenv`, в `.gitignore`. **Это правильный подход для локального dev.**

При деплое SaaS:

| Что | Где хранится | Кто видит |
| --- | --- | --- |
| Ключи MOEX/T-Bank/ЦБ | Backend env vars (Railway/Render secret manager) | Только бэкенд-процесс |
| URL backend API | Frontend env var (Vite `VITE_API_URL`) | Видно в браузере — но это публичный URL |
| API-ключ твоего сервиса (если защищаешь от чужого использования) | Backend env vars + middleware проверки | Только backend |
| Сессии пользователей | JWT в httpOnly cookie | Браузер хранит, JS не читает |

**Жёсткие правила (security 101):**

1. **Никогда не клади ключи в код фронтенда.** Всё, что попадает в JS bundle, видно в браузере через DevTools. Это включает `.env.local` в Next.js (если префикс `NEXT_PUBLIC_*` или `VITE_*` — оно идёт в bundle).
2. **Frontend никогда не вызывает MOEX/T-Bank напрямую.** Только через свой backend. Backend — единственный держатель ключей. Frontend → `POST /api/forecast` → backend → MOEX API → backend → JSON для frontend.
3. **CORS на backend** — разреши только свой frontend-домен (`https://your-app.vercel.app`), не `*`.
4. **Rate limiting** — `slowapi` или `fastapi-limiter`, чтобы кто-то не залил тебя запросами и не выжег твои API-квоты MOEX/T-Bank.

**Для дипломной защиты (демо локально):**

Никаких изменений не нужно — текущий `.env` подход безопасен, потому что:

- Запускается только на твоём ноуте
- Нет публичного URL
- Никто не может вытащить ключи из браузера, потому что они не покидают backend

**Для полноценного SaaS-деплоя (после защиты):**

| Платформа | Где задавать env vars |
| --- | --- |
| Railway | Project → Variables → добавляешь `TINVEST_TOKEN=xxx`, шифруется at rest |
| Render | Service → Environment → Add environment variable |
| Fly.io | `fly secrets set TINVEST_TOKEN=xxx` через CLI |
| Vercel (frontend) | Project Settings → Environment Variables (только публичные `VITE_*`) |

**Дополнительно для prod:**

- Ротация ключей раз в 3–6 месяцев
- Логи — без печати raw API responses (уже отмечено как баг в "Код" → "Логирование сырого ответа API")
- HTTPS обязателен (даёт автоматически Railway/Render/Vercel)
- Sentry/PostHog для error tracking (бесплатные тиры хватает)

Время на security baseline: 2–3 часа на настройку CORS + rate limiting + env vars в провайдере.

**Этап 3 (после SaaS) — T-Bank Sandbox paper-trading**

Развернуть модель как **paper-trading стратегию** через T-Bank Invest API в режиме sandbox. Цель — валидация в условиях, приближенных к реальной торговле: задержка между сигналом и ордером, slippage, исполнение по рынку, реальная ликвидность.

Архитектура (грубо):

```text
Cron 9:50 МСК → Signal generator (модель h=1) → Position sizer
  → Sandbox executor (T-Bank InvestAPI) → PnL tracker (SQLite + Streamlit)
  → Закрытие позиций на 18:30 (для h=1)
```

Стек:

- `tinkoff-invest-python` SDK для работы с InvestAPI sandbox
- SQLite для журнала сделок и daily PnL
- Streamlit-дашборд (отдельный от ML-дашборда) для мониторинга performance
- APScheduler или просто Windows Task Scheduler для cron-запуска
- Переиспользует уже существующие функции `prepare_and_train_model`, `_get_forecast_for_horizon` из `stock_modelv16.py`

Ключевые метрики мониторинга:

- **PnL vs Naive** (Buy&Hold SBER) — главная: если проигрываем Naive, модель не имеет alpha (как и предсказывает `Q_AND_A.md`)
- **Sharpe / Sortino ratio** на 60+ сделок (для статистической значимости)
- **Max drawdown** — реальный риск, не виден в multi-date backtest
- **Live Direction Accuracy vs historical** — должны сходиться, если нет — overfit на backtest dates
- **Hit rate по типам сигналов** — фильтрация confidence HIGH/MED/LOW

Защитимая академическая формулировка для будущей публикации/портфолио: *"Модель валидирована не только историческим бэктестом, но и развёрнута как paper-trading стратегия через T-Bank Invest API в режиме sandbox. Это **demonstration of integration**, не доказательство alpha — для production-применения требуется учёт реальных комиссий, налогов, ликвидности и значительно больший период тестирования (минимум 1 год)."*

Время: 26–40 часов active dev + 1–3 месяца пассивного paper trading.

**Открытые вопросы по стратегии (решать ближе к делу):**

Эти решения определят характер стратегии. На текущем этапе зафиксированы как варианты, конкретные значения выбирать при старте разработки этапа 3.

| Вопрос | Варианты | Что учесть при выборе |
| --- | --- | --- |
| **Какой сигнал использовать?** | (a) только Direction h=1; (b) комбинация Direction h=1 && Direction h=2; (c) Δ% > threshold (например, > 0.3%); (d) фильтр по confidence (только HIGH/MED) | На multi-ticker validation MTSS показал анти-сигнал на h=1 — возможно стоит исключать тикеры с известным анти-сигналом или брать только h=3 для них |
| **На каких тикерах торговать?** | (a) только SBER (стабильный, лучший baseline); (b) корзина 5 голубых фишек (SBER/LKOH/GAZP/MGNT/MTSS); (c) динамический выбор по силе сигнала | MGNT showed Mean Err 3.76% — стоит исключить; MTSS — анти-сигнал; для start лучше SBER |
| **Long-only или Long/Short?** | (a) Long-only (проще, безопаснее); (b) Long/Short (требует sandbox-поддержки коротких позиций) | На бычьем рынке 2025-2026 Long-only достаточно; шорты добавить позже |
| **Position sizing?** | (a) Fixed fraction (10% портфеля на сделку); (b) Kelly criterion (агрессивно); (c) Risk parity (volatility-targeted через CI width); (d) Equal weight | Kelly даёт max growth, но высокий drawdown; для начала — fixed 10%, max 3 одновременных позиций |
| **Когда закрывать?** | (a) End-of-day для h=1 (просто); (b) Hold 2 дня для h=2; (c) При достижении CI границы; (d) Trailing stop | Самый простой — EOD для h=1; матчит модель напрямую |
| **Stop-loss?** | (a) По нижней границе CI (адаптивный); (b) Fixed −2% от входа; (c) Без stop-loss | Адаптивный лучше: учитывает текущую волатильность через GARCH |
| **Take-profit?** | (a) По верхней границе CI; (b) Fixed +1% trailing; (c) Без take-profit (держать до EOD) | Без TP проще — модель сама даёт точку выхода через предсказание |
| **Max drawdown gate?** | (a) −10% портфеля → пауза 1 неделя; (b) −5% → reduce position size; (c) Без gate | Обязательно нужен gate — защищает от затяжного "плохого периода" |
| **Min Δ% для входа?** | (a) > 0.3% (фильтрует слабые сигналы); (b) > 1% (только сильные); (c) Без фильтра | 0.3% разумно: меньше = шум, больше = редкие сделки |
| **Cooldown после серии убытков?** | (a) После 3 убыточных подряд — пауза 1 день; (b) Без cooldown | Психологический gate, но в paper trading менее критичен |

Технические вопросы, которые надо проверить при старте:

1. **Real-time или delayed котировки в sandbox?** — критично для timing открытия позиций
2. **Симуляция ликвидности** — sandbox может исполнять любой объём, что нереалистично для крупных позиций
3. **Учёт комиссий и спреда** — sandbox может игнорировать (надо явно вычитать из PnL вручную)
4. **Rate limits InvestAPI** — ~10 req/sec в free tier, при многотикерной стратегии нужно батчевать запросы
5. **Sandbox vs prod токены** — переключение через одну переменную в `.env`, остальной код идентичен

**Средний приоритет**

- **Configurable start date** — `DATA_START_DATE` в `.env` и `--start-date` в CLI
- **Walk-forward с более мелким шагом** — сейчас 3 сплита (80/85/90%); 5–10 сплитов дают стабильнее OOS-оценку и лучше откалиброванные CI
- **Межтикерные признаки** — спред Urals/Brent для нефтянки (LKOH), цена газа для GAZP; IMOEX как общерыночный фон для всех тикеров
- **Исторические фундаменталии P/E/ROE** — динамика вместо snapshot; дивиденды уже реализованы (v15.9); P/E, ROE требуют платного источника (financemarker.ru)

**На основе multi-date backtest SBER 14.05.2026 (12 дат, 2025-05 — 2026-04)**

Multi-date выявил три аномалии: (1) systematic negative bias растёт с горизонтом
(−0.27% / −0.57% / −0.96%), (2) beats Naive только 33–50% на всех горизонтах,
(3) IC h=2 = −0.448 — модель **анти-предсказывает направление** на 2-дневном
горизонте. Корневая гипотеза: mean-reversion bias из-за SMA/EMA/BB/ZScore
индикаторов на бычьем рынке 2025–2026 → модель «ожидает откат», а тренд
продолжается.

- **Bias correction post-hoc** — после walk-forward вычислять медианный signed
  bias OOS-остатков и вычитать его из финального прогноза. Гарантированно
  убирает systematic bias по уровню, но не лечит направление. ~15 строк кода,
  применять только если momentum-фичи (см. ниже в архиве работ v16) не решили
  проблему. Риск: overfit под недавний рыночный режим.
- **Multi-ticker pooled training** — объединить SBER+LKOH+GAZP+TCSG+YDEX в одну
  обучающую выборку с one-hot тикера. Гипотеза: больше разнообразия режимов
  (нефть/банки/IT) → меньше переподгонки под SBER-специфику и mean-reversion.
  Трудоёмкость ~3–4 часа: data loader, перестройка фич, тесты на shared scaler.
  Хороший пункт для главы «Дальнейшие исследования».
- **Naive baseline как третий base-learner** — добавить Naive (`last_close`) как
  третий вход Ridge meta-learner. Тогда "beats Naive" становится математической
  гарантией на тренировке (Ridge сам присвоит Naive нужный вес). Низкая
  трудоёмкость (~1 час), но рискованно концептуально: meta-learner может стать
  тривиальным копированием Naive, скрывая реальное качество LSTM/XGB.

**Низкий приоритет (академический интерес)**

- **Эмпирически проверить расширение `start_date` за пределы 2014-01-01** — для тикеров с длинной историей (SBER с 2007, LKOH/GAZP с 1990-х) запустить multi-date backtest с `start_date='2007-01-01'` и сравнить с baseline 2014. Ожидание: метрики просядут из-за **distribution shift** (рынок до 2014 ≠ после санкций), на ранний период не покрыты `cbr_rate` (с 2013), `brent_price` через MOEX BRN (с 2016), `imoex` в текущей методологии (с 2017) → NaN/sentinel размывают сигнал. Хороший материал для главы "Обоснование выбора стартовой даты" в дипломе: 2014 — это не лень, а **осознанная граница санкционного режима**, разбавление снижает релевантность недавних данных. Альтернатива — rolling-window training (учиться только на последних 3–5 годах) — требует переработки walk-forward, упомянуто как нерешённое для PLZL.
- **ARIMA-поправка остатков** — закрыло бы Ljung-Box если бы корень был в средней (lag 1–3). Диагностика v15.10 показала — корень в гетероскедастичности (lag 18–20 + |res|), поэтому ARIMA на residuals здесь **не поможет**; правильный путь — GARCH-CI (см. "Средний приоритет"). Оставлено как академический вариант для случая, если diagnostic-сообщение когда-нибудь покажет "momentum в среднем"
- **Attention-механизм** — заменить средний LSTM-слой на Bahdanau attention; модель сама выбирает, какие из 30 прошлых дней важнее; интерпретируемая карта весов — сильный слайд на защите
- **Новостной сентимент** — индекс тональности новостей по тикеру (Интерфакс, РБК) через NLP; рынок реагирует на текст раньше цены; трудоёмко, требует парсинга
- **Настраиваемый горизонт прогноза** — через `.env` или CLI (сейчас жёстко `[1, 2, 3]`)
- **Temporal Fusion Transformer (TFT)** — специально спроектирован для временных рядов с covariates; на практике бьёт LSTM на финансовых данных; высокая трудоёмкость
- **Intraday-derived фичи для дневной модели** — использовать минутные/часовые свечи MOEX (через `moexalgo` с `period='1m'/'1h'`) не для смены таймфрейма прогноза, а для построения **дополнительных дневных признаков**: `daily_realized_volatility` (Σ минутных доходностей², лучшая прокси волатильности, чем Close-to-Close), `intraday_momentum` (доходность последнего часа торгов), `overnight_return` (gap Close[t-1] → Open[t]), `open_to_close_skew` (форма дневной свечи). Это **усиление существующей модели**, не смена парадигмы — таймфрейм прогноза остаётся 1–3 дня, но фичи богаче. Ожидаемый эффект: +5–10% к R² on returns на ликвидных тикерах (SBER, GAZP). Цена: переписать `load_stock_data_moex_test` для двухуровневой загрузки (минутки → агрегация в дневные фичи), история минуток короче (типично 1–2 года), объём данных ×~480. Переход **на** минутный/часовой прогноз = другая задача (market microstructure forecasting), конкуренция с HFT-фирмами на латентности микросекунд — не имеет смысла для академической работы на Python.

### Рефакторинг (кандидаты на вынос из stock_modelv16.py)

| Файл | Функции | Причина |
| --- | --- | --- |
| `optuna_tuning.py` | `optimize_lstm_params`, `optimize_xgboost_params` | Полностью изолированы от пайплайна; Optuna-код меняется независимо |
| `data_loader.py` | `load_stock_data_moex_test` | Единственное место для замены источника данных (MOEX → другой) |

Не трогать пока:

- `run_backtest` — плотно завязан на локальные переменные и logger
- `prepare_and_train_model` — громоздкая сигнатура при выносе, мало выгоды

### Открытые вопросы

1. ~~**GUI / параметры Optuna**~~ — **РЕШЕНО (v16)**: если для тикера нет сохранённых параметров, по умолчанию запрос Optuna теперь отвечает `y`.
2. **Способ демонстрации** — нативный Windows-интерфейс или небольшой веб-сайт для защиты.
3. ~~**Визуальный разрыв на графике**~~ — **РЕШЕНО (v16)**: переход с непрерывной временно́й оси matplotlib на целочисленный индекс торговых дней — выходные/праздники не занимают позицию, прогнозные точки вплотную примыкают к последней реальной цене. Также добавлена адаптивная плотность date-ticks (геометрический шаг).
4. **Multi-ticker multi-date backtest** — сейчас `--multi-backtest` работает на один тикер; для защиты полезен прогон на 5 тикерах × 12 дат с агрегацией по тикерам.

---

## Архив (реализовано)

### v16 — Multi-date backtest, GARCH-CI, relative features, R²-scoring Ridge

- **✅ v16 — Multi-date backtest** — флаг `--multi-backtest N` запускает прогон на N исторических датах (первый торговый день каждого из последних N полных месяцев); агрегирует Direction Accuracy с Wilson 95% CI, IC (Spearman), CI coverage, % beats Naive. Macro/dividend данные загружаются один раз на весь диапазон. Результаты в `outputs/stock_modelv16/multidate/`. Закрывает главную академическую слабость одиночного бэктеста (3 точки → монетка).
- **✅ v16 — GARCH(1,1) для адаптивных CI** — `ci_mode='garch'` через пакет `arch`. GARCH-σ обучается на OOS-остатках в рублях; CI = pred ± 1.645·σ_t (90%). Решает ARCH-эффект, который игнорировали CI постоянной ширины. Каждый горизонт (h=1, 2, 3) обучает свой GARCH на h-специфичных остатках.
- **✅ v16 — Relative features (4 фичи)** — `Close_to_SMA10`, `Close_to_EMA20`, `BB_Position`, `Close_ZScore_20` — масштаб-инвариантные индикаторы режима цены. Вектор XGB: 39 → 43.
- **✅ v16 — R²-scoring для Ridge alpha** — `scoring='r2'` вместо `neg_mean_squared_error` в Optuna. Оживляет LSTM в стэкинге: на SBER ненулевой вес на 3/5 датах вместо 0/5; Avg Model Error 1.30% → 1.27%.
- **✅ v16 — PyTorch cu130** — соответствие системному CUDA Toolkit 13.x на RTX 5070 Ti.
- **✅ v16 — Целочисленная ось X на графиках** — устраняет визуальный разрыв из-за выходных/праздников между концом исторической линии и прогнозом.
- **✅ v16 — Адаптивная плотность date-ticks** — геометрически растущий шаг от текущей даты в прошлое (множитель 1.35 для основного графика, 1.45 для презентационного): последняя пара недель почти ежедневно, дальше еженедельно, ещё дальше — раз в месяц/два.
- **✅ v16 — Русские подписи основного графика** — легенда и оси переведены (`Реальная цена`, `Предсказано (тест)`, `Прогноз (1–3 дня)`, `ДИ (1–3 дня)`, `Дата`, `Цена закрытия, ₽`).
- **✅ v16 — Полное название тикера** — новая функция `get_ticker_shortname()` в `fundamentals_loader.py` запрашивает поле `SHORTNAME` из MOEX ISS; выводится в FORECAST SUMMARY и в подписи графика (`Прогноз цены: SBER — СБЕР`).
- **✅ v16 — Расширенный вывод FORECAST SUMMARY** — для каждого горизонта добавлены: Direction (▲/▼), Δ% (от `last_close`), Confidence (HIGH/MED/LOW по walk-forward R²), Naive baseline, сравнение «модель лучше/хуже Naive».
- **✅ v16 — Стандартизация диалога пользователя** — все 7 вопросов приведены к единому формату; Optuna по умолчанию = `y` (если нет сохранённых параметров); порядок вопроса 4 (show_plot первым).
- **✅ v16 — Per-horizon look_back + per-horizon Optuna** — `LSTM_LOOK_BACK_PER_HORIZON = {1: 30, 2: 30, 3: 60}`. Эксперимент v16-experiments (5 итераций) показал: LB=60 даёт +25pp DA и IC +0.41 на h=3 SBER, но регрессирует h=1/h=2 (включая HP-bound анти-сигналы). Финальный гибрид: baseline LB=30 для h=1/h=2, LB=60 + per-horizon Optuna HP только для h=3. `save_hyperparams`/`load_hyperparams` поддерживают `horizon` параметр, с fallback на общий файл — full backward compat для других тикеров. Helper `_resolve_horizon_hp` инкапсулирует трёхуровневый fallback. Schema versioning + атомарная запись JSON.
- **✅ v16 — Дивидендные фичи исключены из Granger pre-screening** — `_FUND_DIV_COLS` всегда в feature set. Granger стабильно отбрасывал их с p > 0.5 (sentinel 999 ~88% времени + нелинейность дивидендных patterns). XGB-shape вырос с (n, 30, 43) до (n, 30, 46). Реальный эффект на прогноз включается в 45-дневном окне перед реестром. Pre-div run-up за 60+ дней — known limitation (психологический эффект известности, не покрывается моделью из-за защиты от look-ahead bias).
- **✅ v16 — 36/36 тестов pass** — `tests/test_hyperparams.py` (14 новых тестов: paths, save/load roundtrip, schema versioning, fallback цепочка, `_resolve_horizon_hp`, атомарность). Починены 2 предсуществующих падающих теста (`test_benchmark::test_run_benchmark_structure` — устаревшие fake-данные без полей v15.1; `test_macro_loader::test_date_range_coverage` — мокирование кеша `.macro_cache.csv`).

### v15.9 — Дивидендные time-varying признаки

- **✅ v15.9 — Дивидендные признаки** — новый модуль `fundamentals_loader.py`; `load_dividend_features(ticker, start, end)` через MOEX ISS работает для любого тикера MOEX; три признака: `div_days_to_next`, `div_next_amount`, `div_days_since_last`; look-ahead bias исключён `_ANNOUNCE_WINDOW = 45` дней; Granger-скрининг расширен на `_MACRO_COLS + _FUND_DIV_COLS`; вектор признаков: 36–41 → 36–44 динамически

### v15.8 — LSTM Ensemble по seeds (N=3)

- **✅ v15.8 — Ensemble по seeds** — `ENSEMBLE_SEEDS = [42, 7, 123]`; три LSTM с разными инициализациями на каждый walk-forward сплит; предсказания усредняются до Ridge (Вариант A); Ridge по-прежнему `[lstm_avg, xgb_pred]`; время прогона ×3 по блоку LSTM; снижает дисперсию ошибки между запусками

### v15.7 — IMOEX и RTSI как признаки

- **✅ v15.7 — Рыночные индексы IMOEX/RTSI** — `_load_moex_index(index_id, start, end)` в `macro_loader.py`; годовые запросы к MOEX ISS candles (engines/stock/markets/index/boards/SNDX); `imoex` и `rtsi` добавлены в `_MACRO_COLS`; Granger pre-screening автоматически исключает незначимые; вектор признаков: 36–39 → 36–41 динамически

### v15.6 — Optuna оптимизация Ridge alpha

- **✅ v15.6 — Optuna для Ridge alpha** — функция `optimize_ridge_alpha(meta_features, y_true, n_trials=20)`; log-uniform поиск alpha ∈ [1e-3, 100] за 20 итераций Optuna; оценка через 5-fold CV на OOS-предсказаниях walk-forward; финальный Ridge дообучается с best_alpha на данных последнего сплита; на 2 входных признаках best_alpha стремится к нулю (≈ 0.001–0.02), Ridge вырождается в OLS

### v15.5 — оптимизация загрузки, UX и кросс-тикерные бэктесты

- **✅ v15.5 — Фундаментальные данные загружаются один раз** — `fund_data=None` в `prepare_and_train_model`; `shared_funds` загружается до цикла горизонтов в `run_backtest` и `__main__`; Optuna-блок переиспользует `shared_funds`; было 3–4 запроса к T-Bank API, стал 1
- **✅ v15.5 — GPU memory growth** — `tf.config.experimental.set_memory_growth(gpu, True)` для каждого GPU при инициализации; TensorFlow больше не захватывает всю видеопамять
- **✅ v15.5 — Прогресс-бар Optuna** — `show_progress_bar=True` в обоих `study.optimize`; прогресс итераций виден в консоли
- **✅ v15.5 — Кросс-тикерные бэктесты на двух датах** — 5 тикеров (SBER, LKOH, GAZP, TCSG, YDEX) × 2 даты (2024-10-14 и 2024-03-15); результаты в `benchmarks.md`; исключает впечатление cherry-picking на защите

### v15.4 — склейка YNDX+YDEX и Ljung-Box guard

- **✅ v15.4 — Склейка YNDX+YDEX** — при `ticker=YDEX` автоматически загружается и конкатенируется история `YNDX` (до 2024-06-14) + `YDEX` (с 2024-07-24); ratio на стыке 1.019; итого 3053 строки с 2014 года; Error% 2.05%→1.75%, CI 0%→100%, Direction 67%→100%
- **✅ v15.4 — `_load_single_ticker`** — вынесен из `load_stock_data_moex_test`; функция стала обёрткой с логикой склейки
- **✅ v15.4 — Ljung-Box guard** — `lags = min(20, len(residuals) // 2)`; при `lags < 2` тест пропускается; устранён `ValueError` для тикеров с короткой историей
- **✅ v15.4 — v14 → v15** — исправлены строки вывода: заголовок консоли, имена `.jpg`/`.txt` файлов, строка отчёта

### v15.3 — Ridge мета-леарнер

- **✅ v15.3 — Ridge мета-леарнер** — `XGBRegressor` заменён на `Ridge(alpha=1.0)`; при 2 входных признаках линейная модель корректнее: меньше переобучение, веса интерпретируемы; `LSTM=coef_[0], XGB=coef_[1]` логируются на каждом сплите; RMSE 10.08→8.71, Error% 1.34%→0.55% на SBER benchmark
- **✅ v15.3 — `ci_params` вынесен перед циклом** — словарь XGBoost-параметров для CI-моделей создаётся один раз до walk-forward

### v15.2 — признаки сезонности

- **✅ v15.2 — Признаки сезонности** — три новых признака: `day_of_week` (0=пн…4=пт), `month` (1–12), `quarter` (1–4); вычисляются в `update_technical_indicators`; добавлены в оба списка `features`; вектор признаков +3 (итого 36–39 динамически); avg Error% на SBER benchmark: 1.60% → 1.34%; `month` появляется в XGBoost feature importances

### v15.1 — воспроизводимость и новые метрики

- **✅ v15.1 — Random seed** — `RANDOM_SEED = 42`; `random`, `np.random`, `tf.random` инициализируются после импортов; устранён разброс RMSE 8.57–11.77 между прогонами
- **✅ v15.1 — Direction Accuracy** — метрика % правильно угаданных направлений движения цены (вверх/вниз от базы); выводится в бэктест-лог, `.txt`-отчёт и benchmark-вывод
- **✅ v15.1 — Suppress Granger stdout** — `contextlib.redirect_stdout(io.StringIO())` вокруг всех трёх вызовов `grangercausalitytests`; убраны таблицы "number of lags / ssr based F test..."

### v15 — макропризнаки и диагностика

- **✅ v15 — `value_usd` (константный признак) заменён** — три time-varying макропризнака `usd_rub_hist`, `cbr_rate`, `brent_price` через реальные API (ЦБ РФ XML, ЦБ РФ SOAP, MOEX ISS BRN фьючерс); `macro_loader.py` выделен в отдельный модуль
- **✅ v15 — Granger pre-screening** — перед обучением каждого горизонта признаки с p ≥ 0.05 динамически исключаются; на SBER исключены `cbr_rate` и `brent_price`, на GAZP — только `brent_price`
- **✅ v15 — Тест Льюнга–Бокса** — диагностика автокорреляции OOS-остатков после walk-forward (20 лагов, stdout + logger)
- **✅ v15 — Тест Грэнжера** — формальное обоснование выбора топ-15 признаков из XGBoost feature importance (stdout + logger)
- **✅ v15 — Макроданные загружаются один раз** — параметр `macro_data=None` в `prepare_and_train_model`; один вызов `load_macro_data` на весь прогон вместо трёх
- **✅ v15 — `--benchmark` CLI-флаг** — воспроизводимый бэктест без интерактивного ввода

### Критические ошибки

- **✅ v13.2 — Ошибка отступа в точке входа** — `if backtest_mode:` / `else:` находились вне `if __name__ == '__main__':`.
- **✅ v13.5 — SSL-верификация отключена** — оба вызова Tinkoff Invest API переведены на `verify=certifi.where()`; добавлен `import certifi`.
- **✅ v13.2 — XGBoost Optuna обучался на train-данных** — добавлен holdout-split 80/20 внутри функции оптимизации.
- **✅ v14.3/v14.5 — CI-bounds систематически ниже точечного прогноза** — квантильные модели теперь обучаются на OOS-остатках walk-forward тестов; CI добавляется как смещение к прогнозу (`lower = pred + residual_q`).

### Качество модели

- **✅ v13.3 — Большинство фундаментальных признаков = 0** — нулевые фичи закомментированы, вектор сокращён с 39 до 29 признаков.
- **✅ v13.2 — Курс USD/RUB захардкожен** — заменён `get_usd_rub_rate()` с XML API ЦБ РФ; fallback 90.0.
- **✅ v14.0 — Наивная авторегрессия для прогноза 2–3 дней** — заменена Direct Multi-Step Forecasting (MIMO): отдельная модель на каждый горизонт.
- **✅ v13.6 — Нет признаков на основе доходностей** — добавлены `Vol_Return_5/10/20` и `Return_MA_5/10`; вектор расширен до 34 признаков.
- **✅ v13.3 — Прогнозные даты не учитывали торговый календарь** — добавлена `next_business_day()`.
- **✅ v14.1 — Прогноз начинался с пропуском текущего дня** — `base_date = data['Date'].max()` вместо `end_date`.

### Код и зависимости

- **✅ v13.4 — Неиспользуемые импорты** — удалены `TensorBoard`, `ModelCheckpoint`, `interp1d`, `io`, `Path`.
- **✅ v14.1 — Мёртвый код `update_technical_indicators_single_row`** — удалён после перехода на MIMO.
- **✅ v13.5 — `requirements.txt`** — зафиксированы 11 прямых зависимостей с точными версиями.

### Вынесенные модули

- **✅ `config.py`** — все гиперпараметры и константы вынесены из основного скрипта.
- **✅ `presentation_output.py`** — презентационный график для слайдов (16:9, русские подписи, 90% CI, PNG 1920×1080 при 150 DPI); вызывается из `stock_modelv16.py` (и legacy `stock_modelv15.py`) при вопросе 6 интерактивного режима или флаге `--presentation`.

### Прочее

- **✅ Режим воспроизводимого бенчмарка** — `benchmark_mode` в интерактивном меню; фиксированный тикер и дата, дефолтные гиперпараметры, результаты в `benchmarks.md`. Дата старта данных вынесена в `BENCHMARK_START_DATE`.
- **✅ `TINVEST_USE_GRPC`** — переменная удалена из `.env` (использовалась только в legacy gRPC-клиенте, в v14 не нужна).
- **✅ Сравнение с ARIMAX-аналогом** — [Stock-Forecast-ARIMAX](https://github.com/GorelikovMatvey/Stock-Forecast-ARIMAX) изучен; выводы перенесены в подраздел "Из сравнения с ARIMAX".
