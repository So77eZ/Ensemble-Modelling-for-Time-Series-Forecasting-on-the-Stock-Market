import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

import json
import contextlib
import io
import random
import time
import matplotlib
import argparse

from datetime import datetime, timedelta

import requests
import urllib3
import xml.etree.ElementTree as ET

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import torch
import torch.nn as nn
import logging

import optuna

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

# LSTM реализован на PyTorch (CUDA на Windows native, в отличие от TF 2.10+).
# Сохранённые hyperparams (units/dropout/lr) совместимы с PyTorch-архитектурой.

RANDOM_SEED = 42
ENSEMBLE_SEEDS = [42, 7, 123]   # N=3 LSTM с разными инициализациями
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ============================================================================
# CONFIGURATION & LOGGING SETUP
# ============================================================================

from presentation_output import plot_presentation
from macro_loader import load_macro_data
from fundamentals_loader import load_dividend_features, get_ticker_shortname

_MACRO_COLS    = ['usd_rub_hist', 'cbr_rate', 'brent_price', 'imoex', 'rtsi']
_FUND_DIV_COLS = ['div_days_to_next', 'div_next_amount', 'div_days_since_last']

# Подмножество признаков, которое получает LSTM (v15-наследие).
# XGB видит полный набор (43 фичей включая relative features), LSTM — только сырое
# OHLCV + return + контекст волатильности. Идея: дать LSTM уникальное «сырое
# временное» представление, чтобы он не дублировал feature engineering, который
# XGB и так выучит сам. Расширение (15 фич) пробовали для log-return target в v16 —
# не дало эффекта, оставлено наследие v15.
_LSTM_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change_1', 'Vol_Return_10']

from config import (
    LSTM_LOOK_BACK, LSTM_EPOCHS, LSTM_PATIENCE, LSTM_BATCH_SIZE,
    LSTM_LEARNING_RATE, LSTM_DROPOUT_RATE, LSTM_UNITS,
    XGBOOST_N_ESTIMATORS, XGBOOST_MAX_DEPTH, XGBOOST_LEARNING_RATE,
    XGBOOST_SUBSAMPLE, XGBOOST_COLSAMPLE_BYTREE, XGBOOST_RANDOM_STATE,
    XGBOOST_VERBOSITY, XGBOOST_DEVICE, OUTPUT_ROOT,
    META_N_ESTIMATORS, META_MAX_DEPTH, META_LEARNING_RATE,
    META_SUBSAMPLE, META_COLSAMPLE_BYTREE
)

MODEL_VERSION = os.path.splitext(os.path.basename(__file__))[0]
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, MODEL_VERSION)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

log_dir = os.path.join(MODEL_OUTPUT_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'training_detailed_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info(f"MODEL VERSION: {MODEL_VERSION}")
logger.info(f"OUTPUT DIR: {MODEL_OUTPUT_DIR}")
logger.info(f"PyTorch: {torch.__version__}")
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if TORCH_DEVICE.type == 'cuda':
    logger.info(f"GPU Available: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
else:
    logger.info("GPU Available: [] (CPU mode)")
logger.info(f"XGBoost Available: {XGBOOST_AVAILABLE}")
logger.info("=" * 60)

# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================

BENCHMARK_TICKER      = 'SBER'
BENCHMARK_DATE        = '2024-10-14'
BENCHMARK_CI_MODE     = 'wide'
BENCHMARK_APPROX_TIME = '5–10 минут'
BENCHMARK_START_DATE  = '2014-01-01'
BENCHMARKS_FILE       = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', 'benchmarks.md')

# ============================================================================
# HYPERPARAMETERS MANAGEMENT
# ============================================================================

HYPERPARAMS_DIR = os.path.join(MODEL_OUTPUT_DIR, 'hyperparams')
os.makedirs(HYPERPARAMS_DIR, exist_ok=True)

def save_hyperparams(ticker, lstm_params, xgb_params):
    """Сохранение оптимальных гиперпараметров для тикера"""
    params = {
        'timestamp': datetime.now().isoformat(),
        'lstm': lstm_params,
        'xgboost': xgb_params
    }
    filepath = os.path.join(HYPERPARAMS_DIR, f'{ticker}_hyperparams.json')
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Hyperparameters saved: {filepath}")

def load_hyperparams(ticker):
    """Загрузка сохраненных гиперпараметров для тикера"""
    filepath = os.path.join(HYPERPARAMS_DIR, f'{ticker}_hyperparams.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)
        logger.info(f"Loaded hyperparameters from: {filepath}")
        logger.info(f"Saved on: {params['timestamp']}")
        return params['lstm'], params['xgboost']
    return None, None

def get_default_hyperparams():
    """Дефолтные гиперпараметры если Optuna не используется"""
    lstm_params = {
        'units': LSTM_UNITS,
        'dropout': LSTM_DROPOUT_RATE,
        'lr': LSTM_LEARNING_RATE
    }
    xgb_params = {
        'n_estimators': XGBOOST_N_ESTIMATORS,
        'max_depth': XGBOOST_MAX_DEPTH,
        'learning_rate': XGBOOST_LEARNING_RATE,
        'subsample': XGBOOST_SUBSAMPLE,
        'colsample_bytree': XGBOOST_COLSAMPLE_BYTREE
    }
    return lstm_params, xgb_params

# ============================================================================
# USD/RUB RATE
# ============================================================================

def next_business_day(date: datetime, n: int) -> datetime:
    """Сдвиг на n рабочих дней (пропускает субботу и воскресенье)."""
    current = date
    added = 0
    while added < n:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current

def _forecast_dates_for_horizon(base_date: datetime, horizon: int) -> list:
    """Возвращает список из horizon рабочих дней начиная с base_date+1."""
    return [next_business_day(base_date, i + 1) for i in range(horizon)]


def merge_horizon_results(all_results: dict):
    """
    Принимает {h: result_tuple} от трёх вызовов prepare_and_train_model.
    Возвращает (forecasts, confidence_intervals) или None при любом сбое.
    """
    if any(r is None or r[0] is None for r in all_results.values()):
        return None
    forecasts = {h: all_results[h][3][h] for h in all_results}
    confidence_intervals = {h: all_results[h][5][h] for h in all_results}
    return forecasts, confidence_intervals

def _train_quantile_pair(
    meta_features: np.ndarray,
    residuals: np.ndarray,
    lower_alpha: float,
    upper_alpha: float,
    base_params: dict,
) -> tuple:
    """Обучает пару квантильных XGBoost-моделей для границ CI.

    Обе модели обучаются на одной выборке (meta_features, residuals).
    Различие достигается параметром quantile_alpha: pinball-loss
    с разными значениями alpha извлекает разные перцентили условного
    распределения остатков.

    Parameters
    ----------
    meta_features : np.ndarray
        Признаки уровня 1 (объединённые предсказания LSTM и XGBoost
        с тестовых окон walk-forward).
    residuals : np.ndarray
        OOS-остатки (actual - meta_pred) с тех же тестовых окон.
    lower_alpha, upper_alpha : float
        Перцентили для нижней и верхней границы интервала
        (например, 0.05 и 0.95 для 90% CI).
    base_params : dict
        Базовые гиперпараметры XGBoost (берутся от мета-учителя).

    Returns
    -------
    tuple[xgb.XGBRegressor, xgb.XGBRegressor]
        Обученные модели для нижней и верхней границы.
    """
    common = {**base_params, 'objective': 'reg:quantileerror'}
    lower_model = xgb.XGBRegressor(**{**common, 'quantile_alpha': lower_alpha})
    upper_model = xgb.XGBRegressor(**{**common, 'quantile_alpha': upper_alpha})
    lower_model.fit(meta_features, residuals)
    upper_model.fit(meta_features, residuals)
    return lower_model, upper_model

def _get_ci_params(ci_mode: str, X_train: np.ndarray, y_train: np.ndarray):
    """
    Возвращает (X_q, y_q, lower_alpha, upper_alpha) для обучения квантильных моделей.
    narrow: последние 756 строк (~3 торговых года), перцентили 25/75.
    wide:   полная история, перцентили 5/95.
    """
    NARROW_WINDOW = 756
    if ci_mode == 'narrow':
        q_start = max(0, len(X_train) - NARROW_WINDOW)
        return X_train[q_start:], y_train[q_start:], 0.25, 0.75
    return X_train, y_train, 0.05, 0.95


def get_usd_rub_rate() -> float:
    """Актуальный курс USD/RUB от ЦБ РФ. Fallback = 90.0 при ошибке."""
    try:
        resp = requests.get(
            'https://www.cbr.ru/scripts/XML_daily.asp',
            timeout=5
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for valute in root.findall('Valute'):
            if valute.find('CharCode').text == 'USD':
                rate = float(valute.find('Value').text.replace(',', '.'))
                logger.info(f"USD/RUB rate from CBR: {rate}")
                return rate
    except Exception as e:
        logger.warning(f"Could not fetch USD/RUB from CBR: {e}. Using fallback 90.0")
    return 90.0

# ============================================================================
# TINKOFF API FUNDAMENTALS
# ============================================================================

class TinkoffFundamentalLoader:
    def __init__(self, token: str):
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.base_url = (
            "https://invest-public-api.tbank.ru/rest/"
            "tinkoff.public.invest.api.contract.v1.InstrumentsService"
        )

    def get_fundamentals(self, ticker: str) -> dict:
        logger.info(f"--- LOADING TINKOFF FUNDAMENTALS: {ticker} ---")
        asset_uid = self._find_asset_uid(ticker)
        if not asset_uid:
            logger.warning("Asset UID not found. Using empty fundamentals.")
            return self._get_empty_fundamentals()

        raw_funds = self._fetch_fundamentals_by_uid(asset_uid)
        return self._map_to_model_features(raw_funds)

    def _find_asset_uid(self, ticker: str) -> str | None:
        url = f"{self.base_url}/Shares"
        payload = {"instrumentStatus": "INSTRUMENT_STATUS_BASE"}
        try:
            logger.info(f"Searching for ticker {ticker}...")
            resp = requests.post(
                url,
                json=payload,
                headers=self.headers,
                timeout=10,
                verify=False
            )
            if resp.status_code != 200:
                logger.error(f"API Error (Shares): {resp.status_code} {resp.text}")
                return None

            data = resp.json()
            instruments = data.get('instruments', [])
            for item in instruments:
                if (item.get('ticker') == ticker
                        and item.get('classCode') == 'TQBR'):
                    uid = item.get('assetUid')
                    logger.info(
                        f"FOUND ASSET UID: {uid} ({item.get('name')})"
                    )
                    return uid

            logger.warning(f"Ticker {ticker} not found in TQBR class.")
            return None
        except Exception as e:
            logger.error(f"Exception searching ticker: {e}")
            return None

    def _fetch_fundamentals_by_uid(self, asset_uid: str) -> dict:
        url = f"{self.base_url}/GetAssetFundamentals"
        payload = {"assets": [asset_uid]}
        try:
            logger.info(f"Requesting GetAssetFundamentals for {asset_uid}...")
            resp = requests.post(
                url,
                json=payload,
                headers=self.headers,
                timeout=10,
                verify=False
            )
            if resp.status_code != 200:
                logger.error(f"API Error (Fundamentals): {resp.status_code}")
                return self._get_empty_fundamentals()

            data = resp.json()
            funds = data.get('fundamentals', [])
            if not funds:
                logger.warning("API returned empty fundamentals list.")
                return self._get_empty_fundamentals()

            item = funds[0]
            result = {
                'pe_ratio': float(item.get('peRatioTtm', 0) or 0),
                'pb_ratio': float(item.get('priceToBookTtm', 0) or 0),
                'roe': float(item.get('roe', 0) or 0),
                'div_yield': float(item.get('dividendYieldDailyTtm', 0) or 0),
                'beta': float(item.get('beta', 0) or 0),
                'market_cap': float(item.get('marketCapitalization', 0) or 0)
            }
            logger.info(
                "SUCCESS: "
                f"P/E={result['pe_ratio']}, ROE={result['roe']}, "
                f"Div={result['div_yield']}"
            )
            return result
        except Exception as e:
            logger.error(f"Exception loading fundamentals: {e}")
            return self._get_empty_fundamentals()

    def _get_empty_fundamentals(self) -> dict:
        return {
            'pe_ratio': 0.0,
            'pb_ratio': 0.0,
            'roe': 0.0,
            'div_yield': 0.0,
            'beta': 0.0,
            'market_cap': 0.0
        }

    def _map_to_model_features(self, funds: dict) -> dict:
        return {
            'market_cap': funds.get('market_cap', 0.0),
            'roa': 0.0,
            'roe': funds.get('roe', 0.0),
            'debt_equity': 0.0,
            'current_ratio': 0.0,
            'gross_profit_margin': 0.0,
            'dividend_yield': funds.get('div_yield', 0.0),
            'eps_growth': 0.0,
            'sales_growth': 0.0,
            'operating_margin': 0.0,
            'net_profit_margin': 0.0,
            'pe_ratio': funds.get('pe_ratio', 0.0),
            'pb_ratio': funds.get('pb_ratio', 0.0),
            'ps_ratio': 0.0,
            'price_cash_flow': 0.0,
            'value_usd': get_usd_rub_rate(),
            'beta': funds.get('beta', 0.0)
        }

tinkoff_token = os.getenv('TINVEST_TOKEN') or os.getenv('TINKOFF_TOKEN')
tinkoff_loader = TinkoffFundamentalLoader(tinkoff_token) if tinkoff_token else None

# ============================================================================
# STOCK DATA LOADING
# ============================================================================

def _load_single_ticker(ticker_symbol, start_date, end_date):
    # moexalgo with retry (SSL EOF is transient)
    _RETRIES = 3
    for attempt in range(_RETRIES):
        try:
            from moexalgo import Ticker
            stock = Ticker(ticker_symbol)
            data = stock.candles(start=start_date, end=end_date, period='1D')
            if not data.empty:
                data = data[['begin', 'open', 'high', 'low', 'close', 'volume']]
                data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
                logger.info(f" -> moexalgo {ticker_symbol}: {len(data)} rows ({data['Date'].min()} to {data['Date'].max()})")
                return data
            break
        except Exception as e:
            if attempt < _RETRIES - 1:
                delay = 2 ** attempt
                logger.warning(f" moexalgo attempt {attempt+1}/{_RETRIES} ({ticker_symbol}): {e}. Retry in {delay}s...")
                time.sleep(delay)
            else:
                logger.warning(f" moexalgo error ({ticker_symbol}): {e}. Switching to direct API...")

    # Direct MOEX ISS API — try with SSL verification first, then without
    base_url = (
        "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/"
        f"securities/{ticker_symbol}.json"
    )
    params = {'from': start_date, 'till': end_date, 'limit': 100}

    for verify_ssl in [True, False]:
        all_data = []
        offset = 0
        fetch_ok = True

        while True:
            params['start'] = offset
            chunk_ok = False
            for attempt in range(_RETRIES):
                try:
                    response = requests.get(base_url, params=params, timeout=15, verify=verify_ssl)
                    response.raise_for_status()
                    json_data = response.json()
                    columns = json_data['history']['columns']
                    rows = json_data['history']['data']
                    if not rows:
                        chunk_ok = None  # sentinel: pagination done
                    else:
                        all_data.append(pd.DataFrame(rows, columns=columns))
                        offset += len(rows)
                        chunk_ok = True
                    break
                except Exception as e:
                    if attempt < _RETRIES - 1:
                        time.sleep(2 ** attempt)
                    else:
                        logger.warning(f"Direct MOEX API error ({ticker_symbol}): {e}")
                        fetch_ok = False

            if chunk_ok is None or not fetch_ok:
                break

        if all_data:
            data = pd.concat(all_data, ignore_index=True)
            data = data[['TRADEDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date').drop_duplicates()
            if not verify_ssl:
                logger.warning(f" -> REST {ticker_symbol}: SSL verify=False (network fallback)")
            logger.info(f" -> REST {ticker_symbol}: {len(data)} rows ({data['Date'].min()} to {data['Date'].max()})")
            return data

        if verify_ssl and not fetch_ok:
            logger.warning(f"Direct MOEX SSL error ({ticker_symbol}), retrying without SSL verification...")
        else:
            break  # no SSL error but empty data (ticker not found), or verify=False exhausted

    logger.error(f"Could not load data for {ticker_symbol}")
    return None


def load_stock_data_moex_test(ticker_symbol, start_date, end_date):
    logger.info(f"Deep data loading for {ticker_symbol} from {start_date} to {end_date}...")

    # YDEX (МКПАО Яндекс, листинг с 2024-07) склеивается с YNDX (Yandex N.V., до 2024-06).
    # Коэффициент обмена акций 1:1 — нормировка не нужна.
    # Размечаем строки колонкой is_post_restructure: 0 для YNDX-истории, 1 для YDEX —
    # модель видит структурный шок 2024-07 (смена юрисдикции NL→RU, разный shareholder base)
    # и может учить разные режимы вместо смешанного ряда.
    if ticker_symbol == 'YDEX':
        yndx = _load_single_ticker('YNDX', start_date, end_date)
        ydex = _load_single_ticker('YDEX', start_date, end_date)
        parts = []
        if yndx is not None and not yndx.empty:
            yndx = yndx.copy()
            yndx['is_post_restructure'] = 0
            parts.append(yndx)
        if ydex is not None and not ydex.empty:
            ydex = ydex.copy()
            ydex['is_post_restructure'] = 1
            parts.append(ydex)
        if not parts:
            logger.error("YDEX: не удалось загрузить ни YNDX, ни YDEX")
            return None
        if len(parts) == 2:
            last_yndx = float(parts[0]['Close'].iloc[-1])
            first_ydex = float(parts[1]['Close'].iloc[0])
            ratio = first_ydex / last_yndx if last_yndx != 0 else 1.0
            logger.info(f"Стык YNDX/YDEX: последняя YNDX={last_yndx:.2f}, первая YDEX={first_ydex:.2f}, ratio={ratio:.4f}")
        merged = pd.concat(parts, ignore_index=True)
        merged = merged.sort_values('Date').drop_duplicates(subset='Date').reset_index(drop=True)
        logger.info(f"YNDX+YDEX итого: {len(merged)} строк ({merged['Date'].min()} до {merged['Date'].max()}); "
                    f"is_post_restructure: 0 на {(merged['is_post_restructure']==0).sum()} строках, "
                    f"1 на {(merged['is_post_restructure']==1).sum()} строках")
        return merged

    # T (МКПАО Т-Технологии, листинг с 2024-11) — реструктуризация TCSG (TCS Group Holding
    # PLC, делистнут 2024-11-27 после смены юрисдикции Кипр → РФ).
    # Особенность: MOEX отдаёт под тикером 'T' уже **склеенную историю** TCSG+T
    # (1697 строк, 2019-10-28 → сегодня) — отдельная склейка не нужна, в отличие от
    # YDEX. Дамми is_post_restructure ставим по дате (граница 2024-11-28).
    # Принимаем оба имени тикера ('T' и 'TCSG') — оба возвращают одну и ту же серию.
    if ticker_symbol in ('T', 'TCSG'):
        data = _load_single_ticker('T', start_date, end_date)
        if data is None or data.empty:
            # Fallback на старый тикер, если вдруг T недоступен (теоретический сценарий)
            data = _load_single_ticker('TCSG', start_date, end_date)
            if data is None or data.empty:
                logger.error("T/TCSG: не удалось загрузить серию ни под T, ни под TCSG")
                return None
        data = data.copy()
        cutoff = pd.to_datetime('2024-11-28')
        data['is_post_restructure'] = (data['Date'] >= cutoff).astype(int)
        n0 = int((data['is_post_restructure'] == 0).sum())
        n1 = int((data['is_post_restructure'] == 1).sum())
        logger.info(f"T/TCSG: {len(data)} строк ({data['Date'].min()} до {data['Date'].max()}); "
                    f"is_post_restructure: 0 на {n0} строках (TCSG-эпоха), "
                    f"1 на {n1} строках (T-эпоха, с 2024-11-28)")
        return data

    return _load_single_ticker(ticker_symbol, start_date, end_date)

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_sma(data, window=10):
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, span=20):
    return data['Close'].ewm(span=span, adjust=False).mean()

def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, std=2):
    sma = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper = sma + (rolling_std * std)
    lower = sma - (rolling_std * std)
    width = (upper - lower) / sma
    return sma, upper, lower, width

def calculate_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=window).mean()

def calculate_stochastic(data, k_window=14, d_window=3):
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def calculate_adx(data, window=14):
    plus_dm = data['High'].diff()
    minus_dm = -data['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = calculate_atr(data, window=1)
    atr = tr.rolling(window=window).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=window).mean()
    return adx.fillna(0)

def calculate_momentum(data, window=10):
    return data['Close'].diff(window)

def calculate_price_change(data, window=1):
    result = data['Close'].pct_change(window) * 100
    return result

def update_technical_indicators(data):
    data['SMA_10'] = calculate_sma(data, 10)
    data['EMA_20'] = calculate_ema(data, 20)
    data['RSI_14'] = calculate_rsi(data, 14)
    data['MACD'], data['MACD_Signal'], data['MACD_Histogram'] = calculate_macd(data)
    data['BB_Middle'], data['BB_Upper'], data['BB_Lower'], data['BB_Width'] = calculate_bollinger_bands(data)
    data['ATR_14'] = calculate_atr(data, 14)
    data['Stoch_K'], data['Stoch_D'] = calculate_stochastic(data)
    data['ADX_14'] = calculate_adx(data, 14)
    data['Momentum_10'] = calculate_momentum(data, 10)
    data['Price_Change_1'] = calculate_price_change(data, 1)
    data['Price_Change_5'] = calculate_price_change(data, 5)
    returns = data['Close'].pct_change()
    data['Vol_Return_5'] = returns.rolling(5).std()
    data['Vol_Return_10'] = returns.rolling(10).std()
    data['Vol_Return_20'] = returns.rolling(20).std()
    data['Return_MA_5'] = returns.rolling(5).mean()
    data['Return_MA_10'] = returns.rolling(10).mean()
    data['day_of_week'] = data['Date'].dt.dayofweek   # 0=пн … 4=пт
    data['month']       = data['Date'].dt.month        # 1–12
    data['quarter']     = data['Date'].dt.quarter      # 1–4
    # Относительные признаки: цена относительно скользящих средних.
    # Убирают масштаб цены и встраивают логику возврата к среднему.
    data['Close_to_SMA10'] = data['Close'] / data['SMA_10']
    data['Close_to_EMA20'] = data['Close'] / data['EMA_20']
    bb_range = (data['BB_Upper'] - data['BB_Lower']).replace(0, np.nan)
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / bb_range  # [0,1] внутри полос
    roll_std = data['Close'].rolling(20).std().replace(0, np.nan)
    data['Close_ZScore_20'] = (data['Close'] - data['Close'].rolling(20).mean()) / roll_std
    return data

# ============================================================================
# OPTUNA OPTIMIZATION
# ============================================================================

class LSTMRegressor(nn.Module):
    """3-слойная LSTM-регрессия с decreasing dropout, аналог Keras-архитектуры из v15.x.
    Архитектура: LSTM(u1) → Dropout(d) → LSTM(u2) → Dropout(d/2) → LSTM(u2) → Dropout(d) → Dense(32, ReLU) → Dense(1).
    """
    def __init__(self, n_features: int, units_1: int, units_2: int, dropout: float):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, units_1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(units_1, units_2, batch_first=True)
        self.drop2 = nn.Dropout(dropout * 0.5)
        self.lstm3 = nn.LSTM(units_2, units_2, batch_first=True)
        self.drop3 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(units_2, 32)
        self.dense2 = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm1(x); x = self.drop1(x)
        x, _ = self.lstm2(x); x = self.drop2(x)
        x, _ = self.lstm3(x)             # (B, T, units_2)
        x = self.drop3(x[:, -1, :])      # last timestep, ≡ return_sequences=False
        x = torch.relu(self.dense1(x))
        return self.dense2(x).squeeze(-1)


def _to_device_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(TORCH_DEVICE)


def train_lstm_torch(model: LSTMRegressor, X_train: np.ndarray, y_train: np.ndarray,
                     lr: float, epochs: int, patience: int, batch_size: int):
    """Тренировка LSTM с manual EarlyStopping (validation_split=0.2, restore_best_weights=True).
    Возвращает (model, history dict с ключами 'loss'/'val_loss')."""
    model = model.to(TORCH_DEVICE)
    X_t = _to_device_tensor(X_train)
    y_t = _to_device_tensor(y_train)

    n_val = max(1, int(0.2 * len(X_t)))
    X_tr, X_val = X_t[:-n_val], X_t[-n_val:]
    y_tr, y_val = y_t[:-n_val], y_t[-n_val:]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {'loss': [], 'val_loss': []}
    best_val = float('inf')
    best_state = None
    no_improve = 0

    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=TORCH_DEVICE)
        train_loss_sum = 0.0
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            pred = model(X_tr[idx])
            loss = loss_fn(pred, y_tr[idx])
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(idx)
        train_loss = train_loss_sum / len(X_tr)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_lstm_torch(model: LSTMRegressor, X: np.ndarray) -> np.ndarray:
    """Inference на GPU; возвращает 1D numpy."""
    model.eval()
    X_t = _to_device_tensor(X)
    with torch.no_grad():
        out = model(X_t)
    return out.detach().cpu().numpy().flatten()


def optimize_lstm_params(X_train, y_train, n_trials=20):
    def objective(trial):
        units = trial.suggest_int('units', 32, 128)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        # 2-слойная модель для быстрого Optuna trial (как было в Keras-версии)
        class _OptunaLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm1 = nn.LSTM(X_train.shape[2], units, batch_first=True)
                self.drop1 = nn.Dropout(dropout)
                self.lstm2 = nn.LSTM(units, units, batch_first=True)
                self.drop2 = nn.Dropout(dropout)
                self.dense = nn.Linear(units, 1)
            def forward(self, x):
                x, _ = self.lstm1(x); x = self.drop1(x)
                x, _ = self.lstm2(x); x = self.drop2(x[:, -1, :])
                return self.dense(x).squeeze(-1)

        model = _OptunaLSTM()
        _, hist = train_lstm_torch(model, X_train, y_train, lr=lr,
                                    epochs=LSTM_EPOCHS, patience=LSTM_PATIENCE,
                                    batch_size=LSTM_BATCH_SIZE)
        return min(hist['val_loss'])

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params

def optimize_xgboost_params(X_train, y_train, n_trials=20):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': XGBOOST_RANDOM_STATE,
            'verbosity': XGBOOST_VERBOSITY,
            'device': XGBOOST_DEVICE
        }
        split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:split], X_train[split:]
        y_tr, y_val = y_train[:split], y_train[split:]
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def optimize_ridge_alpha(meta_features: np.ndarray, y_true: np.ndarray, n_trials: int = 20) -> float:
    """Подбор Ridge alpha через Optuna с R² scoring.

    v16-B1.1: заменён neg_MSE на R². R² штрафует одновременно и за абсолютную
    ошибку, и за слабую корреляцию (R² = 0 для предсказания константы = mean(y)).
    Это компромисс между MSE-only (vs мы получали LSTM=0) и correlation-only
    (vs модель сжимается к среднему). Стандартная sklearn-метрика, не требует
    custom scorer.
    """
    from sklearn.model_selection import cross_val_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-3, 100.0, log=True)
        scores = cross_val_score(
            Ridge(alpha=alpha, positive=True), meta_features, y_true,
            cv=min(5, len(y_true)), scoring='r2'
        )
        return float(scores.mean())

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return float(study.best_params['alpha'])

# ============================================================================
# MODEL TRAINING
# ============================================================================

def prepare_and_train_model(data, ticker, end_date, best_lstm_params, best_xgb_params, backtest_mode=False, backtest_date=None, horizon: int = 1, ci_mode: str = 'wide', macro_data=None, fund_data=None, div_data=None):
    logger.info("=" * 60)
    logger.info(f"PREPARING DATA FOR {ticker}")
    if backtest_mode:
        logger.info(f"BACKTEST MODE: Training until {backtest_date}")
    logger.info("=" * 60)
    logger.info("Calculating technical indicators...")
    data = update_technical_indicators(data)

    if fund_data is not None:
        tinkoff_funds = fund_data
    elif tinkoff_loader:
        tinkoff_funds = tinkoff_loader.get_fundamentals(ticker)
    else:
        tinkoff_funds = {}

    for key, value in tinkoff_funds.items():
        data[key] = value

    # Макроэкономические time-varying признаки
    if macro_data is None:
        macro_start = data['Date'].min().strftime('%Y-%m-%d')
        logger.info("Loading macro data (USD/RUB history, CBR key rate, Brent, IMOEX, RTSI)...")
        macro_data = load_macro_data(macro_start, end_date)
    data['Date'] = pd.to_datetime(data['Date']).dt.normalize()
    data = data.merge(macro_data, on='Date', how='left')
    data = data.ffill().bfill()

    # Дивидендные time-varying признаки (MOEX ISS, любой тикер)
    if div_data is None:
        div_start = data['Date'].min().strftime('%Y-%m-%d')
        logger.info(f"Loading dividend features for {ticker}...")
        div_data = load_dividend_features(ticker, div_start, end_date)
    div_data['Date'] = pd.to_datetime(div_data['Date']).dt.normalize()
    data = data.merge(div_data, on='Date', how='left')

    data = data.infer_objects(copy=False).fillna(0)

    # Гарантируем наличие фундаментальных колонок (если Tinkoff API недоступен)
    for col in ['market_cap', 'roe', 'dividend_yield', 'pe_ratio', 'pb_ratio', 'beta']:
        if col not in data.columns:
            data[col] = 0.0

    logger.info(f"Data shape after cleaning: {data.shape}")

    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_10', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width',
        'ATR_14', 'Stoch_K', 'Stoch_D', 'ADX_14', 'Momentum_10',
        'Price_Change_1', 'Price_Change_5',
        # --- Скользящая волатильность и momentum доходностей ---
        'Vol_Return_5', 'Vol_Return_10', 'Vol_Return_20',
        'Return_MA_5', 'Return_MA_10',
        # --- Сезонность ---
        'day_of_week', 'month', 'quarter',
        # --- Относительные признаки (v16): цена относительно MA/полос, z-score ---
        'Close_to_SMA10', 'Close_to_EMA20', 'BB_Position', 'Close_ZScore_20',
        # --- Фундаментальные (snapshot из Tinkoff Invest API) ---
        'market_cap', 'roe', 'dividend_yield', 'pe_ratio', 'pb_ratio', 'beta',
        # --- Макроэкономические (time-varying, загружаются из macro_loader) ---
        *[c for c in _MACRO_COLS if c in data.columns and not data[c].isna().all()],
        # --- Дивидендные (time-varying, загружаются из fundamentals_loader) ---
        *[c for c in _FUND_DIV_COLS if c in data.columns and not data[c].isna().all()],
        # --- Дамми реструктуризации (только для склеенных тикеров типа YDEX = YNDX+YDEX) ---
        *(['is_post_restructure'] if 'is_post_restructure' in data.columns else []),
    ]

    # Granger-скрининг: убираем макро- и дивидендные признаки, не предсказывающие Close (p >= 0.05)
    from statsmodels.tsa.stattools import grangercausalitytests as _gct
    _granger_cols = [c for c in _MACRO_COLS + _FUND_DIV_COLS if c in features]
    for _mc in _granger_cols:
        # Исключаем константные признаки (NaN или нулевая дисперсия) — Granger на них не работает
        _col_data = data[_mc].dropna()
        if _col_data.empty or _col_data.std() == 0:
            features.remove(_mc)
            logger.info(f"Granger screening: {_mc} excluded (constant/empty column)")
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _gr = _gct(data[['Close', _mc]].dropna(), maxlag=5)
            _p  = min(r[0]['ssr_ftest'][1] for r in _gr.values())
            if _p >= 0.05:
                features.remove(_mc)
                logger.info(f"Granger screening: {_mc} excluded (p={_p:.3f})")
            else:
                logger.info(f"Granger screening: {_mc} retained (p={_p:.3f})")
        except Exception as e:
            features.remove(_mc)
            logger.warning(f"Granger screening: {_mc} excluded (error: {e})")

    data = data[['Date'] + features].dropna()
    
    # В режиме бэктеста обрезаем данные до указанной даты
    if backtest_mode and backtest_date:
        backtest_dt = datetime.strptime(backtest_date, '%Y-%m-%d')
        data = data[data['Date'] <= backtest_dt]
        logger.info(f"Backtest: Data cut to {data['Date'].max()}")
    
    logger.info(f"Data range for training: {data['Date'].min()} to {data['Date'].max()}")

    logger.info("Normalizing features...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features, index=data.index)
    
    close_scaler = MinMaxScaler()
    close_scaler.fit(data[['Close']])

    # LSTM получает подмножество фич (см. _LSTM_FEATURES); XGB — полный набор.
    lstm_features = [f for f in _LSTM_FEATURES if f in features]
    logger.info(f"LSTM features ({len(lstm_features)}/{len(features)}): {lstm_features}")

    logger.info(f"Preparing sequences with look_back={LSTM_LOOK_BACK}, horizon={horizon}...")
    X, X_lstm, y = [], [], []
    for i in range(LSTM_LOOK_BACK, len(scaled_df) - horizon):
        X.append(scaled_df.iloc[i-LSTM_LOOK_BACK:i].values)
        X_lstm.append(scaled_df[lstm_features].iloc[i-LSTM_LOOK_BACK:i].values)
        y.append(scaled_df['Close'].iloc[i + horizon])
    X, X_lstm, y = np.array(X), np.array(X_lstm), np.array(y)
    logger.info(f"Total sequences: {len(X)} | XGB shape={X.shape}, LSTM shape={X_lstm.shape}")

    # Защитная валидация: walk-forward с 80/85/90% сплитами требует разумного размера выборки.
    # При len(X) < 200 последние 5% дают <10 тестовых семплов — метрики становятся бессмысленными.
    # Реальный пример (TCSG до v15.13): 1187 sequences, последний test=59, R²=-1.0.
    _MIN_SEQUENCES_HARD = 200
    _MIN_SEQUENCES_WARN = 500
    if len(X) < _MIN_SEQUENCES_HARD:
        logger.error(f"Недостаточно данных для walk-forward: {len(X)} sequences (минимум {_MIN_SEQUENCES_HARD}). "
                     f"Метрики и LSTM-обучение не будут стабильными. Прерываю обучение для {ticker}.")
        return None
    if len(X) < _MIN_SEQUENCES_WARN:
        logger.warning(f"Малая выборка: {len(X)} sequences (рекомендуется ≥{_MIN_SEQUENCES_WARN}). "
                       f"OOS-метрики могут быть нестабильны, особенно последний split (~{int(0.1*len(X))} семплов).")

    splits = [
        (int(0.8 * len(X)), int(0.85 * len(X))),
        (int(0.85 * len(X)), int(0.9 * len(X))),
        (int(0.9 * len(X)), len(X))
    ]
    logger.info(f"Splits: {splits}")

    rmses, maes, r2s = [], [], []
    final_pred = []
    models = []
    oos_meta_list: list = []
    oos_residuals_list: list = []
    oos_y_list: list = []
    last_meta_train: np.ndarray | None = None
    last_y_train: np.ndarray | None = None

    ci_params = {
        'n_estimators': META_N_ESTIMATORS,
        'max_depth': META_MAX_DEPTH,
        'learning_rate': META_LEARNING_RATE,
        'subsample': META_SUBSAMPLE,
        'colsample_bytree': META_COLSAMPLE_BYTREE,
        'random_state': XGBOOST_RANDOM_STATE,
        'verbosity': XGBOOST_VERBOSITY,
        'device': XGBOOST_DEVICE,
    }

    for i, (train_end, test_end) in enumerate(splits, 1):
        logger.info("\n" + "=" * 60)
        logger.info(f"WALK-FORWARD SPLIT {i}/{len(splits)}")
        logger.info("=" * 60)

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]
        X_lstm_train = X_lstm[:train_end]
        X_lstm_test  = X_lstm[train_end:test_end]
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        logger.info(f"Training LSTM ensemble ({len(ENSEMBLE_SEEDS)} seeds) on {TORCH_DEVICE}...")
        lstm_units_1 = best_lstm_params['units']
        lstm_units_2 = max(32, lstm_units_1 // 2)
        n_features_lstm = X_lstm_train.shape[2]
        lstm_models_split = []
        for _seed in ENSEMBLE_SEEDS:
            torch.manual_seed(_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(_seed)
            np.random.seed(_seed)
            _m = LSTMRegressor(n_features_lstm, lstm_units_1, lstm_units_2, best_lstm_params['dropout'])
            _m, _hist = train_lstm_torch(
                _m, X_lstm_train, y_train,
                lr=best_lstm_params['lr'],
                epochs=LSTM_EPOCHS,
                patience=LSTM_PATIENCE,
                batch_size=LSTM_BATCH_SIZE,
            )
            lstm_models_split.append(_m)
            logger.info(f"  seed={_seed}: train={_hist['loss'][-1]:.4f}, val={_hist['val_loss'][-1]:.4f}")
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        logger.info("Training XGBoost model...")
        xgb_params = {
            **best_xgb_params,
            'random_state': XGBOOST_RANDOM_STATE,
            'verbosity': XGBOOST_VERBOSITY,
            'device': XGBOOST_DEVICE
        }
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        flat_feature_names = [f"{feat}_t{t}" for t in range(LSTM_LOOK_BACK) for feat in features]
        
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train_flat, y_train)
        logger.info("[OK] XGBoost model trained")

        importances_flat = xgb_model.feature_importances_
        feature_importance_dict = {}
        for idx, feat_name in enumerate(flat_feature_names):
            orig_feat = feat_name.rsplit('_t', 1)[0]
            if orig_feat not in feature_importance_dict:
                feature_importance_dict[orig_feat] = 0
            feature_importance_dict[orig_feat] += importances_flat[idx]
        
        sorted_importances = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top 10 Feature Importances (aggregated):")
        for feat, imp in sorted_importances[:10]:
            logger.info(f" {feat}: {imp:.4f}")

        from statsmodels.tsa.stattools import grangercausalitytests
        top15 = [name for name, _ in sorted_importances[:15]]
        logger.info("Granger causality (top-15 features -> Close, maxlag=5):")
        print("Granger causality (top-15 features -> Close, maxlag=5):")
        for feat in top15:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = grangercausalitytests(data[['Close', feat]].dropna(), maxlag=5)
                min_pval = min(res[0]['ssr_ftest'][1] for res in result.values())
                gc_flag = "[ok]" if min_pval < 0.05 else "[no]"
                msg = f"  Close <- {feat:<22}: p={min_pval:.3f} {gc_flag}"
            except Exception as e:
                msg = f"  Close <- {feat:<22}: [!] тест не удался ({e})"
            print(msg)
            logger.info(msg)

        logger.info("Generating level 0 predictions...")
        lstm_train_preds = np.mean(
            [predict_lstm_torch(_m, X_lstm_train) for _m in lstm_models_split], axis=0
        )
        xgb_train_preds = xgb_model.predict(X_train_flat)

        meta_train = np.column_stack((lstm_train_preds, xgb_train_preds))

        logger.info("Training Meta-Learner (Ridge)...")
        meta_learner = Ridge(alpha=1.0, positive=True)
        meta_learner.fit(meta_train, y_train)
        logger.info(f"[OK] Meta-Learner trained | LSTM={meta_learner.coef_[0]:.3f}, XGB={meta_learner.coef_[1]:.3f}")
        last_meta_train = meta_train
        last_y_train = y_train

        lstm_test_preds = np.mean(
            [predict_lstm_torch(_m, X_lstm_test) for _m in lstm_models_split], axis=0
        )
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        xgb_test_preds = xgb_model.predict(X_test_flat)
        meta_test = np.column_stack((lstm_test_preds, xgb_test_preds))
        test_preds = meta_learner.predict(meta_test)

        # Собираем OOS-остатки для CI (обучение после цикла)
        oos_meta_list.append(meta_test)
        oos_residuals_list.append(y_test - test_preds)
        oos_y_list.append(y_test)

        test_preds_inv = close_scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
        y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        rmse = np.sqrt(mean_squared_error(y_test_inv, test_preds_inv))
        mae = mean_absolute_error(y_test_inv, test_preds_inv)
        r2 = r2_score(y_test_inv, test_preds_inv)

        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)

        final_pred.extend(test_preds_inv)
        models.append((lstm_models_split, xgb_model, meta_learner))

    # Освобождение GPU-памяти: для финального прогноза нужен только последний
    # ансамбль (models[-1]); LSTM-модели предыдущих сплитов больше не используются.
    # На каждый горизонт вызов prepare_and_train_model() создаёт 3 splits × 3 seeds = 9 LSTM,
    # без очистки они копятся в VRAM при последовательных h=1/h=2/h=3 (×3 = 27 моделей).
    if torch.cuda.is_available() and len(models) > 1:
        for _i in range(len(models) - 1):
            _lstm_old, _xgb_old, _meta_old = models[_i]
            models[_i] = ([], _xgb_old, _meta_old)  # обнуляем ссылки на LSTM-тензоры
        torch.cuda.empty_cache()

    avg_rmse = np.mean(rmses)
    avg_mae = np.mean(maes)
    avg_r2 = np.mean(r2s)
    logger.info(f"Average Metrics: RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}, R2={avg_r2:.4f}")

    # Оптимизация alpha Ridge на накопленных OOS-предсказаниях всех сплитов
    oos_meta_all = np.vstack(oos_meta_list)
    oos_y_all = np.concatenate(oos_y_list)
    logger.info("Optimizing Ridge alpha on OOS predictions (Optuna, 20 trials)...")
    best_alpha = optimize_ridge_alpha(oos_meta_all, oos_y_all, n_trials=20)
    logger.info(f"[OK] Best Ridge alpha: {best_alpha:.4f}")
    if last_meta_train is not None:
        best_meta = Ridge(alpha=best_alpha, positive=True)
        best_meta.fit(last_meta_train, last_y_train)
        logger.info(f"[OK] Final Ridge retrained | LSTM={best_meta.coef_[0]:.3f}, XGB={best_meta.coef_[1]:.3f}")
        lstm_ms, xgb_m, _ = models[-1]
        models[-1] = (lstm_ms, xgb_m, best_meta)

    # CI-модели обучаются на OOS-остатках всех walk-forward сплитов.
    # OOS-остатки (actual − pred на тестовых окнах) честно отражают погрешность
    # модели на невиданных данных, в отличие от in-sample остатков.
    logger.info("Training CI quantile models on OOS residuals...")
    oos_meta = np.vstack(oos_meta_list)
    oos_residuals = np.concatenate(oos_residuals_list)

    # Центрирование остатков: вычитаем медианный bias, чтобы CI отражал
    # разброс ошибок, а не их систематическое направление. Без этого
    # при наличии bias на калибровочной выборке точечный прогноз может
    # оказаться вне доверительного интервала.
    residual_bias = float(np.median(oos_residuals))
    oos_residuals_centered = oos_residuals - residual_bias
    logger.info(f"Residual bias (median): {residual_bias:.6f}")

    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_lags = min(20, len(oos_residuals_centered) // 2)
    if lb_lags >= 2:
        lb = acorr_ljungbox(oos_residuals_centered, lags=lb_lags, return_df=True)
        min_pval = lb['lb_pvalue'].min()
        # Сначала проверяем гетероскедастичность (LB на |res| ≈ ARCH-тест) —
        # волатильность кластерами это норма для финансовых рядов, не баг.
        lb_abs = acorr_ljungbox(np.abs(oos_residuals_centered), lags=lb_lags, return_df=True)
        abs_min_pval = lb_abs['lb_pvalue'].min()
        # Лаг минимума автокорреляции: короткие (1–3) = momentum, длинные = месячная структура
        worst_lag = int(lb['lb_pvalue'].idxmin())
        if min_pval < 0.05:
            if abs_min_pval < 0.05 and worst_lag >= 5:
                # классический ARCH-эффект: автокорр на длинных лагах + в |res|
                lb_flag = "ARCH-эффект (норма для фин. рядов; CI постоянной ширины)"
            elif worst_lag <= 3:
                lb_flag = f"momentum в среднем (lag={worst_lag}) — структура не учтена"
            else:
                lb_flag = f"автокорр на lag={worst_lag} (длинная структура / макро-лаг)"
        else:
            lb_flag = "остатки некоррелированы"
        lb_msg = f"Ljung-Box ({lb_lags} лагов): p-min={min_pval:.3f} [{lb_flag}]"
        # топ-3 «худших» лага: где автокорреляция самая значимая
        worst = lb.nsmallest(3, 'lb_pvalue')
        worst_str = ", ".join(f"lag={int(idx)} p={row.lb_pvalue:.3f}"
                              for idx, row in worst.iterrows())
        lb_top_msg = f"  топ-3 лагов: {worst_str}"
        abs_flag = ("гетероскедастичность (волатильность кластерами)"
                    if abs_min_pval < 0.05 else "дисперсия стационарна")
        lb_abs_msg = f"  Ljung-Box на |res|: p-min={abs_min_pval:.3f} [{abs_flag}]"
        # per-fold: чисто ли внутри каждого сплита, или проблема в склейке
        fold_pvals = []
        for i, fold_res in enumerate(oos_residuals_list, 1):
            fr = np.asarray(fold_res, dtype=float)
            fr = fr - np.median(fr)
            fl = min(20, len(fr) // 2)
            if fl >= 2:
                lb_f = acorr_ljungbox(fr, lags=fl, return_df=True)
                fold_pvals.append(f"fold{i}: p-min={lb_f['lb_pvalue'].min():.3f} (n={len(fr)})")
            else:
                fold_pvals.append(f"fold{i}: skip (n={len(fr)})")
        lb_fold_msg = "  per-fold: " + " | ".join(fold_pvals)
    else:
        lb_msg = f"Ljung-Box: недостаточно данных ({len(oos_residuals_centered)} остатков)"
        lb_top_msg = lb_abs_msg = lb_fold_msg = None
    print(lb_msg)
    logger.info(lb_msg)
    for _msg in (lb_top_msg, lb_abs_msg, lb_fold_msg):
        if _msg:
            print(_msg)
            logger.info(_msg)

    meta_q_oos, res_q, lower_alpha, upper_alpha = _get_ci_params(
        ci_mode, oos_meta, oos_residuals_centered
    )
    best_lower_model, best_upper_model = _train_quantile_pair(
        meta_features=meta_q_oos,
        residuals=res_q,
        lower_alpha=lower_alpha,
        upper_alpha=upper_alpha,
        base_params=ci_params,
    )
    logger.info("✓ CI quantile models trained")

    best_lstm_ensemble, best_xgb_model, best_meta_learner = models[-1]

    logger.info("Generating forecasts...")
    forecasts = {}
    confidence_intervals = {}

    if backtest_mode and backtest_date:
        base_date = datetime.strptime(backtest_date, '%Y-%m-%d')
    else:
        # Используем последнюю дату в данных, а не дату запуска:
        # end_date — сегодня, но MOEX возвращает данные до вчера (рынок ещё не закрылся).
        # Если взять end_date, next_business_day пропустит сегодня и прогноз начнётся с послезавтра.
        base_date = pd.to_datetime(data['Date'].max()).to_pydatetime()

    last_features = data[features].tail(LSTM_LOOK_BACK)
    last_scaled = scaler.transform(last_features)
    last_scaled_df = pd.DataFrame(last_scaled, columns=features, index=last_features.index)

    # LSTM-вход — только подмножество фич (см. _LSTM_FEATURES)
    _lstm_input = last_scaled_df[lstm_features].values.reshape(1, LSTM_LOOK_BACK, len(lstm_features))
    lstm_pred_scaled = float(np.mean(
        [predict_lstm_torch(_m, _lstm_input)[0] for _m in best_lstm_ensemble]
    ))

    current_scaled_flat = last_scaled.reshape(1, -1)
    xgb_pred_scaled = best_xgb_model.predict(current_scaled_flat)[0]

    meta_input = np.array([[lstm_pred_scaled, xgb_pred_scaled]])
    pred_close_scaled = best_meta_learner.predict(meta_input)[0]
    pred_close = close_scaler.inverse_transform([[pred_close_scaled]])[0][0]

    lower_residual = best_lower_model.predict(meta_input)[0]
    upper_residual = best_upper_model.predict(meta_input)[0]
    lower = close_scaler.inverse_transform([[pred_close_scaled + lower_residual]])[0][0]
    upper = close_scaler.inverse_transform([[pred_close_scaled + upper_residual]])[0][0]
    if lower > upper:
        lower, upper = upper, lower

    forecasts[horizon] = [pred_close]
    confidence_intervals[horizon] = ([lower], [upper])

    forecast_dates = _forecast_dates_for_horizon(base_date, horizon)

    logger.info(
        f"Horizon {horizon}: Forecast={pred_close:.2f}, "
        f"Lower CI={lower:.2f}, Upper CI={upper:.2f}"
    )

    return (
        data,
        close_scaler.inverse_transform(y[-len(final_pred):].reshape(-1, 1)).flatten(),
        final_pred,
        forecasts,
        forecast_dates,
        confidence_intervals,
        avg_rmse,
        avg_mae,
        avg_r2,
        scaler,
        close_scaler,
        features
    )

# ============================================================================
# BACKTESTING FUNCTIONS
# ============================================================================

def run_backtest(data, ticker, backtest_date, best_lstm_params, best_xgb_params, ci_mode: str = 'wide', macro_data=None, div_data=None):
    """
    Запуск бэктеста: обучение до backtest_date, прогноз на следующие дни,
    сравнение с реальными данными.
    """
    logger.info("\n" + "="*60)
    logger.info("STARTING BACKTEST MODE")
    logger.info("="*60)
    
    # Находим индекс даты бэктеста
    backtest_dt = pd.to_datetime(backtest_date)
    available_data = data[data['Date'] <= backtest_dt]
    future_data = data[data['Date'] > backtest_dt].copy()
    base_price = float(available_data['Close'].iloc[-1]) if not available_data.empty else None
    
    if future_data.empty:
        logger.error(f"No future data available after {backtest_date}")
        return None
    
    logger.info(f"Training data until: {available_data['Date'].max()}")
    logger.info(f"Available future data: {len(future_data)} trading days")
    logger.info(f"Future data dates: {future_data['Date'].min()} to {future_data['Date'].max()}")
    
    # Обучаем модель на данных до backtest_date для трех горизонтов
    macro_start = data['Date'].min().strftime('%Y-%m-%d')
    if macro_data is None:
        logger.info("Loading macro data once for all horizons...")
        macro_data = load_macro_data(macro_start, backtest_date)
    if div_data is None:
        logger.info(f"Loading dividend features once for all horizons ({ticker})...")
        div_data = load_dividend_features(ticker, macro_start, backtest_date)
    logger.info("Loading fundamental data once for all horizons...")
    shared_funds = tinkoff_loader.get_fundamentals(ticker) if tinkoff_loader else {}

    all_results = {}
    for h in [1, 2, 3]:
        all_results[h] = prepare_and_train_model(
            data, ticker, backtest_date,
            best_lstm_params, best_xgb_params,
            backtest_mode=True, backtest_date=backtest_date,
            horizon=h,
            ci_mode=ci_mode,
            macro_data=macro_data,
            fund_data=shared_funds,
            div_data=div_data,
        )

    merged = merge_horizon_results(all_results)
    if merged is None:
        return None

    forecasts, confidence_intervals = merged

    data_train, real_prices, final_pred, _, _, _, \
        rmse_val, mae_val, r2_val, scaler, close_scaler, features = all_results[1]
    forecast_dates = _forecast_dates_for_horizon(
        datetime.strptime(backtest_date, '%Y-%m-%d'), 3
    )

    # Сравниваем прогнозы с реальными данными
    backtest_results = []
    
    # ИСПРАВЛЕНИЕ: создаем словарь для быстрого поиска реальных цен по дате
    future_data['Date_key'] = future_data['Date'].dt.date
    real_prices_dict = dict(zip(future_data['Date_key'], future_data['Close']))
    
    _dates = list(real_prices_dict.keys())
    logger.info(f"\nAvailable real data dates: {len(_dates)} дат, первая {_dates[0]}, последняя {_dates[-1]}" if _dates else "\nAvailable real data dates: пусто")
    
    for horizon in [1, 2, 3]:
        if horizon - 1 >= len(forecast_dates):
            logger.warning(f"Forecast date not generated for {horizon}-day horizon")
            continue
            
        forecast_date = forecast_dates[horizon-1]
        forecast_price = forecasts[horizon][-1]
        lower_ci = confidence_intervals[horizon][0][-1]
        upper_ci = confidence_intervals[horizon][1][-1]
        
        logger.info(f"\nChecking {horizon}-day forecast for date: {forecast_date.strftime('%Y-%m-%d')}")
        
        # ИСПРАВЛЕНИЕ: ищем ближайшую доступную торговую дату
        forecast_date_key = forecast_date.date()
        
        # Если точная дата есть в данных
        if forecast_date_key in real_prices_dict:
            real_price = real_prices_dict[forecast_date_key]
            actual_date = forecast_date
        else:
            # Ищем ближайшую следующую торговую дату
            logger.warning(f"  Exact date {forecast_date_key} not found (likely weekend/holiday)")
            future_dates_after = [d for d in real_prices_dict.keys() if d > forecast_date_key]
            
            if future_dates_after:
                actual_date_key = min(future_dates_after)
                real_price = real_prices_dict[actual_date_key]
                actual_date = pd.Timestamp(actual_date_key)
                logger.info(f"  Using next trading day: {actual_date_key}")
            else:
                logger.warning(f"  No trading data available after {forecast_date_key}")
                continue
        
        # Вычисляем метрики модели
        error = abs(forecast_price - real_price)
        error_pct = (error / real_price) * 100
        in_ci = lower_ci <= real_price <= upper_ci
        forecast_dir = 'up' if (base_price is None or forecast_price >= base_price) else 'down'
        real_dir     = 'up' if (base_price is None or real_price     >= base_price) else 'down'
        dir_correct  = forecast_dir == real_dir

        # Naïve baseline: предсказывает последнюю известную цену без изменений
        naive_price = base_price if base_price is not None else real_price
        naive_error = abs(naive_price - real_price)
        naive_error_pct = (naive_error / real_price) * 100

        # Доходности относительно базовой цены (для R² on returns и IC)
        model_return_pct = (forecast_price - naive_price) / naive_price * 100 if naive_price else 0.0
        real_return_pct  = (real_price    - naive_price) / naive_price * 100 if naive_price else 0.0

        backtest_results.append({
            'horizon': horizon,
            'forecast_date': forecast_date,
            'actual_date': actual_date,
            'forecast': forecast_price,
            'real': real_price,
            'error': error,
            'error_pct': error_pct,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'in_ci': in_ci,
            'forecast_dir': forecast_dir,
            'real_dir': real_dir,
            'dir_correct': dir_correct,
            'naive_price': naive_price,
            'naive_error': naive_error,
            'naive_error_pct': naive_error_pct,
            'model_return_pct': model_return_pct,
            'real_return_pct': real_return_pct,
        })

        logger.info(f"  Forecast date: {forecast_date.strftime('%Y-%m-%d')}")
        logger.info(f"  Actual date used: {actual_date.strftime('%Y-%m-%d')}")
        logger.info(f"  Forecast: {forecast_price:.2f} RUB")
        logger.info(f"  Real: {real_price:.2f} RUB")
        logger.info(f"  Error: {error:.2f} RUB ({error_pct:.2f}%)")
        logger.info(f"  Naïve: {naive_price:.2f} RUB | Error: {naive_error:.2f} RUB ({naive_error_pct:.2f}%)")
        logger.info(f"  CI: [{lower_ci:.2f}, {upper_ci:.2f}]")
        logger.info(f"  Real price in CI: {'✓ YES' if in_ci else '✗ NO'}")
        logger.info(f"  Direction: forecast={forecast_dir}, real={real_dir} {'[ok]' if dir_correct else '[no]'}")
    
    if not backtest_results:
        logger.error("No backtest results generated - no matching dates found")
        return None
    
    return backtest_results, forecasts, forecast_dates, confidence_intervals, all_results

# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

_BENCHMARKS_HEADER = """\
# Benchmarks — SBER, 2024-10-14

## Методология

- Тикер и дата зафиксированы, не меняются между версиями.
- Гиперпараметры: дефолтные из `config.py` (не Optuna).
  Причина: бенчмарк измеряет качество кода, а не подбора параметров.
  Optuna-параметры меняются между прогонами и делают сравнение версий
  непрозрачным; дефолты воспроизводимы без предварительной оптимизации.
- CI-режим: wide (5/95, полная история).

## Результаты

| Версия | Дата запуска | RMSE h1/h2/h3 | avg RMSE | MAE h1/h2/h3 | avg MAE | R² h1/h2/h3 | avg R² | Прогноз h1/h2/h3 | Реал h1/h2/h3 | Ошибка% h1/h2/h3 | avg Ошибка% | CI coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""


def append_benchmark_result(metrics: dict):
    from pathlib import Path as _Path
    path = _Path(BENCHMARKS_FILE)
    if not path.exists():
        path.write_text(_BENCHMARKS_HEADER, encoding='utf-8')
    h = metrics['horizons']
    row = (
        f"| {metrics['version']} "
        f"| {metrics['run_at']} "
        f"| {h[1]['rmse']:.2f}/{h[2]['rmse']:.2f}/{h[3]['rmse']:.2f} "
        f"| {metrics['avg_rmse']:.2f} "
        f"| {h[1]['mae']:.2f}/{h[2]['mae']:.2f}/{h[3]['mae']:.2f} "
        f"| {metrics['avg_mae']:.2f} "
        f"| {h[1]['r2']:.3f}/{h[2]['r2']:.3f}/{h[3]['r2']:.3f} "
        f"| {metrics['avg_r2']:.3f} "
        f"| {h[1]['forecast']:.1f}/{h[2]['forecast']:.1f}/{h[3]['forecast']:.1f} "
        f"| {h[1]['real']:.1f}/{h[2]['real']:.1f}/{h[3]['real']:.1f} "
        f"| {h[1]['error_pct']:.1f}%/{h[2]['error_pct']:.1f}%/{h[3]['error_pct']:.1f}% "
        f"| {metrics['avg_error_pct']:.1f}% "
        f"| {round(metrics['ci_coverage']):.0f}% |\n"
    )
    with open(path, 'a', encoding='utf-8') as f:
        f.write(row)
    return path


def run_benchmark(data) -> dict:
    logger.info("=" * 60)
    logger.info("BENCHMARK MODE: SBER / %s / ci=%s / default params", BENCHMARK_DATE, BENCHMARK_CI_MODE)
    logger.info("=" * 60)

    lstm_p, xgb_p = get_default_hyperparams()
    backtest_return = run_backtest(
        data, BENCHMARK_TICKER, BENCHMARK_DATE,
        lstm_p, xgb_p, ci_mode=BENCHMARK_CI_MODE,
    )

    if backtest_return is None or not backtest_return[0]:
        raise RuntimeError("Benchmark failed: run_backtest returned no results")

    results_list, _, _, _, all_results = backtest_return

    horizons = {}
    for res in results_list:
        h = res['horizon']
        horizons[h] = {
            'forecast':    res['forecast'],
            'real':        res['real'],
            'error_pct':   res['error_pct'],
            'in_ci':       res['in_ci'],
            'dir_correct': res['dir_correct'],
            'forecast_dir': res['forecast_dir'],
            'real_dir':    res['real_dir'],
            'rmse':        all_results[h][6],
            'mae':         all_results[h][7],
            'r2':          all_results[h][8],
        }

    avg_rmse      = sum(horizons[h]['rmse']        for h in [1, 2, 3]) / 3
    avg_mae       = sum(horizons[h]['mae']         for h in [1, 2, 3]) / 3
    avg_r2        = sum(horizons[h]['r2']          for h in [1, 2, 3]) / 3
    avg_error_pct = sum(horizons[h]['error_pct']   for h in [1, 2, 3]) / 3
    ci_coverage   = sum(horizons[h]['in_ci']       for h in [1, 2, 3]) / 3 * 100
    dir_accuracy  = sum(horizons[h]['dir_correct'] for h in [1, 2, 3]) / 3 * 100

    return {
        'version':       MODEL_VERSION,
        'run_at':        datetime.now().strftime('%Y-%m-%d %H:%M'),
        'horizons':      horizons,
        'avg_rmse':      avg_rmse,
        'avg_mae':       avg_mae,
        'avg_r2':        avg_r2,
        'avg_error_pct': avg_error_pct,
        'ci_coverage':   ci_coverage,
        'dir_accuracy':  dir_accuracy,
    }

# ============================================================================
# USER INTERACTION
# ============================================================================

def get_user_inputs():
    """Собираем все параметры от пользователя в начале программы"""
    print("\n" + "="*60)
    print("STOCK PRICE FORECASTING MODEL v16")
    print("="*60)

    # 0. Benchmark mode
    print("\n0. Режим бенчмарка:")
    print("   [1] Нет (обычный режим)")
    print("   [2] Да (воспроизводимый прогон для отслеживания качества)")
    bm_choice = input("   Выбор [1/2, по умолчанию 1]: ").strip() or "1"
    if bm_choice == "2":
        print("\nПАРАМЕТРЫ БЕНЧМАРКА:")
        print(f"  Тикер:           {BENCHMARK_TICKER}")
        print(f"  Дата бэктеста:   {BENCHMARK_DATE}")
        print(f"  CI-режим:        {BENCHMARK_CI_MODE} (5/95, вся история)")
        print(f"  Гиперпараметры:  дефолтные из config.py")
        print(f"  Примерное время: {BENCHMARK_APPROX_TIME}")
        confirm = input("   Продолжить? (y/n, по умолчанию y): ").strip().lower()
        if confirm == 'n':
            print("Выход из программы.")
            exit(0)
        return {'benchmark_mode': True}

    # 1. Тикер
    ticker_input = input(
        "\n1. Тикер (например: SBER, LKOH, GAZP) [по умолчанию SBER]: "
    ).strip().upper()
    ticker = ticker_input if ticker_input else 'SBER'

    # 2. Режим работы
    print("\n2. Режим работы:")
    print("   [1] Прогноз (на будущее)")
    print("   [2] Бэктест (проверка прогноза на исторических данных)")
    mode_input = input("   Выбор [1/2, по умолчанию 1]: ").strip()
    backtest_mode = mode_input == '2'

    backtest_date = None
    if backtest_mode:
        default_backtest = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        backtest_input = input(
            f"   Дата бэктеста (YYYY-MM-DD) [по умолчанию {default_backtest}]: "
        ).strip()
        backtest_date = backtest_input if backtest_input else default_backtest
        print(f"   Бэктест: обучение до {backtest_date}, проверка прогноза после")

    # 3. Гиперпараметры модели
    print("\n3. Гиперпараметры модели:")
    saved_lstm, saved_xgb = load_hyperparams(ticker)

    if saved_lstm and saved_xgb:
        print(f"   Найдены сохранённые параметры для {ticker}:")
        print(f"   LSTM: units={saved_lstm['units']}, dropout={saved_lstm['dropout']:.4f}, lr={saved_lstm['lr']:.6f}")
        print(f"   XGBoost: n_est={saved_xgb['n_estimators']}, depth={saved_xgb['max_depth']}, lr={saved_xgb['learning_rate']:.4f}")
        use_saved = input("   Использовать сохранённые параметры? (y/n, по умолчанию y): ").strip().lower()
        optimize = use_saved == 'n'
    else:
        print(f"   Сохранённые параметры для {ticker} не найдены.")
        optimize_input = input("   Запустить Optuna для поиска оптимальных параметров? (y/n, по умолчанию y): ").strip().lower()
        optimize = optimize_input != 'n'

    if optimize:
        trials_input = input("   Количество итераций Optuna [по умолчанию 20]: ").strip()
        n_trials = int(trials_input) if trials_input.isdigit() else 20
    else:
        n_trials = 0

    # 4. Визуализация
    print("\n4. Визуализация:")
    show_plot = input("   Показывать график после обучения? (y/n, по умолчанию y): ").strip().lower() != 'n'
    show_ci   = input("   Показывать доверительные интервалы на графике? (y/n, по умолчанию y): ").strip().lower() != 'n'

    # 5. Режим доверительных интервалов
    print("\n5. Режим доверительных интервалов:")
    print("   [1] Широкий — 5/95 перцентили, полная история (учитывает кризисы 2022)")
    print("   [2] Узкий   — 25/75 перцентили, последние 3 года (актуальная волатильность)")
    ci_mode_input = input("   Выбор [1/2, по умолчанию 1]: ").strip()
    ci_mode = 'narrow' if ci_mode_input == '2' else 'wide'

    # 6. Презентационный режим
    print("\n6. Презентационный режим:")
    pres_input = input("   Построить дополнительный упрощённый график для слайдов? (y/n, по умолчанию n): ").strip().lower()
    presentation_mode = pres_input == 'y'

    if presentation_mode:
        raw = input("   Окно истории на графике, торговых дней [по умолчанию 90]: ").strip()
        history_window = int(raw) if raw.isdigit() else 90
    else:
        history_window = 90

    print("\n" + "="*60)
    print("ПАРАМЕТРЫ ЗАПУСКА:")
    print(f"  Тикер: {ticker}")
    print(f"  Режим: {'Бэктест' if backtest_mode else 'Прогноз'}")
    if backtest_mode:
        print(f"  Дата бэктеста: {backtest_date}")
    print(f"  Оптимизация: {'Да (' + str(n_trials) + ' итераций)' if optimize else 'Нет (используются сохранённые/дефолтные)'}")
    print(f"  Показать график: {'Да' if show_plot else 'Нет'}")
    print(f"  Доверительные интервалы: {'Да' if show_ci else 'Нет'}")
    ci_label = 'Узкий (25/75, последние 3 года)' if ci_mode == 'narrow' else 'Широкий (5/95, вся история)'
    print(f"  Режим CI: {ci_label}")
    print(f"  Презентационный режим: {'Да (окно ' + str(history_window) + ' дней)' if presentation_mode else 'Нет'}")
    print("="*60)

    confirm = input("\nПродолжить с этими параметрами? (y/n, по умолчанию y): ").strip().lower()
    if confirm == 'n':
        print("Выход из программы.")
        exit(0)

    return {
        'ticker': ticker,
        'backtest_mode': backtest_mode,
        'backtest_date': backtest_date,
        'optimize': optimize,
        'n_trials': n_trials,
        'show_ci': show_ci,
        'show_plot': show_plot,
        'ci_mode': ci_mode,
        'presentation_mode': presentation_mode,
        'history_window': history_window,
        'benchmark_mode': False,
    }

# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stock price forecasting with LSTM + XGBoost ensemble'
    )
    parser.add_argument('--no-gui', action='store_true', help='Headless mode (no plots)')
    parser.add_argument('--ticker', type=str, default=None, help='Ticker symbol')
    parser.add_argument('--backtest', type=str, default=None, help='Backtest date (YYYY-MM-DD)')
    parser.add_argument('--optimize', action='store_true', help='Run Optuna optimization')
    parser.add_argument('--trials', type=int, default=20, help='Optuna trials')
    parser.add_argument('--ci-mode', choices=['wide', 'narrow'], default='wide',
                        help='CI mode: wide=5/95 full history, narrow=25/75 last 3y')
    parser.add_argument('--presentation', action='store_true',
                        help='Построить дополнительный график в презентационном стиле')
    parser.add_argument('--history-window', type=int, default=90,
                        help='Окно истории для презентационного графика, торговых дней (по умолчанию 90)')
    parser.add_argument('--benchmark', action='store_true',
                        help=f'Запустить воспроизводимый бенчмарк ({BENCHMARK_TICKER}, дефолтные параметры)')
    args = parser.parse_args()

    if args.no_gui or args.benchmark:
        try:
            matplotlib.use('Agg')
        except Exception:
            pass

    import matplotlib.pyplot as plt

    # Получаем параметры от пользователя (если не заданы через CLI)
    if args.benchmark:
        user_params = {'benchmark_mode': True}
    elif args.ticker:
        # CLI mode
        user_params = {
            'ticker': args.ticker.upper(),
            'backtest_mode': bool(args.backtest),
            'backtest_date': args.backtest,
            'optimize': args.optimize,
            'n_trials': args.trials,
            'show_ci': False,
            'show_plot': not args.no_gui,
            'ci_mode': args.ci_mode,
            'presentation_mode': args.presentation,
            'history_window': args.history_window,
        }
    else:
        # Interactive mode
        user_params = get_user_inputs()

    benchmark_mode = user_params.get('benchmark_mode', False)

    if benchmark_mode:
        start_date = BENCHMARK_START_DATE
        end_date   = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"Loading data for benchmark: {BENCHMARK_TICKER} {start_date}–{end_date}")
        bm_data = load_stock_data_moex_test(BENCHMARK_TICKER, start_date, end_date)
        if bm_data is None or bm_data.empty:
            print(f"Не удалось загрузить данные для бенчмарка ({BENCHMARK_TICKER}).")
            exit(1)

        metrics = run_benchmark(bm_data)
        out_path = append_benchmark_result(metrics)

        print("\n" + "=" * 60)
        print(f"BENCHMARK RESULTS — {metrics['version']}")
        print("=" * 60)
        for h in [1, 2, 3]:
            hd = metrics['horizons'][h]
            ci_mark  = "[ok]" if hd['in_ci']       else "[no]"
            dir_mark = "[ok]" if hd['dir_correct']  else "[no]"
            print(f"  h={h}: прогноз {hd['forecast']:.2f} / реал {hd['real']:.2f} "
                  f"/ ошибка {hd['error_pct']:.2f}% / CI {ci_mark} "
                  f"/ направление {hd['forecast_dir']} {dir_mark}")
        print(f"  avg RMSE: {metrics['avg_rmse']:.2f} | "
              f"avg MAE: {metrics['avg_mae']:.2f} | "
              f"avg R2: {metrics['avg_r2']:.3f} | "
              f"avg Ошибка%: {metrics['avg_error_pct']:.2f}%")
        print(f"  CI coverage: {metrics['ci_coverage']:.0f}% | "
              f"Direction Accuracy: {metrics['dir_accuracy']:.0f}%")
        print(f"\nРезультаты записаны: {out_path}")
        print("=" * 60)
        exit(0)

    ticker = user_params['ticker']
    backtest_mode = user_params['backtest_mode']
    backtest_date = user_params['backtest_date']
    optimize = user_params['optimize']
    n_trials = user_params['n_trials']
    show_ci = user_params['show_ci']
    show_plot = user_params['show_plot']
    ci_mode = user_params.get('ci_mode', 'wide')

    # Определяем даты загрузки данных
    start_date = '2014-01-01'
    if backtest_mode:
        # Загружаем данные до текущей даты, чтобы было с чем сравнить
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Data load range: {start_date} to {end_date}")

    logger.info("\n" + "#" * 60)
    logger.info(f"# PROCESSING {ticker}")
    logger.info("#" * 60)

    data = load_stock_data_moex_test(ticker, start_date, end_date)
    if data is None or data.empty:
        print(f"Тикер '{ticker}' не найден на MOEX или данные не загружены.")
        exit(1)

    ticker_name = get_ticker_shortname(ticker)

    logger.info("Loading fundamental data once for all horizons...")
    shared_funds = tinkoff_loader.get_fundamentals(ticker) if tinkoff_loader else {}

    # Загружаем macro+div один раз для всего прогона (Optuna + обучение + бэктест)
    _shared_start = data['Date'].min().strftime('%Y-%m-%d')
    _shared_end   = backtest_date if backtest_mode else end_date
    logger.info("Loading macro data (once for entire run)...")
    shared_macro = load_macro_data(_shared_start, _shared_end)
    logger.info(f"Loading dividend features (once for entire run, {ticker})...")
    shared_div = load_dividend_features(ticker, _shared_start, _shared_end)

    # Получаем или оптимизируем гиперпараметры
    if optimize:
        logger.info("\n" + "="*60)
        logger.info("RUNNING OPTUNA OPTIMIZATION")
        logger.info("="*60)
        
        # Подготовка данных для оптимизации
        data_for_opt = update_technical_indicators(data.copy())

        for key, value in shared_funds.items():
            data_for_opt[key] = value

        data_for_opt = data_for_opt.infer_objects(copy=False).fillna(0)

        for col in ['market_cap', 'roe', 'dividend_yield', 'pe_ratio', 'pb_ratio', 'beta']:
            if col not in data_for_opt.columns:
                data_for_opt[col] = 0.0

        data_for_opt['Date'] = pd.to_datetime(data_for_opt['Date']).dt.normalize()
        data_for_opt = data_for_opt.merge(shared_macro, on='Date', how='left')
        _opt_div_norm = shared_div.copy()
        _opt_div_norm['Date'] = pd.to_datetime(_opt_div_norm['Date']).dt.normalize()
        data_for_opt = data_for_opt.merge(_opt_div_norm, on='Date', how='left')
        data_for_opt = data_for_opt.ffill().bfill()

        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width',
            'ATR_14', 'Stoch_K', 'Stoch_D', 'ADX_14', 'Momentum_10',
            'Price_Change_1', 'Price_Change_5',
            # --- Сезонность ---
            'day_of_week', 'month', 'quarter',
            # --- Относительные признаки (v16) ---
            'Close_to_SMA10', 'Close_to_EMA20', 'BB_Position', 'Close_ZScore_20',
            # --- Фундаментальные (snapshot из Tinkoff Invest API) ---
            'market_cap', 'roe', 'dividend_yield', 'pe_ratio', 'pb_ratio', 'beta',
            # --- Макроэкономические (time-varying) ---
            *[c for c in _MACRO_COLS if c in data_for_opt.columns and not data_for_opt[c].isna().all()],
            # --- Дивидендные (time-varying) ---
            *[c for c in _FUND_DIV_COLS if c in data_for_opt.columns and not data_for_opt[c].isna().all()],
        ]

        # Granger-скрининг для Optuna: те же правила, что и в основной ветке
        from statsmodels.tsa.stattools import grangercausalitytests as _gct
        for _mc in [c for c in _MACRO_COLS + _FUND_DIV_COLS if c in features]:
            _col_data = data_for_opt[_mc].dropna()
            if _col_data.empty or _col_data.std() == 0:
                features.remove(_mc)
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    _gr = _gct(data_for_opt[['Close', _mc]].dropna(), maxlag=5)
                _p  = min(r[0]['ssr_ftest'][1] for r in _gr.values())
                if _p >= 0.05:
                    features.remove(_mc)
            except Exception:
                features.remove(_mc)

        # В режиме бэктеста оптимизируем только на данных до backtest_date
        if backtest_mode:
            backtest_dt = pd.to_datetime(backtest_date)
            data_for_opt = data_for_opt[data_for_opt['Date'] <= backtest_dt]

        data_for_opt = data_for_opt[['Date'] + features].dropna()

        scaler_opt = MinMaxScaler()
        scaled_data_opt = scaler_opt.fit_transform(data_for_opt[features])
        scaled_df_opt = pd.DataFrame(scaled_data_opt, columns=features, index=data_for_opt.index)

        X_opt, y_opt = [], []
        for i in range(LSTM_LOOK_BACK, len(scaled_df_opt)):
            X_opt.append(scaled_df_opt.iloc[i-LSTM_LOOK_BACK:i].values)
            y_opt.append(scaled_df_opt['Close'].iloc[i])
        X_opt, y_opt = np.array(X_opt), np.array(y_opt)

        logger.info(f"Optimizing LSTM params with Optuna ({n_trials} trials)...")
        best_lstm_params = optimize_lstm_params(X_opt, y_opt, n_trials=n_trials)

        logger.info(f"Optimizing XGBoost params with Optuna ({n_trials} trials)...")
        best_xgb_params = optimize_xgboost_params(X_opt.reshape(X_opt.shape[0], -1), y_opt, n_trials=n_trials)
        
        # Сохраняем найденные параметры
        save_hyperparams(ticker, best_lstm_params, best_xgb_params)
    else:
        # Загружаем сохраненные или используем дефолтные
        saved_lstm, saved_xgb = load_hyperparams(ticker)
        if saved_lstm and saved_xgb:
            best_lstm_params, best_xgb_params = saved_lstm, saved_xgb
            logger.info("Using saved hyperparameters")
        else:
            best_lstm_params, best_xgb_params = get_default_hyperparams()
            logger.info("Using default hyperparameters")

    # Основной запуск модели
    if backtest_mode:
        # Режим бэктеста
        backtest_results = run_backtest(
            data, ticker, backtest_date, best_lstm_params, best_xgb_params,
            ci_mode=ci_mode, macro_data=shared_macro, div_data=shared_div,
        )

        if backtest_results and backtest_results[0]:
            results_list, forecasts, forecast_dates, confidence_intervals, _ = backtest_results
            print("\n" + "="*60)
            print("BACKTEST RESULTS SUMMARY")
            print("="*60)
            for res in results_list:
                print(f"\n{res['horizon']}-day forecast:")
                if res['forecast_date'] != res['actual_date']:
                    print(f"  Target date:  {res['forecast_date'].strftime('%Y-%m-%d')} (weekend/holiday)")
                    print(f"  Actual date:  {res['actual_date'].strftime('%Y-%m-%d')} (next trading day)")
                else:
                    print(f"  Date:         {res['forecast_date'].strftime('%Y-%m-%d')}")
                print(f"  Real:         {res['real']:.2f} RUB")
                print(f"  Forecast:     {res['forecast']:.2f} RUB  | Error: {res['error']:.2f} RUB ({res['error_pct']:.2f}%)")
                print(f"  Naive:        {res['naive_price']:.2f} RUB  | Error: {res['naive_error']:.2f} RUB ({res['naive_error_pct']:.2f}%)")
                print(f"  Return:       model {res['model_return_pct']:+.2f}%  |  real {res['real_return_pct']:+.2f}%")
                print(f"  CI:           [{res['lower_ci']:.2f}, {res['upper_ci']:.2f}]")
                print(f"  In CI:        {'YES' if res['in_ci'] else 'NO'}")
                dir_mark = "[ok]" if res['dir_correct'] else "[no]"
                print(f"  Direction:    {res['forecast_dir']} (real: {res['real_dir']}) {dir_mark}")

            avg_error = np.mean([r['error'] for r in results_list])
            avg_error_pct = np.mean([r['error_pct'] for r in results_list])
            avg_naive_error = np.mean([r['naive_error'] for r in results_list])
            avg_naive_error_pct = np.mean([r['naive_error_pct'] for r in results_list])
            ci_coverage = sum([r['in_ci'] for r in results_list]) / len(results_list) * 100
            dir_accuracy = sum([r['dir_correct'] for r in results_list]) / len(results_list) * 100
            vs_naive = "лучше Naive" if avg_error_pct < avg_naive_error_pct else "хуже Naive"

            real_rets = [r['real_return_pct'] for r in results_list]
            model_rets = [r['model_return_pct'] for r in results_list]
            r2_on_rets = r2_score(real_rets, model_rets) if len(results_list) >= 2 else float('nan')
            ic_val = pd.Series(model_rets).corr(pd.Series(real_rets), method='spearman') if len(results_list) >= 2 else float('nan')

            print(f"\nAVERAGE METRICS:")
            print(f"  Model Error:    {avg_error:.2f} RUB ({avg_error_pct:.2f}%)")
            print(f"  Naive Error:    {avg_naive_error:.2f} RUB ({avg_naive_error_pct:.2f}%) | модель {vs_naive}")
            print(f"  R2 on returns:  {r2_on_rets:.3f}  (на {len(results_list)} горизонтах)")
            print(f"  IC (Spearman):  {ic_val:.3f}")
            print(f"  CI Coverage:    {ci_coverage:.1f}%")
            print(f"  Direction Accuracy: {dir_accuracy:.1f}%")
            print(f"  Total forecasts checked: {len(results_list)}")
            print("="*60)

            backtest_log_dir = os.path.join(MODEL_OUTPUT_DIR, 'backtest')
            os.makedirs(backtest_log_dir, exist_ok=True)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            backtest_log_file = os.path.join(
                backtest_log_dir,
                f'{ticker}_backtest_{backtest_date}_{timestamp_str}.txt'
            )

            with open(backtest_log_file, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("BACKTEST RESULTS\n")
                f.write("="*60 + "\n")
                f.write(f"Ticker: {ticker}\n")
                f.write(f"Backtest Date: {backtest_date}\n")
                f.write(f"Run Time: {datetime.now().isoformat()}\n")
                f.write(f"Model Version: {MODEL_VERSION}\n\n")
                f.write("HYPERPARAMETERS:\n")
                f.write(f"LSTM: units={best_lstm_params['units']}, dropout={best_lstm_params['dropout']:.4f}, lr={best_lstm_params['lr']:.6f}\n")
                f.write(f"XGBoost: n_est={best_xgb_params['n_estimators']}, depth={best_xgb_params['max_depth']}, lr={best_xgb_params['learning_rate']:.4f}\n\n")
                f.write("FORECAST vs REAL:\n")
                for res in results_list:
                    f.write(f"\n{res['horizon']}-day forecast:\n")
                    if res['forecast_date'] != res['actual_date']:
                        f.write(f"  Target date:  {res['forecast_date'].strftime('%Y-%m-%d')} (weekend/holiday)\n")
                        f.write(f"  Actual date:  {res['actual_date'].strftime('%Y-%m-%d')} (next trading day)\n")
                    else:
                        f.write(f"  Date:         {res['forecast_date'].strftime('%Y-%m-%d')}\n")
                    f.write(f"  Real:         {res['real']:.2f} RUB\n")
                    f.write(f"  Forecast:     {res['forecast']:.2f} RUB | Error: {res['error']:.2f} RUB ({res['error_pct']:.2f}%)\n")
                    f.write(f"  Naive:        {res['naive_price']:.2f} RUB | Error: {res['naive_error']:.2f} RUB ({res['naive_error_pct']:.2f}%)\n")
                    f.write(f"  Return:       model {res['model_return_pct']:+.2f}%  |  real {res['real_return_pct']:+.2f}%\n")
                    f.write(f"  CI:           [{res['lower_ci']:.2f}, {res['upper_ci']:.2f}]\n")
                    f.write(f"  In CI:        {'YES' if res['in_ci'] else 'NO'}\n")
                    dir_mark = "[ok]" if res['dir_correct'] else "[no]"
                    f.write(f"  Direction:    {res['forecast_dir']} (real: {res['real_dir']}) {dir_mark}\n")
                f.write(f"\nAVERAGE METRICS:\n")
                f.write(f"Model Error:  {avg_error:.2f} RUB ({avg_error_pct:.2f}%)\n")
                f.write(f"Naive Error:  {avg_naive_error:.2f} RUB ({avg_naive_error_pct:.2f}%) -- модель {vs_naive}\n")
                f.write(f"R2 on returns: {r2_on_rets:.3f}\n")
                f.write(f"IC (Spearman): {ic_val:.3f}\n")
                f.write(f"CI Coverage: {ci_coverage:.1f}%\n")
                f.write(f"Direction Accuracy: {dir_accuracy:.1f}%\n")
                f.write(f"Total forecasts checked: {len(results_list)}\n")

            logger.info(f"[OK] Backtest results saved: {backtest_log_file}")
        else:
            logger.error("Backtest failed: no results generated")

    else:
        # Обычный режим прогноза
        all_results = {}
        for h in [1, 2, 3]:
            all_results[h] = prepare_and_train_model(
                data, ticker, end_date,
                best_lstm_params, best_xgb_params,
                backtest_mode=False,
                horizon=h,
                ci_mode=ci_mode,
                macro_data=shared_macro,
                fund_data=shared_funds,
                div_data=shared_div,
            )

        merged = merge_horizon_results(all_results)
        if merged is not None:
            forecasts, confidence_intervals = merged

            data_res, real_prices, final_pred, _, _, _, \
                rmse_val, mae_val, r2_val, scaler, close_scaler, features = all_results[1]
            forecast_dates = _forecast_dates_for_horizon(
                datetime.strptime(end_date, '%Y-%m-%d'), 3
            )

            # Текущая известная цена — база для расчёта направления и Δ%
            last_close = float(data_res['Close'].iloc[-1])
            last_date  = data_res['Date'].iloc[-1].strftime('%d.%m.%Y')

            def _confidence_label(r2_val):
                """Качественная оценка уверенности по walk-forward R²."""
                if r2_val >= 0.85:
                    return "HIGH"
                if r2_val >= 0.70:
                    return "MED"
                return "LOW"

            print("\n" + "="*72)
            print(f"FORECAST SUMMARY: {ticker}  —  {ticker_name}")
            print(f"Текущая цена (база для Δ%): {last_close:.2f} RUB ({last_date})")
            print("="*72)
            for horizon in [1, 2, 3]:
                price    = forecasts[horizon][-1]
                date_str = forecast_dates[horizon-1].strftime('%d.%m.%Y')
                delta_pct = (price - last_close) / last_close * 100
                if   delta_pct >  0.05: direction = "UP  "
                elif delta_pct < -0.05: direction = "DOWN"
                else:                   direction = "FLAT"
                conf = _confidence_label(all_results[horizon][8])
                print(f"+{horizon}d  {date_str}  {price:7.2f} RUB  {delta_pct:+5.2f}%  {direction}  [{conf}]")
                if show_ci:
                    lower, upper = confidence_intervals[horizon][0][-1], confidence_intervals[horizon][1][-1]
                    print(f"        CI: [{lower:.2f} - {upper:.2f}]  ширина {upper-lower:.2f}")
            print(f"\nNaive baseline (без модели): {last_close:.2f} RUB для всех горизонтов")
            print("="*72)

            logger.info("\n" + "=" * 88)
            logger.info(f"FORECAST SUMMARY: {ticker} — {ticker_name}  (ci_mode={ci_mode})")
            logger.info(f"Текущая цена: {last_close:.2f} RUB ({last_date})  |  Naive baseline = {last_close:.2f} для всех горизонтов")
            logger.info("=" * 88)
            logger.info("%-5s  %-12s  %9s  %7s  %5s  %5s  %6s  %s",
                        "H", "Date", "Forecast", "Δ%", "Dir", "Conf", "R²", "CI [lower – upper]  width")
            logger.info("-" * 88)
            for h in [1, 2, 3]:
                h_price  = forecasts[h][-1]
                h_lower  = confidence_intervals[h][0][-1]
                h_upper  = confidence_intervals[h][1][-1]
                h_width  = h_upper - h_lower
                h_r2     = all_results[h][8]
                h_date   = all_results[h][4][-1].strftime('%d.%m.%Y')
                h_delta  = (h_price - last_close) / last_close * 100
                if   h_delta >  0.05: h_dir = "UP"
                elif h_delta < -0.05: h_dir = "DOWN"
                else:                 h_dir = "FLAT"
                h_conf   = _confidence_label(h_r2)
                ci_ok    = "✓" if h_lower <= h_price <= h_upper else "✗"
                logger.info("+%dd    %-12s  %9.2f  %+6.2f%%  %5s  %5s  %.3f  [%6.2f – %6.2f]  %5.2f %s",
                            h, h_date, h_price, h_delta, h_dir, h_conf, h_r2,
                            h_lower, h_upper, h_width, ci_ok)
            logger.info("=" * 88)

            graphs_dir = os.path.join(MODEL_OUTPUT_DIR, 'graphs')
            logs_dir = os.path.join(MODEL_OUTPUT_DIR, 'logs')
            os.makedirs(graphs_dir, exist_ok=True)
            os.makedirs(logs_dir, exist_ok=True)

            plt.figure(figsize=(16, 8))
            plt.plot(
                data_res['Date'],
                data_res['Close'],
                label='Real Prices',
                color='blue',
                linewidth=2
            )
            plt.plot(
                data_res['Date'].iloc[-len(final_pred):],
                final_pred,
                label='Predicted (test)',
                color='green',
                linestyle='--',
                linewidth=2
            )

            cum_forecast_dates = forecast_dates[:3]
            cum_forecast_prices = [forecasts[h][-1] for h in [1, 2, 3]]
            plt.plot(cum_forecast_dates, cum_forecast_prices, label='Forecast (1-3 days)',
                     linestyle='-.', linewidth=2, marker='o', color='red')

            if show_ci:
                cum_lower = [confidence_intervals[h][0][-1] for h in [1, 2, 3]]
                cum_upper = [confidence_intervals[h][1][-1] for h in [1, 2, 3]]
                plt.fill_between(cum_forecast_dates, cum_lower, cum_upper, alpha=0.2, label='CI (1-3 days)')

            plt.title(
                f'Прогноз цены: {ticker} — {ticker_name}',
                fontsize=14,
                fontweight='bold'
            )
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price (RUB)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            graph_filename = f'{ticker}_price_forecast_v15_{timestamp_str}.jpg'
            graph_path = os.path.join(graphs_dir, graph_filename)
            plt.savefig(graph_path, dpi=300, bbox_inches='tight')
            logger.info(f"[OK] Graph saved: {graph_path}")

            if show_plot:
                plt.show()
            else:
                plt.close()

            if user_params.get('presentation_mode'):
                pres_path = plot_presentation(
                    history_dates=data['Date'],
                    history_prices=data['Close'],
                    forecast_dates=cum_forecast_dates,
                    forecast_values=cum_forecast_prices,
                    ci_lower=[confidence_intervals[h][0][-1] for h in [1, 2, 3]],
                    ci_upper=[confidence_intervals[h][1][-1] for h in [1, 2, 3]],
                    ticker=ticker,
                    output_dir=graphs_dir,
                    history_window=user_params.get('history_window', 90),
                    timestamp=timestamp_str,
                )
                print(f"\nПрезентационный график сохранён: {pres_path}")
                logger.info(f"[OK] Presentation graph saved: {pres_path}")

            log_filename = f'{ticker}_forecast_v15_{timestamp_str}.txt'
            log_path = os.path.join(logs_dir, log_filename)
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"{'=' * 60}\n")
                f.write('MODELING STOCK PRICE FORECAST: stock_modelv15\n')
                f.write(f"{'=' * 60}\n")
                f.write(f'Ticker: {ticker}\n')
                f.write(f'Run time: {datetime.now().isoformat()}\n')
                f.write(f'Model version: {MODEL_VERSION}\n')
                f.write(f'LSTM Params: units={best_lstm_params["units"]}, dropout={best_lstm_params["dropout"]:.4f}, lr={best_lstm_params["lr"]:.6f}\n')
                f.write(f'XGBoost Params: n_estimators={best_xgb_params["n_estimators"]}, max_depth={best_xgb_params["max_depth"]}, lr={best_xgb_params["learning_rate"]:.4f}\n')
                f.write('\nQUALITY METRICS & FORECASTS (avg over 3 walk-forward splits):\n')
                f.write(f" {'H':<4}  {'Date':<12}  {'Forecast':>9}  {'RMSE':>7}  {'MAE':>7}  {'R²':>6}  CI [lower – upper]  width\n")
                f.write(f" {'-'*78}\n")
                for h in [1, 2, 3]:
                    h_price = forecasts[h][-1]
                    h_lower = confidence_intervals[h][0][-1]
                    h_upper = confidence_intervals[h][1][-1]
                    h_width = h_upper - h_lower
                    h_rmse  = all_results[h][6]
                    h_mae   = all_results[h][7]
                    h_r2    = all_results[h][8]
                    h_date  = all_results[h][4][-1].strftime('%d.%m.%Y')
                    ci_ok   = "✓" if h_lower <= h_price <= h_upper else "✗"
                    f.write(f" +{h}d   {h_date:<12}  {h_price:>9.2f}  {h_rmse:>7.4f}  {h_mae:>7.4f}  {h_r2:>6.4f}  [{h_lower:.2f} – {h_upper:.2f}]  {h_width:.2f} {ci_ok}\n")

            logger.info(f"[OK] Log saved: {log_path}")

        logger.info("\n" + "=" * 60)
        logger.info("MODELING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Results saved in: {MODEL_OUTPUT_DIR}")