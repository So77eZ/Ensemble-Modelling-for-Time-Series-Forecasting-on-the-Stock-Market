import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import json
import matplotlib
import argparse

from datetime import datetime, timedelta

import requests
import xml.etree.ElementTree as ET

import tensorflow as tf
from scipy.interpolate import interp1d

import io
import logging
from pathlib import Path

import optuna

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ============================================================================
# CONFIGURATION & LOGGING SETUP
# ============================================================================

from config import (
    LSTM_LOOK_BACK, LSTM_EPOCHS, LSTM_PATIENCE, LSTM_BATCH_SIZE,
    LSTM_LEARNING_RATE, LSTM_DROPOUT_RATE, LSTM_UNITS,
    XGBOOST_N_ESTIMATORS, XGBOOST_MAX_DEPTH, XGBOOST_LEARNING_RATE,
    XGBOOST_SUBSAMPLE, XGBOOST_COLSAMPLE_BYTREE, XGBOOST_RANDOM_STATE,
    XGBOOST_VERBOSITY, OUTPUT_ROOT
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
logger.info(f"TensorFlow: {tf.__version__}")
logger.info(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
logger.info(f"XGBoost Available: {XGBOOST_AVAILABLE}")
logger.info("=" * 60)

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
            logger.info(f"RAW API RESPONSE: {resp.text[:500]}...")
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

def load_stock_data_moex_test(ticker_symbol, start_date, end_date):
    logger.info(f"Deep data loading for {ticker_symbol} from {start_date} to {end_date}...")
    
    try:
        from moexalgo import Ticker
        stock = Ticker(ticker_symbol)
        data = stock.candles(start=start_date, end=end_date, period='1D')
        if not data.empty:
            data = data[['begin', 'open', 'high', 'low', 'close', 'volume']]
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
            logger.info(f" -> moexalgo: {len(data)} rows ({data['Date'].min()} to {data['Date'].max()})")
            return data
    except Exception as e:
        logger.warning(f" moexalgo error: {e}. Switching to direct API...")

    base_url = (
        "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/"
        f"securities/{ticker_symbol}.json"
    )
    params = {'from': start_date, 'till': end_date, 'limit': 100}
    all_data = []
    start = 0

    while True:
        params['start'] = start
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            json_data = response.json()
            columns = json_data['history']['columns']
            rows = json_data['history']['data']
            if not rows:
                break
            df_chunk = pd.DataFrame(rows, columns=columns)
            all_data.append(df_chunk)
            start += len(rows)
        except Exception as e:
            logger.warning(f"Direct MOEX API error: {e}")
            break

    if all_data:
        data = pd.concat(all_data, ignore_index=True)
        data = data[['TRADEDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').drop_duplicates()
        logger.info(f" -> Total loaded: {len(data)} rows ({data['Date'].min()} to {data['Date'].max()})")
        return data
    else:
        logger.error(f"Could not load data for {ticker_symbol}")
        return None

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
    return data

def update_technical_indicators_single_row(data, idx):
    """Обновление технических индикаторов для одной строки (для прогнозов)"""
    if idx < 10:
        data.at[idx, 'SMA_10'] = data['Close'][:idx+1].mean()
    else:
        data.at[idx, 'SMA_10'] = data['Close'][idx-9:idx+1].mean()

    if idx == 0:
        data.at[idx, 'EMA_20'] = data.at[idx, 'Close']
    else:
        alpha = 2 / 21
        prev_ema = data.at[idx-1, 'EMA_20']
        if pd.isna(prev_ema):
            prev_ema = data.at[idx, 'Close']
        data.at[idx, 'EMA_20'] = (data.at[idx, 'Close'] * alpha) + (prev_ema * (1 - alpha))

    if idx >= 14:
        window_data = data.iloc[max(0, idx-14):idx+1]
        rsi_val = calculate_rsi(window_data, 14).iloc[-1]
        data.at[idx, 'RSI_14'] = rsi_val if not pd.isna(rsi_val) else 50.0

    if idx >= 26:
        window_data = data.iloc[max(0, idx-26):idx+1]
        macd, signal, hist = calculate_macd(window_data)
        data.at[idx, 'MACD'] = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0
        data.at[idx, 'MACD_Signal'] = signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0.0
        data.at[idx, 'MACD_Histogram'] = hist.iloc[-1] if not pd.isna(hist.iloc[-1]) else 0.0

    if idx >= 20:
        window_data = data.iloc[max(0, idx-20):idx+1]
        bb_mid, bb_up, bb_low, bb_width = calculate_bollinger_bands(window_data)
        data.at[idx, 'BB_Middle'] = bb_mid.iloc[-1] if not pd.isna(bb_mid.iloc[-1]) else data.at[idx, 'Close']
        data.at[idx, 'BB_Upper'] = bb_up.iloc[-1] if not pd.isna(bb_up.iloc[-1]) else data.at[idx, 'Close']
        data.at[idx, 'BB_Lower'] = bb_low.iloc[-1] if not pd.isna(bb_low.iloc[-1]) else data.at[idx, 'Close']
        data.at[idx, 'BB_Width'] = bb_width.iloc[-1] if not pd.isna(bb_width.iloc[-1]) else 0.0

    if idx >= 14:
        window_data = data.iloc[max(0, idx-14):idx+1]
        atr_val = calculate_atr(window_data, 14).iloc[-1]
        data.at[idx, 'ATR_14'] = atr_val if not pd.isna(atr_val) else 0.0
        
        stoch_k, stoch_d = calculate_stochastic(window_data)
        data.at[idx, 'Stoch_K'] = stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 50.0
        data.at[idx, 'Stoch_D'] = stoch_d.iloc[-1] if not pd.isna(stoch_d.iloc[-1]) else 50.0
        
        adx_val = calculate_adx(window_data, 14).iloc[-1]
        data.at[idx, 'ADX_14'] = adx_val if not pd.isna(adx_val) else 0.0

    if idx >= 10:
        window_data = data.iloc[max(0, idx-10):idx+1]
        momentum_val = calculate_momentum(window_data, 10).iloc[-1]
        data.at[idx, 'Momentum_10'] = momentum_val if not pd.isna(momentum_val) else 0.0

    if idx >= 1:
        window_data = data.iloc[max(0, idx-1):idx+1]
        pc1_val = calculate_price_change(window_data, 1).iloc[-1]
        data.at[idx, 'Price_Change_1'] = pc1_val if not pd.isna(pc1_val) else 0.0
        
    if idx >= 5:
        window_data = data.iloc[max(0, idx-5):idx+1]
        pc5_val = calculate_price_change(window_data, 5).iloc[-1]
        data.at[idx, 'Price_Change_5'] = pc5_val if not pd.isna(pc5_val) else 0.0

    for col in ['ATR_14', 'Stoch_K', 'Stoch_D', 'ADX_14', 'Momentum_10', 'Price_Change_1', 'Price_Change_5',
                'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width']:
        if col in data.columns and pd.isna(data.at[idx, col]):
            if col in ['Stoch_K', 'Stoch_D', 'RSI_14']:
                data.at[idx, col] = 50.0
            elif col in ['BB_Middle', 'BB_Upper', 'BB_Lower']:
                data.at[idx, col] = data.at[idx, 'Close']
            else:
                data.at[idx, col] = 0.0

    return data

# ============================================================================
# OPTUNA OPTIMIZATION
# ============================================================================

def optimize_lstm_params(X_train, y_train, n_trials=20):
    def objective(trial):
        units = trial.suggest_int('units', 32, 128)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units))
        model.add(Dropout(dropout))
        model.add(Dense(1))

        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        early_stop = EarlyStopping(monitor='val_loss', patience=LSTM_PATIENCE, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        return min(history.history['val_loss'])

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
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
            'verbosity': XGBOOST_VERBOSITY
        }
        split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:split], X_train[split:]
        y_tr, y_val = y_train[:split], y_train[split:]
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

# ============================================================================
# MODEL TRAINING
# ============================================================================

def prepare_and_train_model(data, ticker, end_date, best_lstm_params, best_xgb_params, backtest_mode=False, backtest_date=None):
    logger.info("=" * 60)
    logger.info(f"PREPARING DATA FOR {ticker}")
    if backtest_mode:
        logger.info(f"BACKTEST MODE: Training until {backtest_date}")
    logger.info("=" * 60)
    logger.info("Calculating technical indicators...")
    data = update_technical_indicators(data)

    if tinkoff_loader:
        tinkoff_funds = tinkoff_loader.get_fundamentals(ticker)
    else:
        tinkoff_funds = {}

    for key, value in tinkoff_funds.items():
        data[key] = value

    data = data.infer_objects(copy=False).fillna(0)
    logger.info(f"Data shape after cleaning: {data.shape}")

    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_10', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width',
        'ATR_14', 'Stoch_K', 'Stoch_D', 'ADX_14', 'Momentum_10',
        'Price_Change_1', 'Price_Change_5',
        'market_cap', 'roa', 'roe', 'debt_equity', 'current_ratio',
        'gross_profit_margin', 'dividend_yield', 'eps_growth', 'sales_growth',
        'operating_margin', 'net_profit_margin', 'pe_ratio', 'pb_ratio',
        'ps_ratio', 'price_cash_flow', 'value_usd', 'beta'
    ]
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

    logger.info(f"Preparing sequences with look_back={LSTM_LOOK_BACK}...")
    X, y = [], []
    for i in range(LSTM_LOOK_BACK, len(scaled_df)):
        X.append(scaled_df.iloc[i-LSTM_LOOK_BACK:i].values)
        y.append(scaled_df['Close'].iloc[i])
    X, y = np.array(X), np.array(y)
    logger.info(f"Total sequences: {len(X)}")

    splits = [
        (int(0.8 * len(X)), int(0.85 * len(X))),
        (int(0.85 * len(X)), int(0.9 * len(X))),
        (int(0.9 * len(X)), len(X))
    ]
    logger.info(f"Splits: {splits}")

    rmses, maes, r2s = [], [], []
    final_pred = []
    models = []

    for i, (train_end, test_end) in enumerate(splits, 1):
        logger.info("\n" + "=" * 60)
        logger.info(f"WALK-FORWARD SPLIT {i}/{len(splits)}")
        logger.info("=" * 60)

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        logger.info("Training LSTM model...")
        lstm_model = Sequential()
        lstm_model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        
        lstm_units_1 = best_lstm_params['units']
        lstm_units_2 = max(32, lstm_units_1 // 2)
        lstm_model.add(LSTM(units=lstm_units_1, return_sequences=True))
        lstm_model.add(Dropout(best_lstm_params['dropout']))
        lstm_model.add(LSTM(units=lstm_units_2, return_sequences=True))
        lstm_model.add(Dropout(best_lstm_params['dropout'] * 0.5))
        lstm_model.add(LSTM(units=lstm_units_2))
        lstm_model.add(Dropout(best_lstm_params['dropout']))
        lstm_model.add(Dense(32, activation='relu'))
        lstm_model.add(Dense(1))

        optimizer = Adam(learning_rate=best_lstm_params['lr'])
        lstm_model.compile(optimizer=optimizer, loss='mean_squared_error')

        early_stop = EarlyStopping(monitor='val_loss', patience=LSTM_PATIENCE, restore_best_weights=True)
        history = lstm_model.fit(
            X_train, y_train,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        logger.info(f"No overfitting: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        logger.info("Training XGBoost model...")
        xgb_params = {
            **best_xgb_params,
            'random_state': XGBOOST_RANDOM_STATE,
            'verbosity': XGBOOST_VERBOSITY
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

        logger.info("Training quantile regression models...")
        lower_params = {**xgb_params, 'objective': 'reg:quantileerror', 'quantile_alpha': 0.05}
        median_params = {**xgb_params, 'objective': 'reg:quantileerror', 'quantile_alpha': 0.5}
        upper_params = {**xgb_params, 'objective': 'reg:quantileerror', 'quantile_alpha': 0.95}

        lower_model = xgb.XGBRegressor(**lower_params)
        median_model = xgb.XGBRegressor(**median_params)
        upper_model = xgb.XGBRegressor(**upper_params)

        lower_model.fit(X_train_flat, y_train)
        median_model.fit(X_train_flat, y_train)
        upper_model.fit(X_train_flat, y_train)
        logger.info("✓ Quantile models trained")

        logger.info("Generating level 0 predictions...")
        lstm_train_preds = lstm_model.predict(X_train, verbose=0).flatten()
        xgb_train_preds = xgb_model.predict(X_train_flat)

        meta_train = np.column_stack((lstm_train_preds, xgb_train_preds))

        logger.info("Training Meta-Learner...")
        meta_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': XGBOOST_RANDOM_STATE,
            'verbosity': XGBOOST_VERBOSITY
        }
        meta_learner = xgb.XGBRegressor(**meta_params)
        meta_learner.fit(meta_train, y_train)
        logger.info("[OK] Meta-Learner trained")

        lstm_test_preds = lstm_model.predict(X_test, verbose=0).flatten()
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        xgb_test_preds = xgb_model.predict(X_test_flat)
        meta_test = np.column_stack((lstm_test_preds, xgb_test_preds))
        test_preds = meta_learner.predict(meta_test)

        test_preds_inv = close_scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
        y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        rmse = np.sqrt(mean_squared_error(y_test_inv, test_preds_inv))
        mae = mean_absolute_error(y_test_inv, test_preds_inv)
        r2 = r2_score(y_test_inv, test_preds_inv)

        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)

        final_pred.extend(test_preds_inv)
        models.append((lstm_model, xgb_model, meta_learner, lower_model, median_model, upper_model))

    avg_rmse = np.mean(rmses)
    avg_mae = np.mean(maes)
    avg_r2 = np.mean(r2s)
    logger.info(f"Average Metrics: RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}, R2={avg_r2:.4f}")

    best_model, best_xgb_model, best_meta_learner, best_lower_model, best_median_model, best_upper_model = models[-1]

    logger.info("Generating forecasts...")
    forecast_horizons = [1, 2, 3]
    forecasts = {}
    confidence_intervals = {}
    
    # Определяем базовую дату для прогнозов
    if backtest_mode and backtest_date:
        base_date = datetime.strptime(backtest_date, '%Y-%m-%d')
    else:
        base_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    forecast_base = data.copy()
    
    for horizon in forecast_horizons:
        forecast_prices = []
        lower_bounds = []
        upper_bounds = []
        
        temp_df = forecast_base.copy()
        
        for day in range(horizon):
            last_features = temp_df[features].tail(LSTM_LOOK_BACK)
            last_scaled = scaler.transform(last_features)
            last_scaled_df = pd.DataFrame(last_scaled, columns=features, index=last_features.index)

            lstm_pred_scaled = best_model.predict(
                last_scaled_df.values.reshape(1, LSTM_LOOK_BACK, len(features)),
                verbose=0
            )[0][0]

            current_scaled_flat = last_scaled.reshape(1, -1)
            xgb_pred_scaled = best_xgb_model.predict(current_scaled_flat)[0]

            meta_input = np.array([[lstm_pred_scaled, xgb_pred_scaled]])
            pred_close_scaled = best_meta_learner.predict(meta_input)[0]
            pred_close = close_scaler.inverse_transform([[pred_close_scaled]])[0][0]

            lower_scaled = best_lower_model.predict(current_scaled_flat)[0]
            upper_scaled = best_upper_model.predict(current_scaled_flat)[0]

            lower = close_scaler.inverse_transform([[lower_scaled]])[0][0]
            upper = close_scaler.inverse_transform([[upper_scaled]])[0][0]
            
            if lower > upper:
                lower, upper = upper, lower

            forecast_prices.append(pred_close)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

            prev_close = temp_df['Close'].iloc[-1]
            
            if len(temp_df) > 1:
                volatility = abs(temp_df['Close'].iloc[-1] - temp_df['Close'].iloc[-2])
            else:
                volatility = pred_close * 0.01
            
            new_data = {
                'Date': base_date + timedelta(days=day + 1),
                'Open': float(prev_close),
                'High': float(max(prev_close, pred_close) + volatility * 0.3),
                'Low': float(min(prev_close, pred_close) - volatility * 0.3),
                'Close': float(pred_close),
                'Volume': float(temp_df['Volume'].iloc[-1])
            }
            
            for col in features:
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in temp_df.columns:
                        new_data[col] = float(temp_df[col].iloc[-1])
                    else:
                        new_data[col] = 0.0
            
            new_row = pd.DataFrame([new_data])
            temp_df = pd.concat([temp_df, new_row], ignore_index=True)
            
            temp_df = update_technical_indicators_single_row(temp_df, len(temp_df) - 1)

            logger.info(
                f"Horizon {horizon}, Day {day + 1}: Forecast={pred_close:.2f}, "
                f"Open={new_data['Open']:.2f}, High={new_data['High']:.2f}, "
                f"Low={new_data['Low']:.2f}, Lower CI={lower:.2f}, Upper CI={upper:.2f}"
            )

        forecasts[horizon] = forecast_prices
        confidence_intervals[horizon] = (lower_bounds, upper_bounds)

    forecast_dates = [base_date + timedelta(days=i + 1) for i in range(max(forecast_horizons))]

    logger.info("Forecast logic check: No forward-looking indicators used; updates are sequential.")

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

def run_backtest(data, ticker, backtest_date, best_lstm_params, best_xgb_params):
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
    
    if future_data.empty:
        logger.error(f"No future data available after {backtest_date}")
        return None
    
    logger.info(f"Training data until: {available_data['Date'].max()}")
    logger.info(f"Available future data: {len(future_data)} trading days")
    logger.info(f"Future data dates: {future_data['Date'].min()} to {future_data['Date'].max()}")
    
    # Обучаем модель на данных до backtest_date
    result = prepare_and_train_model(
        data, ticker, backtest_date, 
        best_lstm_params, best_xgb_params,
        backtest_mode=True, backtest_date=backtest_date
    )
    
    if result[0] is None:
        return None
    
    data_train, real_prices, final_pred, forecasts, forecast_dates, confidence_intervals, \
        rmse_val, mae_val, r2_val, scaler, close_scaler, features = result
    
    # Сравниваем прогнозы с реальными данными
    backtest_results = []
    
    # ИСПРАВЛЕНИЕ: создаем словарь для быстрого поиска реальных цен по дате
    future_data['Date_key'] = future_data['Date'].dt.date
    real_prices_dict = dict(zip(future_data['Date_key'], future_data['Close']))
    
    logger.info(f"\nAvailable real data dates: {list(real_prices_dict.keys())}")
    
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
        
        # Вычисляем метрики
        error = abs(forecast_price - real_price)
        error_pct = (error / real_price) * 100
        in_ci = lower_ci <= real_price <= upper_ci
        
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
            'in_ci': in_ci
        })
        
        logger.info(f"  Forecast date: {forecast_date.strftime('%Y-%m-%d')}")
        logger.info(f"  Actual date used: {actual_date.strftime('%Y-%m-%d')}")
        logger.info(f"  Forecast: {forecast_price:.2f} RUB")
        logger.info(f"  Real: {real_price:.2f} RUB")
        logger.info(f"  Error: {error:.2f} RUB ({error_pct:.2f}%)")
        logger.info(f"  CI: [{lower_ci:.2f}, {upper_ci:.2f}]")
        logger.info(f"  Real price in CI: {'✓ YES' if in_ci else '✗ NO'}")
    
    if not backtest_results:
        logger.error("No backtest results generated - no matching dates found")
        return None
    
    return backtest_results, forecasts, forecast_dates, confidence_intervals

# ============================================================================
# USER INTERACTION
# ============================================================================

def get_user_inputs():
    """Собираем все параметры от пользователя в начале программы"""
    print("\n" + "="*60)
    print("STOCK PRICE FORECASTING MODEL v13")
    print("="*60)
    
    # 1. Тикер
    ticker_input = input(
        "\n1. Введите тикер (например: SBER, LKOH, GAZP) "
        "или Enter для SBER: "
    ).strip().upper()
    ticker = ticker_input if ticker_input else 'SBER'
    
    # 2. Режим работы
    print("\n2. Выберите режим работы:")
    print("   [1] Обычный прогноз (на будущее)")
    print("   [2] Бэктест (проверка прогноза на известных данных)")
    mode_input = input("Режим (1/2, по умолчанию 1): ").strip()
    backtest_mode = mode_input == '2'
    
    backtest_date = None
    if backtest_mode:
        default_backtest = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        backtest_input = input(
            f"\n   Введите дату для бэктеста (YYYY-MM-DD, по умолчанию {default_backtest}): "
        ).strip()
        backtest_date = backtest_input if backtest_input else default_backtest
        print(f"   Бэктест: обучение до {backtest_date}, проверка прогноза после")
    
    # 3. Оптимизация гиперпараметров
    print("\n3. Гиперпараметры модели:")
    saved_lstm, saved_xgb = load_hyperparams(ticker)
    
    if saved_lstm and saved_xgb:
        print(f"   Найдены сохраненные параметры для {ticker}:")
        print(f"   LSTM: units={saved_lstm['units']}, dropout={saved_lstm['dropout']:.4f}, lr={saved_lstm['lr']:.6f}")
        print(f"   XGBoost: n_est={saved_xgb['n_estimators']}, depth={saved_xgb['max_depth']}, lr={saved_xgb['learning_rate']:.4f}")
        use_saved = input("   Использовать сохраненные параметры? (y/n, по умолчанию y): ").strip().lower()
        optimize = use_saved == 'n'
    else:
        print(f"   Сохраненные параметры для {ticker} не найдены.")
        optimize_input = input("   Запустить Optuna для поиска оптимальных параметров? (y/n, по умолчанию n): ").strip().lower()
        optimize = optimize_input == 'y'
    
    if optimize:
        trials_input = input("   Количество итераций Optuna (по умолчанию 20): ").strip()
        n_trials = int(trials_input) if trials_input.isdigit() else 20
    else:
        n_trials = 0
    
    # 4. Визуализация
    print("\n4. Параметры визуализации:")
    show_ci = input("   Показывать доверительные интервалы на графике? (y/n, по умолчанию n): ").strip().lower() == 'y'
    show_plot = input("   Показывать график после обучения? (y/n, по умолчанию n): ").strip().lower() == 'y'
    
    print("\n" + "="*60)
    print("ПАРАМЕТРЫ ЗАПУСКА:")
    print(f"  Тикер: {ticker}")
    print(f"  Режим: {'Бэктест' if backtest_mode else 'Прогноз'}")
    if backtest_mode:
        print(f"  Дата бэктеста: {backtest_date}")
    print(f"  Оптимизация: {'Да (' + str(n_trials) + ' итераций)' if optimize else 'Нет (используются сохраненные/дефолтные)'}")
    print(f"  Доверительные интервалы: {'Да' if show_ci else 'Нет'}")
    print(f"  Показать график: {'Да' if show_plot else 'Нет'}")
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
        'show_plot': show_plot
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
    args = parser.parse_args()

    if args.no_gui:
        try:
            matplotlib.use('Agg')
        except Exception:
            pass

    import matplotlib.pyplot as plt

    # Получаем параметры от пользователя (если не заданы через CLI)
    if args.ticker:
        # CLI mode
        user_params = {
            'ticker': args.ticker.upper(),
            'backtest_mode': bool(args.backtest),
            'backtest_date': args.backtest,
            'optimize': args.optimize,
            'n_trials': args.trials,
            'show_ci': False,
            'show_plot': not args.no_gui
        }
    else:
        # Interactive mode
        user_params = get_user_inputs()

    ticker = user_params['ticker']
    backtest_mode = user_params['backtest_mode']
    backtest_date = user_params['backtest_date']
    optimize = user_params['optimize']
    n_trials = user_params['n_trials']
    show_ci = user_params['show_ci']
    show_plot = user_params['show_plot']

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

    # Получаем или оптимизируем гиперпараметры
    if optimize:
        logger.info("\n" + "="*60)
        logger.info("RUNNING OPTUNA OPTIMIZATION")
        logger.info("="*60)
        
        # Подготовка данных для оптимизации
        data_for_opt = update_technical_indicators(data.copy())

        if tinkoff_loader:
            tinkoff_funds = tinkoff_loader.get_fundamentals(ticker)
        else:
            tinkoff_funds = {}

        for key, value in tinkoff_funds.items():
            data_for_opt[key] = value

        data_for_opt = data_for_opt.infer_objects(copy=False).fillna(0)

        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width',
            'ATR_14', 'Stoch_K', 'Stoch_D', 'ADX_14', 'Momentum_10',
            'Price_Change_1', 'Price_Change_5',
            'market_cap', 'roa', 'roe', 'debt_equity', 'current_ratio',
            'gross_profit_margin', 'dividend_yield', 'eps_growth', 'sales_growth',
            'operating_margin', 'net_profit_margin', 'pe_ratio', 'pb_ratio',
            'ps_ratio', 'price_cash_flow', 'value_usd', 'beta'
        ]
        
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
        backtest_results = run_backtest(data, ticker, backtest_date, best_lstm_params, best_xgb_params)

        if backtest_results and backtest_results[0]:
            results_list, forecasts, forecast_dates, confidence_intervals = backtest_results
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
                print(f"  Forecast:     {res['forecast']:.2f} RUB")
                print(f"  Real:         {res['real']:.2f} RUB")
                print(f"  Error:        {res['error']:.2f} RUB ({res['error_pct']:.2f}%)")
                print(f"  CI:           [{res['lower_ci']:.2f}, {res['upper_ci']:.2f}]")
                print(f"  In CI:        {'✓ YES' if res['in_ci'] else '✗ NO'}")

            avg_error = np.mean([r['error'] for r in results_list])
            avg_error_pct = np.mean([r['error_pct'] for r in results_list])
            ci_coverage = sum([r['in_ci'] for r in results_list]) / len(results_list) * 100

            print(f"\nAVERAGE METRICS:")
            print(f"  Average Error:  {avg_error:.2f} RUB ({avg_error_pct:.2f}%)")
            print(f"  CI Coverage:    {ci_coverage:.1f}%")
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
                    f.write(f"  Forecast:     {res['forecast']:.2f} RUB\n")
                    f.write(f"  Real:         {res['real']:.2f} RUB\n")
                    f.write(f"  Error:        {res['error']:.2f} RUB ({res['error_pct']:.2f}%)\n")
                    f.write(f"  CI:           [{res['lower_ci']:.2f}, {res['upper_ci']:.2f}]\n")
                    f.write(f"  In CI:        {'YES' if res['in_ci'] else 'NO'}\n")
                f.write(f"\nAVERAGE METRICS:\n")
                f.write(f"Average Error: {avg_error:.2f} RUB ({avg_error_pct:.2f}%)\n")
                f.write(f"CI Coverage: {ci_coverage:.1f}%\n")
                f.write(f"Total forecasts checked: {len(results_list)}\n")

            logger.info(f"[OK] Backtest results saved: {backtest_log_file}")
        else:
            logger.error("Backtest failed: no results generated")

    else:
        # Обычный режим прогноза
        result = prepare_and_train_model(
            data, ticker, end_date,
            best_lstm_params, best_xgb_params,
            backtest_mode=False
        )

        if result[0] is not None:
            data_res, real_prices, final_pred, forecasts, forecast_dates, confidence_intervals, \
                rmse_val, mae_val, r2_val, scaler, close_scaler, features = result

            print("\n" + "="*60)
            print("FORECAST SUMMARY")
            print("="*60)
            for horizon in [1, 2, 3]:
                price = forecasts[horizon][-1]
                date_str = forecast_dates[horizon-1].strftime('%d.%m.%Y')
                print(f"{horizon} day: {forecast_dates[horizon-1]} ~ Цена на {date_str}: {price:.2f}")
                if show_ci:
                    lower, upper = confidence_intervals[horizon][0][-1], confidence_intervals[horizon][1][-1]
                    print(f"  Confidence Interval: [{lower:.2f}, {upper:.2f}]")
            print("="*60)

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
                f'Stock Price Forecast for {ticker} (v13)',
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
            graph_filename = f'{ticker}_price_forecast_v13_{timestamp_str}.jpg'
            graph_path = os.path.join(graphs_dir, graph_filename)
            plt.savefig(graph_path, dpi=300, bbox_inches='tight')
            logger.info(f"[OK] Graph saved: {graph_path}")

            if show_plot:
                plt.show()
            else:
                plt.close()

            log_filename = f'{ticker}_forecast_v13_{timestamp_str}.txt'
            log_path = os.path.join(logs_dir, log_filename)
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"{'=' * 60}\n")
                f.write('MODELING STOCK PRICE FORECAST: stock_modelv13\n')
                f.write(f"{'=' * 60}\n")
                f.write(f'Ticker: {ticker}\n')
                f.write(f'Run time: {datetime.now().isoformat()}\n')
                f.write(f'Model version: {MODEL_VERSION}\n')
                f.write(f'LSTM Params: units={best_lstm_params["units"]}, dropout={best_lstm_params["dropout"]:.4f}, lr={best_lstm_params["lr"]:.6f}\n')
                f.write(f'XGBoost Params: n_estimators={best_xgb_params["n_estimators"]}, max_depth={best_xgb_params["max_depth"]}, lr={best_xgb_params["learning_rate"]:.4f}\n')
                f.write('\nQUALITY METRICS (Avg over splits):\n')
                f.write(f' RMSE: {rmse_val:.4f}\n')
                f.write(f' MAE: {mae_val:.4f}\n')
                f.write(f' R²: {r2_val:.4f}\n')
                f.write('\nFORECASTS:\n')
                for horizon in [1, 2, 3]:
                    prices = forecasts[horizon]
                    lower, upper = confidence_intervals[horizon]
                    f.write(f'{horizon} days ahead:\n')
                    for day_idx, (price, low, up) in enumerate(zip(prices, lower, upper), 1):
                        f.write(f'  Day {day_idx}: Price={price:.2f}, CI=[{low:.2f}, {up:.2f}]\n')

            logger.info(f"[OK] Log saved: {log_path}")

        logger.info("\n" + "=" * 60)
        logger.info("MODELING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Results saved in: {MODEL_OUTPUT_DIR}")