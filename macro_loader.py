import logging
import os
import certifi
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# Дисковый кэш: CSV рядом с модулем, перезагружает только недостающие дни.
# Последние CACHE_REFRESH_DAYS дней всегда перезагружаются (данные могут уточняться).
_CACHE_PATH        = os.path.join(os.path.dirname(__file__), '.macro_cache.csv')
_CACHE_REFRESH_DAYS = 7


# ── приватные загрузчики ──────────────────────────────────────────────────────

def _load_cbr_usd_rub_history(start: str, end: str) -> pd.Series:
    """Исторический курс USD/RUB из ЦБ РФ XML Dynamic API."""
    d1 = datetime.strptime(start, '%Y-%m-%d').strftime('%d/%m/%Y')
    d2 = datetime.strptime(end,   '%Y-%m-%d').strftime('%d/%m/%Y')
    url = (
        f"https://www.cbr.ru/scripts/XML_dynamic.asp"
        f"?date_req1={d1}&date_req2={d2}&VAL_NM_RQ=R01235"
    )
    try:
        resp = requests.get(url, timeout=15, verify=certifi.where())
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        dates, rates = [], []
        for record in root.findall('Record'):
            date_str = record.attrib.get('Date', '')
            val_node = record.find('Value')
            if date_str and val_node is not None and val_node.text:
                dates.append(datetime.strptime(date_str, '%d.%m.%Y'))
                rates.append(float(val_node.text.replace(',', '.')))
        if not dates:
            raise ValueError("empty response")
        s = pd.Series(rates, index=pd.to_datetime(dates), name='usd_rub_hist')
        logger.info(
            f"usd_rub_hist: {len(s)} записей {s.index.min().date()} — {s.index.max().date()}"
            f", диапазон {s.min():.2f}–{s.max():.2f} руб."
        )
        return s
    except Exception as e:
        logger.warning(f"usd_rub_hist: CBR USD/RUB history unavailable ({e})")
        return pd.Series(dtype=float)


def _load_cbr_key_rate(start: str, end: str) -> pd.Series:
    """Ключевая ставка ЦБ РФ через SOAP API DailyInfo.asmx."""
    soap_body = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
        '<soap:Body>'
        '<KeyRate xmlns="http://web.cbr.ru/">'
        f'<fromDate>{start}T00:00:00</fromDate>'
        f'<ToDate>{end}T00:00:00</ToDate>'
        '</KeyRate>'
        '</soap:Body>'
        '</soap:Envelope>'
    )
    try:
        resp = requests.post(
            'https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx',
            data=soap_body.encode('utf-8'),
            headers={
                'Content-Type': 'text/xml; charset=utf-8',
                'SOAPAction': 'http://web.cbr.ru/KeyRate',
            },
            timeout=3,
            verify=certifi.where(),
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        dates, rates = [], []
        for kr in root.iter('KR'):
            dt_el   = kr.find('DT')
            rate_el = kr.find('Rate')
            if dt_el is not None and rate_el is not None and dt_el.text:
                dates.append(pd.to_datetime(dt_el.text[:10]))
                rates.append(float(rate_el.text))
        if not dates:
            raise ValueError("empty SOAP response")
        s = pd.Series(rates, index=pd.DatetimeIndex(dates), name='cbr_rate')
        logger.info(
            f"cbr_rate: {len(s)} записей {s.index.min().date()} — {s.index.max().date()}"
            f", диапазон {s.min():.2f}%–{s.max():.2f}%"
        )
        return s
    except Exception as e:
        logger.warning(f"cbr_rate: CBR key rate unavailable ({e})")
        return pd.Series(dtype=float)


def _load_moex_brent(start: str, end: str) -> pd.Series:
    """Цена Brent из MOEX ISS (фьючерс BR, ближний контракт).

    Формат secid: BR{letter}{year_digit}, где letter — стандартные коды месяцев
    фьючерсных контрактов (F=янв, G=фев, H=мар, J=апр, K=май, M=июн,
    N=июл, Q=авг, U=сен, V=окт, X=ноя, Z=дек), year_digit — последняя цифра года.
    Запросы выполняются параллельно (ThreadPoolExecutor).
    """
    _MONTH_LETTERS = {
        1: 'F', 2: 'G', 3: 'H', 4: 'J',  5: 'K',  6: 'M',
        7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z',
    }
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt   = datetime.strptime(end,   '%Y-%m-%d')

    # Собираем уникальные secid для запроса
    secids = []
    seen_secids = set()
    year, month = start_dt.year, start_dt.month
    while datetime(year, month, 1) <= end_dt + timedelta(days=62):
        secid = f"BR{_MONTH_LETTERS[month]}{str(year)[-1]}"
        if secid not in seen_secids:
            seen_secids.add(secid)
            secids.append(secid)
        month += 1
        if month > 12:
            month = 1
            year += 1

    def _fetch_brent(secid: str):
        url = (
            f"https://iss.moex.com/iss/history/engines/futures/markets/forts"
            f"/boards/RFUD/securities/{secid}/candles.json"
        )
        try:
            resp = requests.get(
                url,
                params={'interval': 24, 'from': start, 'till': end, 'iss.meta': 'off'},
                timeout=15,
                verify=certifi.where(),
            )
            if resp.status_code == 200:
                j = resp.json()
                cols = j['history']['columns']
                rows = j['history']['data']
                if rows:
                    ci = cols.index('CLOSE')
                    di = cols.index('TRADEDATE')
                    filtered = [r for r in rows if r[ci] is not None]
                    if filtered:
                        return pd.DataFrame({
                            'Date':        pd.to_datetime([r[di] for r in filtered]),
                            'brent_price': [float(r[ci]) for r in filtered],
                        })
        except Exception:
            pass
        return None

    all_frames = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_brent, sid): sid for sid in secids}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                all_frames.append(result)

    if not all_frames:
        logger.warning("brent_price: MOEX BRN futures unavailable — feature excluded")
        return pd.Series(dtype=float)

    combined = (
        pd.concat(all_frames)
        .sort_values('Date')
        .drop_duplicates('Date', keep='first')
    )
    s = pd.Series(combined['brent_price'].values, index=combined['Date'].values, name='brent_price')
    logger.info(
        f"brent_price: {len(s)} записей {pd.Timestamp(s.index.min()).date()} — {pd.Timestamp(s.index.max()).date()}"
        f", диапазон {s.min():.2f}–{s.max():.2f} руб."
    )
    return s


def _load_moex_index(index_id: str, start: str, end: str, board: str = 'SNDX') -> pd.Series:
    """Цена закрытия индекса MOEX через MOEX ISS history candles API.

    Запросы помесячные (~21 торговый день < 100 записей лимит MOEX ISS),
    выполняются параллельно через ThreadPoolExecutor.
    board: 'SNDX' для IMOEX, 'RTSI' для RTSI.
    """
    import calendar

    col_name = index_id.lower()
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt   = datetime.strptime(end,   '%Y-%m-%d')

    url = (
        f"https://iss.moex.com/iss/history/engines/stock/markets/index"
        f"/boards/{board}/securities/{index_id}/candles.json"
    )

    # Строим список месячных чанков
    chunks = []
    year, month = start_dt.year, start_dt.month
    while datetime(year, month, 1) <= end_dt:
        last_day = calendar.monthrange(year, month)[1]
        chunk_start = max(start_dt, datetime(year, month, 1)).strftime('%Y-%m-%d')
        chunk_end   = min(end_dt,   datetime(year, month, last_day)).strftime('%Y-%m-%d')
        chunks.append((chunk_start, chunk_end))
        month += 1
        if month > 12:
            month = 1
            year += 1

    def _fetch_chunk(chunk_start: str, chunk_end: str):
        try:
            resp = requests.get(
                url,
                params={'interval': 24, 'from': chunk_start, 'till': chunk_end, 'iss.meta': 'off'},
                timeout=15,
                verify=certifi.where(),
            )
            if resp.status_code == 200:
                j = resp.json()
                block = j.get('history') or j.get('candles') or {}
                cols = block.get('columns', [])
                rows = block.get('data', [])
                if rows:
                    ci = next((i for i, c in enumerate(cols) if c.lower() == 'close'), None)
                    di = next((i for i, c in enumerate(cols) if c.lower() in ('tradedate', 'begin')), None)
                    if ci is not None and di is not None:
                        filtered = [r for r in rows if r[ci] is not None]
                        if filtered:
                            return pd.DataFrame({
                                'Date':   pd.to_datetime([r[di][:10] for r in filtered]),
                                col_name: [float(r[ci]) for r in filtered],
                            })
        except Exception:
            pass
        return None

    all_frames = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_chunk, cs, ce): (cs, ce) for cs, ce in chunks}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                all_frames.append(result)

    if not all_frames:
        logger.warning(f"{col_name}: MOEX {index_id} index unavailable — feature excluded")
        return pd.Series(dtype=float)

    combined = (
        pd.concat(all_frames)
        .sort_values('Date')
        .drop_duplicates('Date', keep='first')
    )
    s = pd.Series(combined[col_name].values, index=combined['Date'].values, name=col_name)
    logger.info(
        f"{col_name}: {len(s)} записей {pd.Timestamp(s.index.min()).date()} — {pd.Timestamp(s.index.max()).date()}"
        f", диапазон {s.min():.2f}–{s.max():.2f}"
    )
    return s


# ── кэш ──────────────────────────────────────────────────────────────────────

_MACRO_COLS_ALL = ['usd_rub_hist', 'cbr_rate', 'brent_price', 'imoex', 'rtsi']


def _cache_load() -> pd.DataFrame | None:
    """Читает кэш-файл. Возвращает None если файл отсутствует или повреждён."""
    if not os.path.exists(_CACHE_PATH):
        return None
    try:
        df = pd.read_csv(_CACHE_PATH, parse_dates=['Date'])
        if 'Date' not in df.columns or df.empty:
            return None
        return df
    except Exception:
        return None


def _cache_save(df: pd.DataFrame) -> None:
    try:
        df.to_csv(_CACHE_PATH, index=False)
    except Exception as e:
        logger.warning(f"macro cache: не удалось сохранить ({e})")


def _fetch_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Загружает данные за период, все 5 источников параллельно."""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    result = pd.DataFrame({'Date': date_range})

    loaders = {
        'usd_rub_hist': _load_cbr_usd_rub_history,
        'cbr_rate':     _load_cbr_key_rate,
        'brent_price':  _load_moex_brent,
        'imoex':        lambda s, e: _load_moex_index('IMOEX', s, e, board='SNDX'),
        'rtsi':         lambda s, e: _load_moex_index('RTSI',  s, e, board='RTSI'),
    }

    def _run(col_loader):
        col, loader = col_loader
        return col, loader(start_date, end_date)

    with ThreadPoolExecutor(max_workers=len(loaders)) as pool:
        for col, series in pool.map(_run, loaders.items()):
            if series.empty:
                result[col] = float('nan')
            else:
                series.index = pd.to_datetime(series.index).normalize()
                result[col] = series.reindex(date_range).values

    result[_MACRO_COLS_ALL] = result[_MACRO_COLS_ALL].ffill().bfill()
    return result


# ── публичный интерфейс ───────────────────────────────────────────────────────

def load_macro_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Загружает макроэкономические признаки за период [start_date, end_date].

    Возвращает DataFrame с колонками:
        Date, usd_rub_hist, cbr_rate, brent_price, imoex, rtsi

    Результаты кэшируются в .macro_cache.csv рядом с модулем.
    При повторных вызовах дозагружаются только отсутствующие дни.
    Последние CACHE_REFRESH_DAYS дней всегда перезагружаются.
    При недоступности источника колонка содержит NaN и логируется warning.
    """
    start_dt = pd.Timestamp(start_date)
    end_dt   = pd.Timestamp(end_date)

    cached = _cache_load()

    if cached is not None:
        cached_end = cached['Date'].max()
        # Граница перезагрузки: начало "свежей" зоны
        refresh_from = cached_end - pd.Timedelta(days=_CACHE_REFRESH_DAYS)

        if cached['Date'].min() <= start_dt and cached_end >= end_dt:
            # Кэш полностью покрывает запрос — только обновляем свежую зону
            fetch_start = max(refresh_from, start_dt)
            logger.info(
                f"macro cache hit: покрыт {cached['Date'].min().date()}–{cached_end.date()}, "
                f"обновляем {fetch_start.date()}–{end_dt.date()}"
            )
            fresh = _fetch_range(fetch_start.strftime('%Y-%m-%d'), end_date)
            merged = (
                pd.concat([cached[cached['Date'] < fetch_start], fresh])
                .sort_values('Date')
                .drop_duplicates('Date', keep='last')
                .reset_index(drop=True)
            )
            _cache_save(merged)
        else:
            # Кэш не покрывает весь диапазон — дозагружаем недостающее
            fetch_start = min(start_dt, cached['Date'].min())
            fetch_end   = max(end_dt, cached_end)
            miss_start  = (cached_end - pd.Timedelta(days=_CACHE_REFRESH_DAYS)).strftime('%Y-%m-%d')
            logger.info(
                f"macro cache partial: дозагружаем {miss_start}–{fetch_end.date()}"
            )
            fresh = _fetch_range(miss_start, fetch_end.strftime('%Y-%m-%d'))
            merged = (
                pd.concat([cached[cached['Date'] < pd.Timestamp(miss_start)], fresh])
                .sort_values('Date')
                .drop_duplicates('Date', keep='last')
                .reset_index(drop=True)
            )
            _cache_save(merged)
    else:
        # Кэша нет — полная загрузка
        logger.info(f"macro cache miss: загружаем {start_date}–{end_date}")
        merged = _fetch_range(start_date, end_date)
        _cache_save(merged)

    # Нарезаем по запрошенному диапазону
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    result = merged[merged['Date'].between(start_dt, end_dt)].copy()

    # Убеждаемся, что все календарные дни присутствуют (ffill для выходных)
    result = (
        pd.DataFrame({'Date': date_range})
        .merge(result, on='Date', how='left')
    )
    result[_MACRO_COLS_ALL] = result[_MACRO_COLS_ALL].ffill().bfill()

    for col in _MACRO_COLS_ALL:
        if col in result.columns and result[col].isna().all():
            logger.warning(f"{col}: macro source unavailable, feature will be excluded")

    return result
