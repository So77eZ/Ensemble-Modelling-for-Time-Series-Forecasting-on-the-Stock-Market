import logging
import certifi
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


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
            timeout=30,
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
    """
    _MONTH_LETTERS = {
        1: 'F', 2: 'G', 3: 'H', 4: 'J',  5: 'K',  6: 'M',
        7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z',
    }
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt   = datetime.strptime(end,   '%Y-%m-%d')

    all_frames = []
    seen_secids = set()
    year, month = start_dt.year, start_dt.month

    while datetime(year, month, 1) <= end_dt + timedelta(days=62):
        secid = f"BR{_MONTH_LETTERS[month]}{str(year)[-1]}"
        if secid not in seen_secids:
            seen_secids.add(secid)
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
                            all_frames.append(pd.DataFrame({
                                'Date':        pd.to_datetime([r[di] for r in filtered]),
                                'brent_price': [float(r[ci]) for r in filtered],
                            }))
            except Exception:
                pass

        month += 1
        if month > 12:
            month = 1
            year += 1

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


# ── публичный интерфейс ───────────────────────────────────────────────────────

def load_macro_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Загружает макроэкономические признаки за период [start_date, end_date].

    Возвращает DataFrame с колонками:
        Date, usd_rub_hist, cbr_rate, brent_price

    Все даты в диапазоне присутствуют; пропуски заполнены ffill/bfill.
    При недоступности источника колонка содержит NaN и логируется warning.
    """
    # Полный диапазон дат (включая выходные — нужен для корректного ffill)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    result = pd.DataFrame({'Date': date_range})

    loaders = {
        'usd_rub_hist': _load_cbr_usd_rub_history,
        'cbr_rate':     _load_cbr_key_rate,
        'brent_price':  _load_moex_brent,
    }

    for col, loader in loaders.items():
        series = loader(start_date, end_date)
        if series.empty:
            result[col] = float('nan')
        else:
            series.index = pd.to_datetime(series.index).normalize()
            tmp = series.reindex(date_range)
            result[col] = tmp.values

    # ffill/bfill: заполняем пропуски из выходных и праздников
    result[list(loaders.keys())] = (
        result[list(loaders.keys())]
        .ffill()
        .bfill()
    )

    # Предупреждаем если целая колонка осталась NaN
    for col in loaders:
        if result[col].isna().all():
            logger.warning(f"{col}: macro source unavailable, feature will be excluded")

    return result
