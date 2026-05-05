import re
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
        return pd.Series(rates, index=pd.to_datetime(dates), name='usd_rub_hist')
    except Exception as e:
        logger.warning(f"usd_rub_hist: CBR USD/RUB history unavailable ({e})")
        return pd.Series(dtype=float)


def _load_cbr_key_rate(start: str, end: str) -> pd.Series:
    """Ключевая ставка ЦБ РФ из HTML-таблицы (без lxml)."""
    url = (
        "https://www.cbr.ru/hd_base/KeyRate/"
        f"?UniDbQuery.Posted=True&UniDbQuery.From={start}&UniDbQuery.To={end}"
    )
    try:
        resp = requests.get(url, timeout=15, verify=certifi.where())
        resp.raise_for_status()
        # Строки таблицы: <td>ДД.ММ.ГГГГ</td><td>X,XX</td>
        rows = re.findall(
            r'<td[^>]*>\s*(\d{2}\.\d{2}\.\d{4})\s*</td>\s*<td[^>]*>\s*([\d,]+)\s*</td>',
            resp.text
        )
        if not rows:
            raise ValueError("no rows parsed from HTML")
        dates = [datetime.strptime(r[0], '%d.%m.%Y') for r in rows]
        rates = [float(r[1].replace(',', '.')) for r in rows]
        return pd.Series(rates, index=pd.to_datetime(dates), name='cbr_rate')
    except Exception as e:
        logger.warning(f"cbr_rate: CBR key rate unavailable ({e})")
        return pd.Series(dtype=float)


def _load_moex_brent(start: str, end: str) -> pd.Series:
    """Цена Brent из MOEX ISS (фьючерс BRN, ближний контракт по месяцу)."""
    # MOEX использует формат BRN-M.YY, например BRN-3.25 для март 2025
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt   = datetime.strptime(end,   '%Y-%m-%d')

    all_frames = []
    year, month = start_dt.year, start_dt.month

    # Итерируем по месяцам с запасом +2 месяца (чтобы покрыть экспирацию)
    while datetime(year, month, 1) <= end_dt + timedelta(days=62):
        secid = f"BRN-{month}.{str(year)[-2:]}"
        url = (
            f"https://iss.moex.com/iss/history/engines/futures/markets/forts"
            f"/boards/RFUD/securities/{secid}/candles.json"
        )
        try:
            resp = requests.get(
                url,
                params={'interval': 24, 'from': start, 'till': end, 'iss.meta': 'off'},
                timeout=15,
                verify=certifi.where()
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
        .drop_duplicates('Date', keep='first')  # ближний = наименьший secid при сортировке
    )
    return pd.Series(
        combined['brent_price'].values,
        index=combined['Date'].values,
        name='brent_price'
    )


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
