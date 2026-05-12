import logging

import certifi
import requests
import pandas as pd

logger = logging.getLogger(__name__)

# Окно объявления дивиденда (дней до registryclosedate).
# Дивиденды в России объявляются за 2–4 нед. до закрытия реестра.
# 45 дней — консервативный запас, исключающий look-ahead bias.
_ANNOUNCE_WINDOW = 45

# Сентинельные значения для «нет информации»
_NO_UPCOMING = 999   # div_days_to_next когда нет объявленного дивиденда
_NO_PAST     = 999   # div_days_since_last до первой исторической выплаты


def load_dividend_features(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Time-varying дивидендные признаки для тикера из MOEX ISS.

    Работает для любого тикера MOEX (SBER, GAZP, LKOH, GMKN, …).

    Возвращает DataFrame с колонками:
        Date               — каждый календарный день в [start, end]
        div_days_to_next   — дней до следующей даты закрытия реестра.
                             Ненулевое только если объявлен (≤ _ANNOUNCE_WINDOW дней).
                             _NO_UPCOMING (999) = дивиденд не объявлен или информации нет.
        div_next_amount    — размер объявленного дивиденда на акцию (руб.).
                             0.0 если дивиденд не объявлен.
        div_days_since_last — дней с последнего закрытия реестра.
                              Монотонно растёт до следующей выплаты, сбрасывается.
                              _NO_PAST (999) до первой исторической выплаты.

    Look-ahead bias исключён: признак div_days_to_next ненулевой только если
    до даты закрытия реестра не более _ANNOUNCE_WINDOW дней — т.е. дивиденд
    к этому моменту уже точно объявлен (дата закрытия реестра публикуется в
    пресс-релизе СД примерно за 2–4 недели).

    При недоступности источника возвращает DataFrame с сентинельными значениями
    и логирует warning.
    """
    date_range = pd.date_range(start=start, end=end, freq='D')
    empty = pd.DataFrame({
        'Date': date_range,
        'div_days_to_next':    float(_NO_UPCOMING),
        'div_next_amount':     0.0,
        'div_days_since_last': float(_NO_PAST),
    })

    url = f"https://iss.moex.com/iss/securities/{ticker}/dividends.json"
    try:
        resp = requests.get(url, params={'iss.meta': 'off'}, timeout=15, verify=certifi.where())
        resp.raise_for_status()
        j = resp.json()
        cols_list = j['dividends']['columns']
        rows = j['dividends']['data']
    except Exception as e:
        logger.warning(f"dividends {ticker}: MOEX ISS недоступен ({e}) — признаки будут сентинельными")
        return empty

    if not rows:
        logger.warning(f"dividends {ticker}: нет данных о дивидендах — признаки будут сентинельными")
        return empty

    date_i = cols_list.index('registryclosedate')
    val_i  = cols_list.index('value')

    divs = sorted(
        [(pd.Timestamp(row[date_i]), float(row[val_i])) for row in rows],
        key=lambda x: x[0],
    )

    days_to_next_list    = []
    next_amount_list     = []
    days_since_last_list = []

    for current in date_range:
        # Прошедшие выплаты — определяем days_since_last
        past = [(d, a) for d, a in divs if d < current]
        if past:
            last_date, _ = past[-1]
            days_since_last_list.append((current - last_date).days)
        else:
            days_since_last_list.append(float(_NO_PAST))

        # Будущие/текущие выплаты — определяем days_to_next
        future = [(d, a) for d, a in divs if d >= current]
        if future:
            next_date, next_amt = future[0]
            days_to = (next_date - current).days
            if days_to <= _ANNOUNCE_WINDOW:
                days_to_next_list.append(float(days_to))
                next_amount_list.append(next_amt)
            else:
                days_to_next_list.append(float(_NO_UPCOMING))
                next_amount_list.append(0.0)
        else:
            days_to_next_list.append(float(_NO_UPCOMING))
            next_amount_list.append(0.0)

    result = pd.DataFrame({
        'Date':               date_range,
        'div_days_to_next':   days_to_next_list,
        'div_next_amount':    next_amount_list,
        'div_days_since_last': days_since_last_list,
    })

    n_announce = (result['div_days_to_next'] < _NO_UPCOMING).sum()
    logger.info(
        f"dividends {ticker}: {len(divs)} выплат загружено, "
        f"{n_announce} дней в announce-окне ({_ANNOUNCE_WINDOW} дн.), "
        f"диапазон {result['Date'].min().date()} — {result['Date'].max().date()}"
    )
    return result


def get_ticker_shortname(ticker: str) -> str:
    """Краткое наименование тикера из MOEX ISS. При ошибке возвращает сам тикер."""
    url = f"https://iss.moex.com/iss/securities/{ticker}.json"
    try:
        resp = requests.get(url, params={'iss.meta': 'off'}, timeout=10, verify=certifi.where())
        resp.raise_for_status()
        j = resp.json()
        cols = j['description']['columns']
        rows = j['description']['data']
        name_i = cols.index('name')
        val_i  = cols.index('value')
        shortname = next((r[val_i] for r in rows if r[name_i] == 'SHORTNAME'), None)
        if shortname:
            return shortname
    except Exception as e:
        logger.warning(f"shortname {ticker}: недоступен ({e})")
    return ticker
