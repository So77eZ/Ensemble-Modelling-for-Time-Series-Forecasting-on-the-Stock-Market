import pandas as pd
import pytest
from unittest.mock import patch
from macro_loader import load_macro_data


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_cbr_usd_xml(dates_rates: list[tuple]) -> bytes:
    """Генерирует минимальный XML ответ ЦБ РФ для USD/RUB."""
    rows = ""
    for date_str, rate in dates_rates:
        rows += f'<Record Date="{date_str}"><Value>{rate}</Value></Record>\n'
    return f'<ValCurs>{rows}</ValCurs>'.encode()


def _make_cbr_keyrate_html(dates_rates: list[tuple]) -> str:
    """Генерирует минимальный HTML с таблицей ставок ЦБ РФ."""
    rows = "".join(
        f"<tr><td>{d}</td><td>{r}</td></tr>" for d, r in dates_rates
    )
    return f"<html><body><table>{rows}</table></body></html>"


def _make_moex_candles_json(dates_closes: list[tuple]) -> dict:
    """Генерирует минимальный JSON ответ MOEX ISS candles."""
    data = [[d, None, None, None, c, None, None] for d, c in dates_closes]
    return {
        "history": {
            "columns": ["TRADEDATE", "OPEN", "LOW", "HIGH", "CLOSE", "VOLUME", "VALUE"],
            "data": data
        }
    }


# ── тест 1: колонки ───────────────────────────────────────────────────────────

def test_load_macro_data_columns():
    """load_macro_data возвращает DataFrame с нужными колонками."""
    with patch("macro_loader._load_cbr_usd_rub_history") as m_usd, \
         patch("macro_loader._load_cbr_key_rate") as m_cbr, \
         patch("macro_loader._load_moex_brent") as m_brent:

        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        m_usd.return_value = pd.Series([89.5, 90.1], index=idx, name="usd_rub_hist")
        m_cbr.return_value = pd.Series([16.0, 16.0], index=idx, name="cbr_rate")
        m_brent.return_value = pd.Series([75.2, 76.0], index=idx, name="brent_price")

        result = load_macro_data("2024-01-02", "2024-01-03")

    assert set(result.columns) >= {"Date", "usd_rub_hist", "cbr_rate", "brent_price"}
    assert len(result) >= 1


# ── тест 2: fallback при падении источника ───────────────────────────────────

def test_fallback_on_failed_source(caplog):
    """Если один загрузчик падает — его колонка NaN, остальные целы."""
    import logging
    with patch("macro_loader._load_cbr_usd_rub_history") as m_usd, \
         patch("macro_loader._load_cbr_key_rate") as m_cbr, \
         patch("macro_loader._load_moex_brent") as m_brent, \
         caplog.at_level(logging.WARNING, logger="macro_loader"):

        idx = pd.to_datetime(["2024-01-02"])
        m_usd.return_value = pd.Series([89.5], index=idx, name="usd_rub_hist")
        m_cbr.return_value = pd.Series(dtype=float)        # пустой → NaN-колонка
        m_brent.return_value = pd.Series([75.0], index=idx, name="brent_price")

        result = load_macro_data("2024-01-02", "2024-01-02")

    assert "usd_rub_hist" in result.columns
    assert "brent_price" in result.columns
    assert "cbr_rate" in result.columns
    # cbr_rate должен быть NaN (пустой ряд, нечем заполнить)
    assert result["cbr_rate"].isna().all()


# ── тест 3: покрытие дат ─────────────────────────────────────────────────────

def test_date_range_coverage():
    """После ffill/bfill нет NaN при частичных пропусках (выходные)."""
    with patch("macro_loader._load_cbr_usd_rub_history") as m_usd, \
         patch("macro_loader._load_cbr_key_rate") as m_cbr, \
         patch("macro_loader._load_moex_brent") as m_brent:

        # Данные только за пн и ср — вт пропущен (выходной/нет данных)
        idx = pd.to_datetime(["2024-01-08", "2024-01-10"])
        m_usd.return_value = pd.Series([89.5, 90.1], index=idx, name="usd_rub_hist")
        m_cbr.return_value = pd.Series([16.0, 16.0], index=idx, name="cbr_rate")
        m_brent.return_value = pd.Series([75.2, 76.0], index=idx, name="brent_price")

        result = load_macro_data("2024-01-08", "2024-01-10")

    # Все строки с реальными датами (08, 09, 10) должны быть заполнены
    for col in ["usd_rub_hist", "cbr_rate", "brent_price"]:
        assert not result[col].isna().any(), f"{col} contains NaN after ffill/bfill"
