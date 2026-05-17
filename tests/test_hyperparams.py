"""Тесты для save/load/resolve гиперпараметров (per-horizon + legacy)."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stock_modelv16 as smv16


# ---------------------------------------------------------------------------
# _hyperparams_path
# ---------------------------------------------------------------------------

def test_hyperparams_path_common():
    """Без horizon → legacy общий путь."""
    p = smv16._hyperparams_path('SBER')
    assert p.endswith('SBER_hyperparams.json')


def test_hyperparams_path_per_horizon():
    """С horizon → per-horizon путь с суффиксом."""
    for h in [1, 2, 3]:
        p = smv16._hyperparams_path('SBER', horizon=h)
        assert p.endswith(f'SBER_h{h}_hyperparams.json')


def test_hyperparams_path_different_for_horizons():
    """Per-horizon пути для разных горизонтов отличаются."""
    paths = {smv16._hyperparams_path('SBER', horizon=h) for h in [1, 2, 3]}
    assert len(paths) == 3


# ---------------------------------------------------------------------------
# save/load roundtrip
# ---------------------------------------------------------------------------

def _patch_hyperparams_dir(tmpdir, monkeypatch):
    """Подменяем HYPERPARAMS_DIR на tmpdir для изоляции от outputs/."""
    monkeypatch.setattr(smv16, 'HYPERPARAMS_DIR', str(tmpdir))


def test_save_load_common_roundtrip(tmp_path, monkeypatch):
    _patch_hyperparams_dir(tmp_path, monkeypatch)
    lstm = {'units': 64, 'dropout': 0.2, 'lr': 0.001}
    xgb = {'n_estimators': 100, 'max_depth': 3}
    smv16.save_hyperparams('TST', lstm, xgb)
    loaded_lstm, loaded_xgb = smv16.load_hyperparams('TST')
    assert loaded_lstm == lstm
    assert loaded_xgb == xgb


def test_save_load_per_horizon_roundtrip(tmp_path, monkeypatch):
    _patch_hyperparams_dir(tmp_path, monkeypatch)
    for h in [1, 2, 3]:
        lstm = {'units': 50 + h, 'dropout': 0.1 * h, 'lr': 0.001 * h}
        xgb = {'n_estimators': 50 * h}
        smv16.save_hyperparams('TST', lstm, xgb, horizon=h)
    for h in [1, 2, 3]:
        loaded_lstm, loaded_xgb = smv16.load_hyperparams('TST', horizon=h)
        assert loaded_lstm == {'units': 50 + h, 'dropout': 0.1 * h, 'lr': 0.001 * h}
        assert loaded_xgb == {'n_estimators': 50 * h}


def test_save_writes_schema_version(tmp_path, monkeypatch):
    """Сохранённый JSON содержит schema_version и horizon поле."""
    _patch_hyperparams_dir(tmp_path, monkeypatch)
    smv16.save_hyperparams('TST', {'u': 1}, {'n': 2}, horizon=2)
    with open(os.path.join(str(tmp_path), 'TST_h2_hyperparams.json')) as f:
        raw = json.load(f)
    assert raw['schema_version'] == smv16._HYPERPARAMS_SCHEMA_VERSION
    assert raw['horizon'] == 2
    assert 'timestamp' in raw
    assert raw['lstm'] == {'u': 1}
    assert raw['xgboost'] == {'n': 2}


def test_load_legacy_without_timestamp(tmp_path, monkeypatch):
    """Старый файл без timestamp читается без KeyError (.get с default)."""
    _patch_hyperparams_dir(tmp_path, monkeypatch)
    legacy = {'lstm': {'u': 10}, 'xgboost': {'n': 20}}  # без timestamp/schema
    with open(os.path.join(str(tmp_path), 'OLD_hyperparams.json'), 'w') as f:
        json.dump(legacy, f)
    lstm, xgb = smv16.load_hyperparams('OLD')
    assert lstm == {'u': 10}
    assert xgb == {'n': 20}


# ---------------------------------------------------------------------------
# load_hyperparams fallback chain
# ---------------------------------------------------------------------------

def test_load_per_horizon_falls_back_to_common(tmp_path, monkeypatch):
    """Per-horizon файла нет → подхватываем общий."""
    _patch_hyperparams_dir(tmp_path, monkeypatch)
    smv16.save_hyperparams('TST', {'u': 99}, {'n': 88})  # только общий
    lstm, xgb = smv16.load_hyperparams('TST', horizon=1)
    assert lstm == {'u': 99}
    assert xgb == {'n': 88}


def test_load_per_horizon_prefers_specific_over_common(tmp_path, monkeypatch):
    """Per-horizon файл приоритетнее общего."""
    _patch_hyperparams_dir(tmp_path, monkeypatch)
    smv16.save_hyperparams('TST', {'u': 1}, {'n': 1})              # общий
    smv16.save_hyperparams('TST', {'u': 2}, {'n': 2}, horizon=3)   # h=3 специфика
    lstm_h3, xgb_h3 = smv16.load_hyperparams('TST', horizon=3)
    assert lstm_h3 == {'u': 2}
    assert xgb_h3 == {'n': 2}
    # А для горизонта без специфики — общий
    lstm_h1, xgb_h1 = smv16.load_hyperparams('TST', horizon=1)
    assert lstm_h1 == {'u': 1}
    assert xgb_h1 == {'n': 1}


def test_load_returns_none_when_nothing_exists(tmp_path, monkeypatch):
    _patch_hyperparams_dir(tmp_path, monkeypatch)
    lstm, xgb = smv16.load_hyperparams('NOSUCH')
    assert lstm is None and xgb is None
    lstm, xgb = smv16.load_hyperparams('NOSUCH', horizon=1)
    assert lstm is None and xgb is None


# ---------------------------------------------------------------------------
# _resolve_horizon_hp
# ---------------------------------------------------------------------------

def test_resolve_uses_per_horizon_when_exists(tmp_path, monkeypatch):
    _patch_hyperparams_dir(tmp_path, monkeypatch)
    smv16.save_hyperparams('TST', {'u': 5}, {'n': 5}, horizon=2)
    fallback_lstm = {'u': 999}
    fallback_xgb = {'n': 999}
    lstm, xgb = smv16._resolve_horizon_hp('TST', 2, fallback_lstm, fallback_xgb)
    assert lstm == {'u': 5}
    assert xgb == {'n': 5}


def test_resolve_uses_common_when_per_horizon_missing(tmp_path, monkeypatch):
    _patch_hyperparams_dir(tmp_path, monkeypatch)
    smv16.save_hyperparams('TST', {'u': 7}, {'n': 7})  # только общий
    fallback = {'unused': True}
    lstm, xgb = smv16._resolve_horizon_hp('TST', 2, fallback, fallback)
    assert lstm == {'u': 7}
    assert xgb == {'n': 7}


def test_resolve_uses_fallback_when_nothing_saved(tmp_path, monkeypatch):
    _patch_hyperparams_dir(tmp_path, monkeypatch)
    fallback_lstm = {'u': 'fb'}
    fallback_xgb = {'n': 'fb'}
    lstm, xgb = smv16._resolve_horizon_hp('NOSUCH', 1, fallback_lstm, fallback_xgb)
    assert lstm is fallback_lstm
    assert xgb is fallback_xgb


# ---------------------------------------------------------------------------
# атомарность save (упрощённая проверка)
# ---------------------------------------------------------------------------

def test_save_does_not_leave_tmp_file(tmp_path, monkeypatch):
    """После успешного save .tmp файла не остаётся."""
    _patch_hyperparams_dir(tmp_path, monkeypatch)
    smv16.save_hyperparams('TST', {'u': 1}, {'n': 2}, horizon=1)
    files = os.listdir(str(tmp_path))
    assert 'TST_h1_hyperparams.json' in files
    assert all(not f.endswith('.tmp') for f in files)
