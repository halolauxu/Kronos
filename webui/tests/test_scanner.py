import os
import sys
import tempfile
import json
import time
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import Database
from data_provider import DataProvider
from scanner import ScanTask, FREQ_DEFAULTS


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        database = Database(db_path)
        yield database
        database.close()


@pytest.fixture
def provider(db):
    return DataProvider(db)


def _make_mock_predictor():
    predictor = MagicMock()
    pred_df = pd.DataFrame({
        "open": [10.5, 10.8],
        "high": [11.0, 11.2],
        "low": [10.0, 10.5],
        "close": [10.8, 11.0],
        "volume": [1000, 1100],
        "amount": [10800, 12100],
    })
    predictor.predict.return_value = pred_df
    predictor.predict_batch.return_value = [pred_df, pred_df]
    predictor.max_context = 512
    return predictor


def test_freq_defaults():
    """Should have correct default params for daily and 5min."""
    assert FREQ_DEFAULTS["daily"]["lookback"] == 400
    assert FREQ_DEFAULTS["daily"]["pred_len"] == 10
    assert FREQ_DEFAULTS["5min"]["lookback"] == 400
    assert FREQ_DEFAULTS["5min"]["pred_len"] == 48


def test_scan_state_initial():
    """Scan task should start with correct initial state."""
    db = MagicMock()
    provider = MagicMock()
    predictor = _make_mock_predictor()
    task = ScanTask(db, provider, predictor, "000300", "daily", "kronos-small", {})
    state = task.get_state()
    assert state["status"] == "pending"
    assert state["scan_id"] is not None


def test_scan_processes_stocks(db, provider):
    """Scan should process each stock and save results."""
    predictor = _make_mock_predictor()

    db.upsert_constituents("000300", [("000001", "Test A"), ("000002", "Test B")])

    kline_df = pd.DataFrame({
        "dt": pd.bdate_range("2024-01-01", periods=450),
        "open": np.random.uniform(10, 12, 450),
        "high": np.random.uniform(11, 13, 450),
        "low": np.random.uniform(9, 11, 450),
        "close": np.random.uniform(10, 12, 450),
        "volume": np.random.uniform(800, 1500, 450),
        "amount": np.random.uniform(8000, 15000, 450),
    })
    provider.fetch_kline = MagicMock(return_value=kline_df)
    provider.fetch_constituents = MagicMock(return_value=db.get_constituents("000300"))

    task = ScanTask(db, provider, predictor, "000300", "daily", "kronos-small", {})
    task.run()

    state = task.get_state()
    assert state["status"] == "completed"
    assert state["completed"] == 2

    results = db.get_scan_results(state["scan_id"])
    assert len(results) == 2


def test_scan_stop(db, provider):
    """Scan should stop when requested."""
    predictor = _make_mock_predictor()
    db.upsert_constituents("000300", [("000001", "A"), ("000002", "B"), ("000003", "C")])

    kline_df = pd.DataFrame({
        "dt": pd.bdate_range("2024-01-01", periods=450),
        "open": np.random.uniform(10, 12, 450),
        "high": np.random.uniform(11, 13, 450),
        "low": np.random.uniform(9, 11, 450),
        "close": np.random.uniform(10, 12, 450),
        "volume": np.random.uniform(800, 1500, 450),
        "amount": np.random.uniform(8000, 15000, 450),
    })
    provider.fetch_kline = MagicMock(return_value=kline_df)
    provider.fetch_constituents = MagicMock(return_value=db.get_constituents("000300"))

    task = ScanTask(db, provider, predictor, "000300", "daily", "kronos-small", {})

    original_run = task._process_stock
    call_count = [0]
    def stop_after_first(*args, **kwargs):
        result = original_run(*args, **kwargs)
        call_count[0] += 1
        if call_count[0] >= 1:
            task.stop()
        return result
    task._process_stock = stop_after_first

    task.run()
    state = task.get_state()
    assert state["status"] == "stopped"
    assert state["completed"] >= 1
