import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import Database


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        database = Database(db_path)
        yield database
        database.close()


def test_tables_created(db):
    """All three tables should exist after init."""
    tables = db.query("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = {row[0] for row in tables}
    assert "index_constituents" in table_names
    assert "kline_data" in table_names
    assert "prediction_results" in table_names


def test_upsert_constituents(db):
    """Should insert and update constituent stocks."""
    stocks = [("000001", "Ping An Bank"), ("000002", "Vanke A")]
    db.upsert_constituents("000300", stocks)
    rows = db.get_constituents("000300")
    assert len(rows) == 2
    assert rows[0]["stock_code"] == "000001"


def test_upsert_kline(db):
    """Should insert kline rows and support incremental queries."""
    import pandas as pd
    data = pd.DataFrame({
        "dt": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "open": [10.0, 10.5],
        "high": [10.5, 11.0],
        "low": [9.5, 10.0],
        "close": [10.2, 10.8],
        "volume": [1000, 1200],
        "amount": [10200, 12960],
    })
    db.upsert_kline("000001", "daily", data)
    last_dt = db.get_last_kline_dt("000001", "daily")
    assert str(last_dt) == "2026-01-02 00:00:00"


def test_get_kline(db):
    """Should retrieve kline data as DataFrame."""
    import pandas as pd
    data = pd.DataFrame({
        "dt": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "open": [10.0, 10.5, 11.0],
        "high": [10.5, 11.0, 11.5],
        "low": [9.5, 10.0, 10.5],
        "close": [10.2, 10.8, 11.2],
        "volume": [1000, 1200, 1100],
        "amount": [10200, 12960, 12320],
    })
    db.upsert_kline("000001", "daily", data)
    df = db.get_kline("000001", "daily", limit=2)
    assert len(df) == 2
    # Should return the last 2 rows (most recent)
    assert df.iloc[-1]["close"] == 11.2


def test_save_and_get_prediction(db):
    """Should save and retrieve prediction results."""
    import json
    db.save_prediction(
        scan_id="20260412_153000",
        stock_code="000001",
        stock_name="Ping An Bank",
        freq="daily",
        model_key="kronos-small",
        last_close=10.0,
        pred_close=10.5,
        pred_change_pct=5.0,
        pred_data=json.dumps([{"open": 10.1, "close": 10.5}]),
        params=json.dumps({"T": 1.0}),
    )
    results = db.get_scan_results("20260412_153000")
    assert len(results) == 1
    assert results[0]["pred_change_pct"] == 5.0


def test_get_scan_history(db):
    """Should list distinct scan IDs."""
    import json
    for scan_id in ["20260412_100000", "20260412_110000"]:
        db.save_prediction(
            scan_id=scan_id, stock_code="000001", stock_name="Test",
            freq="daily", model_key="kronos-small",
            last_close=10.0, pred_close=10.5, pred_change_pct=5.0,
            pred_data=json.dumps([]), params=json.dumps({}),
        )
    history = db.get_scan_history()
    assert len(history) == 2
