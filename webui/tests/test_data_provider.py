import os
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import Database
from data_provider import DataProvider


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


def _mock_constituents_df():
    return pd.DataFrame({
        "成分券代码": ["000001", "000002", "600519"],
        "成分券名称": ["平安银行", "万科A", "贵州茅台"],
    })


def test_get_index_list(provider):
    """Should return the supported index list."""
    indices = provider.get_index_list()
    assert len(indices) >= 3
    assert any(i["code"] == "000300" for i in indices)


@patch("data_provider.ak")
def test_fetch_constituents(mock_ak, provider):
    """Should fetch and cache constituents via akshare."""
    mock_ak.index_stock_cons_csindex.return_value = _mock_constituents_df()
    stocks = provider.fetch_constituents("000300")
    assert len(stocks) == 3
    assert stocks[0]["stock_code"] == "000001"
    cached = provider.db.get_constituents("000300")
    assert len(cached) == 3


@patch("data_provider.ak")
def test_fetch_constituents_uses_cache(mock_ak, provider):
    """Should use cached data if updated today."""
    mock_ak.index_stock_cons_csindex.return_value = _mock_constituents_df()
    provider.fetch_constituents("000300")
    provider.fetch_constituents("000300")
    assert mock_ak.index_stock_cons_csindex.call_count == 1


def _mock_daily_kline_df():
    return pd.DataFrame({
        "日期": pd.date_range("2025-01-01", periods=5, freq="B"),
        "开盘": [10.0, 10.5, 11.0, 10.8, 11.2],
        "最高": [10.5, 11.0, 11.5, 11.2, 11.5],
        "最低": [9.5, 10.0, 10.5, 10.5, 10.8],
        "收盘": [10.2, 10.8, 11.2, 10.9, 11.3],
        "成交量": [1000, 1200, 1100, 900, 1300],
        "成交额": [10200, 12960, 12320, 9810, 14690],
    })


@patch("data_provider.ak")
def test_fetch_kline_daily(mock_ak, provider):
    """Should fetch daily kline and store in DB."""
    mock_ak.stock_zh_a_hist.return_value = _mock_daily_kline_df()
    df = provider.fetch_kline("000001", "daily")
    assert len(df) == 5
    assert "open" in df.columns
    stored = provider.db.get_kline("000001", "daily")
    assert len(stored) == 5


@patch("data_provider.ak")
def test_fetch_kline_incremental(mock_ak, provider):
    """Should only fetch new data on second call."""
    mock_ak.stock_zh_a_hist.return_value = _mock_daily_kline_df()
    provider.fetch_kline("000001", "daily")

    extended = pd.DataFrame({
        "日期": pd.date_range("2025-01-01", periods=8, freq="B"),
        "开盘": [10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.3, 11.8],
        "最高": [10.5, 11.0, 11.5, 11.2, 11.5, 12.0, 11.8, 12.2],
        "最低": [9.5, 10.0, 10.5, 10.5, 10.8, 11.0, 11.0, 11.5],
        "收盘": [10.2, 10.8, 11.2, 10.9, 11.3, 11.8, 11.5, 12.0],
        "成交量": [1000, 1200, 1100, 900, 1300, 1400, 1100, 1500],
        "成交额": [10200, 12960, 12320, 9810, 14690, 16520, 12650, 18000],
    })
    mock_ak.stock_zh_a_hist.return_value = extended
    df = provider.fetch_kline("000001", "daily")
    stored = provider.db.get_kline("000001", "daily", limit=100)
    assert len(stored) == 8


def test_clean_kline_data(provider):
    """Should handle column renaming and type conversion."""
    raw = pd.DataFrame({
        "日期": ["2025-01-01"],
        "开盘": ["10.0"],
        "最高": ["10.5"],
        "最低": ["9.5"],
        "收盘": ["10.2"],
        "成交量": ["1,000"],
        "成交额": ["10,200"],
    })
    cleaned = provider._clean_kline(raw, "daily")
    assert cleaned["volume"].iloc[0] == 1000
    assert cleaned["amount"].iloc[0] == 10200


def test_apply_price_limits(provider):
    """Should clip prices within +-10%."""
    pred_df = pd.DataFrame({
        "open": [12.0], "high": [13.0], "low": [8.0], "close": [12.5],
    })
    result = provider.apply_price_limits(pred_df, last_close=10.0, limit_rate=0.1)
    assert result["high"].iloc[0] <= 11.0
    assert result["low"].iloc[0] >= 9.0
