# A-Share Batch Prediction Web UI Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Kronos Web UI to batch-predict Chinese A-share index constituent stocks, rank them by predicted price change, and show per-stock K-line detail views.

**Architecture:** Flask backend extended with 3 new modules (db.py, data_provider.py, scanner.py). SQLite for persistence. akshare for data. Background thread for batch scanning. Three HTML pages: scan/ranking home, stock detail, search.

**Tech Stack:** Python/Flask, SQLite, akshare, Plotly.js, existing Kronos model (`predict` / `predict_batch`)

**Spec:** `docs/superpowers/specs/2026-04-12-a-share-batch-prediction-webui-design.md`

---

## File Structure Overview

```
webui/
├── app.py                    # MODIFY: add new routes + API endpoints
├── run.py                    # UNCHANGED
├── db.py                     # CREATE: SQLite database manager
├── data_provider.py          # CREATE: akshare data fetching + cleaning + incremental update
├── scanner.py                # CREATE: batch scan background task
├── data/                     # auto-created at runtime for kronos.db
├── templates/
│   ├── index.html            # MODIFY: add scan panel + ranking table (new tab/section)
│   ├── stock_detail.html     # CREATE: single stock detail page
│   └── search.html           # CREATE: search prediction page
└── prediction_results/       # UNCHANGED (backward compat)
```

---

## Task 1: Database Module (`db.py`)

**Files:**
- Create: `webui/db.py`
- Create: `webui/tests/test_db.py`

- [ ] **Step 1: Write failing tests for db module**

Create `webui/tests/__init__.py` (empty) and `webui/tests/test_db.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/xukun/Documents/freqtrade/Kronos && python -m pytest webui/tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'db'`

- [ ] **Step 3: Implement `db.py`**

```python
import os
import sqlite3
import threading
import json
from datetime import datetime

import pandas as pd


class Database:
    def __init__(self, db_path=None):
        if db_path is None:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "kronos.db")
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS index_constituents (
                    index_code TEXT,
                    stock_code TEXT,
                    stock_name TEXT,
                    updated_at DATETIME,
                    PRIMARY KEY (index_code, stock_code)
                );
                CREATE TABLE IF NOT EXISTS kline_data (
                    stock_code TEXT,
                    freq TEXT,
                    dt DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    PRIMARY KEY (stock_code, freq, dt)
                );
                CREATE TABLE IF NOT EXISTS prediction_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id TEXT,
                    stock_code TEXT,
                    stock_name TEXT,
                    freq TEXT,
                    model_key TEXT,
                    predicted_at DATETIME,
                    last_close REAL,
                    pred_close REAL,
                    pred_change_pct REAL,
                    pred_data JSON,
                    params JSON
                );
                CREATE INDEX IF NOT EXISTS idx_pred_scan ON prediction_results(scan_id);
                CREATE INDEX IF NOT EXISTS idx_pred_stock ON prediction_results(stock_code, freq, predicted_at DESC);
                CREATE INDEX IF NOT EXISTS idx_kline_lookup ON kline_data(stock_code, freq, dt DESC);
            """)
            conn.close()

    def query(self, sql, params=None):
        """Execute a read-only query. No lock needed (WAL allows concurrent reads)."""
        conn = self._get_conn()
        try:
            return conn.execute(sql, params or ()).fetchall()
        finally:
            conn.close()

    def execute(self, sql, params=None):
        """Execute a write query with lock."""
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(sql, params or ())
                conn.commit()
                return cursor
            finally:
                conn.close()

    def close(self):
        pass  # connections are opened/closed per operation

    # --- Constituents ---

    def upsert_constituents(self, index_code, stocks):
        """stocks: list of (stock_code, stock_name) tuples."""
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                # Clear old entries for this index
                conn.execute("DELETE FROM index_constituents WHERE index_code = ?", (index_code,))
                conn.executemany(
                    "INSERT INTO index_constituents (index_code, stock_code, stock_name, updated_at) VALUES (?, ?, ?, ?)",
                    [(index_code, code, name, now) for code, name in stocks],
                )
                conn.commit()
            finally:
                conn.close()

    def get_constituents(self, index_code):
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM index_constituents WHERE index_code = ? ORDER BY stock_code",
                (index_code,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_constituents_updated_at(self, index_code):
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT MAX(updated_at) as updated_at FROM index_constituents WHERE index_code = ?",
                (index_code,),
            ).fetchone()
            return row["updated_at"] if row else None
        finally:
            conn.close()

    # --- Kline ---

    def upsert_kline(self, stock_code, freq, df):
        """df must have columns: dt, open, high, low, close, volume, amount."""
        with self._lock:
            conn = self._get_conn()
            try:
                for _, row in df.iterrows():
                    conn.execute(
                        """INSERT OR REPLACE INTO kline_data
                           (stock_code, freq, dt, open, high, low, close, volume, amount)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (stock_code, freq, str(row["dt"]), float(row["open"]),
                         float(row["high"]), float(row["low"]), float(row["close"]),
                         float(row["volume"]), float(row["amount"])),
                    )
                conn.commit()
            finally:
                conn.close()

    def get_last_kline_dt(self, stock_code, freq):
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT MAX(dt) as last_dt FROM kline_data WHERE stock_code = ? AND freq = ?",
                (stock_code, freq),
            ).fetchone()
            return row["last_dt"] if row and row["last_dt"] else None
        finally:
            conn.close()

    def get_kline(self, stock_code, freq, limit=400):
        """Return the most recent `limit` kline rows as DataFrame."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT dt, open, high, low, close, volume, amount
                   FROM kline_data WHERE stock_code = ? AND freq = ?
                   ORDER BY dt DESC LIMIT ?""",
                (stock_code, freq, limit),
            ).fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame([dict(r) for r in rows])
            df["dt"] = pd.to_datetime(df["dt"])
            df = df.sort_values("dt").reset_index(drop=True)
            return df
        finally:
            conn.close()

    # --- Predictions ---

    def save_prediction(self, scan_id, stock_code, stock_name, freq, model_key,
                        last_close, pred_close, pred_change_pct, pred_data, params):
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO prediction_results
                       (scan_id, stock_code, stock_name, freq, model_key, predicted_at,
                        last_close, pred_close, pred_change_pct, pred_data, params)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (scan_id, stock_code, stock_name, freq, model_key, now,
                     last_close, pred_close, pred_change_pct, pred_data, params),
                )
                conn.commit()
            finally:
                conn.close()

    def get_scan_results(self, scan_id, page=1, page_size=50):
        conn = self._get_conn()
        try:
            offset = (page - 1) * page_size
            rows = conn.execute(
                """SELECT * FROM prediction_results WHERE scan_id = ?
                   ORDER BY pred_change_pct DESC LIMIT ? OFFSET ?""",
                (scan_id, page_size, offset),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_scan_results_count(self, scan_id):
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM prediction_results WHERE scan_id = ?",
                (scan_id,),
            ).fetchone()
            return row["cnt"]
        finally:
            conn.close()

    def get_scan_history(self, page=1, page_size=50):
        conn = self._get_conn()
        try:
            offset = (page - 1) * page_size
            rows = conn.execute(
                """SELECT scan_id, freq, model_key, MIN(predicted_at) as started_at,
                          COUNT(*) as total_stocks
                   FROM prediction_results
                   GROUP BY scan_id
                   ORDER BY started_at DESC LIMIT ? OFFSET ?""",
                (page_size, offset),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_latest_prediction(self, stock_code, freq):
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT * FROM prediction_results
                   WHERE stock_code = ? AND freq = ?
                   ORDER BY predicted_at DESC LIMIT 1""",
                (stock_code, freq),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def search_stocks(self, query, limit=20):
        """Search stocks by code or name in cached constituents."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT DISTINCT stock_code, stock_name FROM index_constituents
                   WHERE stock_code LIKE ? OR stock_name LIKE ?
                   ORDER BY stock_code LIMIT ?""",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_stock_name(self, stock_code):
        """Look up stock name from constituents cache."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT stock_name FROM index_constituents WHERE stock_code = ? LIMIT 1",
                (stock_code,),
            ).fetchone()
            return row["stock_name"] if row else ""
        finally:
            conn.close()

    def get_scan_results_summary(self, scan_id, page=1, page_size=50):
        """Get scan results without pred_data (lightweight for ranking table)."""
        conn = self._get_conn()
        try:
            offset = (page - 1) * page_size
            rows = conn.execute(
                """SELECT scan_id, stock_code, stock_name, freq, model_key,
                          predicted_at, last_close, pred_close, pred_change_pct
                   FROM prediction_results WHERE scan_id = ?
                   ORDER BY pred_change_pct DESC LIMIT ? OFFSET ?""",
                (scan_id, page_size, offset),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/xukun/Documents/freqtrade/Kronos && python -m pytest webui/tests/test_db.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add webui/db.py webui/tests/__init__.py webui/tests/test_db.py
git commit -m "feat(webui): add SQLite database module for A-share data persistence"
```

---

## Task 2: Data Provider Module (`data_provider.py`)

**Files:**
- Create: `webui/data_provider.py`
- Create: `webui/tests/test_data_provider.py`

**Dependencies:** Task 1 (db.py)

- [ ] **Step 1: Write failing tests for data_provider**

Create `webui/tests/test_data_provider.py`:

```python
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


# --- Index constituents ---

INDEX_CONSTITUENTS = {
    "000300": "index_stock_cons_csindex",
}


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
    # Should be cached in DB
    cached = provider.db.get_constituents("000300")
    assert len(cached) == 3


@patch("data_provider.ak")
def test_fetch_constituents_uses_cache(mock_ak, provider):
    """Should use cached data if updated today."""
    mock_ak.index_stock_cons_csindex.return_value = _mock_constituents_df()
    provider.fetch_constituents("000300")  # first call populates cache
    provider.fetch_constituents("000300")  # second call should use cache
    assert mock_ak.index_stock_cons_csindex.call_count == 1


# --- Kline data ---

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
    # Check stored in DB
    stored = provider.db.get_kline("000001", "daily")
    assert len(stored) == 5


@patch("data_provider.ak")
def test_fetch_kline_incremental(mock_ak, provider):
    """Should only fetch new data on second call."""
    mock_ak.stock_zh_a_hist.return_value = _mock_daily_kline_df()
    provider.fetch_kline("000001", "daily")

    # Second call: mock returns more data (8 rows)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/xukun/Documents/freqtrade/Kronos && python -m pytest webui/tests/test_data_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_provider'`

- [ ] **Step 3: Implement `data_provider.py`**

```python
import time
from datetime import datetime, date

import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None


# Supported indices
INDEX_CONFIG = [
    {"code": "000300", "name": "CSI 300 (沪深300)", "count_approx": 300},
    {"code": "000905", "name": "CSI 500 (中证500)", "count_approx": 500},
    {"code": "000852", "name": "CSI 1000 (中证1000)", "count_approx": 1000},
]

# Column mapping from akshare Chinese names to English
DAILY_COL_MAP = {
    "日期": "dt",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}

MIN_COL_MAP = {
    "时间": "dt",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


class DataProvider:
    def __init__(self, db):
        self.db = db

    def get_index_list(self):
        return INDEX_CONFIG

    # --- Constituents ---

    def fetch_constituents(self, index_code, force=False):
        """Fetch index constituents. Uses cache if updated today unless force=True."""
        if not force:
            updated_at = self.db.get_constituents_updated_at(index_code)
            if updated_at:
                updated_date = datetime.fromisoformat(updated_at).date()
                if updated_date == date.today():
                    cached = self.db.get_constituents(index_code)
                    if cached:
                        return cached

        if ak is None:
            raise ImportError("akshare is not installed. Run: pip install akshare")

        try:
            df = ak.index_stock_cons_csindex(symbol=index_code)
        except Exception as e:
            # Fallback to cache
            cached = self.db.get_constituents(index_code)
            if cached:
                return cached
            raise RuntimeError(f"Failed to fetch constituents for {index_code}: {e}")

        code_col = "成分券代码" if "成分券代码" in df.columns else df.columns[0]
        name_col = "成分券名称" if "成分券名称" in df.columns else df.columns[1]

        stocks = [(str(row[code_col]).zfill(6), str(row[name_col])) for _, row in df.iterrows()]
        self.db.upsert_constituents(index_code, stocks)
        return self.db.get_constituents(index_code)

    # --- Kline ---

    def fetch_kline(self, stock_code, freq, max_retries=3):
        """Fetch kline data with incremental update. Returns DataFrame from DB."""
        last_dt = self.db.get_last_kline_dt(stock_code, freq)

        raw_df = self._fetch_kline_from_akshare(stock_code, freq, max_retries)
        if raw_df is None or raw_df.empty:
            # Return whatever is in DB
            return self.db.get_kline(stock_code, freq, limit=600)

        cleaned = self._clean_kline(raw_df, freq)

        # Incremental: only insert rows after last_dt
        if last_dt:
            last_dt_parsed = pd.to_datetime(last_dt)
            cleaned = cleaned[cleaned["dt"] > last_dt_parsed]

        if not cleaned.empty:
            self.db.upsert_kline(stock_code, freq, cleaned)

        return self.db.get_kline(stock_code, freq, limit=600)

    def _fetch_kline_from_akshare(self, stock_code, freq, max_retries=3):
        if ak is None:
            raise ImportError("akshare is not installed. Run: pip install akshare")

        for attempt in range(1, max_retries + 1):
            try:
                if freq == "daily":
                    df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="")
                else:  # 5min
                    df = ak.stock_zh_a_hist_min_em(symbol=stock_code, period="5", adjust="")
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass
            time.sleep(1.0)
        return None

    def _clean_kline(self, df, freq):
        """Clean and normalize kline data from akshare."""
        col_map = DAILY_COL_MAP if freq == "daily" else MIN_COL_MAP
        rename_map = {}
        for cn, en in col_map.items():
            if cn in df.columns:
                rename_map[cn] = en
        df = df.rename(columns=rename_map)

        df["dt"] = pd.to_datetime(df["dt"])
        df = df.sort_values("dt").reset_index(drop=True)

        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .str.replace(",", "", regex=False)
                    .replace({"--": None, "": None})
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fix invalid open values
        open_bad = (df["open"] == 0) | (df["open"].isna())
        if open_bad.any():
            df.loc[open_bad, "open"] = df["close"].shift(1)
            df["open"] = df["open"].fillna(df["close"])

        # Fix missing amount
        if "amount" not in df.columns or df["amount"].isna().all() or (df["amount"] == 0).all():
            df["amount"] = df["close"] * df["volume"]

        # Ensure volume column exists
        if "volume" not in df.columns:
            df["volume"] = 0

        df = df.dropna(subset=["open", "high", "low", "close"])
        return df[["dt", "open", "high", "low", "close", "volume", "amount"]]

    # --- Price limits ---

    @staticmethod
    def apply_price_limits(pred_df, last_close, limit_rate=0.1):
        """Apply A-share +-10% daily price limits to predictions."""
        pred_df = pred_df.reset_index(drop=True)
        cols = ["open", "high", "low", "close"]
        pred_df[cols] = pred_df[cols].astype("float64")

        for i in range(len(pred_df)):
            limit_up = last_close * (1 + limit_rate)
            limit_down = last_close * (1 - limit_rate)
            for col in cols:
                value = pred_df.at[i, col]
                if pd.notna(value):
                    pred_df.at[i, col] = float(max(min(value, limit_up), limit_down))
            last_close = float(pred_df.at[i, "close"])

        return pred_df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/xukun/Documents/freqtrade/Kronos && python -m pytest webui/tests/test_data_provider.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add webui/data_provider.py webui/tests/test_data_provider.py
git commit -m "feat(webui): add akshare data provider with incremental update and price limits"
```

---

## Task 3: Scanner Module (`scanner.py`)

**Files:**
- Create: `webui/scanner.py`
- Create: `webui/tests/test_scanner.py`

**Dependencies:** Task 1 (db.py), Task 2 (data_provider.py)

- [ ] **Step 1: Write failing tests for scanner**

Create `webui/tests/test_scanner.py`:

```python
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
    # predict returns a DataFrame
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

    # Pre-populate constituents
    db.upsert_constituents("000300", [("000001", "Test A"), ("000002", "Test B")])

    # Mock data_provider.fetch_kline to return sufficient data
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
    task.run()  # Run synchronously for testing

    state = task.get_state()
    assert state["status"] == "completed"
    assert state["completed"] == 2

    # Check predictions saved
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

    # Simulate stop after first stock by patching
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/xukun/Documents/freqtrade/Kronos && python -m pytest webui/tests/test_scanner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scanner'`

- [ ] **Step 3: Implement `scanner.py`**

```python
import json
import threading
from datetime import datetime

import numpy as np
import pandas as pd

from data_provider import DataProvider


FREQ_DEFAULTS = {
    "daily": {
        "lookback": 400,
        "pred_len": 10,
        "T": 1.0,
        "top_p": 0.9,
        "sample_count": 1,
    },
    "5min": {
        "lookback": 400,
        "pred_len": 48,
        "T": 1.0,
        "top_p": 0.9,
        "sample_count": 1,
    },
}

BATCH_SIZE = 8  # Number of stocks to predict together


class ScanTask:
    def __init__(self, db, data_provider, predictor, index_code, freq, model_key, params_override):
        self.db = db
        self.data_provider = data_provider
        self.predictor = predictor
        self.index_code = index_code
        self.freq = freq
        self.model_key = model_key

        defaults = FREQ_DEFAULTS.get(freq, FREQ_DEFAULTS["daily"]).copy()
        defaults.update(params_override)
        # Validate lookback < max_context
        max_ctx = getattr(predictor, 'max_context', 512)
        if defaults["lookback"] >= max_ctx:
            defaults["lookback"] = max_ctx - 112  # Leave room for autoregressive generation
        self.params = defaults

        self.scan_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._state = {
            "scan_id": self.scan_id,
            "status": "pending",
            "total": 0,
            "completed": 0,
            "current_stock": "",
            "errors": [],
            "started_at": None,
        }
        self._thread = None

    def get_state(self):
        return dict(self._state)

    def start(self):
        """Start scan in background thread."""
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        self._state["status"] = "stopped"

    def run(self):
        """Run the scan. Can be called directly for testing or via start() for background."""
        self._state["status"] = "running"
        self._state["started_at"] = datetime.now().isoformat()

        try:
            # Get constituent stocks
            constituents = self.data_provider.fetch_constituents(self.index_code)
            self._state["total"] = len(constituents)

            lookback = self.params["lookback"]
            pred_len = self.params["pred_len"]

            # Process in batches
            batch = []
            for stock in constituents:
                if self._state["status"] == "stopped":
                    break

                code = stock["stock_code"]
                name = stock["stock_name"]
                self._state["current_stock"] = f"{code} {name}"

                try:
                    # Fetch/update kline data
                    kline_df = self.data_provider.fetch_kline(code, self.freq)
                    if kline_df is None or len(kline_df) < lookback:
                        self._state["errors"].append(f"{code}: insufficient data ({len(kline_df) if kline_df is not None else 0} rows)")
                        self._state["completed"] += 1
                        continue

                    batch.append({
                        "code": code,
                        "name": name,
                        "kline_df": kline_df,
                    })

                    # Process batch when full or last stock
                    if len(batch) >= BATCH_SIZE or stock == constituents[-1]:
                        self._process_batch(batch, lookback, pred_len)
                        batch = []

                except Exception as e:
                    self._state["errors"].append(f"{code}: {str(e)}")
                    self._state["completed"] += 1

            if self._state["status"] == "running":
                self._state["status"] = "completed"
            self._state["current_stock"] = ""

        except Exception as e:
            self._state["status"] = "error"
            self._state["errors"].append(f"Scan error: {str(e)}")

    def _process_batch(self, batch, lookback, pred_len):
        """Process a batch of stocks through predict_batch."""
        df_list = []
        x_timestamp_list = []
        y_timestamp_list = []
        last_closes = []

        for item in batch:
            kline_df = item["kline_df"]
            x_df = kline_df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
            x_ts = pd.Series(kline_df.iloc[-lookback:]["dt"].values)

            last_close = float(kline_df.iloc[-1]["close"])
            last_closes.append(last_close)

            if self.freq == "daily":
                y_ts = pd.Series(pd.bdate_range(
                    start=kline_df["dt"].iloc[-1] + pd.Timedelta(days=1),
                    periods=pred_len
                ))
            else:
                last_dt = kline_df["dt"].iloc[-1]
                time_diff = kline_df["dt"].iloc[-1] - kline_df["dt"].iloc[-2] if len(kline_df) > 1 else pd.Timedelta(minutes=5)
                y_ts = pd.Series(pd.date_range(
                    start=last_dt + time_diff,
                    periods=pred_len,
                    freq=time_diff
                ))

            df_list.append(x_df)
            x_timestamp_list.append(x_ts)
            y_timestamp_list.append(y_ts)

        try:
            if len(df_list) == 1:
                pred_dfs = [self.predictor.predict(
                    df=df_list[0],
                    x_timestamp=x_timestamp_list[0],
                    y_timestamp=y_timestamp_list[0],
                    pred_len=pred_len,
                    T=self.params["T"],
                    top_p=self.params["top_p"],
                    sample_count=self.params["sample_count"],
                )]
            else:
                pred_dfs = self.predictor.predict_batch(
                    df_list=df_list,
                    x_timestamp_list=x_timestamp_list,
                    y_timestamp_list=y_timestamp_list,
                    pred_len=pred_len,
                    T=self.params["T"],
                    top_p=self.params["top_p"],
                    sample_count=self.params["sample_count"],
                )
        except Exception as e:
            # Fallback to single predictions
            pred_dfs = []
            for i in range(len(df_list)):
                try:
                    pred_df = self.predictor.predict(
                        df=df_list[i],
                        x_timestamp=x_timestamp_list[i],
                        y_timestamp=y_timestamp_list[i],
                        pred_len=pred_len,
                        T=self.params["T"],
                        top_p=self.params["top_p"],
                        sample_count=self.params["sample_count"],
                    )
                    pred_dfs.append(pred_df)
                except Exception:
                    pred_dfs.append(None)

        for i, item in enumerate(batch):
            if self._state["status"] == "stopped":
                break
            self._process_stock(item, pred_dfs[i] if i < len(pred_dfs) else None, last_closes[i])

    def _process_stock(self, item, pred_df, last_close):
        """Save a single stock's prediction result."""
        code = item["code"]
        name = item["name"]

        if pred_df is None:
            self._state["errors"].append(f"{code}: prediction failed")
            self._state["completed"] += 1
            return

        # Apply price limits
        pred_df = DataProvider.apply_price_limits(pred_df, last_close)

        pred_close = float(pred_df.iloc[-1]["close"])
        pred_change_pct = ((pred_close - last_close) / last_close) * 100

        # Serialize prediction data
        pred_data = pred_df.reset_index(drop=True).to_dict(orient="records")

        self.db.save_prediction(
            scan_id=self.scan_id,
            stock_code=code,
            stock_name=name,
            freq=self.freq,
            model_key=self.model_key,
            last_close=last_close,
            pred_close=pred_close,
            pred_change_pct=round(pred_change_pct, 4),
            pred_data=json.dumps(pred_data),
            params=json.dumps(self.params),
        )
        self._state["completed"] += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/xukun/Documents/freqtrade/Kronos && python -m pytest webui/tests/test_scanner.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add webui/scanner.py webui/tests/test_scanner.py
git commit -m "feat(webui): add batch scanner with batched prediction and stop support"
```

---

## Task 4: Backend API Routes (`app.py`)

**Files:**
- Modify: `webui/app.py`

**Dependencies:** Task 1, 2, 3

- [ ] **Step 1: Add imports and global state at top of `app.py`**

After the existing imports (line 11), add:

```python
from db import Database
from data_provider import DataProvider
from scanner import ScanTask
```

After `predictor = None` (line 31), add:

```python
# A-share modules (lazy initialized)
database = None
data_provider = None
current_scan = None  # Active ScanTask instance


def get_db():
    """Lazy-init database and data provider."""
    global database, data_provider
    if database is None:
        database = Database()
        data_provider = DataProvider(database)
    return database, data_provider
```

- [ ] **Step 2: Add `/api/index-list` endpoint**

After the existing `/api/model-status` route (~line 698), add:

```python
@app.route('/api/index-list')
def get_index_list():
    """Get available index list with constituent counts."""
    _, dp = get_db()
    indices = dp.get_index_list()
    return jsonify({'indices': indices})
```

- [ ] **Step 3: Add scan API endpoints**

```python
@app.route('/api/scan', methods=['POST'])
def start_scan():
    """Start batch scan."""
    global current_scan
    if current_scan and current_scan.get_state()["status"] == "running":
        return jsonify({'error': 'A scan is already running'}), 400
    if predictor is None:
        return jsonify({'error': 'Model not loaded. Please load a model first.'}), 400

    data = request.get_json()
    index_code = data.get('index_code', '000300')
    freq = data.get('freq', 'daily')
    model_key = data.get('model_key', 'kronos-small')
    params_override = {}
    for key in ['T', 'top_p', 'sample_count', 'lookback', 'pred_len']:
        if key in data:
            params_override[key] = float(data[key]) if key in ('T', 'top_p') else int(data[key])

    db, dp = get_db()
    current_scan = ScanTask(db, dp, predictor, index_code, freq, model_key, params_override)
    current_scan.start()
    return jsonify({'success': True, 'scan_id': current_scan.scan_id, 'message': 'Scan started'})


@app.route('/api/scan/status')
def scan_status():
    """Get scan progress with lightweight results (no pred_data)."""
    if current_scan is None:
        return jsonify({'status': 'idle', 'message': 'No scan running'})
    state = current_scan.get_state()
    # Include lightweight results (no pred_data blobs)
    if state["completed"] > 0:
        db, _ = get_db()
        results = db.get_scan_results_summary(state["scan_id"], page=1, page_size=1000)
        state["results"] = results
    return jsonify(state)


@app.route('/api/scan/stop', methods=['POST'])
def stop_scan():
    """Stop running scan."""
    if current_scan is None or current_scan.get_state()["status"] != "running":
        return jsonify({'error': 'No scan running'}), 400
    current_scan.stop()
    return jsonify({'success': True, 'message': 'Scan stop requested'})


@app.route('/api/scan/history')
def scan_history():
    """List historical scans."""
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 50))
    db, _ = get_db()
    history = db.get_scan_history(page, page_size)
    return jsonify({'history': history})


@app.route('/api/scan/<scan_id>/results')
def scan_results(scan_id):
    """Get results for a specific scan."""
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 50))
    db, _ = get_db()
    results = db.get_scan_results(scan_id, page, page_size)
    total = db.get_scan_results_count(scan_id)
    return jsonify({'results': results, 'total': total, 'page': page, 'page_size': page_size})
```

- [ ] **Step 4: Add single stock prediction and search endpoints**

```python
@app.route('/api/predict-symbol', methods=['POST'])
def predict_symbol():
    """Predict a single stock by code."""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 400

    data = request.get_json()
    stock_code = data.get('stock_code', '').strip()
    freq = data.get('freq', 'daily')

    if not stock_code:
        return jsonify({'error': 'stock_code is required'}), 400

    from scanner import FREQ_DEFAULTS
    params = FREQ_DEFAULTS.get(freq, FREQ_DEFAULTS["daily"]).copy()
    for key in ['T', 'top_p', 'sample_count']:
        if key in data:
            params[key] = float(data[key]) if key in ('T', 'top_p') else int(data[key])

    try:
        _, dp = get_db()
        kline_df = dp.fetch_kline(stock_code, freq)
        if kline_df is None or len(kline_df) < params["lookback"]:
            return jsonify({'error': f'Insufficient data for {stock_code}'}), 400

        lookback = params["lookback"]
        pred_len = params["pred_len"]
        x_df = kline_df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
        x_ts = pd.Series(kline_df.iloc[-lookback:]["dt"].values)

        if freq == "daily":
            y_ts = pd.Series(pd.bdate_range(
                start=kline_df["dt"].iloc[-1] + pd.Timedelta(days=1), periods=pred_len
            ))
        else:
            last_dt = kline_df["dt"].iloc[-1]
            td = kline_df["dt"].iloc[-1] - kline_df["dt"].iloc[-2] if len(kline_df) > 1 else pd.Timedelta(minutes=5)
            y_ts = pd.Series(pd.date_range(start=last_dt + td, periods=pred_len, freq=td))

        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=params["T"], top_p=params["top_p"],
            sample_count=params["sample_count"],
        )

        last_close = float(kline_df.iloc[-1]["close"])
        pred_df = DataProvider.apply_price_limits(pred_df, last_close)

        pred_close = float(pred_df.iloc[-1]["close"])
        pred_change_pct = ((pred_close - last_close) / last_close) * 100

        # Build chart data
        hist_data = kline_df.iloc[-lookback:].to_dict(orient="records")
        for row in hist_data:
            row["dt"] = str(row["dt"])
        pred_data = pred_df.reset_index(drop=True).to_dict(orient="records")
        pred_timestamps = [str(t) for t in y_ts]

        db, dp = get_db()
        stock_name = db.get_stock_name(stock_code)

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'stock_name': stock_name,
            'freq': freq,
            'last_close': last_close,
            'pred_close': pred_close,
            'pred_change_pct': round(pred_change_pct, 4),
            'historical': hist_data,
            'predictions': pred_data,
            'pred_timestamps': pred_timestamps,
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/search-stock')
def search_stock():
    """Search stock by code or name in cached constituents."""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'results': []})
    db, _ = get_db()
    results = db.search_stocks(query)
    return jsonify({'results': results})
```

- [ ] **Step 5: Add page routes for detail and search**

```python
@app.route('/stock/<symbol>')
def stock_detail(symbol):
    """Stock detail page."""
    freq = request.args.get('freq', 'daily')
    db, _ = get_db()
    prediction = db.get_latest_prediction(symbol, freq)
    return render_template('stock_detail.html', symbol=symbol, freq=freq, prediction=prediction)


@app.route('/search')
def search_page():
    """Search prediction page."""
    return render_template('search.html')
```

- [ ] **Step 6: Add model lock for scan safety**

In the existing `load_model()` function (~line 626), add a guard at the top:

```python
    # Block model switching during scan
    if current_scan and current_scan.get_state()["status"] == "running":
        return jsonify({'error': 'Cannot switch model while a scan is running'}), 400
```

- [ ] **Step 7: Commit**

```bash
git add webui/app.py
git commit -m "feat(webui): add A-share scan, predict-symbol, and search API endpoints"
```

---

## Task 5: Frontend — Scan Panel & Ranking Table (`index.html`)

**Files:**
- Modify: `webui/templates/index.html`

**Dependencies:** Task 4

- [ ] **Step 1: Add navigation tabs after the header section**

After the `<div class="header">` block (~line 446), add a tab navigation bar:

```html
        <!-- Navigation -->
        <div style="display:flex; gap:10px; margin-bottom:20px; justify-content:center;">
            <a href="/" class="btn" style="width:auto; padding:10px 24px; text-decoration:none;">Scan & Ranking</a>
            <a href="/search" class="btn btn-secondary" style="width:auto; padding:10px 24px; text-decoration:none;">Search Prediction</a>
        </div>
```

- [ ] **Step 2: Add A-share scan panel section in control panel**

Before the existing `<hr>` after model loading (~line 482), add a new section:

```html
                <hr style="margin: 20px 0; border: 1px solid #e2e8f0;">

                <!-- A-Share Batch Scan Section -->
                <h2 style="color:#4a5568; margin-bottom:15px; font-size:1.3rem;">A-Share Batch Scan</h2>

                <div class="form-group">
                    <label for="index-select">Select Index:</label>
                    <select id="index-select">
                        <option value="">Loading...</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="freq-select">Frequency:</label>
                    <select id="freq-select">
                        <option value="daily">Daily (日线)</option>
                        <option value="5min">5-Minute (5分钟线)</option>
                    </select>
                </div>

                <button id="scan-btn" class="btn btn-success" disabled>
                    Start Batch Scan
                </button>
                <button id="stop-scan-btn" class="btn btn-warning" style="display:none;">
                    Stop Scan
                </button>

                <!-- Scan Progress -->
                <div id="scan-progress" style="display:none; margin-top:15px;">
                    <div style="background:#e2e8f0; border-radius:8px; height:24px; overflow:hidden;">
                        <div id="scan-progress-bar" style="background:linear-gradient(135deg,#667eea,#764ba2); height:100%; width:0%; transition:width 0.3s; border-radius:8px; display:flex; align-items:center; justify-content:center; color:white; font-size:12px; font-weight:600;">
                            0%
                        </div>
                    </div>
                    <p id="scan-progress-text" style="font-size:12px; color:#718096; margin-top:5px;">Preparing...</p>
                </div>
```

- [ ] **Step 3: Add ranking table in the chart area**

After the chart container's closing `</div>` (~line 632), add:

```html
        <!-- Ranking Table -->
        <div id="ranking-section" class="chart-container" style="display:none; margin-top:20px;">
            <h2>Prediction Ranking</h2>
            <div style="margin-bottom:10px;">
                <select id="sort-select" style="padding:8px; border:1px solid #e2e8f0; border-radius:6px;">
                    <option value="pred_change_pct_desc">Predicted Change (High to Low)</option>
                    <option value="pred_change_pct_asc">Predicted Change (Low to High)</option>
                    <option value="stock_code_asc">Stock Code (Ascending)</option>
                </select>
            </div>
            <div style="max-height:600px; overflow-y:auto;">
                <table class="comparison-table" id="ranking-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Code</th>
                            <th>Name</th>
                            <th>Last Close</th>
                            <th>Predicted Change</th>
                            <th>Target Price</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody id="ranking-tbody"></tbody>
                </table>
            </div>
        </div>

        <!-- Scan History -->
        <div id="history-section" class="chart-container" style="margin-top:20px;">
            <h2>Scan History</h2>
            <div id="history-list" style="max-height:300px; overflow-y:auto;">
                <p style="color:#718096;">No scan history yet.</p>
            </div>
        </div>
```

- [ ] **Step 4: Add JavaScript for scan functionality**

At the end of the `<script>` section (before `</script>`), add:

```javascript
        // ===== A-Share Scan Functions =====
        let scanPollingInterval = null;

        async function loadIndexList() {
            try {
                const response = await axios.get('/api/index-list');
                const select = document.getElementById('index-select');
                select.innerHTML = '';
                response.data.indices.forEach(idx => {
                    const opt = document.createElement('option');
                    opt.value = idx.code;
                    opt.textContent = `${idx.name} (~${idx.count_approx} stocks)`;
                    select.appendChild(opt);
                });
            } catch (e) {
                console.error('Failed to load index list:', e);
            }
        }

        async function startScan() {
            const indexCode = document.getElementById('index-select').value;
            const freq = document.getElementById('freq-select').value;
            const modelKey = document.getElementById('model-select').value;
            const T = parseFloat(document.getElementById('temperature').value);
            const topP = parseFloat(document.getElementById('top-p').value);
            const sampleCount = parseInt(document.getElementById('sample-count').value);

            if (!modelKey || !modelLoaded) {
                showStatus('error', 'Please load a model first');
                return;
            }

            try {
                const response = await axios.post('/api/scan', {
                    index_code: indexCode, freq: freq, model_key: modelKey,
                    T: T, top_p: topP, sample_count: sampleCount,
                });
                if (response.data.success) {
                    document.getElementById('scan-btn').style.display = 'none';
                    document.getElementById('stop-scan-btn').style.display = 'block';
                    document.getElementById('scan-progress').style.display = 'block';
                    document.getElementById('ranking-section').style.display = 'block';
                    showStatus('success', 'Scan started: ' + response.data.scan_id);
                    scanPollingInterval = setInterval(pollScanStatus, 2000);
                }
            } catch (e) {
                showStatus('error', e.response?.data?.error || 'Failed to start scan');
            }
        }

        async function pollScanStatus() {
            try {
                const response = await axios.get('/api/scan/status');
                const state = response.data;
                if (state.status === 'idle') return;

                const pct = state.total > 0 ? Math.round((state.completed / state.total) * 100) : 0;
                const bar = document.getElementById('scan-progress-bar');
                bar.style.width = pct + '%';
                bar.textContent = pct + '%';
                document.getElementById('scan-progress-text').textContent =
                    `${state.completed}/${state.total} - ${state.current_stock || 'Processing...'}`;

                if (state.results) {
                    renderRanking(state.results);
                }

                if (state.status !== 'running') {
                    clearInterval(scanPollingInterval);
                    document.getElementById('scan-btn').style.display = 'block';
                    document.getElementById('stop-scan-btn').style.display = 'none';
                    showStatus(state.status === 'completed' ? 'success' : 'warning',
                        `Scan ${state.status}. ${state.completed}/${state.total} stocks processed.` +
                        (state.errors.length > 0 ? ` ${state.errors.length} errors.` : ''));
                    loadScanHistory();
                }
            } catch (e) {
                console.error('Poll error:', e);
            }
        }

        async function stopScan() {
            try {
                await axios.post('/api/scan/stop');
                showStatus('info', 'Stopping scan...');
            } catch (e) {
                showStatus('error', 'Failed to stop scan');
            }
        }

        function renderRanking(results) {
            // Cache results for re-sorting
            document.getElementById('ranking-tbody').dataset.lastResults = JSON.stringify(results);
            const sortKey = document.getElementById('sort-select').value;
            let sorted = [...results];
            if (sortKey === 'pred_change_pct_desc') sorted.sort((a, b) => b.pred_change_pct - a.pred_change_pct);
            else if (sortKey === 'pred_change_pct_asc') sorted.sort((a, b) => a.pred_change_pct - b.pred_change_pct);
            else if (sortKey === 'stock_code_asc') sorted.sort((a, b) => a.stock_code.localeCompare(b.stock_code));

            const tbody = document.getElementById('ranking-tbody');
            tbody.innerHTML = '';
            sorted.forEach((r, i) => {
                const color = r.pred_change_pct >= 0 ? '#EF5350' : '#26A69A';
                const sign = r.pred_change_pct >= 0 ? '+' : '';
                const targetPrice = r.last_close * (1 + r.pred_change_pct / 100);
                tbody.innerHTML += `<tr>
                    <td>${i + 1}</td>
                    <td>${r.stock_code}</td>
                    <td>${r.stock_name}</td>
                    <td>${r.last_close.toFixed(2)}</td>
                    <td style="color:${color}; font-weight:600;">${sign}${r.pred_change_pct.toFixed(2)}%</td>
                    <td>${targetPrice.toFixed(2)}</td>
                    <td><a href="/stock/${r.stock_code}?freq=${r.freq}" style="color:#667eea;">Detail</a></td>
                </tr>`;
            });
        }

        async function loadScanHistory() {
            try {
                const response = await axios.get('/api/scan/history');
                const div = document.getElementById('history-list');
                if (response.data.history.length === 0) {
                    div.innerHTML = '<p style="color:#718096;">No scan history yet.</p>';
                    return;
                }
                div.innerHTML = response.data.history.map(h =>
                    `<div style="padding:10px; border-bottom:1px solid #e2e8f0; cursor:pointer;" onclick="loadHistoryScan('${h.scan_id}')">
                        <strong>${h.scan_id}</strong> - ${h.freq} - ${h.model_key} - ${h.total_stocks} stocks
                        <span style="color:#718096; font-size:12px;"> ${h.started_at}</span>
                    </div>`
                ).join('');
            } catch (e) {
                console.error('Failed to load history:', e);
            }
        }

        async function loadHistoryScan(scanId) {
            try {
                const response = await axios.get(`/api/scan/${scanId}/results?page_size=1000`);
                document.getElementById('ranking-section').style.display = 'block';
                renderRanking(response.data.results);
            } catch (e) {
                showStatus('error', 'Failed to load scan results');
            }
        }

        // Wire up buttons and init
        document.addEventListener('DOMContentLoaded', function() {
            // Existing init calls are already here, add:
            loadIndexList();
            loadScanHistory();

            document.getElementById('scan-btn').addEventListener('click', startScan);
            document.getElementById('stop-scan-btn').addEventListener('click', stopScan);
            document.getElementById('sort-select').addEventListener('change', () => {
                // Re-sort cached results (from last poll or history load)
                const tbody = document.getElementById('ranking-tbody');
                if (tbody.dataset.lastResults) {
                    renderRanking(JSON.parse(tbody.dataset.lastResults));
                }
            });
        });

        // Note: also add `document.getElementById('scan-btn').disabled = false;`
        // inside the existing loadModel() function's success branch (after line 721:
        // `document.getElementById('predict-btn').disabled = false;`)
```

- [ ] **Step 5: Commit**

```bash
git add webui/templates/index.html
git commit -m "feat(webui): add A-share scan panel, ranking table, and progress UI"
```

---

## Task 6: Frontend — Stock Detail Page (`stock_detail.html`)

**Files:**
- Create: `webui/templates/stock_detail.html`

**Dependencies:** Task 4

- [ ] **Step 1: Create `stock_detail.html`**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kronos - {{ symbol }} Detail</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); min-height:100vh; color:#333; }
        .container { max-width:1400px; margin:0 auto; padding:20px; }
        .header { text-align:center; margin-bottom:20px; color:white; }
        .header h1 { font-size:2rem; margin-bottom:5px; }
        .card { background:white; border-radius:15px; padding:25px; box-shadow:0 10px 30px rgba(0,0,0,0.2); margin-bottom:20px; }
        .stock-info { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:15px; }
        .info-item { text-align:center; }
        .info-item .label { font-size:12px; color:#718096; }
        .info-item .value { font-size:1.5rem; font-weight:600; }
        .btn { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; border:none; padding:10px 20px; border-radius:8px; font-size:14px; font-weight:600; cursor:pointer; }
        .btn:hover { transform:translateY(-2px); box-shadow:0 5px 15px rgba(0,0,0,0.2); }
        .btn-secondary { background:linear-gradient(135deg,#718096 0%,#4a5568 100%); }
        #chart { width:100%; height:500px; }
        .pred-table { width:100%; border-collapse:collapse; margin-top:15px; }
        .pred-table th, .pred-table td { border:1px solid #e2e8f0; padding:8px; text-align:center; font-size:12px; }
        .pred-table th { background:#f7fafc; font-weight:600; }
        nav { display:flex; gap:10px; justify-content:center; margin-bottom:20px; }
        nav a { color:white; text-decoration:none; padding:8px 16px; border:1px solid rgba(255,255,255,0.3); border-radius:8px; }
        nav a:hover { background:rgba(255,255,255,0.1); }
        .spinner { border:4px solid #f3f3f3; border-top:4px solid #667eea; border-radius:50%; width:40px; height:40px; animation:spin 1s linear infinite; margin:20px auto; }
        @keyframes spin { 0%{transform:rotate(0deg);} 100%{transform:rotate(360deg);} }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ symbol }} Stock Prediction</h1>
        </div>
        <nav>
            <a href="/">Back to Scan</a>
            <a href="/search">Search</a>
        </nav>

        <div class="card">
            <div class="stock-info" id="stock-info">
                <div class="info-item"><div class="label">Code</div><div class="value">{{ symbol }}</div></div>
                <div class="info-item"><div class="label">Frequency</div><div class="value" id="info-freq">{{ freq }}</div></div>
                <div class="info-item"><div class="label">Last Close</div><div class="value" id="info-last-close">-</div></div>
                <div class="info-item"><div class="label">Predicted Change</div><div class="value" id="info-change">-</div></div>
                <div class="info-item"><div class="label">Target Price</div><div class="value" id="info-target">-</div></div>
            </div>
        </div>

        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                <h2 style="color:#4a5568;">K-Line Chart</h2>
                <button class="btn" id="repredict-btn" onclick="repredict()">Re-Predict</button>
            </div>
            <div id="loading" style="display:none;"><div class="spinner"></div><p style="text-align:center;">Predicting...</p></div>
            <div id="chart"></div>
        </div>

        <div class="card" id="pred-detail" style="display:none;">
            <h2 style="color:#4a5568; margin-bottom:15px;">Prediction Detail</h2>
            <div style="max-height:400px; overflow-y:auto;">
                <table class="pred-table">
                    <thead><tr><th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th></tr></thead>
                    <tbody id="pred-tbody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const symbol = "{{ symbol }}";
        const freq = "{{ freq }}";

        document.addEventListener('DOMContentLoaded', loadPrediction);

        async function loadPrediction() {
            document.getElementById('loading').style.display = 'block';
            try {
                const response = await axios.post('/api/predict-symbol', {
                    stock_code: symbol, freq: freq, sample_count: 3
                });
                if (response.data.success) {
                    renderResult(response.data);
                }
            } catch (e) {
                alert('Prediction failed: ' + (e.response?.data?.error || e.message));
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        function renderResult(data) {
            // Info bar
            document.getElementById('info-last-close').textContent = data.last_close.toFixed(2);
            const changePct = data.pred_change_pct;
            const changeEl = document.getElementById('info-change');
            changeEl.textContent = (changePct >= 0 ? '+' : '') + changePct.toFixed(2) + '%';
            changeEl.style.color = changePct >= 0 ? '#EF5350' : '#26A69A';
            document.getElementById('info-target').textContent = data.pred_close.toFixed(2);

            // Chart
            const hist = data.historical;
            const preds = data.predictions;
            const predTs = data.pred_timestamps;

            const histTrace = {
                x: hist.map(r => r.dt), open: hist.map(r => r.open),
                high: hist.map(r => r.high), low: hist.map(r => r.low), close: hist.map(r => r.close),
                type: 'candlestick', name: 'Historical',
                increasing: {line: {color: '#EF5350'}}, decreasing: {line: {color: '#26A69A'}},
            };
            const predTrace = {
                x: predTs, open: preds.map(r => r.open),
                high: preds.map(r => r.high), low: preds.map(r => r.low), close: preds.map(r => r.close),
                type: 'candlestick', name: 'Predicted',
                increasing: {line: {color: '#FF7043'}}, decreasing: {line: {color: '#66BB6A'}},
            };
            Plotly.newPlot('chart', [histTrace, predTrace], {
                title: `${symbol} - ${freq}`,
                xaxis: {title: 'Date', rangeslider: {visible: false}},
                yaxis: {title: 'Price'},
                template: 'plotly_white', height: 500,
            });

            // Detail table
            const tbody = document.getElementById('pred-tbody');
            tbody.innerHTML = '';
            preds.forEach((p, i) => {
                tbody.innerHTML += `<tr>
                    <td>${predTs[i]}</td>
                    <td>${p.open.toFixed(2)}</td><td>${p.high.toFixed(2)}</td>
                    <td>${p.low.toFixed(2)}</td><td>${p.close.toFixed(2)}</td>
                </tr>`;
            });
            document.getElementById('pred-detail').style.display = 'block';
        }

        async function repredict() {
            await loadPrediction();
        }
    </script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add webui/templates/stock_detail.html
git commit -m "feat(webui): add stock detail page with K-line chart and prediction"
```

---

## Task 7: Frontend — Search Page (`search.html`)

**Files:**
- Create: `webui/templates/search.html`

**Dependencies:** Task 4

- [ ] **Step 1: Create `search.html`**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kronos - Search Stock</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); min-height:100vh; color:#333; }
        .container { max-width:1200px; margin:0 auto; padding:20px; }
        .header { text-align:center; margin-bottom:20px; color:white; }
        .header h1 { font-size:2rem; margin-bottom:5px; }
        .card { background:white; border-radius:15px; padding:25px; box-shadow:0 10px 30px rgba(0,0,0,0.2); margin-bottom:20px; }
        .search-bar { display:flex; gap:10px; margin-bottom:15px; }
        .search-bar input { flex:1; padding:12px; border:2px solid #e2e8f0; border-radius:8px; font-size:16px; }
        .search-bar input:focus { outline:none; border-color:#667eea; }
        .search-bar select { padding:12px; border:2px solid #e2e8f0; border-radius:8px; font-size:14px; }
        .btn { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; border:none; padding:12px 24px; border-radius:8px; font-size:16px; font-weight:600; cursor:pointer; }
        .btn:hover { transform:translateY(-2px); box-shadow:0 5px 15px rgba(0,0,0,0.2); }
        .btn:disabled { opacity:0.6; cursor:not-allowed; }
        .suggestions { border:1px solid #e2e8f0; border-radius:8px; max-height:200px; overflow-y:auto; display:none; }
        .suggestion-item { padding:10px 15px; cursor:pointer; border-bottom:1px solid #f0f0f0; }
        .suggestion-item:hover { background:#f7fafc; }
        #chart { width:100%; height:500px; }
        .info-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px; margin-bottom:15px; }
        .info-item { text-align:center; padding:10px; background:#f7fafc; border-radius:8px; }
        .info-item .label { font-size:11px; color:#718096; }
        .info-item .value { font-size:1.3rem; font-weight:600; }
        nav { display:flex; gap:10px; justify-content:center; margin-bottom:20px; }
        nav a { color:white; text-decoration:none; padding:8px 16px; border:1px solid rgba(255,255,255,0.3); border-radius:8px; }
        nav a:hover { background:rgba(255,255,255,0.1); }
        .spinner { border:4px solid #f3f3f3; border-top:4px solid #667eea; border-radius:50%; width:40px; height:40px; animation:spin 1s linear infinite; margin:20px auto; }
        @keyframes spin { 0%{transform:rotate(0deg);} 100%{transform:rotate(360deg);} }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Search & Predict</h1>
            <p style="opacity:0.9;">Enter a stock code to predict</p>
        </div>
        <nav>
            <a href="/">Back to Scan</a>
            <a href="/search">Search</a>
        </nav>

        <div class="card">
            <div class="search-bar">
                <input type="text" id="search-input" placeholder="Enter stock code, e.g. 000001, 600519">
                <select id="search-freq">
                    <option value="daily">Daily</option>
                    <option value="5min">5-Min</option>
                </select>
                <button class="btn" id="search-btn" onclick="searchAndPredict()">Predict</button>
            </div>
            <div class="suggestions" id="suggestions"></div>
        </div>

        <!-- Recent Searches -->
        <div class="card" id="recent-card">
            <h3 style="color:#4a5568; margin-bottom:10px;">Recent Searches</h3>
            <div id="recent-list" style="color:#718096;">No recent searches.</div>
        </div>

        <div id="loading" style="display:none;"><div class="spinner"></div><p style="text-align:center; color:white;">Fetching data and predicting...</p></div>

        <div class="card" id="result-card" style="display:none;">
            <div class="info-grid" id="result-info"></div>
            <div id="chart"></div>
        </div>
    </div>

    <script>
        let searchTimeout = null;
        const input = document.getElementById('search-input');

        // Recent searches (localStorage)
        function getRecentSearches() {
            try { return JSON.parse(localStorage.getItem('kronos_recent_searches') || '[]'); } catch { return []; }
        }
        function addRecentSearch(code) {
            let recent = getRecentSearches().filter(c => c !== code);
            recent.unshift(code);
            if (recent.length > 10) recent = recent.slice(0, 10);
            localStorage.setItem('kronos_recent_searches', JSON.stringify(recent));
            renderRecentSearches();
        }
        function renderRecentSearches() {
            const recent = getRecentSearches();
            const div = document.getElementById('recent-list');
            if (recent.length === 0) { div.innerHTML = 'No recent searches.'; return; }
            div.innerHTML = recent.map(c =>
                `<span style="display:inline-block; padding:5px 12px; margin:3px; background:#f7fafc; border:1px solid #e2e8f0; border-radius:6px; cursor:pointer;" onclick="selectStock('${c}')">${c}</span>`
            ).join('');
        }
        document.addEventListener('DOMContentLoaded', renderRecentSearches);

        input.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            const q = this.value.trim();
            if (q.length < 1) { document.getElementById('suggestions').style.display = 'none'; return; }
            searchTimeout = setTimeout(() => fetchSuggestions(q), 300);
        });

        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') searchAndPredict();
        });

        async function fetchSuggestions(q) {
            try {
                const response = await axios.get('/api/search-stock?q=' + encodeURIComponent(q));
                const div = document.getElementById('suggestions');
                if (response.data.results.length === 0) { div.style.display = 'none'; return; }
                div.innerHTML = response.data.results.map(r =>
                    `<div class="suggestion-item" onclick="selectStock('${r.stock_code}')">${r.stock_code} - ${r.stock_name}</div>`
                ).join('');
                div.style.display = 'block';
            } catch (e) { /* ignore */ }
        }

        function selectStock(code) {
            input.value = code;
            document.getElementById('suggestions').style.display = 'none';
        }

        async function searchAndPredict() {
            const code = input.value.trim();
            const freq = document.getElementById('search-freq').value;
            if (!code) return;

            addRecentSearch(code);
            document.getElementById('suggestions').style.display = 'none';
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result-card').style.display = 'none';

            try {
                const response = await axios.post('/api/predict-symbol', {
                    stock_code: code, freq: freq, sample_count: 3,
                });
                if (response.data.success) {
                    renderSearchResult(response.data);
                }
            } catch (e) {
                alert('Failed: ' + (e.response?.data?.error || e.message));
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        function renderSearchResult(data) {
            const changePct = data.pred_change_pct;
            const color = changePct >= 0 ? '#EF5350' : '#26A69A';
            const sign = changePct >= 0 ? '+' : '';

            document.getElementById('result-info').innerHTML = `
                <div class="info-item"><div class="label">Code</div><div class="value">${data.stock_code}</div></div>
                <div class="info-item"><div class="label">Last Close</div><div class="value">${data.last_close.toFixed(2)}</div></div>
                <div class="info-item"><div class="label">Predicted Change</div><div class="value" style="color:${color}">${sign}${changePct.toFixed(2)}%</div></div>
                <div class="info-item"><div class="label">Target Price</div><div class="value">${data.pred_close.toFixed(2)}</div></div>
            `;

            const hist = data.historical;
            const preds = data.predictions;
            const predTs = data.pred_timestamps;

            Plotly.newPlot('chart', [
                { x: hist.map(r => r.dt), open: hist.map(r => r.open), high: hist.map(r => r.high),
                  low: hist.map(r => r.low), close: hist.map(r => r.close),
                  type: 'candlestick', name: 'Historical',
                  increasing: {line: {color: '#EF5350'}}, decreasing: {line: {color: '#26A69A'}} },
                { x: predTs, open: preds.map(r => r.open), high: preds.map(r => r.high),
                  low: preds.map(r => r.low), close: preds.map(r => r.close),
                  type: 'candlestick', name: 'Predicted',
                  increasing: {line: {color: '#FF7043'}}, decreasing: {line: {color: '#66BB6A'}} },
            ], {
                title: `${data.stock_code} - ${data.freq}`,
                xaxis: { title: 'Date', rangeslider: {visible: false} },
                yaxis: { title: 'Price' },
                template: 'plotly_white', height: 500,
            });

            document.getElementById('result-card').style.display = 'block';
        }
    </script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add webui/templates/search.html
git commit -m "feat(webui): add search prediction page with autocomplete"
```

---

## Task 8: Update Requirements & Final Integration

**Files:**
- Create or modify: `webui/requirements.txt`

**Dependencies:** All previous tasks

- [ ] **Step 1: Append akshare to requirements**

Check existing `webui/requirements.txt` (or project root `requirements.txt`). Do NOT overwrite — only append `akshare` if not already present:

```bash
# Check if akshare is already listed
grep -q "akshare" webui/requirements.txt || echo "akshare" >> webui/requirements.txt
```

- [ ] **Step 2: Run a smoke test — start the app and check endpoints**

Run: `cd /Users/xukun/Documents/freqtrade/Kronos && python -c "from webui.db import Database; from webui.data_provider import DataProvider; from webui.scanner import ScanTask; print('All modules import OK')"`
Expected: "All modules import OK"

- [ ] **Step 3: Run all tests**

Run: `cd /Users/xukun/Documents/freqtrade/Kronos && python -m pytest webui/tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add webui/requirements.txt
git commit -m "chore(webui): add akshare to requirements for A-share support"
```

---

## Task Summary & Dependencies

```
Task 1: db.py              (no deps)
Task 2: data_provider.py   (depends on Task 1)
Task 3: scanner.py         (depends on Task 1, 2)
Task 4: app.py routes      (depends on Task 1, 2, 3)
Task 5: index.html scan UI (depends on Task 4)
Task 6: stock_detail.html  (depends on Task 4)
Task 7: search.html        (depends on Task 4)
Task 8: requirements       (depends on all)
```

Tasks 1-3 are backend and must be sequential. Tasks 5, 6, 7 are independent frontend tasks and can be parallelized after Task 4. Task 8 is final integration.
