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
                    params JSON,
                    backtest_dir_acc REAL DEFAULT NULL,
                    backtest_overall_correct INTEGER DEFAULT NULL
                );
                CREATE TABLE IF NOT EXISTS scan_jobs (
                    scan_id TEXT PRIMARY KEY,
                    index_code TEXT,
                    freq TEXT,
                    model_key TEXT,
                    status TEXT,
                    total INTEGER DEFAULT 0,
                    completed INTEGER DEFAULT 0,
                    errors TEXT DEFAULT '[]',
                    current_stock TEXT DEFAULT '',
                    started_at DATETIME,
                    updated_at DATETIME
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
                        last_close, pred_close, pred_change_pct, pred_data, params,
                        backtest_dir_acc=None, backtest_overall_correct=None):
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO prediction_results
                       (scan_id, stock_code, stock_name, freq, model_key, predicted_at,
                        last_close, pred_close, pred_change_pct, pred_data, params,
                        backtest_dir_acc, backtest_overall_correct)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (scan_id, stock_code, stock_name, freq, model_key, now,
                     last_close, pred_close, pred_change_pct, pred_data, params,
                     backtest_dir_acc, backtest_overall_correct),
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

    # --- Scan Jobs (persistent progress) ---

    def create_scan_job(self, scan_id, index_code, freq, model_key, total):
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO scan_jobs
                       (scan_id, index_code, freq, model_key, status, total, completed, errors, current_stock, started_at, updated_at)
                       VALUES (?, ?, ?, ?, 'running', ?, 0, '[]', '', ?, ?)""",
                    (scan_id, index_code, freq, model_key, total, now, now),
                )
                conn.commit()
            finally:
                conn.close()

    def update_scan_job(self, scan_id, **kwargs):
        if not kwargs:
            return
        kwargs["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [scan_id]
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(f"UPDATE scan_jobs SET {set_clause} WHERE scan_id = ?", values)
                conn.commit()
            finally:
                conn.close()

    def get_scan_job(self, scan_id):
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM scan_jobs WHERE scan_id = ?", (scan_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_latest_scan_job(self):
        """Get the most recent scan job (for auto-resume)."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM scan_jobs ORDER BY started_at DESC LIMIT 1").fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_scan_results_summary(self, scan_id, page=1, page_size=50):
        """Get scan results without pred_data (lightweight for ranking table)."""
        conn = self._get_conn()
        try:
            offset = (page - 1) * page_size
            rows = conn.execute(
                """SELECT scan_id, stock_code, stock_name, freq, model_key,
                          predicted_at, last_close, pred_close, pred_change_pct,
                          backtest_dir_acc, backtest_overall_correct
                   FROM prediction_results WHERE scan_id = ?
                   ORDER BY pred_change_pct DESC LIMIT ? OFFSET ?""",
                (scan_id, page_size, offset),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
