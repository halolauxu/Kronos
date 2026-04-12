# A Share Batch Prediction Web UI Design

## Overview

Extend the existing Flask-based Kronos Web UI to support batch prediction of Chinese A-share index constituent stocks, with a ranking dashboard, single-stock detail view, and search-based prediction. Data sourced from akshare, persisted in SQLite for incremental updates.

## Requirements

- Batch predict all constituent stocks of CSI 300 / CSI 500 / CSI 1000
- Support both daily and 5-minute K-line frequencies
- Display results as a ranked table sorted by predicted price change
- Click any stock to view detailed K-line chart with prediction overlay
- Search by stock code for on-demand single-stock prediction
- Persist K-line data and prediction results in SQLite to avoid redundant fetching
- Incremental data updates (only fetch new data after last stored record)
- Background scanning with real-time progress reporting

## Architecture

### Approach

Extend the existing `webui/app.py` Flask application. Use `threading.Thread` for background batch scanning. SQLite for persistence. No new infrastructure dependencies.

### File Structure

```
webui/
├── app.py                    # Main application (extend with new routes and APIs)
├── run.py                    # Entry point (unchanged)
├── db.py                     # NEW: SQLite database management
├── data_provider.py          # NEW: akshare data fetching + incremental update
├── scanner.py                # NEW: Batch scan task logic (background thread)
├── data/
│   └── kronos.db             # NEW: SQLite database file (auto-created)
├── templates/
│   ├── index.html            # MODIFY: Add scan panel, ranking table
│   ├── stock_detail.html     # NEW: Single stock detail page
│   └── search.html           # NEW: Search prediction page
└── prediction_results/       # Retained for backward compatibility
```

### Module Responsibilities

- **db.py**: Database connection management (`check_same_thread=False` + write lock), table creation/migration, generic CRUD methods for all three tables.
- **data_provider.py**: Wraps all akshare calls. Fetches index constituents, daily/5-min K-line data. Handles column renaming, type conversion, data cleaning (reused from `prediction_cn_markets_day.py`). Implements incremental update logic (query max `dt` from DB, fetch only newer data). Includes rate limiting (0.5-1s delay between requests) and retry (3 attempts).
- **scanner.py**: `ScanTask` class managing the background thread. Iterates constituent stocks, calls `data_provider` for data update, runs model prediction, applies price limits, computes predicted change %, writes results to DB, updates progress state.
- **app.py**: New routes and API endpoints. Calls the above modules. Retains all existing functionality.

## Database Schema

```sql
-- Index constituent stock cache
CREATE TABLE index_constituents (
    index_code TEXT,        -- "000300" / "000905" / "000852"
    stock_code TEXT,        -- "000001"
    stock_name TEXT,        -- e.g. "Ping An Bank"
    updated_at DATETIME,
    PRIMARY KEY (index_code, stock_code)
);

-- K-line data
CREATE TABLE kline_data (
    stock_code TEXT,
    freq TEXT,              -- "daily" / "5min"
    dt DATETIME,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    amount REAL,
    PRIMARY KEY (stock_code, freq, dt)
);

-- Prediction results
CREATE TABLE prediction_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT,           -- Batch scan ID, e.g. "20260412_153000"
    stock_code TEXT,
    stock_name TEXT,
    freq TEXT,
    model_key TEXT,
    predicted_at DATETIME,
    last_close REAL,
    pred_close REAL,
    pred_change_pct REAL,   -- Predicted change percentage
    pred_data JSON,         -- Full predicted K-line data
    params JSON             -- Prediction parameters (T, top_p, sample_count, etc.)
);
```

### Data Update Strategy

- **Constituents list**: Refresh at most once per day (check `updated_at`).
- **K-line data**: Incremental — query max `dt` from DB, fetch only newer records. First run fetches full history.
- **Prediction results**: Each scan produces a `scan_id`. Historical scan results are retained.

## Pages

### 1. Home / Scan Page (`/`)

- Index selector: CSI 300 / CSI 500 / CSI 1000
- Frequency selector: Daily / 5-min
- Model selector: kronos-mini / kronos-small / kronos-base
- Prediction parameter controls: Temperature, top_p, sample_count
- "Start Scan" button
- Real-time progress bar with current stock name
- Ranking table (appears as results come in):

| Rank | Code | Name | Last Close | Predicted Change | Target Price | Action |
|------|------|------|-----------|-----------------|-------------|--------|
| 1 | 600519 | ... | 1680.00 | +5.2% | 1767.36 | Detail |

- Default sort: predicted change descending
- Sortable columns: change, code, name
- A-share color scheme: red for up, green for down
- Historical scan list: view past scan results

### 2. Stock Detail Page (`/stock/<symbol>`)

- Stock info bar: code, name, latest price, predicted change
- Plotly K-line chart: historical candles + predicted candles (different color)
- Prediction detail table: each predicted candle's OHLC values
- "Re-predict" button with adjustable parameters
- Data priority: read from `prediction_results` table first; if absent or stale, predict on the fly

### 3. Search Page (`/search`)

- Input: stock code (e.g. 000001) or name fuzzy search
- On search: incremental update K-line data, run prediction, display K-line chart
- Recent search history (from DB)

## API Endpoints

### New Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/index-list` | Available indices and constituent counts |
| POST | `/api/scan` | Start batch scan (params: index, freq, model, prediction params) |
| GET | `/api/scan/status` | Get scan progress and completed results |
| POST | `/api/scan/stop` | Stop running scan |
| GET | `/api/scan/history` | List historical scan records |
| GET | `/api/scan/<scan_id>/results` | Get results for a specific scan |
| POST | `/api/predict-symbol` | Single stock prediction (input: code, freq) |
| GET | `/api/search-stock` | Search stock by code or name |

### Existing Endpoints (Retained)

All existing endpoints (`/api/load-model`, `/api/predict`, `/api/data-files`, etc.) remain unchanged for backward compatibility.

## Scan State Management

```python
scan_state = {
    "scan_id": "20260412_153000",
    "status": "running",      # running / completed / stopped / error
    "total": 300,
    "completed": 45,
    "current_stock": "000858 ...",
    "errors": [],
    "started_at": "...",
}
```

Frontend polls `/api/scan/status` every 2 seconds.

## Prediction Parameters by Frequency

`lookback` must be strictly less than `max_context` to leave room for autoregressive generation. For kronos-small/kronos-base (max_context=512), 5-min lookback is capped at 400. kronos-mini (max_context=2048) can use larger lookback values.

| Parameter | Daily | 5-min |
|-----------|-------|-------|
| lookback | 400 | 400 |
| pred_len | 10 | 48 |
| max_context | 512 | 512 |
| T | 1.0 | 1.0 |
| top_p | 0.9 | 0.9 |
| sample_count | 1 | 1 |

Note: `sample_count` defaults to 1 for batch scanning to keep scan time manageable. Single-stock predictions on the detail page can use higher values (e.g. 3-5).

## Key Implementation Details

- **Batched prediction**: The model provides `predict_batch()` in `model/kronos.py` which supports parallel inference on multiple time series. The scanner should use this: first fetch data for a batch of stocks (e.g. 8-16), then run `predict_batch()` on the batch together. This significantly reduces total scan time, especially on GPU. Data fetching remains serial due to akshare rate limits.
- **A-share price limits**: Apply +/-10% daily limit to all predictions (reuse logic from `prediction_cn_markets_day.py`).
- **A-share color scheme**: Red for up (`#EF5350`), green for down (`#26A69A`). Applied only to the new A-share pages (scan, detail, search). The existing CSV-based prediction page retains its original international color scheme.
- **SQLite concurrency**: Use `PRAGMA journal_mode=WAL` for better read/write concurrency. A single global `threading.Lock` guards all write operations. Read operations use separate connections and do not acquire the lock.
- **Rate limiting**: 0.5-1s delay between akshare API calls. 3 retries on failure. Skip stock on persistent failure.
- **Model reuse**: Model loaded once globally, shared across all predictions. Model switching is blocked while a scan is running.
- **Scan cancellation**: The background thread checks `scan_state["status"]` before processing each stock. On "stopped", it commits partial results and exits cleanly.
- **5-min data depth**: For 5-minute frequency, only fetch the last 60 trading days of history (sufficient for 400 lookback). Full history is not needed.
- **Constituent list fallback**: Use `ak.index_stock_cons_csindex(symbol=...)` for fetching. If the API call fails, fall back to the cached list in the database (if available).
- **Pagination**: `/api/scan/<scan_id>/results` and `/api/scan/history` support `page` and `page_size` query parameters. Default page_size=50.

## Database Indices

```sql
CREATE INDEX idx_pred_scan ON prediction_results(scan_id);
CREATE INDEX idx_pred_stock ON prediction_results(stock_code, freq, predicted_at DESC);
CREATE INDEX idx_kline_lookup ON kline_data(stock_code, freq, dt DESC);
```

## Dependencies

- `akshare` — must be added to `webui/requirements.txt` (currently only used in `examples/`)
- `sqlite3` — Python built-in
- No other new external dependencies
