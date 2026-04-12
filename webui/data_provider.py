import time
import datetime as _dt
from datetime import datetime, date

import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None


INDEX_CONFIG = [
    {"code": "000300", "name": "CSI 300 (沪深300)", "count_approx": 300},
    {"code": "000905", "name": "CSI 500 (中证500)", "count_approx": 500},
    {"code": "000852", "name": "CSI 1000 (中证1000)", "count_approx": 1000},
]

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


def generate_ashare_timestamps(last_dt, freq, n):
    """Generate *n* future A-share trading timestamps after *last_dt*.

    A-share 5-minute bar schedule (bar timestamp = period end):
      Morning : 09:35, 09:40, ..., 11:25, 11:30   (24 bars)
      Afternoon: 13:05, 13:10, ..., 14:55, 15:00   (24 bars)
      Total per day: 48 bars.

    For daily frequency, returns business-day timestamps only.
    Weekends (Sat/Sun) are skipped; public holidays are NOT handled.
    """
    if freq == "daily":
        return pd.Series(
            pd.bdate_range(
                start=pd.Timestamp(last_dt) + pd.Timedelta(days=1), periods=n
            )
        )

    # --- 5-minute bars ---
    MORNING_FIRST = _dt.time(9, 35)
    MORNING_LAST = _dt.time(11, 30)
    AFTERNOON_FIRST = _dt.time(13, 5)
    AFTERNOON_LAST = _dt.time(15, 0)
    INTERVAL = pd.Timedelta(minutes=5)

    timestamps = []
    current = pd.Timestamp(last_dt) + INTERVAL

    while len(timestamps) < n:
        t = current.time()
        wd = current.weekday()

        # Skip weekends
        if wd >= 5:
            days_ahead = 7 - wd  # Mon
            current = current.normalize() + pd.Timedelta(days=days_ahead, hours=9, minutes=35)
            continue

        if t < MORNING_FIRST:
            # Before morning session -> jump to 09:35
            current = current.normalize() + pd.Timedelta(hours=9, minutes=35)
            continue

        if MORNING_FIRST <= t <= MORNING_LAST:
            timestamps.append(current)
            current += INTERVAL
            # If next tick crosses into lunch break, skip to afternoon
            if current.time() > MORNING_LAST and current.time() < AFTERNOON_FIRST:
                current = current.normalize() + pd.Timedelta(hours=13, minutes=5)
            continue

        if MORNING_LAST < t < AFTERNOON_FIRST:
            # Lunch break -> jump to 13:05
            current = current.normalize() + pd.Timedelta(hours=13, minutes=5)
            continue

        if AFTERNOON_FIRST <= t <= AFTERNOON_LAST:
            timestamps.append(current)
            current += INTERVAL
            # If next tick crosses market close, jump to next trading day
            if current.time() > AFTERNOON_LAST or current.time() < MORNING_FIRST:
                next_day = current.normalize() + pd.Timedelta(days=1)
                while next_day.weekday() >= 5:
                    next_day += pd.Timedelta(days=1)
                current = next_day + pd.Timedelta(hours=9, minutes=35)
            continue

        # After market close -> next trading day
        next_day = current.normalize() + pd.Timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += pd.Timedelta(days=1)
        current = next_day + pd.Timedelta(hours=9, minutes=35)

    return pd.Series(timestamps)


class DataProvider:
    def __init__(self, db):
        self.db = db

    def get_index_list(self):
        return INDEX_CONFIG

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
            cached = self.db.get_constituents(index_code)
            if cached:
                return cached
            raise RuntimeError(f"Failed to fetch constituents for {index_code}: {e}")

        code_col = "成分券代码" if "成分券代码" in df.columns else df.columns[0]
        name_col = "成分券名称" if "成分券名称" in df.columns else df.columns[1]

        stocks = [(str(row[code_col]).zfill(6), str(row[name_col])) for _, row in df.iterrows()]
        self.db.upsert_constituents(index_code, stocks)
        return self.db.get_constituents(index_code)

    def fetch_kline(self, stock_code, freq, max_retries=5):
        """Fetch kline data with incremental update. Returns DataFrame from DB.

        Strategy: try to fetch from akshare, but always fall back to DB cache.
        This means the first scan may have gaps, but subsequent scans fill them in.
        """
        last_dt = self.db.get_last_kline_dt(stock_code, freq)

        raw_df = self._fetch_kline_from_akshare(stock_code, freq, max_retries)
        if raw_df is not None and not raw_df.empty:
            cleaned = self._clean_kline(raw_df, freq)
            if last_dt:
                last_dt_parsed = pd.to_datetime(last_dt)
                cleaned = cleaned[cleaned["dt"] > last_dt_parsed]
            if not cleaned.empty:
                self.db.upsert_kline(stock_code, freq, cleaned)

        # Always return from DB (may have data from previous runs even if fetch failed)
        return self.db.get_kline(stock_code, freq, limit=600)

    def _fetch_kline_from_akshare(self, stock_code, freq, max_retries=5):
        if ak is None:
            raise ImportError("akshare is not installed. Run: pip install akshare")

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                if freq == "daily":
                    df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="")
                else:
                    df = ak.stock_zh_a_hist_min_em(symbol=stock_code, period="5", adjust="")
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                last_error = e
            # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s
            time.sleep(min(0.5 * (2 ** (attempt - 1)), 8))
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

        open_bad = (df["open"] == 0) | (df["open"].isna())
        if open_bad.any():
            df.loc[open_bad, "open"] = df["close"].shift(1)
            df["open"] = df["open"].fillna(df["close"])

        if "amount" not in df.columns or df["amount"].isna().all() or (df["amount"] == 0).all():
            df["amount"] = df["close"] * df["volume"]

        if "volume" not in df.columns:
            df["volume"] = 0

        df = df.dropna(subset=["open", "high", "low", "close"])
        return df[["dt", "open", "high", "low", "close", "volume", "amount"]]

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
