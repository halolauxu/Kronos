import time
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

    def fetch_kline(self, stock_code, freq, max_retries=3):
        """Fetch kline data with incremental update. Returns DataFrame from DB."""
        last_dt = self.db.get_last_kline_dt(stock_code, freq)

        raw_df = self._fetch_kline_from_akshare(stock_code, freq, max_retries)
        if raw_df is None or raw_df.empty:
            return self.db.get_kline(stock_code, freq, limit=600)

        cleaned = self._clean_kline(raw_df, freq)

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
                else:
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
