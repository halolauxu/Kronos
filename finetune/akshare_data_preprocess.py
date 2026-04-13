"""
akshare_data_preprocess.py

Replaces qlib_data_preprocess.py — uses akshare (163 source) to fetch
up-to-date A-share daily data for CSI300 constituent stocks.

Data is split into train/val/test and saved as pickle files compatible
with the existing dataset.py.

Usage:
    python3 akshare_data_preprocess.py
"""

import os
import sys
import time
import pickle
import pandas as pd
import akshare as ak
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config

config = Config()


def stock_code_to_163(code):
    """Convert 6-digit code to 163-style: sz000001 / sh600519."""
    code = str(code).zfill(6)
    return ('sh' + code) if code.startswith(('6', '9')) else ('sz' + code)


def fetch_constituents(index_code="000300"):
    """Fetch stock list. If index_code is 'all', use Qlib local list for full A-share."""
    if index_code == "all":
        return fetch_all_ashare_from_qlib()

    print(f"Fetching constituents for index {index_code}...")
    df = ak.index_stock_cons_csindex(symbol=index_code)
    code_col = "成分券代码" if "成分券代码" in df.columns else df.columns[0]
    name_col = "成分券名称" if "成分券名称" in df.columns else df.columns[1]
    stocks = [(str(row[code_col]).zfill(6), str(row[name_col])) for _, row in df.iterrows()]
    print(f"Found {len(stocks)} constituent stocks")
    return stocks


def fetch_all_ashare_from_qlib():
    """Get full A-share stock list from local Qlib data (no network needed)."""
    import os
    qlib_path = os.path.expanduser(config.qlib_data_path)
    all_file = os.path.join(qlib_path, "instruments", "all.txt")

    stocks = []
    with open(all_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            symbol = parts[0]  # e.g. SH600000, SZ000001
            # Skip indices (SH000xxx, SZ399xxx)
            if symbol.startswith("SH000") or symbol.startswith("SZ399"):
                continue
            # Convert to 6-digit code
            code = symbol[2:]
            stocks.append((code, ""))  # name will be empty, that's OK for training
    print(f"Found {len(stocks)} A-share stocks from Qlib local data")
    return stocks


def fetch_daily_kline(stock_code, max_retries=3):
    """Fetch daily kline from akshare 163 source."""
    symbol_163 = stock_code_to_163(stock_code)
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_a_daily(symbol=symbol_163)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        time.sleep(0.5)
    return None


def clean_kline(df, stock_code):
    """Clean and format kline data to match Qlib format."""
    df = df.copy()

    # Ensure date column, index named 'datetime' to match Qlib/dataset.py format
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.index.name = "datetime"

    # Select and rename columns to match Qlib format: open,high,low,close,vol,amt
    result = pd.DataFrame(index=df.index)
    result["open"] = pd.to_numeric(df["open"], errors="coerce")
    result["high"] = pd.to_numeric(df["high"], errors="coerce")
    result["low"] = pd.to_numeric(df["low"], errors="coerce")
    result["close"] = pd.to_numeric(df["close"], errors="coerce")
    result["vol"] = pd.to_numeric(df["volume"], errors="coerce")

    if "amount" in df.columns:
        result["amt"] = pd.to_numeric(df["amount"], errors="coerce")
    else:
        result["amt"] = result["close"] * result["vol"]

    # Fix zeros
    for col in ["open", "high", "low", "close"]:
        bad = (result[col] == 0) | result[col].isna()
        if bad.any():
            result.loc[bad, col] = result["close"].shift(1)
    result = result.dropna()

    return result


def main():
    # Config
    instrument = config.instrument
    index_map = {"csi300": "000300", "csi500": "000905", "csi800": "000906", "csi1000": "000852", "all": "all"}
    index_code = index_map.get(instrument, "000300")

    train_start, train_end = config.train_time_range
    val_start, val_end = config.val_time_range
    test_start, test_end = config.test_time_range

    print(f"=== Akshare Data Preprocessing ===")
    print(f"Index: {instrument} ({index_code})")
    print(f"Train: {train_start} ~ {train_end}")
    print(f"Val:   {val_start} ~ {val_end}")
    print(f"Test:  {test_start} ~ {test_end}")
    print()

    # Fetch constituent stocks
    stocks = fetch_constituents(index_code)

    # Fetch all kline data
    all_data = {}
    failed = []
    for code, name in tqdm(stocks, desc="Fetching kline data"):
        df = fetch_daily_kline(code)
        if df is not None and not df.empty:
            cleaned = clean_kline(df, code)
            if len(cleaned) >= config.lookback_window + config.predict_window:
                # Use SH/SZ prefix as key to match Qlib format
                prefix = "SH" if code.startswith(("6", "9")) else "SZ"
                key = f"{prefix}{code}"
                all_data[key] = cleaned
        else:
            failed.append(code)
        time.sleep(0.3)  # Rate limit

    print(f"\nFetched {len(all_data)} stocks, {len(failed)} failed")

    # Split into train/val/test
    train_data = {}
    val_data = {}
    test_data = {}

    for key, df in all_data.items():
        train_mask = (df.index >= train_start) & (df.index <= train_end)
        val_mask = (df.index >= val_start) & (df.index <= val_end)
        test_mask = (df.index >= test_start) & (df.index <= test_end)

        train_df = df[train_mask]
        val_df = df[val_mask]
        test_df = df[test_mask]

        if len(train_df) > 0:
            train_data[key] = train_df
        if len(val_df) > 0:
            val_data[key] = val_df
        if len(test_df) > 0:
            test_data[key] = test_df

    # Save
    save_dir = config.dataset_path
    os.makedirs(save_dir, exist_ok=True)

    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = os.path.join(save_dir, f"{name}_data.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        total_rows = sum(len(v) for v in data.values())
        non_empty = sum(1 for v in data.values() if len(v) > 0)
        if non_empty > 0:
            dates = []
            for v in data.values():
                if len(v) > 0:
                    dates.extend([v.index.min(), v.index.max()])
            print(f"  {name}: {non_empty} stocks, {total_rows:,} rows, {min(dates).date()} ~ {max(dates).date()}")
        else:
            print(f"  {name}: empty")

    print("\nDone! Datasets saved to", save_dir)


if __name__ == "__main__":
    main()
