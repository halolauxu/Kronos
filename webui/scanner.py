import json
import threading
from datetime import datetime

import numpy as np
import pandas as pd

from data_provider import DataProvider, generate_ashare_timestamps


FREQ_DEFAULTS = {
    "daily": {
        "lookback": 400,
        "pred_len": 3,       # 3 trading days
        "T": 0.6,            # Lower T = less random, more conservative predictions
        "top_p": 0.9,
        "sample_count": 5,   # Average 5 samples for more stable results
    },
    "5min": {
        "lookback": 400,
        "pred_len": 48,      # 1 trading day
        "T": 0.6,
        "top_p": 0.9,
        "sample_count": 3,
    },
}

BATCH_SIZE = 8


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
            defaults["lookback"] = max_ctx - 112
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

    def _sync_to_db(self):
        """Persist current scan state to DB so page refresh doesn't lose progress."""
        try:
            self.db.update_scan_job(
                self.scan_id,
                status=self._state["status"],
                completed=self._state["completed"],
                current_stock=self._state["current_stock"],
                errors=json.dumps(self._state["errors"][-50:]),  # keep last 50
            )
        except Exception:
            pass  # non-critical

    def run(self):
        """Run the scan synchronously. Called directly for testing or via start() for background."""
        self._state["status"] = "running"
        self._state["started_at"] = datetime.now().isoformat()

        try:
            constituents = self.data_provider.fetch_constituents(self.index_code)
            self._state["total"] = len(constituents)

            # Persist scan job to DB
            self.db.create_scan_job(
                self.scan_id, self.index_code, self.freq, self.model_key, len(constituents)
            )

            lookback = self.params["lookback"]
            pred_len = self.params["pred_len"]

            batch = []
            for stock in constituents:
                if self._state["status"] == "stopped":
                    break

                code = stock["stock_code"]
                name = stock["stock_name"]
                self._state["current_stock"] = f"{code} {name}"

                try:
                    kline_df = self.data_provider.fetch_kline(code, self.freq)
                    # Rate limit: sleep between stocks to avoid akshare IP ban
                    import time as _time
                    _time.sleep(0.3)  # 300ms between each stock

                    if kline_df is None or len(kline_df) < lookback:
                        self._state["errors"].append(f"{code}: data insufficient ({len(kline_df) if kline_df is not None else 0})")
                        self._state["completed"] += 1
                        self._sync_to_db()
                        continue

                    batch.append({
                        "code": code,
                        "name": name,
                        "kline_df": kline_df,
                    })

                    if len(batch) >= BATCH_SIZE or stock == constituents[-1]:
                        self._process_batch(batch, lookback, pred_len)
                        self._sync_to_db()
                        batch = []

                except Exception as e:
                    self._state["errors"].append(f"{code}: {str(e)}")
                    self._state["completed"] += 1
                    self._sync_to_db()

            if self._state["status"] == "running":
                self._state["status"] = "completed"
            self._state["current_stock"] = ""
            self._sync_to_db()

        except Exception as e:
            self._state["status"] = "error"
            self._state["errors"].append(f"Scan error: {str(e)}")
            self._sync_to_db()

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

            y_ts = generate_ashare_timestamps(kline_df["dt"].iloc[-1], self.freq, pred_len)

            df_list.append(x_df)
            x_timestamp_list.append(x_ts)
            y_timestamp_list.append(y_ts)

        try:
            if len(df_list) == 1:
                pred_dfs = [self.predictor.predict(
                    df=df_list[0], x_timestamp=x_timestamp_list[0], y_timestamp=y_timestamp_list[0],
                    pred_len=pred_len, T=self.params["T"], top_p=self.params["top_p"],
                    sample_count=self.params["sample_count"],
                )]
            else:
                pred_dfs = self.predictor.predict_batch(
                    df_list=df_list, x_timestamp_list=x_timestamp_list, y_timestamp_list=y_timestamp_list,
                    pred_len=pred_len, T=self.params["T"], top_p=self.params["top_p"],
                    sample_count=self.params["sample_count"],
                )
        except Exception as e:
            # Fallback to single predictions
            pred_dfs = []
            for i in range(len(df_list)):
                try:
                    pred_df = self.predictor.predict(
                        df=df_list[i], x_timestamp=x_timestamp_list[i], y_timestamp=y_timestamp_list[i],
                        pred_len=pred_len, T=self.params["T"], top_p=self.params["top_p"],
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
        """Save a single stock's prediction result, with backtest accuracy."""
        code = item["code"]
        name = item["name"]
        kline_df = item["kline_df"]

        if pred_df is None:
            self._state["errors"].append(f"{code}: prediction failed")
            self._state["completed"] += 1
            return

        pred_df = DataProvider.apply_price_limits(pred_df, last_close)

        pred_close = float(pred_df.iloc[-1]["close"])
        pred_change_pct = ((pred_close - last_close) / last_close) * 100

        pred_data = pred_df.reset_index(drop=True).to_dict(orient="records")

        # --- Backtest: predict a past period and compare with actual ---
        backtest_dir_acc = None
        backtest_overall_correct = None
        try:
            backtest_dir_acc, backtest_overall_correct = self._run_backtest(kline_df, code)
        except Exception:
            pass  # Backtest failure is non-critical

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
            backtest_dir_acc=backtest_dir_acc,
            backtest_overall_correct=backtest_overall_correct,
        )
        self._state["completed"] += 1

    def _run_backtest(self, kline_df, code):
        """Run backtest over the most recent 10 trading days to assess model accuracy."""
        lookback = self.params["lookback"]
        bt_pred_len = 5  # Backtest 5 trading days (1 week)

        if len(kline_df) < lookback + bt_pred_len:
            return None, None

        split_idx = len(kline_df) - bt_pred_len
        hist_df = kline_df.iloc[:split_idx]
        actual_df = kline_df.iloc[split_idx:]

        x_df = hist_df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
        x_ts = pd.Series(hist_df.iloc[-lookback:]["dt"].values)
        y_ts = pd.Series(actual_df["dt"].values)

        bt_pred_df = self.predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=bt_pred_len, T=self.params["T"], top_p=self.params["top_p"],
            sample_count=1,  # Use 1 sample for speed in batch backtest
        )

        bt_last_close = float(hist_df.iloc[-1]["close"])
        bt_pred_df = DataProvider.apply_price_limits(bt_pred_df, bt_last_close)

        actual_closes = actual_df["close"].values.astype(float)
        pred_closes = bt_pred_df["close"].values.astype(float)
        min_len = min(len(actual_closes), len(pred_closes))

        if min_len < 2:
            return None, None

        # Per-bar direction accuracy: each bar up/down vs previous bar
        actual_dirs = np.diff(actual_closes[:min_len]) > 0
        pred_dirs = np.diff(pred_closes[:min_len]) > 0
        dir_acc = float(np.mean(actual_dirs == pred_dirs) * 100)

        # Overall direction: did the model get the 10-day trend right?
        actual_change = actual_closes[min_len - 1] - actual_closes[0]
        pred_change = pred_closes[min_len - 1] - bt_last_close
        overall_correct = 1 if (actual_change > 0) == (pred_change > 0) else 0

        return round(dir_acc, 1), overall_correct
