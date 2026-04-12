import os
import sys
import datetime

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_provider import generate_ashare_timestamps


class TestFiveMinuteBasic:
    """Basic sanity checks for 5-minute bar generation."""

    def test_single_bar(self):
        last = pd.Timestamp("2025-06-02 09:30")  # Monday
        ts = generate_ashare_timestamps(last, "5min", 1)
        assert len(ts) == 1
        assert ts.iloc[0] == pd.Timestamp("2025-06-02 09:35")

    def test_full_morning_from_open(self):
        """Starting from 09:30, should yield 24 morning bars (09:35 .. 11:30)."""
        last = pd.Timestamp("2025-06-02 09:30")
        ts = generate_ashare_timestamps(last, "5min", 24)
        assert len(ts) == 24
        assert ts.iloc[0] == pd.Timestamp("2025-06-02 09:35")
        assert ts.iloc[-1] == pd.Timestamp("2025-06-02 11:30")

    def test_full_day_48_bars(self):
        """Starting from 09:30, 48 bars = entire day."""
        last = pd.Timestamp("2025-06-02 09:30")
        ts = generate_ashare_timestamps(last, "5min", 48)
        assert len(ts) == 48
        assert ts.iloc[0] == pd.Timestamp("2025-06-02 09:35")
        assert ts.iloc[23] == pd.Timestamp("2025-06-02 11:30")
        assert ts.iloc[24] == pd.Timestamp("2025-06-02 13:05")
        assert ts.iloc[-1] == pd.Timestamp("2025-06-02 15:00")

    def test_bars_span_lunch_break(self):
        """Last bar at 11:25 -> next bar should be 11:30 then 13:05."""
        last = pd.Timestamp("2025-06-02 11:25")
        ts = generate_ashare_timestamps(last, "5min", 2)
        assert ts.iloc[0] == pd.Timestamp("2025-06-02 11:30")
        assert ts.iloc[1] == pd.Timestamp("2025-06-02 13:05")

    def test_bars_span_market_close(self):
        """Last bar at 14:55 -> next should be 15:00 then next day 09:35."""
        last = pd.Timestamp("2025-06-02 14:55")  # Monday
        ts = generate_ashare_timestamps(last, "5min", 2)
        assert ts.iloc[0] == pd.Timestamp("2025-06-02 15:00")
        assert ts.iloc[1] == pd.Timestamp("2025-06-03 09:35")  # Tuesday

    def test_all_bars_within_trading_hours(self):
        """Every timestamp must be within morning or afternoon session."""
        last = pd.Timestamp("2025-06-02 09:30")
        ts = generate_ashare_timestamps(last, "5min", 200)
        for t in ts:
            t_time = t.time()
            in_morning = datetime.time(9, 35) <= t_time <= datetime.time(11, 30)
            in_afternoon = datetime.time(13, 5) <= t_time <= datetime.time(15, 0)
            assert in_morning or in_afternoon, f"Timestamp {t} is outside trading hours"

    def test_bars_5min_aligned(self):
        """Every timestamp minute should be a multiple of 5."""
        last = pd.Timestamp("2025-06-02 09:30")
        ts = generate_ashare_timestamps(last, "5min", 200)
        for t in ts:
            assert t.minute % 5 == 0, f"Timestamp {t} minute not aligned to 5"

    def test_48_bars_per_trading_day(self):
        """Generate 96 bars (2 days) and verify 48 per day."""
        last = pd.Timestamp("2025-06-02 09:30")  # Monday
        ts = generate_ashare_timestamps(last, "5min", 96)
        day1 = [t for t in ts if t.date() == datetime.date(2025, 6, 2)]
        day2 = [t for t in ts if t.date() == datetime.date(2025, 6, 3)]
        assert len(day1) == 48
        assert len(day2) == 48


class TestWeekendSkip:
    """Verify weekends are skipped."""

    def test_friday_close_to_monday(self):
        """Last bar Friday 15:00 -> next bar Monday 09:35."""
        last = pd.Timestamp("2025-06-06 15:00")  # Friday
        ts = generate_ashare_timestamps(last, "5min", 1)
        assert ts.iloc[0] == pd.Timestamp("2025-06-09 09:35")  # Monday

    def test_friday_afternoon_to_monday(self):
        last = pd.Timestamp("2025-06-06 14:55")  # Friday
        ts = generate_ashare_timestamps(last, "5min", 2)
        assert ts.iloc[0] == pd.Timestamp("2025-06-06 15:00")
        assert ts.iloc[1] == pd.Timestamp("2025-06-09 09:35")  # Monday

    def test_no_weekend_timestamps(self):
        """5 full trading days should produce 240 bars, none on weekends."""
        last = pd.Timestamp("2025-06-02 09:30")  # Monday
        ts = generate_ashare_timestamps(last, "5min", 240)
        for t in ts:
            assert t.weekday() < 5, f"Weekend timestamp: {t}"


class TestEdgeCases:
    """Edge cases for starting positions."""

    def test_start_before_market(self):
        """last_dt is before market open."""
        last = pd.Timestamp("2025-06-02 08:00")
        ts = generate_ashare_timestamps(last, "5min", 1)
        assert ts.iloc[0] == pd.Timestamp("2025-06-02 09:35")

    def test_start_at_lunch(self):
        """last_dt during lunch break."""
        last = pd.Timestamp("2025-06-02 12:00")
        ts = generate_ashare_timestamps(last, "5min", 1)
        assert ts.iloc[0] == pd.Timestamp("2025-06-02 13:05")

    def test_start_after_close(self):
        """last_dt after market close."""
        last = pd.Timestamp("2025-06-02 16:00")
        ts = generate_ashare_timestamps(last, "5min", 1)
        assert ts.iloc[0] == pd.Timestamp("2025-06-03 09:35")

    def test_start_at_1130(self):
        """last_dt at 11:30 (last morning bar) -> next is 13:05."""
        last = pd.Timestamp("2025-06-02 11:30")
        ts = generate_ashare_timestamps(last, "5min", 1)
        assert ts.iloc[0] == pd.Timestamp("2025-06-02 13:05")

    def test_start_at_1500(self):
        """last_dt at 15:00 (last bar of day) -> next day 09:35."""
        last = pd.Timestamp("2025-06-02 15:00")
        ts = generate_ashare_timestamps(last, "5min", 1)
        assert ts.iloc[0] == pd.Timestamp("2025-06-03 09:35")

    def test_start_on_saturday(self):
        """last_dt on Saturday -> jump to Monday."""
        last = pd.Timestamp("2025-06-07 10:00")  # Saturday
        ts = generate_ashare_timestamps(last, "5min", 1)
        assert ts.iloc[0] == pd.Timestamp("2025-06-09 09:35")

    def test_start_on_sunday(self):
        """last_dt on Sunday -> jump to Monday."""
        last = pd.Timestamp("2025-06-08 10:00")  # Sunday
        ts = generate_ashare_timestamps(last, "5min", 1)
        assert ts.iloc[0] == pd.Timestamp("2025-06-09 09:35")


class TestDailyFrequency:
    """Daily frequency should use business day range."""

    def test_daily_basic(self):
        last = pd.Timestamp("2025-06-06")  # Friday
        ts = generate_ashare_timestamps(last, "daily", 3)
        assert len(ts) == 3
        assert ts.iloc[0] == pd.Timestamp("2025-06-09")  # Monday
        assert ts.iloc[1] == pd.Timestamp("2025-06-10")
        assert ts.iloc[2] == pd.Timestamp("2025-06-11")

    def test_daily_returns_series(self):
        last = pd.Timestamp("2025-06-02")
        ts = generate_ashare_timestamps(last, "daily", 5)
        assert isinstance(ts, pd.Series)
        assert len(ts) == 5


class TestReturnType:
    """Verify the function returns pd.Series."""

    def test_returns_series_5min(self):
        ts = generate_ashare_timestamps(pd.Timestamp("2025-06-02 09:30"), "5min", 10)
        assert isinstance(ts, pd.Series)

    def test_returns_series_daily(self):
        ts = generate_ashare_timestamps(pd.Timestamp("2025-06-02"), "daily", 10)
        assert isinstance(ts, pd.Series)
