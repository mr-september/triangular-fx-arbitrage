"""
Data cleaning utilities for FX OHLC data.

Handles bad tick detection, gap filling, and quality reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    """Summary of data quality metrics."""

    total_bars: int
    bad_ticks_removed: int
    gaps_filled: int
    weekend_gaps: int
    date_range: tuple[str, str]

    def __str__(self) -> str:
        return (
            f"Data Quality Report:\n"
            f"  Date range: {self.date_range[0]} to {self.date_range[1]}\n"
            f"  Total bars: {self.total_bars:,}\n"
            f"  Bad ticks removed: {self.bad_ticks_removed:,} "
            f"({100 * self.bad_ticks_removed / max(1, self.total_bars):.3f}%)\n"
            f"  Gaps filled: {self.gaps_filled:,}\n"
            f"  Weekend gaps: {self.weekend_gaps:,}"
        )


def detect_bad_ticks(
    df: pd.DataFrame,
    zscore_threshold: float = 10.0,
    window: int = 100,
) -> pd.Series:
    """
    Detect bad ticks using rolling Z-score on log returns.

    A "bad tick" is a bar where the log return or range is an extreme outlier
    compared to recent history, indicating erroneous data.

    Args:
        df: DataFrame with OHLC columns
        zscore_threshold: Z-score threshold for flagging (default: 10.0)
        window: Rolling window size for computing baseline statistics

    Returns:
        Boolean Series where True indicates a bad tick
    """
    # Compute log returns and log range
    log_return = np.log(df["close"] / df["open"])
    log_range = np.log(df["high"] / df["low"])

    # Rolling statistics
    ret_mean = log_return.rolling(window=window, min_periods=10).mean()
    ret_std = log_return.rolling(window=window, min_periods=10).std()
    range_mean = log_range.rolling(window=window, min_periods=10).mean()
    range_std = log_range.rolling(window=window, min_periods=10).std()

    # Z-scores
    ret_zscore = (log_return - ret_mean).abs() / ret_std.replace(0, np.nan)
    range_zscore = (log_range - range_mean).abs() / range_std.replace(0, np.nan)

    # Flag as bad if either Z-score exceeds threshold
    is_bad = (ret_zscore > zscore_threshold) | (range_zscore > zscore_threshold)

    # Also flag negative ranges (high < low) or inverted OHLC
    is_bad = is_bad | (df["high"] < df["low"])
    is_bad = is_bad | (df["high"] < df["open"])
    is_bad = is_bad | (df["high"] < df["close"])
    is_bad = is_bad | (df["low"] > df["open"])
    is_bad = is_bad | (df["low"] > df["close"])

    return is_bad.fillna(False)


def fill_gaps(
    df: pd.DataFrame,
    max_gap_bars: int = 5,
    method: str = "ffill",
) -> tuple[pd.DataFrame, int]:
    """
    Fill small gaps in the time series using forward fill.

    Large gaps (> max_gap_bars) are left as NaN and should be handled
    by the caller (typically by dropping or segmenting).

    Args:
        df: DataFrame with datetime index
        max_gap_bars: Maximum number of consecutive missing bars to fill
        method: Fill method ('ffill' for forward fill)

    Returns:
        Tuple of (filled DataFrame, number of gaps filled)
    """
    # Detect gaps by looking at the expected frequency
    freq = pd.infer_freq(df.index)
    if freq is None:
        # Try to estimate from median difference
        diffs = df.index.to_series().diff()
        median_diff = diffs.median()
    else:
        median_diff = pd.Timedelta(freq)

    # Reindex to complete time series
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=median_diff)
    df_reindexed = df.reindex(full_idx)

    # Count gaps before filling
    gaps_before = df_reindexed["open"].isna().sum()

    # Forward fill only small gaps
    if method == "ffill":
        df_filled = df_reindexed.ffill(limit=max_gap_bars)
    else:
        df_filled = df_reindexed.fillna(method=method, limit=max_gap_bars)

    # Count remaining gaps (these are large gaps we didn't fill)
    gaps_after = df_filled["open"].isna().sum()
    gaps_filled = gaps_before - gaps_after

    return df_filled, int(gaps_filled)


def remove_weekends(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove weekend bars from FX data.

    FX markets are closed from Friday ~22:00 UTC to Sunday ~22:00 UTC.
    Bars during this period are likely stale or erroneous.

    Args:
        df: DataFrame with datetime index

    Returns:
        Tuple of (filtered DataFrame, number of weekend bars removed)
    """
    # Friday after 22:00 UTC and Saturday are weekend
    # Sunday before 22:00 UTC is also weekend
    is_weekend = (
        (df.index.dayofweek == 4) & (df.index.hour >= 22)  # Friday after 22:00
        | (df.index.dayofweek == 5)  # Saturday
        | (df.index.dayofweek == 6) & (df.index.hour < 22)  # Sunday before 22:00
    )

    weekend_count = is_weekend.sum()
    df_filtered = df[~is_weekend]

    return df_filtered, int(weekend_count)


def clean_ohlc(
    df: pd.DataFrame,
    remove_bad_ticks: bool = True,
    fill_small_gaps: bool = True,
    drop_weekends: bool = True,
    zscore_threshold: float = 10.0,
    max_gap_bars: int = 5,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Comprehensive cleaning pipeline for OHLC data.

    Applies in order:
    1. Bad tick detection and removal
    2. Weekend removal
    3. Small gap filling

    Args:
        df: Raw DataFrame with OHLC columns and datetime index
        remove_bad_ticks: Whether to remove detected bad ticks
        fill_small_gaps: Whether to forward-fill small gaps
        drop_weekends: Whether to remove weekend bars
        zscore_threshold: Z-score threshold for bad tick detection
        max_gap_bars: Maximum consecutive bars to forward-fill

    Returns:
        Tuple of (cleaned DataFrame, DataQualityReport)
    """
    original_len = len(df)
    bad_ticks_removed = 0
    weekend_gaps = 0
    gaps_filled = 0

    df_clean = df.copy()

    # Step 1: Remove bad ticks
    if remove_bad_ticks:
        bad_mask = detect_bad_ticks(df_clean, zscore_threshold=zscore_threshold)
        bad_ticks_removed = bad_mask.sum()
        df_clean = df_clean[~bad_mask]

    # Step 2: Remove weekends
    if drop_weekends:
        df_clean, weekend_gaps = remove_weekends(df_clean)

    # Step 3: Fill small gaps
    if fill_small_gaps:
        df_clean, gaps_filled = fill_gaps(df_clean, max_gap_bars=max_gap_bars)
        # Drop remaining NaN rows (large gaps)
        df_clean = df_clean.dropna()

    # Build report
    report = DataQualityReport(
        total_bars=original_len,
        bad_ticks_removed=int(bad_ticks_removed),
        gaps_filled=gaps_filled,
        weekend_gaps=weekend_gaps,
        date_range=(
            df_clean.index.min().strftime("%Y-%m-%d"),
            df_clean.index.max().strftime("%Y-%m-%d"),
        ),
    )

    return df_clean, report
