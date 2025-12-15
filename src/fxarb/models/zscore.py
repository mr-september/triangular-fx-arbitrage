"""
Z-score based mean reversion strategy.

The simplest and most interpretable approach for trading cointegrated spreads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class ZScoreStrategy:
    """
    Configuration for a Z-score based mean reversion strategy.

    Trading rules:
    - ENTER SHORT when Z-score > entry_threshold (spread is expensive)
    - ENTER LONG when Z-score < -entry_threshold (spread is cheap)
    - EXIT when Z-score crosses exit_threshold (reverts toward mean)

    Optionally:
    - STOP LOSS if Z-score exceeds stop_loss_threshold
    """

    lookback: int = 100  # Rolling window for mean/std
    entry_threshold: float = 2.0  # Enter when |z| > threshold
    exit_threshold: float = 0.0  # Exit when z crosses this level
    stop_loss_threshold: float | None = 4.0  # Stop loss level (None to disable)

    def __str__(self) -> str:
        return (
            f"ZScoreStrategy(lookback={self.lookback}, "
            f"entry=±{self.entry_threshold}, exit={self.exit_threshold}"
            + (f", stop={self.stop_loss_threshold})" if self.stop_loss_threshold else ")")
        )


def compute_zscore(
    spread: pd.Series,
    lookback: int = 100,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Compute rolling Z-score of the spread.

    Z = (spread - rolling_mean) / rolling_std

    Args:
        spread: The cointegrated spread series
        lookback: Rolling window size for mean and std
        min_periods: Minimum observations required (defaults to lookback // 2)

    Returns:
        Z-score series
    """
    if min_periods is None:
        min_periods = max(10, lookback // 2)

    rolling_mean = spread.rolling(window=lookback, min_periods=min_periods).mean()
    rolling_std = spread.rolling(window=lookback, min_periods=min_periods).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    zscore = (spread - rolling_mean) / rolling_std
    zscore.name = "zscore"

    return zscore


def generate_signals(
    zscore: pd.Series,
    strategy: ZScoreStrategy | None = None,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_loss_threshold: float | None = 4.0,
    # Asymmetric threshold support
    long_entry_threshold: float | None = None,  # Override for long entries (z < -threshold)
    short_entry_threshold: float | None = None,  # Override for short entries (z > threshold)
    long_stop_loss: float | None = None,  # Override stop loss for longs
    short_stop_loss: float | None = None,  # Override stop loss for shorts
    max_duration: int | None = None,  # Maximum bars to hold a position
) -> pd.DataFrame:
    """
    Generate trading signals from Z-score series.

    Signal values:
    - 1: Long the spread (buy undervalued)
    - -1: Short the spread (sell overvalued)
    - 0: No position

    Args:
        zscore: Z-score series
        strategy: ZScoreStrategy config (if provided, overrides other params)
        entry_threshold: Enter when |z| exceeds this (symmetric, default)
        exit_threshold: Exit when z crosses this
        stop_loss_threshold: Stop loss level (symmetric, default)
        long_entry_threshold: Override for long entries (z < -threshold)
        short_entry_threshold: Override for short entries (z > threshold)
        long_stop_loss: Override stop loss for longs (z < -stop_loss)
        short_stop_loss: Override stop loss for shorts (z > stop_loss)

    Returns:
        DataFrame with columns:
        - signal: Trading signal (1, 0, -1)
        - position: Actual position (accounts for entry/exit logic)
        - entry: Entry signals only
        - exit: Exit signals only
    """
    if strategy is not None:
        entry_threshold = strategy.entry_threshold
        exit_threshold = strategy.exit_threshold
        stop_loss_threshold = strategy.stop_loss_threshold

    # Resolve asymmetric thresholds (use symmetric if not specified)
    long_entry = long_entry_threshold if long_entry_threshold is not None else entry_threshold
    short_entry = short_entry_threshold if short_entry_threshold is not None else entry_threshold
    long_stop = long_stop_loss if long_stop_loss is not None else stop_loss_threshold
    short_stop = short_stop_loss if short_stop_loss is not None else stop_loss_threshold

    n = len(zscore)
    signals = pd.DataFrame(index=zscore.index)
    signals["zscore"] = zscore

    # Initialize position tracking
    position = np.zeros(n)
    entry_signals = np.zeros(n)
    exit_signals = np.zeros(n)

    current_position = 0
    bars_in_position = 0

    for i in range(n):
        z = zscore.iloc[i]

        if np.isnan(z):
            position[i] = current_position
            continue
            
        # Update duration
        if current_position != 0:
            bars_in_position += 1

        # Check for exit conditions first
        if current_position != 0:
            # Time-based exit
            if max_duration is not None and bars_in_position >= max_duration:
                exit_signals[i] = current_position  # Signal exit (1 if long, -1 if short)
                current_position = 0
                bars_in_position = 0
                
            # Exit on mean reversion
            elif current_position == 1 and z >= exit_threshold:
                exit_signals[i] = 1
                current_position = 0
                bars_in_position = 0
            elif current_position == -1 and z <= exit_threshold:
                exit_signals[i] = -1
                current_position = 0
                bars_in_position = 0

            # Stop loss (asymmetric)
            elif current_position == 1 and long_stop is not None and z < -long_stop:
                exit_signals[i] = 1
                current_position = 0
                bars_in_position = 0
            elif current_position == -1 and short_stop is not None and z > short_stop:
                exit_signals[i] = -1
                current_position = 0
                bars_in_position = 0

        # Check for entry conditions (only if flat) - asymmetric thresholds
        if current_position == 0:
            if z < -long_entry:  # Long when z is very negative
                entry_signals[i] = 1
                current_position = 1
                bars_in_position = 0
            elif z > short_entry:  # Short when z is very positive
                entry_signals[i] = -1
                current_position = -1
                bars_in_position = 0

        position[i] = current_position

    signals["position"] = position
    signals["entry"] = entry_signals
    signals["exit"] = exit_signals
    signals["signal"] = signals["entry"] + signals["exit"]

    return signals


def analyze_signals(signals: pd.DataFrame) -> dict:
    """
    Analyze trading signal statistics.

    Args:
        signals: DataFrame from generate_signals()

    Returns:
        Dictionary of signal statistics
    """
    position = signals["position"]
    entries = signals["entry"]

    # Count trades
    n_long_entries = (entries == 1).sum()
    n_short_entries = (entries == -1).sum()
    n_trades = n_long_entries + n_short_entries

    # Time in market
    time_in_market = (position != 0).mean()
    time_long = (position == 1).mean()
    time_short = (position == -1).mean()

    # Average trade duration (approximate)
    position_changes = position.diff().abs()
    n_position_changes = (position_changes > 0).sum()
    avg_trade_duration = len(position) / max(n_position_changes, 1)

    return {
        "n_trades": n_trades,
        "n_long_entries": n_long_entries,
        "n_short_entries": n_short_entries,
        "time_in_market": time_in_market,
        "time_long": time_long,
        "time_short": time_short,
        "avg_trade_duration_bars": avg_trade_duration,
    }
