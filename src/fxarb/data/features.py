"""
Feature engineering for FX data.

Includes session indicators, overlap flags, and log price computation.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class TradingSession(Enum):
    """Major FX trading sessions with UTC hour ranges."""

    ASIA = (0, 9)  # Tokyo: 00:00-09:00 UTC
    EUROPE = (8, 17)  # London: 08:00-17:00 UTC
    NORTH_AMERICA = (13, 22)  # New York: 13:00-22:00 UTC


def get_session(hour: int) -> list[str]:
    """
    Determine which trading session(s) are active for a given UTC hour.

    Args:
        hour: Hour of day in UTC (0-23)

    Returns:
        List of active session names
    """
    sessions = []
    for session in TradingSession:
        start, end = session.value
        if start <= hour < end:
            sessions.append(session.name.lower())
    return sessions if sessions else ["off_hours"]


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trading session indicator columns to DataFrame.

    Creates binary columns for each session and overlap periods.

    Args:
        df: DataFrame with datetime index

    Returns:
        DataFrame with additional session feature columns:
        - session_asia: 1 if Tokyo session active
        - session_europe: 1 if London session active
        - session_north_america: 1 if New York session active
        - overlap_asia_europe: 1 if Tokyo-London overlap (08:00-09:00 UTC)
        - overlap_europe_na: 1 if London-NY overlap (13:00-17:00 UTC)
        - primary_session: categorical of the "main" session
    """
    df = df.copy()
    hours = df.index.hour

    # Individual session flags
    df["session_asia"] = ((hours >= 0) & (hours < 9)).astype(int)
    df["session_europe"] = ((hours >= 8) & (hours < 17)).astype(int)
    df["session_north_america"] = ((hours >= 13) & (hours < 22)).astype(int)

    # Overlap periods (high liquidity)
    df["overlap_asia_europe"] = ((hours >= 8) & (hours < 9)).astype(int)
    df["overlap_europe_na"] = ((hours >= 13) & (hours < 17)).astype(int)

    # Primary session (for grouping analysis)
    def _get_primary_session(hour: int) -> str:
        if 13 <= hour < 17:
            return "overlap_eu_na"  # Most liquid
        elif 8 <= hour < 9:
            return "overlap_asia_eu"
        elif 0 <= hour < 8:
            return "asia"
        elif 9 <= hour < 13:
            return "europe"
        elif 17 <= hour < 22:
            return "north_america"
        else:
            return "off_hours"

    df["primary_session"] = pd.Categorical(
        [_get_primary_session(h) for h in hours],
        categories=["asia", "overlap_asia_eu", "europe", "overlap_eu_na", "north_america", "off_hours"],
        ordered=True,
    )

    return df


def compute_log_prices(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Add log price column for cointegration analysis.

    Log prices are standard in cointegration because:
    1. Returns are additive in log space
    2. Percentage changes become symmetric
    3. Required for proper spread construction

    Args:
        df: DataFrame with OHLC columns
        price_col: Which price to use ('close', 'open', 'mid')

    Returns:
        DataFrame with additional log_price column
    """
    df = df.copy()

    if price_col == "mid":
        # Midpoint of high and low
        price = (df["high"] + df["low"]) / 2
    else:
        price = df[price_col]

    df["log_price"] = np.log(price)

    return df


def compute_returns(
    df: pd.DataFrame,
    price_col: str = "close",
    periods: int = 1,
) -> pd.DataFrame:
    """
    Compute log returns for analysis.

    Args:
        df: DataFrame with price columns
        price_col: Which price to use
        periods: Number of periods for return calculation

    Returns:
        DataFrame with additional log_return column
    """
    df = df.copy()

    if price_col == "mid":
        price = (df["high"] + df["low"]) / 2
    else:
        price = df[price_col]

    df["log_return"] = np.log(price / price.shift(periods))

    return df


def compute_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualize: bool = True,
    bars_per_day: int = 288,  # 5-min bars
) -> pd.DataFrame:
    """
    Compute rolling realized volatility.

    Args:
        df: DataFrame with log_return column (or will compute from close)
        window: Rolling window size
        annualize: Whether to annualize the volatility
        bars_per_day: Number of bars per trading day for annualization

    Returns:
        DataFrame with volatility column
    """
    df = df.copy()

    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    vol = df["log_return"].rolling(window=window).std()

    if annualize:
        # Annualize assuming ~252 trading days
        vol = vol * np.sqrt(bars_per_day * 252)

    df["volatility"] = vol

    return df
