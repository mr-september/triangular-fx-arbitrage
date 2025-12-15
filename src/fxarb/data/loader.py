"""
Data loader for FX OHLC data from pickle files.

Handles loading, timestamp parsing, and resampling of 1-minute bar data.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

# Default data directory (can be overridden)
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw"


def parse_histdata_timestamp(ts: str) -> datetime:
    """
    Parse HistData timestamp format 'YYYYMMDD HHMMSS' to datetime.

    Args:
        ts: Timestamp string in format 'YYYYMMDD HHMMSS'

    Returns:
        Parsed datetime object (UTC assumed)

    Example:
        >>> parse_histdata_timestamp('20220101 093000')
        datetime(2022, 1, 1, 9, 30, 0)
    """
    return datetime.strptime(ts.strip(), "%Y%m%d %H%M%S")


def load_pair(
    pair: str,
    data_dir: Path | str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load a single FX pair from pickle file.

    Args:
        pair: Currency pair code (e.g., 'eurusd', 'gbpusd')
        data_dir: Directory containing pair subdirectories with pickle files.
                  Defaults to data/raw/ relative to package.
        start_date: Optional start date filter (YYYY-MM-DD format)
        end_date: Optional end date filter (YYYY-MM-DD format)

    Returns:
        DataFrame with columns: datetime (index), open, high, low, close, volume

    Raises:
        FileNotFoundError: If pickle file doesn't exist
        ValueError: If pair format is invalid
    """
    pair = pair.lower().strip()
    if len(pair) != 6 or not pair.isalpha():
        raise ValueError(f"Invalid pair format: {pair}. Expected 6-letter code like 'eurusd'")

    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    pickle_path = data_dir / pair / f"{pair}.pkl"

    if not pickle_path.exists():
        raise FileNotFoundError(f"Data file not found: {pickle_path}")

    # Load with zip compression (HistData format)
    df = pd.read_pickle(pickle_path, compression="zip")

    # Parse timestamps and set as index
    df["datetime"] = df["timestamp"].apply(parse_histdata_timestamp)
    df = df.set_index("datetime").drop(columns=["timestamp"])

    # Apply date filters if specified
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    # Ensure proper column order
    df = df[["open", "high", "low", "close", "volume"]]

    return df


def resample_ohlc(
    df: pd.DataFrame,
    timeframe: str = "5min",
    agg_volume: Literal["sum", "mean", "last"] = "sum",
) -> pd.DataFrame:
    """
    Resample OHLC data to a different timeframe.

    Uses proper OHLC aggregation: first open, max high, min low, last close.

    Args:
        df: DataFrame with OHLC columns and datetime index
        timeframe: Target timeframe (e.g., '5min', '15min', '1h', '4h', '1d')
        agg_volume: How to aggregate volume ('sum', 'mean', or 'last')

    Returns:
        Resampled DataFrame with same columns
    """
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": agg_volume,
    }

    resampled = df.resample(timeframe).agg(agg_dict)

    # Drop bars with no data (e.g., weekends)
    resampled = resampled.dropna(subset=["open"])

    return resampled


def load_triplet(
    pair1: str,
    pair2: str,
    pair3: str,
    data_dir: Path | str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    timeframe: str = "5min",
) -> dict[str, pd.DataFrame]:
    """
    Load and align three FX pairs for triplet analysis.

    All pairs are resampled to the specified timeframe and aligned to common
    timestamps using forward-fill for small gaps.

    Args:
        pair1, pair2, pair3: Currency pair codes
        data_dir: Directory containing data files
        start_date: Optional start date filter
        end_date: Optional end date filter
        timeframe: Target timeframe for resampling

    Returns:
        Dictionary mapping pair codes to aligned DataFrames
    """
    pairs = [pair1.lower(), pair2.lower(), pair3.lower()]
    data = {}

    for pair in pairs:
        df = load_pair(pair, data_dir, start_date, end_date)
        df = resample_ohlc(df, timeframe)
        data[pair] = df

    # Find common index (intersection of all timestamps)
    common_idx = data[pairs[0]].index
    for pair in pairs[1:]:
        common_idx = common_idx.intersection(data[pair].index)

    # Align all DataFrames to common index
    for pair in pairs:
        data[pair] = data[pair].loc[common_idx]

    return data


def get_available_pairs(data_dir: Path | str | None = None) -> list[str]:
    """
    List all available currency pairs in the data directory.

    Args:
        data_dir: Directory containing pair subdirectories

    Returns:
        List of pair codes (e.g., ['eurusd', 'gbpusd', ...])
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    if not data_dir.exists():
        return []

    pairs = []
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            pkl_file = subdir / f"{subdir.name}.pkl"
            if pkl_file.exists():
                pairs.append(subdir.name)

    return sorted(pairs)
