#!/usr/bin/env python3
"""
HistData.com Data Downloader

This script demonstrates how to download and process FX tick data from HistData.com,
a free source of historical forex data.

NOTE: For this project, we use pre-downloaded data. This script is provided for
educational purposes and to show the reproducibility of the data pipeline.

Usage:
    python download_histdata.py --pair eurusd --start-year 2020 --end-year 2022

Data Source: https://www.histdata.com/
License: Free for personal/academic use
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# HistData.com URL patterns
# They provide data in monthly chunks
BASE_URL = "https://www.histdata.com/download/free/forex/ascii/tick_data_quotes/{pair}/{year}/{month}"
DOWNLOAD_URL = "https://www.histdata.com/download/free/forex/{timeframe}/{pair}/{year}"

# Available pairs on HistData
AVAILABLE_PAIRS = [
    "eurusd", "eurchf", "eurgbp", "eurjpy", "euraud",
    "usdcad", "usdchf", "usdjpy",
    "gbpchf", "gbpjpy", "gbpusd",
    "audjpy", "audusd",
    "chfjpy",
    "nzdjpy", "nzdusd",
    "xauusd",
    "eurcad", "audcad", "cadjpy",
    "eurnzd", "audchf", "nzdchf",
    "gbpaud", "gbpcad", "gbpnzd",
    "audnzd", "cadchf",
]


def parse_histdata_timestamp(ts_str: str) -> datetime:
    """
    Parse HistData timestamp format.
    
    HistData uses format: YYYYMMDD HHMMSS
    """
    return datetime.strptime(ts_str.strip(), "%Y%m%d %H%M%S")


def download_month(
    pair: str,
    year: int,
    month: int,
    output_dir: Path,
    timeframe: str = "M1",
) -> pd.DataFrame | None:
    """
    Download one month of data from HistData.
    
    Note: HistData requires manual download or uses a form-based download system.
    This function simulates what the data structure looks like.
    
    In practice, you would:
    1. Go to histdata.com
    2. Navigate to the pair and timeframe
    3. Download the monthly zip file
    4. Extract and process
    
    Args:
        pair: Currency pair (e.g., 'eurusd')
        year: Year (e.g., 2022)
        month: Month (1-12)
        output_dir: Directory to save data
        timeframe: 'M1' for 1-minute, 'TICK' for tick data
        
    Returns:
        DataFrame with OHLCV data, or None if download fails
    """
    # HistData naming convention
    month_str = f"{month:02d}"
    filename = f"DAT_ASCII_{pair.upper()}_{timeframe}_{year}{month_str}.csv"
    
    print(f"  Processing {pair.upper()} {year}-{month_str}...")
    
    # In a real implementation, this would download from histdata.com
    # For demonstration, we show the expected file format
    
    # Expected columns for M1 data:
    # Date (YYYYMMDD), Time (HHMMSS), Open, High, Low, Close, Volume
    
    return None  # Placeholder


def process_histdata_csv(filepath: Path) -> pd.DataFrame:
    """
    Process a HistData CSV file into a standardized DataFrame.
    
    HistData M1 format:
    - No header row
    - Columns: DateTime, Open, High, Low, Close, Volume
    - DateTime format: YYYYMMDD HHMMSS
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with columns: datetime (index), open, high, low, close, volume
    """
    # HistData CSVs have no header
    df = pd.read_csv(
        filepath,
        sep=";",
        header=None,
        names=["datetime", "open", "high", "low", "close", "volume"],
    )
    
    # Parse datetime
    df["datetime"] = df["datetime"].apply(parse_histdata_timestamp)
    df = df.set_index("datetime")
    
    # Ensure numeric types
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    
    return df


def consolidate_to_pickle(
    pair: str,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """
    Consolidate multiple CSV files into a single pickle file.
    
    Args:
        pair: Currency pair
        input_dir: Directory containing monthly CSV files
        output_dir: Directory to save consolidated pickle
        
    Returns:
        Path to the output pickle file
    """
    csv_files = sorted(input_dir.glob(f"*{pair.upper()}*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for {pair}")
    
    dfs = []
    for csv_file in tqdm(csv_files, desc=f"Processing {pair}"):
        try:
            df = process_histdata_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Failed to process {csv_file}: {e}")
            continue
    
    if not dfs:
        raise ValueError(f"No valid data found for {pair}")
    
    # Concatenate and sort
    combined = pd.concat(dfs).sort_index()
    
    # Remove duplicates (can occur at month boundaries)
    combined = combined[~combined.index.duplicated(keep="first")]
    
    # Save as compressed pickle
    output_path = output_dir / pair / f"{pair}.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_pickle(output_path, compression="zip")
    
    print(f"Saved {len(combined):,} rows to {output_path}")
    return output_path


def main():
    """Main entry point for HistData download script."""
    parser = argparse.ArgumentParser(
        description="Download FX data from HistData.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download EUR/USD for 2020-2022
    python download_histdata.py --pair eurusd --start-year 2020 --end-year 2022
    
    # Download multiple pairs
    python download_histdata.py --pair eurusd gbpusd eurgbp --start-year 2020
    
    # List available pairs
    python download_histdata.py --list-pairs
""",
    )
    
    parser.add_argument(
        "--pair",
        nargs="+",
        help="Currency pair(s) to download (e.g., eurusd gbpusd)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="Start year (default: 2020)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2022,
        help="End year (default: 2022)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--list-pairs",
        action="store_true",
        help="List available currency pairs",
    )
    
    args = parser.parse_args()
    
    if args.list_pairs:
        print("Available pairs on HistData.com:")
        for pair in sorted(AVAILABLE_PAIRS):
            print(f"  {pair.upper()}")
        return
    
    if not args.pair:
        parser.error("Please specify at least one --pair or use --list-pairs")
    
    print("=" * 60)
    print("HistData.com Data Downloader")
    print("=" * 60)
    print()
    print("NOTE: HistData.com requires manual download through their website.")
    print("This script demonstrates the data processing pipeline.")
    print()
    print("To download data manually:")
    print("1. Visit https://www.histdata.com/download-free-forex-data/")
    print("2. Select your pair and timeframe (M1 for 1-minute)")
    print("3. Download the monthly zip files")
    print("4. Extract CSVs to the input directory")
    print("5. Run this script to consolidate into pickle format")
    print()
    print("=" * 60)
    
    for pair in args.pair:
        pair = pair.lower()
        if pair not in AVAILABLE_PAIRS:
            print(f"Warning: {pair} may not be available on HistData")
        
        print(f"\nProcessing {pair.upper()}...")
        print(f"  Years: {args.start_year} - {args.end_year}")
        print(f"  Output: {args.output_dir / pair / f'{pair}.pkl'}")
        
        # In a real implementation, this would download and process
        # For demonstration, we just show what would happen
        

if __name__ == "__main__":
    main()
