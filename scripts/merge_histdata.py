#!/usr/bin/env python3
"""
Merge new HistData downloads (2023-2024) with existing pickle files.

Usage:
    1. Download ZIP files from HistData.com for each pair/month
    2. Place them in: data/histdata_downloads/
    3. Run: python scripts/merge_histdata.py
    
The script will:
    - Extract and parse all CSV files
    - Merge with existing pickle data in E:\nevergiveup\data
    - Save updated pickles to data/raw/
"""

import os
import sys
import zipfile
from pathlib import Path
from datetime import datetime
from io import StringIO

import pandas as pd
from tqdm import tqdm


# Configuration
DOWNLOAD_DIR = Path("data/histdata_downloads")
OUTPUT_DIR = Path("data/raw")
LEGACY_DIR = Path("E:/nevergiveup/data")

# Pairs to process (lowercase, no separator)
TARGET_PAIRS = [
    "eurusd", "gbpusd", "eurgbp",  # Primary triplet
    "usdjpy", "eurjpy", "gbpjpy",  # JPY triplet
    "audusd", "nzdusd", "audnzd",  # AUD/NZD triplet
]


def parse_histdata_timestamp(timestamp_str: str) -> pd.Timestamp:
    """Parse HistData timestamp format: YYYYMMDD HHMMSS"""
    try:
        return pd.to_datetime(timestamp_str, format="%Y%m%d %H%M%S")
    except:
        return pd.NaT


def process_csv_content(content: str) -> pd.DataFrame:
    """Process HistData CSV content into DataFrame."""
    # HistData format: DateTime,Open,High,Low,Close,Volume
    # No header in the file
    df = pd.read_csv(
        StringIO(content),
        header=None,
        names=["timestamp", "open", "high", "low", "close", "volume"],
        dtype={
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
        },
    )
    
    # Parse timestamps
    df["timestamp"] = df["timestamp"].astype(str).apply(parse_histdata_timestamp)
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    
    return df


def extract_pair_from_filename(filename: str) -> str | None:
    """Extract pair name from HistData filename."""
    # Format: HISTDATA_COM_ASCII_EURUSD_M1_202301.zip
    filename = filename.upper()
    for pair in [p.upper() for p in TARGET_PAIRS]:
        if pair in filename:
            return pair.lower()
    return None


def load_existing_data(pair: str) -> pd.DataFrame | None:
    """Load existing pickle data for a pair."""
    # Check output dir first (already processed)
    output_path = OUTPUT_DIR / f"{pair}.pkl"
    if output_path.exists():
        return pd.read_pickle(output_path)
    
    # Check legacy dir
    legacy_path = LEGACY_DIR / f"{pair}.pkl"
    if legacy_path.exists():
        return pd.read_pickle(legacy_path)
    
    return None


def merge_dataframes(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge existing and new data, removing duplicates."""
    if existing is None:
        return new
    
    # Concatenate and remove duplicates
    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    
    return combined


def main():
    print("=" * 60)
    print("HistData Merge Utility")
    print("=" * 60)
    
    # Create directories
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all ZIP files
    zip_files = list(DOWNLOAD_DIR.glob("*.zip"))
    
    if not zip_files:
        print(f"\nNo ZIP files found in {DOWNLOAD_DIR}")
        print("\nTo download data from HistData.com:")
        print("  1. Go to https://www.histdata.com/download-free-forex-data/")
        print("  2. Select: ASCII / 1-Minute Bars")
        print("  3. Download each pair/month you need (2023 onwards)")
        print(f"  4. Place ZIP files in: {DOWNLOAD_DIR.absolute()}")
        print("  5. Run this script again")
        return
    
    print(f"\nFound {len(zip_files)} ZIP files to process")
    
    # Group by pair
    pair_data: dict[str, list[pd.DataFrame]] = {p: [] for p in TARGET_PAIRS}
    
    for zip_path in tqdm(zip_files, desc="Processing ZIPs"):
        pair = extract_pair_from_filename(zip_path.name)
        if pair is None:
            print(f"  Skipping unknown pair: {zip_path.name}")
            continue
        
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for csv_name in zf.namelist():
                    if csv_name.endswith(".csv"):
                        content = zf.read(csv_name).decode("utf-8")
                        df = process_csv_content(content)
                        if len(df) > 0:
                            pair_data[pair].append(df)
        except Exception as e:
            print(f"  Error processing {zip_path.name}: {e}")
    
    # Merge and save each pair
    print("\nMerging with existing data...")
    
    for pair in TARGET_PAIRS:
        new_dfs = pair_data[pair]
        if not new_dfs:
            continue
        
        # Concatenate all new data for this pair
        new_combined = pd.concat(new_dfs)
        new_combined = new_combined[~new_combined.index.duplicated(keep="last")]
        new_combined = new_combined.sort_index()
        
        # Load existing
        existing = load_existing_data(pair)
        
        # Merge
        final = merge_dataframes(existing, new_combined)
        
        # Save
        output_path = OUTPUT_DIR / f"{pair}.pkl"
        final.to_pickle(output_path)
        
        print(f"  {pair.upper()}: {len(final):,} bars "
              f"({final.index.min().date()} to {final.index.max().date()})")
    
    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
