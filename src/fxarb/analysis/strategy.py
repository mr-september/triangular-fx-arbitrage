"""
Strategy analysis and signal generation utilities.
"""
import numpy as np
import pandas as pd


def generate_signals(
    z_score_series: pd.Series, 
    entry_threshold: float = 2.0, 
    exit_threshold: float = 0.0,
    long_entry_threshold: float | None = None,
    short_entry_threshold: float | None = None,
    max_duration: int | None = None
) -> pd.DataFrame:
    """
    Generate trading signals based on Z-score crossing thresholds.

    Args:
        z_score_series: Series of Z-scores
        entry_threshold: Absolute Z-score threshold (symmetric) if specific thresholds not provided
        exit_threshold: Absolute Z-score threshold to exit a position
        long_entry_threshold: Specific Z-score level to enter Long (e.g. -2.0)
        short_entry_threshold: Specific Z-score level to enter Short (e.g. 2.0)
        max_duration: Maximum number of bars to hold a position before forced exit

    Returns:
        DataFrame with columns 'z', 'position', 'entry'
    """
    signals = pd.DataFrame(index=z_score_series.index)
    signals['z'] = z_score_series
    z_vals = z_score_series.values

    # Determine thresholds
    # If specific thresholds provided, use them. Otherwise use symmetric entry_threshold.
    long_thresh = long_entry_threshold if long_entry_threshold is not None else -entry_threshold
    short_thresh = short_entry_threshold if short_entry_threshold is not None else entry_threshold

    long_entry = np.asarray(z_vals < long_thresh)
    long_exit = np.asarray(z_vals > -exit_threshold)
    short_entry = np.asarray(z_vals > short_thresh)
    short_exit = np.asarray(z_vals < exit_threshold)
    
    position = np.zeros(len(z_vals))
    current = 0.0
    bars_in_trade = 0

    for i in range(len(z_vals)):
        # Check max duration exit first
        if current != 0 and max_duration is not None:
            if bars_in_trade >= max_duration:
                current = 0.0
                bars_in_trade = 0
            else:
                bars_in_trade += 1

        if current == 0:
            if long_entry[i]: 
                current = 1.0
                bars_in_trade = 1
            elif short_entry[i]: 
                current = -1.0
                bars_in_trade = 1
        elif current == 1.0:
            if long_exit[i]: 
                current = 0.0
                bars_in_trade = 0
        elif current == -1.0:
            if short_exit[i]: 
                current = 0.0
                bars_in_trade = 0
        
        position[i] = current
        
    signals['position'] = position
    signals['entry'] = np.abs(np.diff(position, prepend=0))
    return signals


def calculate_equity_with_costs(signals: pd.DataFrame, price_series: pd.Series, cost_bps: float = 1.0, position_size: float = 1.0) -> pd.Series:
    """
    Calculate equity curve considering transaction costs.

    Args:
        signals: DataFrame with 'position' and 'entry' columns
        price_series: Series of asset prices (or spread values)
        cost_bps: Transaction cost in basis points per trade (one-way)
        position_size: Size of the position

    Returns:
        Series representing cumulative equity
    """
    delta_p = price_series.diff().fillna(0)
    # Gross PnL: Position held from previous step * change in price
    gross_pnl = (signals['position'].shift(1).fillna(0) * delta_p) * position_size
    
    # Cost penalty: applied on every entry/exit (change in position)
    # trades is abs(change in position)
    trades = signals['entry'] * position_size
    # 1 bp = 0.0001
    cost_penalty = trades * (cost_bps * 0.0001)
    
    net_pnl = gross_pnl - cost_penalty
    equity = (1 + net_pnl).cumprod()
    return equity
