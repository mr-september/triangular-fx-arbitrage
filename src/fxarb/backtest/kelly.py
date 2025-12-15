"""
Kelly Criterion for position sizing.

The Kelly Criterion provides the optimal bet size to maximize long-term
geometric growth rate of capital.

Full Kelly: f* = (bp - q) / b = p - q/b
Where:
    p = probability of win
    q = 1 - p = probability of loss
    b = win/loss ratio (average win / average loss)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KellyResult:
    """Results from Kelly Criterion calculation."""

    full_kelly: float  # Optimal fraction (can be > 1)
    half_kelly: float  # Conservative: 50% of optimal
    quarter_kelly: float  # Very conservative: 25% of optimal

    win_rate: float
    win_loss_ratio: float  # Average win / Average loss
    edge: float  # Expected value per unit bet

    def __format__(self, format_spec: str) -> str:
        """Format the full_kelly value."""
        return self.full_kelly.__format__(format_spec)

    def __str__(self) -> str:
        return (
            f"Kelly Criterion Results\n"
            f"{'=' * 40}\n"
            f"Win Rate: {self.win_rate:.1%}\n"
            f"Win/Loss Ratio: {self.win_loss_ratio:.2f}\n"
            f"Edge (EV per unit): {self.edge:.4f}\n"
            f"\nOptimal Position Size:\n"
            f"  Full Kelly: {self.full_kelly:.1%}\n"
            f"  Half Kelly: {self.half_kelly:.1%}\n"
            f"  Quarter Kelly: {self.quarter_kelly:.1%}"
        )


def kelly_criterion(
    trades: pd.DataFrame | None = None,
    win_rate: float | None = None,
    avg_win: float | None = None,
    avg_loss: float | None = None,
) -> KellyResult:
    """
    Calculate Kelly Criterion optimal position size.

    Can calculate from either a trades DataFrame or directly from
    win rate and average win/loss values.

    Args:
        trades: DataFrame with 'return' column or Series of returns
        win_rate: Probability of winning trade (if trades not provided)
        avg_win: Average winning trade return (if trades not provided)
        avg_loss: Average losing trade return (should be negative)

    Returns:
        KellyResult with optimal position sizes
    """
    returns = None
    if trades is not None:
        if isinstance(trades, pd.Series):
            returns = trades
        elif isinstance(trades, pd.DataFrame) and "return" in trades.columns:
            returns = trades["return"]

    if returns is not None:
        # Calculate from trades
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            # Cannot calculate Kelly without both wins and losses
            return KellyResult(
                full_kelly=0.0,
                half_kelly=0.0,
                quarter_kelly=0.0,
                win_rate=len(wins) / max(1, len(returns)),
                win_loss_ratio=0.0,
                edge=returns.mean(),
            )

        win_rate = len(wins) / len(returns)
        avg_win = wins.mean()
        avg_loss = losses.mean()  # Already negative

    elif win_rate is None or avg_win is None or avg_loss is None:
        raise ValueError("Must provide either trades DataFrame/Series or all of win_rate, avg_win, avg_loss")

    # Ensure avg_loss is positive for calculation
    avg_loss_abs = abs(avg_loss)

    if avg_loss_abs == 0:
        # No losses, Kelly is undefined (would be infinite)
        return KellyResult(
            full_kelly=1.0,  # Cap at 100%
            half_kelly=0.5,
            quarter_kelly=0.25,
            win_rate=win_rate,
            win_loss_ratio=float("inf"),
            edge=win_rate * avg_win,
        )

    # Win/loss ratio
    b = avg_win / avg_loss_abs

    # Kelly formula: f* = p - q/b = p - (1-p)/b
    p = win_rate
    q = 1 - p

    full_kelly = p - q / b

    # Expected value per unit bet
    edge = p * avg_win + q * avg_loss  # avg_loss is negative

    return KellyResult(
        full_kelly=full_kelly,
        half_kelly=full_kelly / 2,
        quarter_kelly=full_kelly / 4,
        win_rate=win_rate,
        win_loss_ratio=b,
        edge=edge,
    )


def optimal_position_size(
    kelly_result: KellyResult,
    fraction: float = 0.25,
    max_position: float = 1.0,
    min_edge_required: float = 0.0,
) -> float:
    """
    Get recommended position size from Kelly result.

    Args:
        kelly_result: Result from kelly_criterion()
        fraction: Fraction of full Kelly to use (default 0.25 = quarter Kelly)
        max_position: Maximum allowed position size (cap)
        min_edge_required: Minimum edge required to take a position

    Returns:
        Recommended position size (0 to max_position)
    """
    if kelly_result.edge < min_edge_required:
        return 0.0

    if kelly_result.full_kelly <= 0:
        # Negative Kelly means no edge, don't bet
        return 0.0

    position = kelly_result.full_kelly * fraction

    # Apply bounds
    position = max(0, min(position, max_position))

    return position


def rolling_kelly(
    trades: pd.DataFrame,
    window: int = 50,
    min_trades: int = 20,
) -> pd.DataFrame:
    """
    Calculate rolling Kelly Criterion over trade history.

    Useful for seeing how optimal position size changes over time.

    Args:
        trades: DataFrame with 'return' column and datetime index
        window: Number of trades to use for calculation
        min_trades: Minimum trades required before calculating

    Returns:
        DataFrame with rolling Kelly values
    """
    results = []

    for i in range(min_trades, len(trades)):
        window_trades = trades.iloc[max(0, i - window) : i]

        try:
            kelly = kelly_criterion(window_trades)
            results.append(
                {
                    "trade_index": i,
                    "date": trades.index[i] if hasattr(trades.index, "__iter__") else i,
                    "full_kelly": kelly.full_kelly,
                    "half_kelly": kelly.half_kelly,
                    "quarter_kelly": kelly.quarter_kelly,
                    "win_rate": kelly.win_rate,
                    "edge": kelly.edge,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(results)
