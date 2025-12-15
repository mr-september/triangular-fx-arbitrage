"""
Performance metrics for strategy evaluation.

Implements standard quantitative finance metrics:
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Calmar ratio
- Win rate and profit factor
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a strategy."""

    # Return metrics
    total_return: float
    cagr: float  # Compound Annual Growth Rate
    annualized_volatility: float

    # Risk-adjusted metrics
    sharpe: float
    sortino: float
    calmar: float

    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int  # In bars

    # Trade metrics (optional)
    win_rate: float | None = None
    profit_factor: float | None = None
    avg_win: float | None = None
    avg_loss: float | None = None

    def __str__(self) -> str:
        lines = [
            "Performance Metrics",
            "=" * 40,
            f"Total Return: {self.total_return:>12.2%}",
            f"CAGR: {self.cagr:>18.2%}",
            f"Volatility (ann.): {self.annualized_volatility:>8.2%}",
            "",
            "Risk-Adjusted:",
            f"  Sharpe Ratio: {self.sharpe:>10.2f}",
            f"  Sortino Ratio: {self.sortino:>9.2f}",
            f"  Calmar Ratio: {self.calmar:>10.2f}",
            "",
            "Drawdown:",
            f"  Maximum: {self.max_drawdown:>14.2%}",
            f"  Max Duration: {self.max_drawdown_duration:>9} bars",
        ]

        if self.win_rate is not None:
            lines.extend(
                [
                    "",
                    "Trade Statistics:",
                    f"  Win Rate: {self.win_rate:>13.1%}",
                    f"  Profit Factor: {self.profit_factor:>8.2f}",
                    f"  Avg Win: {self.avg_win:>14.4%}",
                    f"  Avg Loss: {self.avg_loss:>13.4%}",
                ]
            )

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "annualized_volatility": self.annualized_volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
        }


def compute_metrics(
    returns: pd.Series,
    equity_curve: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
    bars_per_year: int = 252 * 288,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Compute comprehensive performance metrics.

    Args:
        returns: Strategy returns per bar
        equity_curve: Optional equity curve (will compute if not provided)
        trades: Optional DataFrame with 'return' column for trade metrics
        bars_per_year: Number of bars per year for annualization
        risk_free_rate: Annual risk-free rate (default 0)

    Returns:
        PerformanceMetrics object
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return PerformanceMetrics(
            total_return=0.0,
            cagr=0.0,
            annualized_volatility=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
        )

    # Compute equity curve if not provided
    if equity_curve is None:
        equity_curve = (1 + returns).cumprod()

    # Total return
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

    # Annualized metrics
    n_bars = len(returns)
    years = n_bars / bars_per_year

    # CAGR
    if years > 0 and equity_curve.iloc[-1] > 0:
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
    else:
        cagr = 0.0

    # Volatility
    annualized_vol = returns.std() * np.sqrt(bars_per_year)

    # Sharpe ratio
    rf_per_bar = (1 + risk_free_rate) ** (1 / bars_per_year) - 1
    excess_returns = returns - rf_per_bar
    sharpe = (
        excess_returns.mean() / returns.std() * np.sqrt(bars_per_year)
        if returns.std() > 0
        else 0.0
    )

    # Sortino ratio (downside deviation only)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
    sortino = (
        excess_returns.mean() / downside_std * np.sqrt(bars_per_year)
        if downside_std > 0
        else 0.0
    )

    # Drawdown analysis
    max_dd, max_dd_duration = compute_drawdown(equity_curve)

    # Calmar ratio
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0.0

    # Trade metrics
    win_rate = None
    profit_factor = None
    avg_win = None
    avg_loss = None

    if trades is not None and len(trades) > 0 and "return" in trades.columns:
        wins = trades[trades["return"] > 0]["return"]
        losses = trades[trades["return"] < 0]["return"]

        win_rate = len(wins) / len(trades)
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0

        total_wins = wins.sum()
        total_losses = abs(losses.sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        annualized_volatility=annualized_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )


def compute_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    """
    Compute maximum drawdown and its duration.

    Args:
        equity_curve: Equity curve series

    Returns:
        Tuple of (max_drawdown, max_drawdown_duration)
    """
    # Running maximum
    running_max = equity_curve.cummax()

    # Drawdown at each point
    drawdown = (equity_curve - running_max) / running_max

    # Maximum drawdown
    max_dd = drawdown.min()  # This is negative

    # Drawdown duration
    # Find periods in drawdown
    in_drawdown = drawdown < 0
    prev_in_drawdown = in_drawdown.astype(float).shift(1).fillna(0.0).astype(bool)
    drawdown_starts = in_drawdown & ~prev_in_drawdown
    drawdown_ends = ~in_drawdown & prev_in_drawdown

    # Calculate duration of each drawdown
    max_duration = 0
    current_duration = 0

    for i in range(len(in_drawdown)):
        if in_drawdown.iloc[i]:
            current_duration += 1
        else:
            if current_duration > max_duration:
                max_duration = current_duration
            current_duration = 0

    # Check if we ended in a drawdown
    if current_duration > max_duration:
        max_duration = current_duration

    return float(max_dd), max_duration


def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 252 * 24,  # ~1 month of 5-min bars
    bars_per_year: int = 252 * 288,
) -> pd.Series:
    """
    Compute rolling Sharpe ratio.

    Args:
        returns: Strategy returns
        window: Rolling window size
        bars_per_year: For annualization

    Returns:
        Rolling Sharpe ratio series
    """
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()

    sharpe = rolling_mean / rolling_std * np.sqrt(bars_per_year)
    sharpe.name = "rolling_sharpe"

    return sharpe
