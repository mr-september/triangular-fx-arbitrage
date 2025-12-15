"""Visualization utilities for spreads, signals, and performance."""

from .spreads import plot_spread, plot_zscore, plot_signals
from .performance import plot_equity_curve, plot_drawdown, plot_returns_distribution

__all__ = [
    "plot_spread",
    "plot_zscore",
    "plot_signals",
    "plot_equity_curve",
    "plot_drawdown",
    "plot_returns_distribution",
]
