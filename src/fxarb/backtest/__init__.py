"""Backtesting framework for strategy evaluation."""

from .engine import Backtester, BacktestResult, run_backtest
from .metrics import compute_metrics, PerformanceMetrics
from .wfo import WalkForwardOptimizer, WFOResult
from .kelly import kelly_criterion, optimal_position_size

__all__ = [
    "Backtester",
    "BacktestResult",
    "run_backtest",
    "compute_metrics",
    "PerformanceMetrics",
    "WalkForwardOptimizer",
    "WFOResult",
    "kelly_criterion",
    "optimal_position_size",
]
