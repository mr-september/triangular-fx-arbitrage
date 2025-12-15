"""
Walk-Forward Optimization (WFO) for robust strategy validation.

WFO iteratively:
1. Optimizes parameters on a training window
2. Tests on a subsequent out-of-sample window
3. Rolls forward and repeats

This provides realistic simulation of periodic recalibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .engine import Backtester, BacktestResult
from .metrics import PerformanceMetrics, compute_metrics


@dataclass
class WFOResult:
    """Results from Walk-Forward Optimization."""

    # Aggregated out-of-sample equity curve
    oos_equity_curve: pd.Series

    # Performance metrics (out-of-sample)
    oos_metrics: PerformanceMetrics

    # Per-window results
    window_results: list[dict]

    # Optimal parameters per window
    optimal_params: list[dict]

    # In-sample vs out-of-sample comparison
    is_sharpe_mean: float
    is_sharpe_std: float
    oos_sharpe_mean: float
    oos_sharpe_std: float

    def __str__(self) -> str:
        return (
            f"Walk-Forward Optimization Results\n"
            f"{'=' * 50}\n"
            f"Number of windows: {len(self.window_results)}\n"
            f"\nIn-Sample Sharpe: {self.is_sharpe_mean:.2f} ± {self.is_sharpe_std:.2f}\n"
            f"Out-of-Sample Sharpe: {self.oos_sharpe_mean:.2f} ± {self.oos_sharpe_std:.2f}\n"
            f"\nAggregated OOS Performance:\n"
            f"{self.oos_metrics}"
        )


@dataclass
class WalkForwardOptimizer:
    """
    Walk-Forward Optimization framework.

    Provides robust strategy validation by iteratively training and testing
    on rolling windows.
    """

    # Window configuration
    train_size: int  # Number of bars for training
    test_size: int  # Number of bars for testing
    step_size: int | None = None  # Step between windows (default = test_size)

    # Optimization settings
    metric: str = "sharpe"  # Metric to optimize ('sharpe', 'calmar', 'return')
    n_jobs: int = 1  # Parallel jobs (not implemented yet)
    transaction_cost_pips: float = 0.0  # Cost per trade in pips

    def __post_init__(self):
        if self.step_size is None:
            self.step_size = self.test_size

    def optimize(
        self,
        spread: pd.Series,
        signal_generator: Callable,
        param_grid: dict[str, list],
        min_train_bars: int | None = None,
    ) -> WFOResult:
        """
        Run walk-forward optimization.

        Args:
            spread: The trading spread series
            signal_generator: Function(spread, **params) -> signals DataFrame
            param_grid: Dict of parameter names to lists of values to try

        Returns:
            WFOResult with aggregated performance
        """
        n_bars = len(spread)
        if min_train_bars is None:
            min_train_bars = self.train_size

        # Generate windows
        windows = []
        start = 0

        while start + self.train_size + self.test_size <= n_bars:
            train_start = start
            train_end = start + self.train_size
            test_start = train_end
            test_end = min(test_start + self.test_size, n_bars)

            windows.append(
                {
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )

            start += self.step_size

        if len(windows) == 0:
            raise ValueError(
                f"Not enough data for WFO. Need {self.train_size + self.test_size} bars, "
                f"have {n_bars}"
            )

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        window_results = []
        optimal_params_list = []
        oos_returns_list = []

        for window in tqdm(windows, desc="WFO Windows"):
            train_spread = spread.iloc[window["train_start"] : window["train_end"]]
            test_spread = spread.iloc[window["test_start"] : window["test_end"]]

            # In-sample optimization
            best_params = None
            best_metric = -np.inf

            for params in param_combinations:
                param_dict = dict(zip(param_names, params))

                try:
                    # Generate signals and backtest on training data
                    signals = signal_generator(train_spread, **param_dict)
                    result = self._backtest(train_spread, signals)

                    metric_value = self._get_metric(result, self.metric)

                    if metric_value > best_metric:
                        best_metric = metric_value
                        best_params = param_dict

                except Exception:
                    continue

            if best_params is None:
                # No valid parameters found, skip window
                continue

            optimal_params_list.append(best_params)

            # Out-of-sample test with best parameters
            test_signals = signal_generator(test_spread, **best_params)
            test_result = self._backtest(test_spread, test_signals)

            # Store results
            window_results.append(
                {
                    "train_start": spread.index[window["train_start"]],
                    "train_end": spread.index[window["train_end"] - 1],
                    "test_start": spread.index[window["test_start"]],
                    "test_end": spread.index[window["test_end"] - 1],
                    "is_sharpe": best_metric if self.metric == "sharpe" else None,
                    "oos_sharpe": test_result.sharpe,
                    "oos_return": test_result.total_return,
                    "params": best_params,
                }
            )

            oos_returns_list.append(test_result.returns)

        if len(oos_returns_list) == 0:
            raise ValueError("No valid windows found during optimization")

        # Aggregate OOS returns
        oos_returns = pd.concat(oos_returns_list)
        oos_equity = (1 + oos_returns).cumprod()
        oos_metrics = compute_metrics(oos_returns, oos_equity)

        # Compute summary statistics
        is_sharpes = [w["is_sharpe"] for w in window_results if w["is_sharpe"] is not None]
        oos_sharpes = [w["oos_sharpe"] for w in window_results]

        return WFOResult(
            oos_equity_curve=oos_equity,
            oos_metrics=oos_metrics,
            window_results=window_results,
            optimal_params=optimal_params_list,
            is_sharpe_mean=np.mean(is_sharpes) if is_sharpes else 0,
            is_sharpe_std=np.std(is_sharpes) if is_sharpes else 0,
            oos_sharpe_mean=np.mean(oos_sharpes),
            oos_sharpe_std=np.std(oos_sharpes),
        )

    def _backtest(self, spread: pd.Series, signals: pd.DataFrame) -> BacktestResult:
        """Run backtest helper."""
        from .engine import run_backtest

        return run_backtest(spread, signals, transaction_cost_pips=self.transaction_cost_pips)

    def _get_metric(self, result: BacktestResult, metric: str) -> float:
        """Extract metric value from backtest result."""
        if metric == "sharpe":
            return result.sharpe
        elif metric == "calmar":
            return result.calmar
        elif metric == "return":
            return result.total_return
        else:
            raise ValueError(f"Unknown metric: {metric}")


def create_param_grid(**kwargs) -> dict[str, list]:
    """
    Convenience function to create parameter grid.

    Example:
        grid = create_param_grid(
            lookback=[50, 100, 200],
            entry_threshold=[1.5, 2.0, 2.5]
        )
    """
    return kwargs
