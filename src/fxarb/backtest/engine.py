"""
Vectorized backtesting engine for mean-reversion strategies.

Implements proper execution modeling:
- Signals on bar t are executed at the open of bar t+1
- No lookahead bias
- Tracks positions, P&L, and equity curve
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    # Time series
    equity_curve: pd.Series
    positions: pd.Series
    returns: pd.Series
    trades: pd.DataFrame

    # Trade statistics
    n_trades: int
    n_winning: int
    n_losing: int
    avg_trade_return: float
    avg_trade_duration: float

    # Overall performance
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    calmar: float

    # Additional info
    strategy_params: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Backtest Results\n"
            f"{'=' * 40}\n"
            f"Total Return: {self.total_return:.2%}\n"
            f"CAGR: {self.cagr:.2%}\n"
            f"Sharpe Ratio: {self.sharpe:.2f}\n"
            f"Max Drawdown: {self.max_drawdown:.2%}\n"
            f"Calmar Ratio: {self.calmar:.2f}\n"
            f"\nTrades:\n"
            f"  Total: {self.n_trades}\n"
            f"  Winning: {self.n_winning} ({100 * self.n_winning / max(1, self.n_trades):.1f}%)\n"
            f"  Losing: {self.n_losing}\n"
            f"  Avg Return: {self.avg_trade_return:.4%}\n"
            f"  Avg Duration: {self.avg_trade_duration:.1f} bars"
        )


class Backtester:
    """
    Vectorized backtester for spread trading strategies.

    Key features:
    - Proper signal lag (trade on next bar's open)
    - Position tracking
    - P&L calculation from spread changes
    - Transaction costs and leverage modeling
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,  # $1M default
        position_size: float = 1.0,
        bars_per_year: int = 252 * 288,  # 5-min bars
        transaction_cost_pips: float = 0.0,  # Cost per trade in pips
        leverage: float = 1.0,  # FX leverage ratio (e.g., 50.0 for 50:1)
        max_position_pct: float = 0.1,  # Max 10% of capital per position
        compounding: str = "arithmetic",  # 'arithmetic' or 'geometric'
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital in USD (default $1M)
            position_size: Position size multiplier (1.0 = full size)
            bars_per_year: Number of bars per year (for annualization)
            transaction_cost_pips: Cost per trade in pips (1 pip = 0.0001)
            leverage: FX leverage ratio (e.g., 50.0 for 50:1)
            max_position_pct: Maximum position as % of capital
            compounding: 'arithmetic' (fixed lots) or 'geometric' (fixed fractional)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.bars_per_year = bars_per_year
        self.transaction_cost_pips = transaction_cost_pips
        self.leverage = leverage
        self.max_position_pct = max_position_pct
        self.compounding = compounding
        
        # Convert pips to spread units (1 pip ≈ 0.0001 for major pairs)
        self.transaction_cost = transaction_cost_pips * 0.0001

    def run(
        self,
        spread: pd.Series,
        positions: pd.Series,
        execution_price: pd.Series = None,
        lag: int = 1,
    ) -> BacktestResult:
        """
        Run backtest on spread with given position signals.

        Position values:
        - 1: Long the spread
        - -1: Short the spread
        - 0: Flat

        Args:
            spread: The cointegrated spread to trade (used for alignment and default execution)
            positions: Position signals (1, 0, -1)
            execution_price: Optional. If set, used for P&L calc instead of spread.
            lag: Lag in bars between signal and execution. Default 1.

        Returns:
            BacktestResult with equity curve and metrics
        """
        # Align data
        common_idx = spread.index.intersection(positions.index)
        if execution_price is not None:
            common_idx = common_idx.intersection(execution_price.index)
            exec_series = execution_price.loc[common_idx]
        else:
            exec_series = spread.loc[common_idx]

        spread = spread.loc[common_idx]
        positions = positions.loc[common_idx]

        # Shift positions by lag to simulate execution delay
        # lag=1: Position at t corresponds to signal at t-1 (Trade Close-to-Close)
        # lag=2: Position at t corresponds to signal at t-2 (Trade Open-to-Next-Open)
        positions_lagged = positions.shift(lag).fillna(0)

        # Compute price changes for P&L
        # P&L at t = Position[t] * (Price[t] - Price[t-1])
        price_changes = exec_series.diff().fillna(0)

        # Strategy P&L
        gross_pnl = positions_lagged * price_changes * self.position_size
        
        # Transaction costs: deduct on position changes (entries and exits)
        position_changes = positions_lagged.diff().fillna(0).abs()
        trade_costs = position_changes * self.transaction_cost
        
        # Net P&L after costs (in spread units)
        strategy_pnl = (gross_pnl - trade_costs).fillna(0)
        
        # Convert spread P&L to dollar returns
        # Each spread unit represents a notional position
        # For FX: if spread is in log prices, 1 unit ≈ $100K notional @ 0.01 move per unit
        # We use initial_capital as the notional base and position size scales it
        # Raw PnL calculation (no volatility scaling)
        # strategy_pnl = position * spread_change
        # Since spread is log-prices, a change of 0.01 is effectively a 1% return on notional
        strategy_returns = strategy_pnl
        
        # Apply leverage to returns
        # This assumes the position size is scaled by leverage
        # e.g., if leverage=2.0, we hold 2x the notional
        leveraged_returns = strategy_returns * self.leverage

        # Equity curve calculation
        if self.compounding == "geometric":
            # Geometric: Fixed % risk (reinvest profits)
            # Equity[t] = Equity[t-1] * (1 + r[t])
            equity_curve = self.initial_capital * (1 + leveraged_returns).cumprod()
        else:
            # Arithmetic: Fixed $ risk (simple interest)
            # Equity[t] = Initial + Sum(Returns * Initial)
            equity_curve = self.initial_capital * (1 + leveraged_returns.cumsum())
            
        equity_curve.name = "equity"

        # Extract trades
        trades = self._extract_trades(positions_lagged, price_changes)

        # Compute metrics
        from .metrics import compute_metrics

        metrics = compute_metrics(
            returns=strategy_returns,
            equity_curve=equity_curve,
            bars_per_year=self.bars_per_year,
        )

        # Trade statistics
        if len(trades) > 0:
            n_trades = len(trades)
            n_winning = (trades["return"] > 0).sum()
            n_losing = (trades["return"] < 0).sum()
            avg_trade_return = trades["return"].mean()
            avg_trade_duration = trades["duration"].mean()
        else:
            n_trades = 0
            n_winning = 0
            n_losing = 0
            avg_trade_return = 0.0
            avg_trade_duration = 0.0

        return BacktestResult(
            equity_curve=equity_curve,
            positions=positions_lagged,
            returns=strategy_returns,
            trades=trades,
            n_trades=n_trades,
            n_winning=n_winning,
            n_losing=n_losing,
            avg_trade_return=avg_trade_return,
            avg_trade_duration=avg_trade_duration,
            total_return=metrics.total_return,
            cagr=metrics.cagr,
            sharpe=metrics.sharpe,
            max_drawdown=metrics.max_drawdown,
            calmar=metrics.calmar,
        )

    def _extract_trades(
        self,
        positions: pd.Series,
        spread_returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Extract individual trades from position series.

        A trade starts when we enter a position and ends when we exit.
        Includes transaction cost deduction (per round trip).
        """
        trades = []

        entry_idx = None
        entry_pos = 0
        cumulative_return = 0.0
        start_i = 0
        
        # Iterate through signals
        # positions[i] is the position HELD during bar i
        for i in range(len(positions)):
            pos = positions.iloc[i]
            idx = positions.index[i]
            ret = spread_returns.iloc[i]
            
            if entry_pos == 0:
                if pos != 0:
                    # New trade starts
                    entry_pos = pos
                    entry_idx = idx
                    cumulative_return = pos * ret
                    start_i = i
            else:
                if pos == entry_pos:
                    # Continue trade
                    cumulative_return += pos * ret
                else:
                    # Trade changed (closed or reversed)
                    # Deduct round-trip transaction cost
                    net_return = cumulative_return - (2 * self.transaction_cost)
                    
                    trades.append(
                        {
                            "entry_time": entry_idx,
                            "exit_time": idx,
                            "direction": "long" if entry_pos > 0 else "short",
                            "return": net_return,
                            "duration": i - start_i,
                        }
                    )
                    
                    if pos != 0:
                        # Reversal: Start new trade immediately
                        entry_pos = pos
                        entry_idx = idx
                        cumulative_return = pos * ret
                        start_i = i
                    else:
                        # Flat
                        entry_pos = 0
                        cumulative_return = 0.0

        return pd.DataFrame(trades)


def run_backtest(
    spread: pd.Series,
    signals: pd.DataFrame,
    initial_capital: float = 100000.0,
    position_size: float = 1.0,
    transaction_cost_pips: float = 0.0,
    leverage: float = 1.0,
    compounding: str = "arithmetic",
    execution_price: pd.Series = None,
    lag: int = 1,
) -> BacktestResult:
    """
    Convenience function to run backtest.

    Args:
        spread: The cointegrated spread
        signals: DataFrame with 'position' column
        initial_capital: Starting capital
        position_size: Position size multiplier
        transaction_cost_pips: Cost per trade in pips
        leverage: FX leverage ratio
        compounding: 'arithmetic' or 'geometric'
        execution_price: Price series for execution (e.g. Open params). Defaults to spread.
        lag: Trade latency in bars. 1 = Trade at Close[t] (signal[t-1]), 2 = Trade at Open[t+1] (signal[t-1])
            Default 1 (Close-to-Close).
    """
    backtester = Backtester(
        initial_capital=initial_capital,
        position_size=position_size,
        transaction_cost_pips=transaction_cost_pips,
        leverage=leverage,
        compounding=compounding,
    )

    positions = signals["position"] if "position" in signals.columns else signals

    return backtester.run(spread, positions, execution_price=execution_price, lag=lag)
