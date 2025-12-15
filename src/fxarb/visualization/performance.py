"""
Performance visualization utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Equity Curve",
    figsize: tuple[int, int] = (14, 6),
    log_scale: bool = False,
    ax: Axes | None = None,
) -> Figure | Axes:
    """
    Plot equity curve with optional benchmark.

    Args:
        equity_curve: Strategy equity curve
        benchmark: Optional benchmark equity curve
        title: Plot title
        figsize: Figure size
        log_scale: Use logarithmic y-axis
        ax: Optional existing axes

    Returns:
        Figure or Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot strategy
    ax.plot(
        equity_curve.index,
        equity_curve.values,
        linewidth=1.2,
        label="Strategy",
        color="blue",
    )

    # Plot benchmark if provided
    if benchmark is not None:
        ax.plot(
            benchmark.index,
            benchmark.values,
            linewidth=1,
            label="Benchmark",
            color="gray",
            alpha=0.7,
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add some basic stats as text
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    ax.text(
        0.02,
        0.98,
        f"Total Return: {total_return:.1f}%",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    return fig if ax is None else ax


def plot_drawdown(
    equity_curve: pd.Series,
    title: str = "Drawdown",
    figsize: tuple[int, int] = (14, 4),
    ax: Axes | None = None,
) -> Figure | Axes:
    """
    Plot drawdown from peak.

    Args:
        equity_curve: Equity curve series
        title: Plot title
        figsize: Figure size
        ax: Optional existing axes

    Returns:
        Figure or Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Calculate drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max

    # Plot
    ax.fill_between(
        drawdown.index,
        0,
        drawdown.values * 100,
        color="red",
        alpha=0.5,
    )
    ax.plot(drawdown.index, drawdown.values * 100, linewidth=0.8, color="darkred")

    # Mark max drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd = drawdown.min()
    ax.axhline(max_dd * 100, color="darkred", linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(
        f"Max DD: {max_dd * 100:.1f}%",
        xy=(max_dd_idx, max_dd * 100),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=9,
        color="darkred",
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_ylim(min(drawdown.min() * 100 * 1.1, -1), 1)
    ax.grid(True, alpha=0.3)

    return fig if ax is None else ax


def plot_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution",
    figsize: tuple[int, int] = (12, 5),
    bins: int = 50,
) -> Figure:
    """
    Plot returns distribution with statistics.

    Args:
        returns: Returns series
        title: Plot title
        figsize: Figure size
        bins: Number of histogram bins

    Returns:
        Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    returns_pct = returns * 100  # Convert to percentage

    # Histogram
    ax1 = axes[0]
    ax1.hist(
        returns_pct.dropna(),
        bins=bins,
        edgecolor="black",
        alpha=0.7,
        density=True,
    )

    # Add normal distribution overlay
    mean = returns_pct.mean()
    std = returns_pct.std()
    x = np.linspace(returns_pct.min(), returns_pct.max(), 100)
    from scipy import stats

    normal = stats.norm.pdf(x, mean, std)
    ax1.plot(x, normal, "r-", linewidth=2, label="Normal fit")

    ax1.axvline(mean, color="black", linestyle="--", linewidth=1, label=f"Mean: {mean:.4f}%")
    ax1.set_xlabel("Return (%)")
    ax1.set_ylabel("Density")
    ax1.set_title("Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Statistics box
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()

    stats_text = (
        f"Mean: {mean:.4f}%\n"
        f"Std: {std:.4f}%\n"
        f"Skewness: {skewness:.2f}\n"
        f"Kurtosis: {kurtosis:.2f}\n"
        f"VaR (95%): {var_95 * 100:.4f}%\n"
        f"CVaR (95%): {cvar_95 * 100:.4f}%"
    )
    ax1.text(
        0.98,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Q-Q plot
    ax2 = axes[1]
    stats.probplot(returns.dropna(), dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot (vs Normal)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_monthly_returns(
    returns: pd.Series,
    title: str = "Monthly Returns Heatmap",
    figsize: tuple[int, int] = (14, 6),
) -> Figure:
    """
    Plot monthly returns heatmap.

    Args:
        returns: Daily returns series with datetime index
        title: Plot title
        figsize: Figure size

    Returns:
        Figure
    """
    # Resample to monthly
    monthly = (1 + returns).resample("ME").prod() - 1

    # Create pivot table: years as rows, months as columns
    monthly_df = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        }
    )
    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ][: len(pivot.columns)]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    import seaborn as sns

    sns.heatmap(
        pivot * 100,
        annot=True,
        fmt=".1f",
        center=0,
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Return (%)"},
    )

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    plt.tight_layout()
    return fig


def create_performance_report(
    equity_curve: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame | None = None,
    title: str = "Performance Report",
    figsize: tuple[int, int] = (14, 12),
) -> Figure:
    """
    Create a comprehensive performance report figure.

    Args:
        equity_curve: Strategy equity curve
        returns: Strategy returns
        trades: Optional trades DataFrame
        title: Report title
        figsize: Figure size

    Returns:
        Figure with multiple subplots
    """
    fig = plt.figure(figsize=figsize)

    # Layout: 3 rows
    # Row 1: Equity curve (full width)
    # Row 2: Drawdown (full width)
    # Row 3: Returns distribution | Monthly returns or trade stats

    ax1 = fig.add_subplot(3, 1, 1)
    plot_equity_curve(equity_curve, ax=ax1, title="Equity Curve")

    ax2 = fig.add_subplot(3, 1, 2)
    plot_drawdown(equity_curve, ax=ax2, title="Drawdown")

    ax3 = fig.add_subplot(3, 2, 5)
    ax3.hist(returns.dropna() * 100, bins=40, edgecolor="black", alpha=0.7)
    ax3.set_xlabel("Return (%)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Returns Distribution")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(3, 2, 6)
    if trades is not None and len(trades) > 0:
        # Trade statistics
        wins = (trades["return"] > 0).sum()
        losses = (trades["return"] < 0).sum()
        ax4.bar(["Wins", "Losses"], [wins, losses], color=["green", "red"], alpha=0.7)
        ax4.set_title(f"Trade Outcomes (n={len(trades)})")
        ax4.set_ylabel("Count")
    else:
        # Rolling Sharpe
        rolling_sharpe = returns.rolling(window=252 * 24).mean() / returns.rolling(window=252 * 24).std()
        rolling_sharpe = rolling_sharpe * np.sqrt(252 * 288)  # Annualize
        ax4.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=0.8)
        ax4.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax4.set_title("Rolling Sharpe (30-day)")
        ax4.set_ylabel("Sharpe Ratio")
        ax4.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    return fig
