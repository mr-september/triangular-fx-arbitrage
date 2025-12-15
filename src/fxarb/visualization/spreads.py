"""
Visualization for spreads, Z-scores, and trading signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_spread(
    spread: pd.Series,
    title: str = "Cointegrated Spread",
    figsize: tuple[int, int] = (14, 6),
    ax: Axes | None = None,
) -> Figure | Axes:
    """
    Plot the spread with mean and standard deviation bands.

    Args:
        spread: The spread series
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

    # Calculate statistics
    mean = spread.mean()
    std = spread.std()

    # Plot spread
    ax.plot(spread.index, spread.values, linewidth=0.8, alpha=0.8, label="Spread")

    # Plot mean and bands
    ax.axhline(mean, color="black", linestyle="--", linewidth=1, label="Mean")
    ax.axhline(mean + 2 * std, color="red", linestyle=":", linewidth=1, label="±2σ")
    ax.axhline(mean - 2 * std, color="red", linestyle=":", linewidth=1)
    ax.axhline(mean + std, color="orange", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(mean - std, color="orange", linestyle=":", linewidth=0.8, alpha=0.7)

    # Shade the bands
    ax.fill_between(
        spread.index, mean - std, mean + std, alpha=0.1, color="green", label="±1σ band"
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread Value")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if ax is None:
        plt.tight_layout()

    return fig if ax is None else ax


def plot_zscore(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    title: str = "Z-Score",
    figsize: tuple[int, int] = (14, 5),
    ax: Axes | None = None,
) -> Figure | Axes:
    """
    Plot Z-score with entry/exit thresholds.

    Args:
        zscore: Z-score series
        entry_threshold: Entry threshold (positive)
        exit_threshold: Exit threshold
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

    # Plot Z-score
    ax.plot(zscore.index, zscore.values, linewidth=0.8, color="blue", alpha=0.8)

    # Entry thresholds
    ax.axhline(entry_threshold, color="red", linestyle="--", linewidth=1, label=f"Entry ±{entry_threshold}")
    ax.axhline(-entry_threshold, color="red", linestyle="--", linewidth=1)

    # Exit threshold
    ax.axhline(exit_threshold, color="green", linestyle="-", linewidth=1, label=f"Exit ({exit_threshold})")

    # Shading for entry regions
    ax.fill_between(
        zscore.index,
        entry_threshold,
        zscore.max() + 1,
        alpha=0.1,
        color="red",
        label="Short entry zone",
    )
    ax.fill_between(
        zscore.index,
        -entry_threshold,
        zscore.min() - 1,
        alpha=0.1,
        color="blue",
        label="Long entry zone",
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Z-Score")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(zscore.min() - 0.5, zscore.max() + 0.5)

    return fig if ax is None else ax


def plot_signals(
    spread: pd.Series,
    signals: pd.DataFrame,
    title: str = "Trading Signals",
    figsize: tuple[int, int] = (14, 8),
) -> Figure:
    """
    Plot spread with trading signals overlay.

    Args:
        spread: The spread series
        signals: DataFrame with 'position', 'entry', 'exit' columns
        title: Plot title
        figsize: Figure size

    Returns:
        Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[2, 1])

    # Upper plot: Spread with entry/exit markers
    ax1 = axes[0]
    ax1.plot(spread.index, spread.values, linewidth=0.8, alpha=0.7, label="Spread")

    # Mark entries
    if "entry" in signals.columns:
        long_entries = signals[signals["entry"] == 1].index
        short_entries = signals[signals["entry"] == -1].index

        ax1.scatter(
            long_entries,
            spread.loc[long_entries],
            marker="^",
            color="green",
            s=50,
            label="Long Entry",
            zorder=5,
        )
        ax1.scatter(
            short_entries,
            spread.loc[short_entries],
            marker="v",
            color="red",
            s=50,
            label="Short Entry",
            zorder=5,
        )

    # Mark exits
    if "exit" in signals.columns:
        exits = signals[signals["exit"] != 0].index
        ax1.scatter(
            exits,
            spread.loc[exits],
            marker="x",
            color="black",
            s=40,
            label="Exit",
            zorder=5,
        )

    ax1.set_title(title)
    ax1.set_ylabel("Spread Value")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Lower plot: Position
    ax2 = axes[1]
    if "position" in signals.columns:
        ax2.fill_between(
            signals.index,
            0,
            signals["position"],
            where=signals["position"] > 0,
            color="green",
            alpha=0.5,
            label="Long",
        )
        ax2.fill_between(
            signals.index,
            0,
            signals["position"],
            where=signals["position"] < 0,
            color="red",
            alpha=0.5,
            label="Short",
        )
        ax2.axhline(0, color="black", linewidth=0.5)

    ax2.set_ylabel("Position")
    ax2.set_xlabel("Date")
    ax2.set_ylim(-1.5, 1.5)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spread_components(
    data: dict[str, pd.DataFrame],
    hedge_ratios: dict[str, float] | None = None,
    title: str = "Spread Components",
    figsize: tuple[int, int] = (14, 10),
) -> Figure:
    """
    Plot individual price series and their contributions to the spread.

    Args:
        data: Dict mapping pair codes to DataFrames
        hedge_ratios: Optional hedge ratios (will show contribution)
        title: Plot title
        figsize: Figure size

    Returns:
        Figure
    """
    n_pairs = len(data)
    fig, axes = plt.subplots(n_pairs + 1, 1, figsize=figsize, sharex=True)

    for i, (pair, df) in enumerate(data.items()):
        ax = axes[i]

        if "log_price" in df.columns:
            price = df["log_price"]
            ylabel = "Log Price"
        else:
            price = np.log(df["close"])
            ylabel = "Log Close"

        ax.plot(price.index, price.values, linewidth=0.8)
        ax.set_ylabel(f"{pair.upper()}\n{ylabel}")
        ax.grid(True, alpha=0.3)

        if hedge_ratios and pair in hedge_ratios:
            ax.set_title(f"{pair.upper()} (β = {hedge_ratios[pair]:.4f})")

    # Bottom: spread
    ax_spread = axes[-1]

    if hedge_ratios:
        # Calculate spread
        spread = pd.Series(0.0, index=list(data.values())[0].index)
        for pair, df in data.items():
            if "log_price" in df.columns:
                price = df["log_price"]
            else:
                price = np.log(df["close"])

            coef = hedge_ratios.get(pair, 0)
            spread = spread + coef * price

        spread = spread - spread.mean()
        ax_spread.plot(spread.index, spread.values, linewidth=0.8, color="purple")
        ax_spread.axhline(0, color="black", linestyle="--", linewidth=0.5)

    ax_spread.set_ylabel("Spread")
    ax_spread.set_xlabel("Date")
    ax_spread.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, y=0.98)
    plt.tight_layout()
    return fig
