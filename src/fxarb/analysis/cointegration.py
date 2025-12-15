"""
Cointegration analysis for FX triplets.

Implements Johansen test and spread construction for triangular arbitrage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass
class CointegrationResult:
    """Results from Johansen cointegration test."""

    # Test statistics
    trace_stats: np.ndarray  # Trace statistics for each rank
    trace_crit_90: np.ndarray  # 90% critical values
    trace_crit_95: np.ndarray  # 95% critical values
    trace_crit_99: np.ndarray  # 99% critical values

    max_eig_stats: np.ndarray  # Max eigenvalue statistics
    max_eig_crit_90: np.ndarray
    max_eig_crit_95: np.ndarray
    max_eig_crit_99: np.ndarray

    # Cointegrating vectors (eigenvectors)
    eigenvectors: np.ndarray  # Shape: (n_vars, n_vars)

    # Number of cointegrating relationships
    n_cointegrating: int  # At 95% confidence

    # Column names for reference
    columns: list[str]

    def __str__(self) -> str:
        lines = ["Johansen Cointegration Test Results", "=" * 40]

        lines.append("\nTrace Test:")
        lines.append(f"{'Rank':<6} {'Stat':<12} {'90%':<10} {'95%':<10} {'99%':<10}")
        for i, (stat, c90, c95, c99) in enumerate(
            zip(self.trace_stats, self.trace_crit_90, self.trace_crit_95, self.trace_crit_99)
        ):
            sig = "*" if stat > c95 else ""
            lines.append(f"{i:<6} {stat:<12.4f} {c90:<10.4f} {c95:<10.4f} {c99:<10.4f} {sig}")

        lines.append(f"\nCointegrating relationships (95%): {self.n_cointegrating}")

        if self.n_cointegrating > 0:
            lines.append("\nFirst cointegrating vector (hedge ratios):")
            vec = self.eigenvectors[:, 0]
            # Normalize so first element is 1
            vec_norm = vec / vec[0]
            for col, val in zip(self.columns, vec_norm):
                lines.append(f"  {col}: {val:.6f}")

        return "\n".join(lines)

    @property
    def hedge_ratios(self) -> dict[str, float]:
        """
        Get normalized hedge ratios from first cointegrating vector.

        Returns dict mapping column names to coefficients, normalized so
        the first column has coefficient 1.0.
        """
        if self.n_cointegrating == 0:
            raise ValueError("No cointegrating relationship found")

        vec = self.eigenvectors[:, 0]
        vec_norm = vec / vec[0]

        return {col: float(val) for col, val in zip(self.columns, vec_norm)}


def johansen_test(
    data: pd.DataFrame | dict[str, pd.DataFrame],
    det_order: Literal[-1, 0, 1] = 0,
    k_ar_diff: int = 1,
    significance: float = 0.05,
) -> CointegrationResult:
    """
    Perform Johansen cointegration test on multiple time series.

    The Johansen test determines whether a group of non-stationary time series
    are cointegrated, meaning a linear combination of them is stationary.

    Args:
        data: Either a DataFrame with log prices as columns, or a dict of
              DataFrames (will use 'log_price' or 'close' column from each)
        det_order: Deterministic term order
            -1: No deterministic terms
             0: Constant term (default, recommended for FX)
             1: Constant and linear trend
        k_ar_diff: Number of lagged differences in the VECM
        significance: Significance level for counting cointegrating relationships

    Returns:
        CointegrationResult with test statistics and hedge ratios

    Example:
        >>> data = load_triplet('eurusd', 'gbpusd', 'eurgbp')
        >>> result = johansen_test(data)
        >>> print(result)
    """
    # Convert dict of DataFrames to single DataFrame of log prices
    if isinstance(data, dict):
        prices = {}
        for name, df in data.items():
            if "log_price" in df.columns:
                prices[name] = df["log_price"]
            else:
                prices[name] = np.log(df["close"])
        df_prices = pd.DataFrame(prices)
    else:
        df_prices = data

    # Ensure we have the data as numpy array
    columns = list(df_prices.columns)
    X = df_prices.dropna().values

    if X.shape[0] < 50:
        raise ValueError(f"Insufficient data points: {X.shape[0]}. Need at least 50.")

    # Run Johansen test
    result = coint_johansen(X, det_order=det_order, k_ar_diff=k_ar_diff)

    # Count significant cointegrating relationships at chosen level
    if significance <= 0.01:
        crit_idx = 2  # 99%
    elif significance <= 0.05:
        crit_idx = 1  # 95%
    else:
        crit_idx = 0  # 90%

    n_coint = 0
    for i, stat in enumerate(result.lr1):  # Trace statistics
        if stat > result.cvt[i, crit_idx]:
            n_coint += 1
        else:
            break  # Sequential testing stops at first non-rejection

    return CointegrationResult(
        trace_stats=result.lr1,
        trace_crit_90=result.cvt[:, 0],
        trace_crit_95=result.cvt[:, 1],
        trace_crit_99=result.cvt[:, 2],
        max_eig_stats=result.lr2,
        max_eig_crit_90=result.cvm[:, 0],
        max_eig_crit_95=result.cvm[:, 1],
        max_eig_crit_99=result.cvm[:, 2],
        eigenvectors=result.evec,
        n_cointegrating=n_coint,
        columns=columns,
    )


def construct_spread(
    data: pd.DataFrame | dict[str, pd.DataFrame],
    hedge_ratios: dict[str, float] | None = None,
    normalize: bool = True,
) -> pd.Series:
    """
    Construct the cointegrated spread from multiple price series.

    The spread is: log(P1) - β2*log(P2) - β3*log(P3) - ... - μ

    If no hedge ratios provided, will run Johansen test to estimate them.

    Args:
        data: DataFrame with log prices or dict of DataFrames
        hedge_ratios: Dict mapping column names to coefficients.
                     First column assumed to have coefficient 1.0.
        normalize: If True, demean the spread

    Returns:
        Series containing the spread values
    """
    # Convert dict to DataFrame
    if isinstance(data, dict):
        prices = {}
        for name, df in data.items():
            if "log_price" in df.columns:
                prices[name] = df["log_price"]
            else:
                prices[name] = np.log(df["close"])
        df_prices = pd.DataFrame(prices)
    else:
        df_prices = data

    # Estimate hedge ratios if not provided
    if hedge_ratios is None:
        result = johansen_test(df_prices)
        if result.n_cointegrating == 0:
            raise ValueError("No cointegrating relationship found. Cannot construct spread.")
        hedge_ratios = result.hedge_ratios

    # Construct spread
    spread = pd.Series(0.0, index=df_prices.index)
    for col, coef in hedge_ratios.items():
        if col in df_prices.columns:
            spread = spread + coef * df_prices[col]

    if normalize:
        spread = spread - spread.mean()

    spread.name = "spread"
    return spread


def estimate_half_life(spread: pd.Series) -> float:
    """
    Estimate the half-life of mean reversion using OLS.

    Uses the relationship: spread_t = ρ * spread_{t-1} + ε
    Half-life = -log(2) / log(ρ)

    Args:
        spread: Stationary spread series

    Returns:
        Estimated half-life in number of bars

    Raises:
        ValueError: If spread appears non-stationary (ρ >= 1)
    """
    spread_clean = spread.dropna()

    # Regress spread on lagged spread
    y = spread_clean.iloc[1:].values
    X = spread_clean.iloc[:-1].values.reshape(-1, 1)

    model = OLS(y, X).fit()
    rho = model.params[0]

    if rho >= 1:
        raise ValueError(
            f"Spread appears non-stationary (ρ={rho:.4f} >= 1). "
            "Half-life is undefined."
        )

    if rho <= 0:
        # Very fast mean reversion, essentially crosses every bar
        return 0.5

    half_life = -np.log(2) / np.log(rho)

    return float(half_life)
