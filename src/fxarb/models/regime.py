"""
Regime detection for filtering mean-reversion trades.

Mean-reversion strategies work best in range-bound markets. This module
provides tools to detect market regimes and filter trades accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HurstRegimeFilter:
    """
    Regime filter based on the Hurst exponent.

    Hurst exponent interpretation:
    - H < 0.5: Mean-reverting (good for our strategy)
    - H ≈ 0.5: Random walk (neutral)
    - H > 0.5: Trending (bad for mean-reversion)

    We only allow trades when H is significantly below 0.5.
    """

    window: int = 100  # Window for Hurst calculation
    threshold: float = 0.45  # Maximum H to allow trading
    min_periods: int = 50  # Minimum observations before filtering

    def __str__(self) -> str:
        return (
            f"HurstRegimeFilter(window={self.window}, "
            f"threshold={self.threshold})"
        )


def compute_hurst_exponent(
    series: pd.Series,
    max_lag: int | None = None,
    method: str = "rs",
) -> float:
    """
    Compute the Hurst exponent using the rescaled range (R/S) method.

    The Hurst exponent measures the long-term memory of a time series:
    - H < 0.5: Anti-persistent (mean-reverting)
    - H = 0.5: Random walk (no memory)
    - H > 0.5: Persistent (trending)

    Args:
        series: Time series to analyze
        max_lag: Maximum lag for R/S calculation (default: len/4)
        method: 'rs' for rescaled range

    Returns:
        Estimated Hurst exponent (0 to 1)
    """
    ts = series.dropna().values
    n = len(ts)

    if n < 20:
        return 0.5  # Not enough data, assume random walk

    if max_lag is None:
        max_lag = min(n // 4, 100)

    # Range of lags to consider
    lags = range(10, max_lag + 1)
    rs_values = []

    for lag in lags:
        # Split into subseries of length 'lag'
        n_subseries = n // lag
        rs_list = []

        for i in range(n_subseries):
            subseries = ts[i * lag : (i + 1) * lag]

            # Calculate R/S for this subseries
            mean = np.mean(subseries)
            std = np.std(subseries, ddof=1)

            if std < 1e-10:
                continue

            # Cumulative deviation from mean
            cumdev = np.cumsum(subseries - mean)

            # Range
            R = np.max(cumdev) - np.min(cumdev)

            # Rescaled range
            rs = R / std
            rs_list.append(rs)

        if rs_list:
            rs_values.append((lag, np.mean(rs_list)))

    if len(rs_values) < 3:
        return 0.5

    # Fit log(R/S) = H * log(n) + c
    log_lags = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])

    # Simple linear regression
    n_points = len(log_lags)
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_rs)
    sum_xy = np.sum(log_lags * log_rs)
    sum_xx = np.sum(log_lags**2)

    H = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_xx - sum_x**2)

    # Clamp to valid range
    return float(np.clip(H, 0.0, 1.0))


def rolling_hurst(
    series: pd.Series,
    window: int = 100,
    min_periods: int = 50,
) -> pd.Series:
    """
    Compute rolling Hurst exponent.

    Args:
        series: Time series to analyze
        window: Rolling window size
        min_periods: Minimum observations required

    Returns:
        Series of Hurst exponent values
    """
    hurst_values = pd.Series(index=series.index, dtype=float)
    hurst_values[:] = np.nan

    for i in range(min_periods, len(series)):
        start_idx = max(0, i - window)
        window_data = series.iloc[start_idx:i]

        if len(window_data) >= min_periods:
            hurst_values.iloc[i] = compute_hurst_exponent(window_data)

    hurst_values.name = "hurst_exponent"
    return hurst_values


def apply_regime_filter(
    signals: pd.DataFrame,
    hurst: pd.Series,
    threshold: float = 0.45,
) -> pd.DataFrame:
    """
    Apply Hurst-based regime filter to trading signals.

    Sets position to 0 when Hurst exponent indicates trending regime.

    Args:
        signals: DataFrame with 'position' column
        hurst: Series of Hurst exponent values
        threshold: Maximum Hurst to allow trading

    Returns:
        DataFrame with filtered positions
    """
    signals = signals.copy()

    # Align indices
    common_idx = signals.index.intersection(hurst.index)
    hurst_aligned = hurst.loc[common_idx]

    # Create regime mask
    is_mean_reverting = hurst_aligned < threshold

    # Filter positions
    original_position = signals.loc[common_idx, "position"].copy()
    signals.loc[common_idx, "position_unfiltered"] = original_position
    signals.loc[common_idx, "position"] = original_position * is_mean_reverting.astype(int)
    signals.loc[common_idx, "hurst"] = hurst_aligned
    signals.loc[common_idx, "regime_filter"] = is_mean_reverting.astype(int)

    return signals


# Optional: HMM-based regime detection
try:
    from hmmlearn import hmm

    def fit_regime_hmm(
        returns: pd.Series,
        n_regimes: int = 2,
        n_iter: int = 100,
    ) -> tuple[np.ndarray, pd.Series]:
        """
        Fit Hidden Markov Model to detect market regimes.

        Args:
            returns: Return series
            n_regimes: Number of hidden states (regimes)
            n_iter: Number of EM iterations

        Returns:
            Tuple of (transition matrix, regime predictions)
        """
        returns_clean = returns.dropna().values.reshape(-1, 1)

        # Fit Gaussian HMM
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
        )
        model.fit(returns_clean)

        # Predict regimes
        regimes = model.predict(returns_clean)

        # Identify which regime is "low volatility" (mean-reverting favorable)
        regime_vols = [np.sqrt(model.covars_[i][0, 0]) for i in range(n_regimes)]
        low_vol_regime = np.argmin(regime_vols)

        # Create series
        regime_series = pd.Series(
            regimes == low_vol_regime,
            index=returns.dropna().index,
            name="favorable_regime",
        ).astype(int)

        return model.transmat_, regime_series

except ImportError:
    # HMM not available
    def fit_regime_hmm(*args, **kwargs):
        raise ImportError("hmmlearn not installed. Install with: pip install hmmlearn")
