"""
GARCH volatility modeling for dynamic threshold adjustment.

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models
capture time-varying volatility, allowing for adaptive trading thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model


@dataclass
class GARCHResult:
    """Results from GARCH model fitting."""

    omega: float  # Constant term
    alpha: float  # ARCH term (lagged squared residuals)
    beta: float  # GARCH term (lagged variance)
    conditional_volatility: pd.Series  # Fitted volatility series
    forecasted_volatility: float  # One-step-ahead forecast

    def __str__(self) -> str:
        persistence = self.alpha + self.beta
        return (
            f"GARCH(1,1) Results:\n"
            f"  ω (omega): {self.omega:.8f}\n"
            f"  α (alpha): {self.alpha:.6f}\n"
            f"  β (beta): {self.beta:.6f}\n"
            f"  Persistence (α+β): {persistence:.6f}\n"
            f"  Forecasted volatility: {self.forecasted_volatility:.6f}"
        )

    @property
    def persistence(self) -> float:
        """Volatility persistence. Close to 1 means shocks have long-lasting effects."""
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        """Long-run (unconditional) variance."""
        if self.persistence >= 1:
            return float("inf")
        return self.omega / (1 - self.persistence)


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    mean: str = "Zero",
    dist: str = "normal",
    rescale: bool = True,
) -> GARCHResult:
    """
    Fit a GARCH model to return series.

    Args:
        returns: Return series (typically log returns of the spread)
        p: Order of GARCH terms (lagged variance)
        q: Order of ARCH terms (lagged squared residuals)
        mean: Mean model ('Zero', 'Constant', 'AR')
        dist: Error distribution ('normal', 't', 'skewt')
        rescale: Whether to rescale data for numerical stability

    Returns:
        GARCHResult with fitted parameters and volatility forecasts
    """
    returns_clean = returns.dropna()

    if len(returns_clean) < 100:
        raise ValueError("Need at least 100 observations for GARCH estimation")

    # Scale returns for numerical stability (arch library works better with scaled data)
    if rescale:
        scale = 100  # Convert to percentage returns
        returns_scaled = returns_clean * scale
    else:
        returns_scaled = returns_clean
        scale = 1

    # Fit GARCH model
    model = arch_model(
        returns_scaled,
        mean=mean,
        vol="GARCH",
        p=p,
        q=q,
        dist=dist,
    )

    result = model.fit(disp="off", show_warning=False)

    # Extract parameters (accounting for scale)
    omega = result.params.get("omega", 0) / (scale**2)
    alpha = result.params.get("alpha[1]", 0)
    beta = result.params.get("beta[1]", 0)

    # Conditional volatility (rescale back)
    conditional_vol = result.conditional_volatility / scale
    conditional_vol.name = "conditional_volatility"

    # Forecast next-period volatility
    forecast = result.forecast(horizon=1)
    forecasted_var = forecast.variance.iloc[-1, 0] / (scale**2)
    forecasted_vol = np.sqrt(forecasted_var)

    return GARCHResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        conditional_volatility=conditional_vol,
        forecasted_volatility=forecasted_vol,
    )


def forecast_volatility(
    returns: pd.Series,
    horizon: int = 1,
    window: int | None = None,
) -> pd.Series:
    """
    Generate rolling volatility forecasts using GARCH.

    For each point in time, fits GARCH on the trailing window and forecasts
    the next period's volatility.

    Args:
        returns: Return series
        horizon: Forecast horizon (1 = next period)
        window: Rolling window size (None = use all history)

    Returns:
        Series of volatility forecasts
    """
    returns_clean = returns.dropna()

    if window is None:
        window = min(500, len(returns_clean) // 2)

    forecasts = pd.Series(index=returns_clean.index, dtype=float)
    forecasts[:] = np.nan

    for i in range(window, len(returns_clean)):
        try:
            # Use trailing window
            window_returns = returns_clean.iloc[i - window : i]

            # Fit GARCH and forecast
            result = fit_garch(window_returns, rescale=True)
            forecasts.iloc[i] = result.forecasted_volatility

        except Exception:
            # Use simple historical volatility as fallback
            forecasts.iloc[i] = window_returns.std()

    forecasts.name = "volatility_forecast"
    return forecasts


def compute_dynamic_threshold(
    volatility_forecast: pd.Series,
    base_threshold: float = 2.0,
    vol_multiplier: float = 1.0,
    min_threshold: float = 1.0,
    max_threshold: float = 4.0,
) -> pd.Series:
    """
    Compute dynamic Z-score thresholds based on volatility forecasts.

    Higher volatility → higher thresholds (more conservative)
    Lower volatility → lower thresholds (more sensitive)

    Args:
        volatility_forecast: Series of volatility forecasts
        base_threshold: Threshold when volatility is at its mean
        vol_multiplier: How much to adjust threshold per unit of volatility
        min_threshold: Minimum threshold (floor)
        max_threshold: Maximum threshold (ceiling)

    Returns:
        Series of dynamic thresholds
    """
    # Normalize volatility relative to its rolling median
    vol_median = volatility_forecast.rolling(window=100, min_periods=20).median()
    vol_ratio = volatility_forecast / vol_median

    # Adjust threshold based on volatility ratio
    threshold = base_threshold * (1 + vol_multiplier * (vol_ratio - 1))

    # Apply bounds
    threshold = threshold.clip(lower=min_threshold, upper=max_threshold)
    threshold.name = "dynamic_threshold"

    return threshold
