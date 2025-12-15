"""
Ornstein-Uhlenbeck process modeling for mean-reverting spreads.

The OU process is the canonical model for mean reversion:
    dX_t = θ(μ - X_t)dt + σ dW_t

Parameters:
    θ (theta): Speed of mean reversion
    μ (mu): Long-term mean
    σ (sigma): Volatility
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class OUProcess:
    """
    Ornstein-Uhlenbeck process parameters.

    The half-life of mean reversion is: t_half = ln(2) / θ
    """

    theta: float  # Speed of mean reversion
    mu: float  # Long-term mean
    sigma: float  # Volatility

    def __str__(self) -> str:
        return (
            f"OU Process:\n"
            f"  θ (mean reversion speed): {self.theta:.6f}\n"
            f"  μ (long-term mean): {self.mu:.6f}\n"
            f"  σ (volatility): {self.sigma:.6f}\n"
            f"  Half-life: {self.half_life:.2f} bars"
        )

    @property
    def half_life(self) -> float:
        """Half-life of mean reversion in number of bars."""
        if self.theta <= 0:
            return float("inf")
        return np.log(2) / self.theta

    def expected_time_to_mean(self, current_value: float, tolerance: float = 0.1) -> float:
        """
        Estimate expected time to return within tolerance of the mean.

        Args:
            current_value: Current spread value
            tolerance: How close to mean is considered "at mean"

        Returns:
            Expected time in bars
        """
        distance = abs(current_value - self.mu)
        if distance < tolerance:
            return 0.0

        # From OU dynamics: E[X_t] = μ + (X_0 - μ)e^(-θt)
        # Solve for t when |E[X_t] - μ| = tolerance
        if self.theta <= 0:
            return float("inf")

        t = -np.log(tolerance / distance) / self.theta
        return max(0, t)

    def simulate(
        self,
        n_steps: int,
        dt: float = 1.0,
        x0: float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Simulate the OU process using Euler-Maruyama discretization.

        Args:
            n_steps: Number of time steps to simulate
            dt: Time step size
            x0: Initial value (defaults to mu)
            seed: Random seed for reproducibility

        Returns:
            Array of simulated values
        """
        if seed is not None:
            np.random.seed(seed)

        if x0 is None:
            x0 = self.mu

        x = np.zeros(n_steps)
        x[0] = x0

        # Euler-Maruyama scheme
        sqrt_dt = np.sqrt(dt)
        for t in range(1, n_steps):
            dx = self.theta * (self.mu - x[t - 1]) * dt + self.sigma * sqrt_dt * np.random.randn()
            x[t] = x[t - 1] + dx

        return x


def fit_ou_mle(
    series: pd.Series,
    dt: float = 1.0,
) -> OUProcess:
    """
    Fit Ornstein-Uhlenbeck parameters using Maximum Likelihood Estimation.

    Uses the exact discrete-time transition density of the OU process.

    Args:
        series: Time series of the spread
        dt: Time step between observations (default 1 for bars)

    Returns:
        Fitted OUProcess parameters
    """
    x = series.dropna().values
    n = len(x)

    if n < 10:
        raise ValueError("Need at least 10 observations to fit OU process")

    # Method 1: OLS-based estimation (fast and robust)
    # From AR(1) representation: X_t = c + ρ*X_{t-1} + ε
    # where ρ = e^(-θ*dt) and c = μ(1 - ρ)

    x_lag = x[:-1]
    x_curr = x[1:]

    # OLS regression
    n_obs = len(x_lag)
    x_mean = np.mean(x_lag)
    y_mean = np.mean(x_curr)

    # Slope and intercept
    cov = np.sum((x_lag - x_mean) * (x_curr - y_mean)) / n_obs
    var = np.sum((x_lag - x_mean) ** 2) / n_obs

    if var < 1e-10:
        raise ValueError("Series has near-zero variance")

    rho = cov / var
    intercept = y_mean - rho * x_mean

    # Handle edge cases
    if rho >= 1:
        # Non-mean-reverting, return degenerate case
        return OUProcess(theta=0.0, mu=np.mean(x), sigma=np.std(x))

    if rho <= 0:
        # Very fast mean reversion
        rho = 0.01

    # Convert to OU parameters
    theta = -np.log(rho) / dt
    mu = intercept / (1 - rho)

    # Estimate sigma from residuals
    residuals = x_curr - (intercept + rho * x_lag)
    var_residuals = np.var(residuals)

    # From OU: Var(ε) = σ²(1 - e^(-2θ*dt)) / (2θ)
    if theta > 0:
        sigma_sq = var_residuals * 2 * theta / (1 - np.exp(-2 * theta * dt))
        sigma = np.sqrt(max(sigma_sq, 1e-10))
    else:
        sigma = np.std(x)

    return OUProcess(theta=theta, mu=mu, sigma=sigma)


def fit_ou_mle_optimized(
    series: pd.Series,
    dt: float = 1.0,
) -> OUProcess:
    """
    Fit OU parameters using numerical MLE optimization.

    More accurate than the OLS method but slower.

    Args:
        series: Time series of the spread
        dt: Time step between observations

    Returns:
        Fitted OUProcess parameters
    """
    x = series.dropna().values
    n = len(x)

    # Get initial estimates from OLS method
    initial = fit_ou_mle(series, dt)

    def neg_log_likelihood(params: np.ndarray) -> float:
        """Negative log-likelihood of OU process."""
        theta, mu, sigma = params

        if theta <= 0 or sigma <= 0:
            return 1e10

        # Transition distribution parameters
        rho = np.exp(-theta * dt)
        var = sigma**2 * (1 - rho**2) / (2 * theta)

        if var <= 0:
            return 1e10

        # Log-likelihood
        x_pred = mu + rho * (x[:-1] - mu)
        residuals = x[1:] - x_pred

        ll = -0.5 * n * np.log(2 * np.pi * var) - 0.5 * np.sum(residuals**2) / var

        return -ll

    # Optimize
    x0 = np.array([initial.theta, initial.mu, initial.sigma])
    bounds = [(1e-6, 10), (None, None), (1e-6, None)]

    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
    )

    if result.success:
        theta, mu, sigma = result.x
        return OUProcess(theta=theta, mu=mu, sigma=sigma)
    else:
        # Fall back to OLS estimates
        return initial
