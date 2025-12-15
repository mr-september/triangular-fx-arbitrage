"""
Kalman filter for dynamic hedge ratio estimation.

The Kalman filter treats hedge ratios as hidden states that evolve over time,
providing adaptive estimates that track changing market conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pykalman import KalmanFilter


@dataclass
class KalmanHedgeRatio:
    """
    Kalman filter for estimating time-varying hedge ratios.

    State-space model:
        β_t = β_{t-1} + w_t  (state transition: random walk)
        y_t = X_t * β_t + v_t  (observation equation)

    Where:
        β: Hedge ratios (hidden states)
        y: Dependent variable (e.g., log price of pair 1)
        X: Independent variables (e.g., log prices of pairs 2, 3)
    """

    n_states: int  # Number of hedge ratios to estimate
    delta: float = 1e-5  # State transition covariance multiplier
    observation_cov: float = 1.0  # Observation noise variance

    # Fitted attributes (set after fit())
    state_means: np.ndarray | None = None  # Shape: (n_obs, n_states)
    state_covs: np.ndarray | None = None  # Shape: (n_obs, n_states, n_states)

    def fit(
        self,
        y: pd.Series | np.ndarray,
        X: pd.DataFrame | np.ndarray,
    ) -> "KalmanHedgeRatio":
        """
        Fit the Kalman filter to estimate time-varying hedge ratios.

        Args:
            y: Dependent variable (n_obs,)
            X: Independent variables (n_obs, n_features)

        Returns:
            Self with fitted state estimates
        """
        y = np.asarray(y).flatten()
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_obs, n_features = X.shape
        self.n_states = n_features

        # Set up Kalman filter
        # Transition matrix (identity - random walk)
        transition_matrix = np.eye(n_features)

        # Transition covariance (how much states can change each step)
        transition_covariance = self.delta * np.eye(n_features)

        # Observation matrix changes each time step (it's X_t)
        # We'll handle this manually in the filtering

        # Initialize filter
        kf = KalmanFilter(
            transition_matrices=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=np.array([[self.observation_cov]]),
            initial_state_mean=np.zeros(n_features),
            initial_state_covariance=np.eye(n_features),
        )

        # Run filter with time-varying observation matrices
        state_means = np.zeros((n_obs, n_features))
        state_covs = np.zeros((n_obs, n_features, n_features))

        # Initial state
        current_mean = np.zeros(n_features)
        current_cov = np.eye(n_features)

        for t in range(n_obs):
            # Observation matrix for this time step
            obs_matrix = X[t : t + 1, :]

            # Predict step
            pred_mean = transition_matrix @ current_mean
            pred_cov = transition_matrix @ current_cov @ transition_matrix.T + transition_covariance

            # Update step (if we have an observation)
            if not np.isnan(y[t]):
                # Kalman gain
                S = obs_matrix @ pred_cov @ obs_matrix.T + self.observation_cov
                K = pred_cov @ obs_matrix.T / S

                # Update
                innovation = y[t] - obs_matrix @ pred_mean
                current_mean = pred_mean + K.flatten() * innovation
                current_cov = (np.eye(n_features) - K @ obs_matrix) @ pred_cov
            else:
                current_mean = pred_mean
                current_cov = pred_cov

            state_means[t] = current_mean
            state_covs[t] = current_cov

        self.state_means = state_means
        self.state_covs = state_covs

        return self

    def get_hedge_ratios(self, index: pd.Index | None = None) -> pd.DataFrame:
        """
        Get the estimated hedge ratios as a DataFrame.

        Args:
            index: Optional index for the DataFrame

        Returns:
            DataFrame with hedge ratio columns
        """
        if self.state_means is None:
            raise ValueError("Must call fit() before getting hedge ratios")

        cols = [f"beta_{i}" for i in range(self.n_states)]
        df = pd.DataFrame(self.state_means, columns=cols, index=index)

        return df

    def get_spread(
        self,
        y: pd.Series | np.ndarray,
        X: pd.DataFrame | np.ndarray,
    ) -> pd.Series:
        """
        Compute the spread using time-varying hedge ratios.

        spread_t = y_t - X_t @ β_t

        Args:
            y: Dependent variable
            X: Independent variables

        Returns:
            Spread series
        """
        if self.state_means is None:
            self.fit(y, X)

        y = np.asarray(y).flatten()
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Compute spread: y - X @ beta
        spread = y - np.sum(X * self.state_means, axis=1)

        if isinstance(y, pd.Series):
            return pd.Series(spread, index=y.index, name="kalman_spread")
        return pd.Series(spread, name="kalman_spread")


def compare_static_vs_dynamic(
    data: dict[str, pd.DataFrame],
    dependent_pair: str,
) -> pd.DataFrame:
    """
    Compare static OLS hedge ratios vs dynamic Kalman hedge ratios.

    Args:
        data: Dict mapping pair codes to DataFrames with log_price column
        dependent_pair: Which pair is the dependent variable

    Returns:
        DataFrame with both spreads for comparison
    """
    pairs = list(data.keys())
    if dependent_pair not in pairs:
        raise ValueError(f"{dependent_pair} not in data")

    # Extract log prices
    y = data[dependent_pair]["log_price"]
    X_pairs = [p for p in pairs if p != dependent_pair]
    X = pd.DataFrame({p: data[p]["log_price"] for p in X_pairs})

    # Align indices
    common_idx = y.index.intersection(X.index)
    y = y.loc[common_idx]
    X = X.loc[common_idx]

    # Static hedge ratios (OLS)
    from statsmodels.regression.linear_model import OLS
    import statsmodels.api as sm

    X_with_const = sm.add_constant(X)
    ols_result = OLS(y, X_with_const).fit()
    static_spread = ols_result.resid

    # Dynamic hedge ratios (Kalman)
    kf = KalmanHedgeRatio(n_states=len(X_pairs))
    kf.fit(y.values, X.values)
    dynamic_spread = kf.get_spread(y, X)

    result = pd.DataFrame(
        {
            "static_spread": static_spread,
            "dynamic_spread": dynamic_spread,
        },
        index=common_idx,
    )

    # Add hedge ratio columns
    hedge_ratios = kf.get_hedge_ratios(index=common_idx)
    for col in hedge_ratios.columns:
        result[f"kalman_{col}"] = hedge_ratios[col]

    return result
