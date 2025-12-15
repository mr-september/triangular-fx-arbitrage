"""
Stationarity tests for time series analysis.

Implements ADF and KPSS tests for testing spread stationarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class StationarityResult:
    """Results from stationarity tests."""

    test_name: str
    statistic: float
    pvalue: float
    critical_values: dict[str, float]
    is_stationary: bool  # At 5% significance
    lags_used: int | None = None

    def __str__(self) -> str:
        status = "STATIONARY" if self.is_stationary else "NON-STATIONARY"
        lines = [
            f"{self.test_name} Test: {status}",
            f"  Statistic: {self.statistic:.6f}",
            f"  P-value: {self.pvalue:.6f}",
            "  Critical values:",
        ]
        for level, value in self.critical_values.items():
            lines.append(f"    {level}: {value:.6f}")
        if self.lags_used is not None:
            lines.append(f"  Lags used: {self.lags_used}")
        return "\n".join(lines)


def adf_test(
    series: pd.Series,
    regression: Literal["c", "ct", "ctt", "n"] = "c",
    maxlag: int | None = None,
    autolag: str = "AIC",
) -> StationarityResult:
    """
    Augmented Dickey-Fuller test for unit root.

    Null hypothesis: The series has a unit root (non-stationary).
    Alternative: The series is stationary.

    A low p-value (< 0.05) means we reject the null → series is stationary.

    Args:
        series: Time series to test
        regression: Regression type:
            'c': Constant only (default)
            'ct': Constant and trend
            'ctt': Constant, linear, and quadratic trend
            'n': No constant or trend
        maxlag: Maximum number of lags to use
        autolag: Method to select number of lags ('AIC', 'BIC', 't-stat', None)

    Returns:
        StationarityResult with test statistics and interpretation
    """
    series_clean = series.dropna()

    result = adfuller(
        series_clean,
        regression=regression,
        maxlag=maxlag,
        autolag=autolag,
    )

    statistic = result[0]
    pvalue = result[1]
    lags_used = result[2]
    critical_values = result[4]

    # Reject null (series is stationary) if statistic < critical value
    # or equivalently if p-value < significance level
    is_stationary = pvalue < 0.05

    return StationarityResult(
        test_name="ADF",
        statistic=statistic,
        pvalue=pvalue,
        critical_values=critical_values,
        is_stationary=is_stationary,
        lags_used=lags_used,
    )


def kpss_test(
    series: pd.Series,
    regression: Literal["c", "ct"] = "c",
    nlags: int | str = "auto",
) -> StationarityResult:
    """
    KPSS test for stationarity.

    Null hypothesis: The series is stationary.
    Alternative: The series has a unit root (non-stationary).

    A high p-value (> 0.05) means we fail to reject null → series is stationary.

    Note: ADF and KPSS have opposite null hypotheses. Using both provides
    more robust inference:
    - ADF rejects + KPSS fails to reject → Stationary
    - ADF fails to reject + KPSS rejects → Non-stationary
    - Both reject or both fail to reject → Inconclusive

    Args:
        series: Time series to test
        regression: 'c' for level stationarity, 'ct' for trend stationarity
        nlags: Number of lags or 'auto' for automatic selection

    Returns:
        StationarityResult with test statistics and interpretation
    """
    series_clean = series.dropna()

    result = kpss(series_clean, regression=regression, nlags=nlags)

    statistic = result[0]
    pvalue = result[1]
    lags_used = result[2]
    critical_values = result[3]

    # Fail to reject null (series is stationary) if p-value > significance level
    is_stationary = pvalue > 0.05

    return StationarityResult(
        test_name="KPSS",
        statistic=statistic,
        pvalue=pvalue,
        critical_values=critical_values,
        is_stationary=is_stationary,
        lags_used=lags_used,
    )


def test_stationarity(
    series: pd.Series,
    verbose: bool = True,
) -> tuple[StationarityResult, StationarityResult, str]:
    """
    Run both ADF and KPSS tests for robust stationarity inference.

    Args:
        series: Time series to test
        verbose: Whether to print results

    Returns:
        Tuple of (adf_result, kpss_result, conclusion)
        conclusion is one of: 'stationary', 'non_stationary', 'inconclusive'
    """
    adf_result = adf_test(series)
    kpss_result = kpss_test(series)

    # Interpret combined results
    if adf_result.is_stationary and kpss_result.is_stationary:
        conclusion = "stationary"
    elif not adf_result.is_stationary and not kpss_result.is_stationary:
        conclusion = "non_stationary"
    else:
        conclusion = "inconclusive"

    if verbose:
        print(adf_result)
        print()
        print(kpss_result)
        print()
        print(f"Combined conclusion: {conclusion.upper()}")

    return adf_result, kpss_result, conclusion
