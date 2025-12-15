"""Tests for statistical analysis modules."""

import numpy as np
import pandas as pd
import pytest

from fxarb.analysis.stationarity import adf_test, kpss_test
from fxarb.analysis.cointegration import estimate_half_life


class TestStationarityTests:
    """Tests for ADF and KPSS stationarity tests."""

    @pytest.fixture
    def stationary_series(self):
        """Create a stationary series (white noise)."""
        np.random.seed(42)
        return pd.Series(np.random.randn(500))

    @pytest.fixture
    def non_stationary_series(self):
        """Create a non-stationary series (random walk)."""
        np.random.seed(42)
        return pd.Series(np.random.randn(500).cumsum())

    def test_adf_stationary(self, stationary_series):
        """ADF should reject null (confirm stationarity) for stationary series."""
        result = adf_test(stationary_series)
        assert result.is_stationary == True  # noqa: E712

    def test_adf_non_stationary(self, non_stationary_series):
        """ADF should fail to reject null for non-stationary series."""
        result = adf_test(non_stationary_series)
        assert result.is_stationary == False  # noqa: E712

    def test_kpss_stationary(self, stationary_series):
        """KPSS should fail to reject null (confirm stationarity) for stationary series."""
        result = kpss_test(stationary_series)
        assert result.is_stationary == True  # noqa: E712


class TestHalfLife:
    """Tests for half-life estimation."""

    def test_half_life_mean_reverting(self):
        """Test half-life estimation for mean-reverting series."""
        np.random.seed(42)
        
        # Generate OU-like process
        n = 1000
        theta = 0.1  # Expected half-life = ln(2)/0.1 ≈ 6.9
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = x[t-1] - theta * x[t-1] + np.random.randn() * 0.1
        
        spread = pd.Series(x)
        half_life = estimate_half_life(spread)
        
        # Should be reasonably close to theoretical value
        expected = np.log(2) / theta
        assert abs(half_life - expected) < 5  # Within 5 bars

    def test_half_life_non_stationary_raises(self):
        """Test that non-stationary series raises ValueError."""
        np.random.seed(42)
        # Create a strongly non-stationary series (random walk with drift)
        # This ensures rho will be >= 1, triggering the ValueError
        n = 500
        random_walk = pd.Series(np.arange(n) + np.random.randn(n) * 0.1)
        
        with pytest.raises(ValueError, match="non-stationary"):
            estimate_half_life(random_walk)
