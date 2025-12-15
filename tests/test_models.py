"""Tests for trading models."""

import numpy as np
import pandas as pd
import pytest

from fxarb.models.zscore import compute_zscore, generate_signals, ZScoreStrategy


class TestZScore:
    """Tests for Z-score computation."""

    @pytest.fixture
    def sample_spread(self):
        """Create sample spread data."""
        np.random.seed(42)
        idx = pd.date_range("2022-01-01", periods=500, freq="5min")
        return pd.Series(np.random.randn(500), index=idx, name="spread")

    def test_compute_zscore_mean_zero(self, sample_spread):
        """Z-score should have approximately zero mean."""
        zscore = compute_zscore(sample_spread, lookback=50)
        # Skip NaN values at the beginning
        zscore_clean = zscore.dropna()
        assert abs(zscore_clean.mean()) < 0.5

    def test_compute_zscore_std_one(self, sample_spread):
        """Z-score should have approximately unit std."""
        zscore = compute_zscore(sample_spread, lookback=50)
        zscore_clean = zscore.dropna()
        assert 0.5 < zscore_clean.std() < 1.5

    def test_generate_signals_long_entry(self):
        """Test that long entry signal is generated when z-score is very negative."""
        # Create spread that becomes very negative
        idx = pd.date_range("2022-01-01", periods=200, freq="5min")
        spread = pd.Series(np.zeros(200), index=idx)
        spread.iloc[150:] = -3  # Sudden drop
        
        zscore = compute_zscore(spread, lookback=50)
        signals = generate_signals(zscore, entry_threshold=2.0)
        
        # Should have at least one long entry
        assert (signals["entry"] == 1).any()


class TestZScoreStrategy:
    """Tests for ZScoreStrategy configuration."""

    def test_strategy_string_representation(self):
        """Test strategy string representation."""
        strategy = ZScoreStrategy(lookback=100, entry_threshold=2.5)
        assert "lookback=100" in str(strategy)
        assert "2.5" in str(strategy)
