"""Tests for data loading and processing."""

import numpy as np
import pandas as pd
import pytest

from fxarb.data.loader import parse_histdata_timestamp, resample_ohlc
from fxarb.data.cleaning import detect_bad_ticks, clean_ohlc
from fxarb.data.features import add_session_features, compute_log_prices


class TestTimestampParsing:
    """Tests for timestamp parsing."""

    def test_parse_histdata_timestamp(self):
        """Test parsing of HistData timestamp format."""
        from datetime import datetime
        
        result = parse_histdata_timestamp("20220101 093000")
        assert result == datetime(2022, 1, 1, 9, 30, 0)

    def test_parse_timestamp_with_leading_space(self):
        """Test parsing handles leading/trailing whitespace."""
        from datetime import datetime
        
        result = parse_histdata_timestamp(" 20220601 143015 ")
        assert result == datetime(2022, 6, 1, 14, 30, 15)


class TestResample:
    """Tests for OHLC resampling."""

    @pytest.fixture
    def sample_1min_data(self):
        """Create sample 1-minute OHLC data."""
        idx = pd.date_range("2022-01-01 09:00", periods=60, freq="1min")
        return pd.DataFrame(
            {
                "open": np.random.randn(60).cumsum() + 100,
                "high": np.random.randn(60).cumsum() + 101,
                "low": np.random.randn(60).cumsum() + 99,
                "close": np.random.randn(60).cumsum() + 100,
                "volume": np.random.randint(100, 1000, 60),
            },
            index=idx,
        )

    def test_resample_5min(self, sample_1min_data):
        """Test resampling from 1-min to 5-min."""
        resampled = resample_ohlc(sample_1min_data, timeframe="5min")
        
        # Should have 12 bars (60 / 5)
        assert len(resampled) == 12

    def test_resample_uses_first_open(self, sample_1min_data):
        """Test that resampling uses first open price."""
        resampled = resample_ohlc(sample_1min_data, timeframe="5min")
        
        # First 5-min bar's open should equal first 1-min bar's open
        assert resampled["open"].iloc[0] == sample_1min_data["open"].iloc[0]


class TestBadTickDetection:
    """Tests for bad tick detection."""

    @pytest.fixture
    def data_with_bad_tick(self):
        """Create OHLC data with a bad tick."""
        idx = pd.date_range("2022-01-01", periods=200, freq="5min")
        np.random.seed(42)
        
        # Normal data
        base = np.random.randn(200).cumsum() + 100
        df = pd.DataFrame(
            {
                "open": base,
                "high": base + abs(np.random.randn(200) * 0.01),
                "low": base - abs(np.random.randn(200) * 0.01),
                "close": base + np.random.randn(200) * 0.01,
            },
            index=idx,
        )
        
        # Insert a bad tick (huge spike)
        df.loc[df.index[100], "close"] = df["close"].iloc[99] * 2
        
        return df

    def test_detect_bad_ticks(self, data_with_bad_tick):
        """Test that bad tick is detected."""
        bad_mask = detect_bad_ticks(data_with_bad_tick, zscore_threshold=5.0)
        
        # Should detect the spike at index 100
        assert bad_mask.iloc[100] == True  # noqa: E712


class TestSessionFeatures:
    """Tests for session feature engineering."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data spanning multiple sessions."""
        idx = pd.date_range("2022-01-03 00:00", periods=288, freq="5min")  # Monday
        return pd.DataFrame(
            {
                "open": np.ones(288),
                "high": np.ones(288),
                "low": np.ones(288),
                "close": np.ones(288),
            },
            index=idx,
        )

    def test_add_session_features(self, sample_data):
        """Test that session features are added correctly."""
        result = add_session_features(sample_data)
        
        assert "session_asia" in result.columns
        assert "session_europe" in result.columns
        assert "session_north_america" in result.columns
        assert "overlap_europe_na" in result.columns

    def test_asia_session_hours(self, sample_data):
        """Test Asia session covers correct hours (00:00-09:00 UTC)."""
        result = add_session_features(sample_data)
        
        # 00:00-09:00 should be Asia
        asia_hours = result[result["session_asia"] == 1].index.hour.unique()
        assert all(h < 9 for h in asia_hours)


class TestLogPrices:
    """Tests for log price computation."""

    def test_compute_log_prices(self):
        """Test log price calculation."""
        idx = pd.date_range("2022-01-01", periods=10, freq="1h")
        df = pd.DataFrame(
            {
                "open": [1.0] * 10,
                "high": [1.1] * 10,
                "low": [0.9] * 10,
                "close": [1.05] * 10,
            },
            index=idx,
        )
        
        result = compute_log_prices(df)
        
        assert "log_price" in result.columns
        assert np.isclose(result["log_price"].iloc[0], np.log(1.05))
