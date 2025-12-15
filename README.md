# Triangular FX Statistical Arbitrage

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A quantit finance project demonstrating **statistical arbitrage** on FX currency triplets using cointegration analysis and mean-reversion strategies.

## Overview

This project implements a complete statistical arbitrage framework for trading cointegrated FX triplets. Unlike pure triangular arbitrage (which is arbitraged away in milliseconds), this approach exploits the **statistical** tendency of related currency pairs to revert to their long-term equilibrium relationship.

### Key Features

- **Cointegration Analysis**: Johansen test for identifying stable equilibrium relationships between currency triplets
- **Mean-Reversion Modeling**: Z-score, Ornstein-Uhlenbeck process, and GARCH volatility models
- **Adaptive Strategies**: Kalman filter for dynamic hedge ratios, Hurst exponent for regime detection
- **Robust Backtesting**: Walk-forward optimization to prevent overfitting
- **Professional Visualization**: Publication-quality charts for spreads, signals, and performance

## The Strategy

### Theoretical Foundation

For a currency triplet like EUR/USD, GBP/USD, and EUR/GBP, we expect:

```
EUR/GBP ≈ EUR/USD ÷ GBP/USD
```

Or in log space:

```
log(EUR/GBP) ≈ log(EUR/USD) - log(GBP/USD)
```

When these pairs are **cointegrated**, deviations from this relationship are temporary and tend to mean-revert. The strategy:

1. **Detects** when the spread between actual and theoretical prices is unusually large
2. **Enters** a position betting on mean reversion
3. **Exits** when the spread returns to normal

### Why It Works (Sometimes)

- **Persistent Equilibrium**: The economic forces that create these relationships (e.g., trade flows, interest rate differentials) are slow-moving
- **Mean Reversion**: Statistical forces push prices back toward equilibrium
- **Multiple Instruments**: Trading three legs provides natural hedging

### Why It's Risky

- **Regime Changes**: The cointegrating relationship can break down during crises
- **Transaction Costs**: Frequent trading erodes profits (not modeled in this backtest)
- **Execution Risk**: Slippage in fast markets (not modeled)
- **Model Risk**: Parameters estimated from historical data may not hold

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/triangular-fx-arbitrage.git
cd triangular-fx-arbitrage

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,notebooks]"
```

## Data

This project uses 1-minute OHLC bar data from [HistData.com](https://www.histdata.com/), a free source of historical forex data.

### Obtaining Data

See `scripts/download_histdata.py` for the data acquisition pipeline:

```bash
python scripts/download_histdata.py --pair eurusd gbpusd eurgbp --start-year 2020
```

### Data Format

Data is stored as compressed pickle files:
```
data/raw/
├── eurusd/eurusd.pkl
├── gbpusd/gbpusd.pkl
└── eurgbp/eurgbp.pkl
```

Each file contains:
- `datetime` index (UTC)
- `open`, `high`, `low`, `close` prices
- `volume` (tick count)

## Quick Start

```python
from fxarb.data import load_triplet, resample_ohlc
from fxarb.analysis import johansen_test, construct_spread
from fxarb.models import compute_zscore, generate_signals
from fxarb.backtest import run_backtest

# Load and prepare data
data = load_triplet("eurusd", "gbpusd", "eurgbp", timeframe="5min")

# Test for cointegration
coint_result = johansen_test(data)
print(coint_result)

# Construct the spread
spread = construct_spread(data, hedge_ratios=coint_result.hedge_ratios)

# Generate trading signals
zscore = compute_zscore(spread, lookback=100)
signals = generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.0)

# Run backtest
result = run_backtest(spread, signals)
print(result)
```

## Project Structure

```
triangular-fx-arbitrage/
├── src/fxarb/
│   ├── data/           # Data loading and preprocessing
│   │   ├── loader.py   # Load from pickle/CSV, resample timeframes
│   │   ├── cleaning.py # Bad tick detection, gap filling
│   │   └── features.py # Session indicators, log prices
│   │
│   ├── analysis/       # Statistical analysis
│   │   ├── cointegration.py  # Johansen test, spread construction
│   │   ├── stationarity.py   # ADF, KPSS tests
│   │   ├── triplets.py       # Triplet discovery and scoring
│   │   └── strategy.py       # Strategy logic and parameters
│   │
│   ├── models/         # Trading models
│   │   ├── zscore.py      # Z-score mean reversion
│   │   ├── ou_process.py  # Ornstein-Uhlenbeck estimation
│   │   ├── garch.py       # GARCH volatility
│   │   ├── kalman.py      # Dynamic hedge ratios
│   │   └── regime.py      # Hurst exponent, HMM
│   │
│   ├── backtest/       # Backtesting framework
│   │   ├── engine.py   # Vectorized backtest engine
│   │   ├── metrics.py  # Sharpe, Sortino, drawdown
│   │   ├── wfo.py      # Walk-forward optimization
│   │   └── kelly.py    # Position sizing
│   │
│   └── visualization/  # Plotting utilities
│
├── notebooks/          # Jupyter notebooks
│   └── research_report_v2.ipynb # Interactive research report
├── reports/            # Generated figures and execution summaries
├── scripts/            # Automation scripts
│   ├── download_histdata.py  # Fetch data from HistData.com
│   ├── merge_histdata.py     # Merge monthly files
│   ├── generate_report.py    # Headless report execution
│   └── generate_report_wfo.py # Walk-forward optimization runner
└── tests/              # Unit tests
```

## Key Concepts

### Johansen Cointegration Test

The Johansen test determines if multiple non-stationary time series share a long-term equilibrium. For a triplet, we look for at least one cointegrating vector.

```python
from fxarb.analysis import johansen_test

result = johansen_test(data)
print(f"Cointegrating relationships: {result.n_cointegrating}")
print(f"Hedge ratios: {result.hedge_ratios}")
```

### Ornstein-Uhlenbeck Process

The OU process models mean-reverting dynamics:

```
dX = θ(μ - X)dt + σdW
```

The **half-life** of mean reversion, `ln(2)/θ`, tells us how long positions typically last.

```python
from fxarb.models import fit_ou_mle

ou = fit_ou_mle(spread)
print(f"Half-life: {ou.half_life:.1f} bars")
```

### Walk-Forward Optimization

To avoid overfitting, we use walk-forward optimization:
1. Optimize on training window
2. Test on out-of-sample window
3. Roll forward and repeat

```python
from fxarb.backtest import WalkForwardOptimizer

wfo = WalkForwardOptimizer(
    train_size=10000,  # ~35 days of 5-min bars
    test_size=2000,    # ~7 days
)
result = wfo.optimize(spread, signal_generator, param_grid)
```

## Important Disclaimers

### Transaction Costs

**This backtest does NOT include transaction costs.** In practice:
- Spread costs: 0.5-2 pips per leg (3 legs = 1.5-6 pips round trip)
- Commission: Varies by broker
- Slippage: Can be significant in fast markets

A realistic implementation would need to clear a minimum profit threshold of ~2-5 pips per trade.

### Execution Reality

- **Signal lag**: We execute on the next bar's open (realistic)
- **Market impact**: Not modeled (would matter at scale)
- **Partial fills**: Not modeled

### Not Financial Advice

This is an educational project demonstrating quantitative finance concepts. Past performance does not indicate future results. Do not trade real money based on this code without extensive additional validation.

## References

1. Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. *Econometrica*, 55(2), 251-276.

2. Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6), 1551-1580.

3. Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading Insights and Techniques*. Wiley.

4. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

## License

MIT License - see [LICENSE](LICENSE) for details.
