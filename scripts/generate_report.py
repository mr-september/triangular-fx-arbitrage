#!/usr/bin/env python3
"""
Generate analysis report with visualizations for the FX arbitrage project.

Produces:
- Cointegration analysis plots
- Spread and Z-score visualizations
- Backtest performance report
- Summary statistics

Outputs saved to: reports/
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure reports directory exists
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10


def main():
    print("=" * 60)
    print("FX Triangular Arbitrage - Analysis Report Generator")
    print("=" * 60)
    
    # Import after setting up
    from fxarb.data import load_pair, resample_ohlc
    from fxarb.data.features import compute_log_prices, add_session_features
    from fxarb.analysis import johansen_test, construct_spread, estimate_half_life
    from fxarb.analysis.stationarity import adf_test, kpss_test
    from fxarb.models import compute_zscore, generate_signals, ZScoreStrategy
    from fxarb.models.ou_process import fit_ou_mle
    from fxarb.backtest import run_backtest
    
    # =========================================================================
    # 1. Load and Prepare Data
    # =========================================================================
    print("\n[1/5] Loading data...")
    
    # Use 2020 data for analysis
    pairs = ["eurusd", "gbpusd", "eurgbp"]
    raw_data = {}
    for pair in pairs:
        raw_data[pair] = load_pair(pair, start_date="2020-01-01", end_date="2020-12-31")
        print(f"  {pair.upper()}: {len(raw_data[pair]):,} bars")
    
    # Resample to 5-minute bars
    print("\n  Resampling to 5-minute bars...")
    data_5m = {pair: resample_ohlc(df, "5min") for pair, df in raw_data.items()}
    
    # Align timestamps
    common_idx = data_5m["eurusd"].index
    for pair in pairs[1:]:
        common_idx = common_idx.intersection(data_5m[pair].index)
    
    data = {}
    for pair in pairs:
        df = data_5m[pair].loc[common_idx].copy()
        df = compute_log_prices(df)
        df = add_session_features(df)
        data[pair] = df
    
    print(f"  Aligned: {len(common_idx):,} common timestamps")
    
    # =========================================================================
    # 2. Generate Price Overview Plot
    # =========================================================================
    print("\n[2/5] Generating price overview...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    for i, (pair, df) in enumerate(data.items()):
        ax = axes[i]
        ax.plot(df.index, df["close"], linewidth=0.5, alpha=0.8)
        ax.set_ylabel(f"{pair.upper()}\nClose Price")
        ax.grid(True, alpha=0.3)
        
        # Add stats
        pct_change = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
        ax.text(0.02, 0.98, f"2020 Change: {pct_change:+.1f}%", 
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    axes[0].set_title("FX Pair Prices - 2020 (5-minute bars)")
    axes[-1].set_xlabel("Date")
    
    # Set explicit x-axis limits
    axes[0].set_xlim(common_idx.min(), common_idx.max())
    
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "01_price_overview.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {REPORTS_DIR / '01_price_overview.png'}")
    
    # =========================================================================
    # 3. Cointegration Analysis
    # =========================================================================
    print("\n[3/5] Running cointegration analysis...")
    
    coint_result = johansen_test(data)
    print(f"  Cointegrating relationships: {coint_result.n_cointegrating}")
    
    # Construct spread
    spread = construct_spread(data, hedge_ratios=coint_result.hedge_ratios)
    
    # Stationarity tests
    adf = adf_test(spread)
    kpss = kpss_test(spread)
    
    # Half-life
    try:
        hl = estimate_half_life(spread)
        print(f"  Half-life: {hl:.1f} bars ({hl * 5:.0f} minutes)")
    except ValueError:
        hl = None
        print("  Half-life: Could not estimate (non-stationary)")
    
    # OU process fitting
    try:
        ou = fit_ou_mle(spread)
        print(f"  OU theta: {ou.theta:.4f}, OU half-life: {ou.half_life:.1f} bars")
    except Exception as e:
        ou = None
        print(f"  OU fitting failed: {e}")
    
    # Generate spread analysis plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Panel 1: Spread
    ax1 = axes[0]
    ax1.plot(spread.index, spread.values, linewidth=0.5, alpha=0.8, color="purple")
    ax1.axhline(spread.mean(), color="black", linestyle="--", linewidth=1, label="Mean")
    ax1.axhline(spread.mean() + 2*spread.std(), color="red", linestyle=":", linewidth=1, label="±2σ")
    ax1.axhline(spread.mean() - 2*spread.std(), color="red", linestyle=":", linewidth=1)
    ax1.fill_between(spread.index, spread.mean() - spread.std(), spread.mean() + spread.std(),
                     alpha=0.1, color="green")
    ax1.set_ylabel("Spread Value")
    ax1.set_title("Cointegrated Spread: EUR/USD - β₁·GBP/USD - β₂·EUR/GBP")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # Add hedge ratios and stats
    hr = coint_result.hedge_ratios
    stats_text = (f"Hedge Ratios:\n"
                  f"  EUR/USD: {hr['eurusd']:.4f}\n"
                  f"  GBP/USD: {hr['gbpusd']:.4f}\n"
                  f"  EUR/GBP: {hr['eurgbp']:.4f}\n\n"
                  f"ADF p-value: {adf.pvalue:.4f}\n"
                  f"KPSS p-value: {kpss.pvalue:.4f}\n"
                  f"Half-life: {hl:.1f} bars" if hl else "")
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9), family="monospace")
    
    # Panel 2: Z-Score
    zscore = compute_zscore(spread, lookback=100)
    ax2 = axes[1]
    ax2.plot(zscore.index, zscore.values, linewidth=0.5, alpha=0.8, color="blue")
    ax2.axhline(2.0, color="red", linestyle="--", linewidth=1, label="Entry ±2.0")
    ax2.axhline(-2.0, color="red", linestyle="--", linewidth=1)
    ax2.axhline(0, color="green", linestyle="-", linewidth=1, label="Exit (0)")
    ax2.fill_between(zscore.index, 2, zscore.max()+1, alpha=0.05, color="red")
    ax2.fill_between(zscore.index, -2, zscore.min()-1, alpha=0.05, color="blue")
    ax2.set_ylabel("Z-Score")
    ax2.set_ylim(zscore.min()-0.5, zscore.max()+0.5)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Spread distribution
    ax3 = axes[2]
    spread_clean = spread.dropna()
    ax3.hist(spread_clean, bins=100, density=True, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax3.axvline(spread_clean.mean(), color="black", linestyle="--", linewidth=2, label="Mean")
    ax3.axvline(spread_clean.mean() + 2*spread_clean.std(), color="red", linestyle=":", linewidth=1.5)
    ax3.axvline(spread_clean.mean() - 2*spread_clean.std(), color="red", linestyle=":", linewidth=1.5, label="±2σ")
    ax3.set_xlabel("Spread Value")
    ax3.set_ylabel("Density")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    
    # Set explicit x-axis limits for time-series panels
    axes[0].set_xlim(spread.index.min(), spread.index.max())
    
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "02_cointegration_analysis.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {REPORTS_DIR / '02_cointegration_analysis.png'}")
    
    # =========================================================================
    # 4. Strategy Backtest
    # =========================================================================
    print("\n[4/5] Running backtest...")
    
    signals = generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.0, stop_loss_threshold=4.0)
    result = run_backtest(spread, signals)
    
    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Number of Trades: {result.n_trades}")
    
    # Generate backtest report
    fig = plt.figure(figsize=(14, 14))
    
    # Panel 1: Equity Curve
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(result.equity_curve.index, result.equity_curve.values, linewidth=1, color="blue")
    ax1.set_ylabel("Equity")
    ax1.set_title("Backtest Results: Z-Score Mean Reversion Strategy")
    ax1.grid(True, alpha=0.3)
    
    # Add return annotation
    ax1.text(0.02, 0.98, f"Total Return: {result.total_return:.2%}\nSharpe: {result.sharpe:.2f}",
             transform=ax1.transAxes, va="top", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    
    # Panel 2: Drawdown
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    running_max = result.equity_curve.cummax()
    drawdown = (result.equity_curve - running_max) / running_max * 100
    ax2.fill_between(drawdown.index, 0, drawdown.values, color="red", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(drawdown.min() * 1.1, 1)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Position
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    pos = result.positions
    ax3.fill_between(pos.index, 0, pos.values, where=pos > 0, color="green", alpha=0.5, label="Long")
    ax3.fill_between(pos.index, 0, pos.values, where=pos < 0, color="red", alpha=0.5, label="Short")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylabel("Position")
    ax3.set_ylim(-1.5, 1.5)
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Returns distribution
    ax4 = fig.add_subplot(4, 1, 4)
    returns_pct = result.returns.dropna() * 100
    ax4.hist(returns_pct, bins=100, density=True, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax4.axvline(returns_pct.mean(), color="black", linestyle="--", linewidth=2)
    ax4.set_xlabel("Return per Bar (%)")
    ax4.set_ylabel("Density")
    ax4.grid(True, alpha=0.3)
    
    # Stats box
    stats_text = (f"Backtest Statistics:\n"
                  f"  Trades: {result.n_trades}\n"
                  f"  Win Rate: {100*result.n_winning/max(1,result.n_trades):.1f}%\n"
                  f"  Avg Trade: {result.avg_trade_return:.4%}\n"
                  f"  Max Drawdown: {result.max_drawdown:.2%}\n"
                  f"  Calmar Ratio: {result.calmar:.2f}")
    ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes, va="top", ha="right", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9), family="monospace")
    
    # Set explicit x-axis limits
    ax1.set_xlim(result.equity_curve.index.min(), result.equity_curve.index.max())
    
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "03_backtest_results.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {REPORTS_DIR / '03_backtest_results.png'}")
    
    # =========================================================================
    # 5. Session Analysis
    # =========================================================================
    print("\n[5/5] Analyzing by trading session...")
    
    # Merge session info with returns
    session_returns = pd.DataFrame({
        "return": result.returns,
        "position": result.positions,
        "session": data["eurusd"]["primary_session"]
    }).dropna()
    
    # Calculate returns by session
    session_stats = session_returns.groupby("session").agg({
        "return": ["mean", "std", "count"]
    }).round(6)
    session_stats.columns = ["mean_return", "std_return", "n_bars"]
    
    # Calculate Sharpe by session (annualized)
    bars_per_year = 252 * 288
    session_stats["sharpe"] = (session_stats["mean_return"] / session_stats["std_return"] * 
                               np.sqrt(bars_per_year))
    
    # Generate session analysis plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of session performance
    ax1 = axes[0]
    sessions = session_stats.index.tolist()
    sharpes = session_stats["sharpe"].values
    colors = ["green" if s > 0 else "red" for s in sharpes]
    bars = ax1.bar(range(len(sessions)), sharpes, color=colors, alpha=0.7, edgecolor="black")
    ax1.set_xticks(range(len(sessions)))
    ax1.set_xticklabels([s.replace("_", "\n") for s in sessions], rotation=0)
    ax1.set_ylabel("Sharpe Ratio (Annualized)")
    ax1.set_title("Strategy Performance by Trading Session")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Time of day analysis
    ax2 = axes[1]
    session_returns["hour"] = session_returns.index.hour
    hourly = session_returns.groupby("hour")["return"].mean() * 100
    ax2.bar(hourly.index, hourly.values, color="steelblue", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Hour (UTC)")
    ax2.set_ylabel("Mean Return (%)")
    ax2.set_title("Mean Return by Hour of Day")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "04_session_analysis.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {REPORTS_DIR / '04_session_analysis.png'}")
    
    # =========================================================================
    # Summary Report
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    
    summary = f"""
Triangular FX Arbitrage Analysis Report
Date: 2020-01-01 to 2020-12-31
Timeframe: 5-minute bars

TRIPLET: EUR/USD, GBP/USD, EUR/GBP
=========================================

COINTEGRATION ANALYSIS
----------------------
Johansen Test:
  Cointegrating relationships: {coint_result.n_cointegrating}
  
Hedge Ratios:
  EUR/USD: {hr['eurusd']:.6f}
  GBP/USD: {hr['gbpusd']:.6f}
  EUR/GBP: {hr['eurgbp']:.6f}

Stationarity Tests:
  ADF p-value: {adf.pvalue:.6f} ({'STATIONARY' if adf.is_stationary else 'NON-STATIONARY'})
  KPSS p-value: {kpss.pvalue:.6f} ({'STATIONARY' if kpss.is_stationary else 'NON-STATIONARY'})

Mean Reversion:
  Half-life: {f'{hl:.1f}' if hl else 'N/A'} bars ({f'{hl*5:.0f}' if hl else 'N/A'} minutes) 
  OU theta: {f'{ou.theta:.6f}' if ou else 'N/A'}


STRATEGY: Z-Score Mean Reversion
--------------------------------
Parameters:
  Lookback: 100 bars
  Entry threshold: ±2.0
  Exit threshold: 0.0
  Stop-loss: ±4.0

BACKTEST RESULTS (No transaction costs)
-----------------------------------------
Performance:
  Total Return: {result.total_return:.2%}
  CAGR: {result.cagr:.2%}
  Sharpe Ratio: {result.sharpe:.2f}
  Max Drawdown: {result.max_drawdown:.2%}
  Calmar Ratio: {result.calmar:.2f}

Trades:
  Total Trades: {result.n_trades}
  Winning Trades: {result.n_winning} ({100*result.n_winning/max(1,result.n_trades):.1f}%)
  Average Trade Return: {result.avg_trade_return:.4%}
  Average Trade Duration: {result.avg_trade_duration:.1f} bars


IMPORTANT DISCLAIMERS
---------------------
* Transaction costs NOT included (would reduce returns by ~1-3 pips per trade)
* Slippage NOT modeled
* This is a simplified simulation for educational/CV purposes
* Past performance does not guarantee future results


Generated Files:
  - reports/01_price_overview.png
  - reports/02_cointegration_analysis.png
  - reports/03_backtest_results.png
  - reports/04_session_analysis.png
"""
    
    print(summary)
    
    # Save text report
    with open(REPORTS_DIR / "summary_report.txt", "w") as f:
        f.write(summary)
    print(f"\nSaved: {REPORTS_DIR / 'summary_report.txt'}")
    
    print("\n" + "=" * 60)
    print("Report generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
