#!/usr/bin/env python3
"""
Generate analysis report using Walk-Forward Optimization (WFO).

This script avoids data leakage by:
1. Training cointegration/hedge ratios only on past data
2. Using rolling windows for parameter estimation
3. Testing on truly out-of-sample periods

Outputs saved to: reports/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

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
    print("FX Triangular Arbitrage - WFO Report Generator")
    print("=" * 60)
    
    from fxarb.data import load_pair, resample_ohlc
    from fxarb.data.features import compute_log_prices, add_session_features
    from fxarb.analysis import johansen_test, construct_spread
    from fxarb.models import compute_zscore, generate_signals
    from fxarb.backtest import run_backtest
    
    # =========================================================================
    # 1. Load and Prepare Data
    # =========================================================================
    print("\n[1/4] Loading data...")
    
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
    # 2. Walk-Forward Backtest (No Data Leakage)
    # =========================================================================
    print("\n[2/4] Running walk-forward backtest...")
    
    # Configuration
    train_days = 30  # 1 month training window
    test_days = 7    # 1 week test window
    bars_per_day = 288  # 5-min bars per day
    
    train_size = train_days * bars_per_day
    test_size = test_days * bars_per_day
    
    n_bars = len(common_idx)
    
    all_oos_returns = []
    all_oos_positions = []
    window_results = []
    
    start_idx = 0
    window_num = 0
    
    pbar = tqdm(total=(n_bars - train_size) // test_size, desc="WFO Windows")
    
    while start_idx + train_size + test_size <= n_bars:
        window_num += 1
        
        # Define window boundaries
        train_start = start_idx
        train_end = start_idx + train_size
        test_start = train_end
        test_end = min(test_start + test_size, n_bars)
        
        # Extract training data
        train_idx = common_idx[train_start:train_end]
        train_data = {p: data[p].loc[train_idx] for p in pairs}
        
        # Extract test data
        test_idx = common_idx[test_start:test_end]
        test_data = {p: data[p].loc[test_idx] for p in pairs}
        
        # Fit cointegration on TRAINING data only
        try:
            coint_result = johansen_test(train_data)
            hedge_ratios = coint_result.hedge_ratios
        except Exception:
            # Skip if cointegration fails
            start_idx += test_size
            pbar.update(1)
            continue
        
        # Construct spread for TEST period using TRAINING hedge ratios
        spread_test = construct_spread(test_data, hedge_ratios=hedge_ratios)
        
        # For Z-score, we need some history - use last 100 bars of training spread
        spread_train = construct_spread(train_data, hedge_ratios=hedge_ratios)
        combined_spread = pd.concat([spread_train.iloc[-100:], spread_test])
        
        # Compute Z-score on combined (but only use test portion for signals)
        zscore_combined = compute_zscore(combined_spread, lookback=100)
        zscore_test = zscore_combined.loc[test_idx]
        
        # Generate signals
        signals_test = generate_signals(zscore_test, entry_threshold=2.0, exit_threshold=0.0)
        
        # Run backtest on test period
        result = run_backtest(spread_test, signals_test)
        
        # Store results
        all_oos_returns.append(result.returns)
        all_oos_positions.append(result.positions)
        
        window_results.append({
            "window": window_num,
            "train_start": train_idx[0],
            "train_end": train_idx[-1],
            "test_start": test_idx[0],
            "test_end": test_idx[-1],
            "n_coint": coint_result.n_cointegrating,
            "sharpe": result.sharpe,
            "return": result.total_return,
            "trades": result.n_trades,
        })
        
        start_idx += test_size
        pbar.update(1)
    
    pbar.close()
    
    # Aggregate OOS results
    oos_returns = pd.concat(all_oos_returns)
    oos_positions = pd.concat(all_oos_positions)
    
    # Equity curve: cumulative sum of normalized returns (additive, not multiplicative)
    oos_equity = 1.0 + oos_returns.cumsum()
    
    # Replace any NaN/Inf with forward fill
    oos_equity = oos_equity.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
    
    # Compute aggregate metrics
    total_return = oos_equity.iloc[-1] - 1
    ann_vol = oos_returns.std() * np.sqrt(252 * bars_per_day)
    sharpe = (oos_returns.mean() / oos_returns.std()) * np.sqrt(252 * bars_per_day) if oos_returns.std() > 0 else 0
    
    running_max = oos_equity.cummax()
    drawdown = (oos_equity - running_max) / running_max.replace(0, 1)  # Avoid div by zero
    max_dd = drawdown.min()
    
    print(f"\n  Windows processed: {len(window_results)}")
    print(f"  Total OOS Return: {total_return:.2%}")
    print(f"  Annualized Sharpe: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.2%}")
    
    # =========================================================================
    # 3. Generate Visualizations
    # =========================================================================
    print("\n[3/4] Generating visualizations...")
    
    # Figure 1: Equity Curve and Drawdown
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    ax1 = axes[0]
    ax1.plot(oos_equity.index, oos_equity.values, linewidth=1, color="blue")
    ax1.set_ylabel("Equity (Normalized)")
    ax1.set_title("Walk-Forward Backtest Results (No Data Leakage)")
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f"Total Return: {total_return:.2%}\nSharpe: {sharpe:.2f}",
             transform=ax1.transAxes, va="top", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    
    ax2 = axes[1]
    ax2.fill_between(drawdown.index, 0, drawdown.values * 100, color="red", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    dd_min = drawdown.min() * 100
    if np.isfinite(dd_min):
        ax2.set_ylim(dd_min * 1.1 if dd_min < 0 else -1, 1)
    else:
        ax2.set_ylim(-10, 1)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    ax3.fill_between(oos_positions.index, 0, oos_positions.values, 
                     where=oos_positions > 0, color="green", alpha=0.5, label="Long")
    ax3.fill_between(oos_positions.index, 0, oos_positions.values, 
                     where=oos_positions < 0, color="red", alpha=0.5, label="Short")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylabel("Position")
    ax3.set_ylim(-1.5, 1.5)
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Date")
    
    # Set explicit x-axis limits
    ax1.set_xlim(oos_equity.index.min(), oos_equity.index.max())
    
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "05_wfo_backtest_results.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {REPORTS_DIR / '05_wfo_backtest_results.png'}")
    
    # Figure 2: Per-Window Performance
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    window_df = pd.DataFrame(window_results)
    
    ax1 = axes[0]
    colors = ["green" if r > 0 else "red" for r in window_df["return"]]
    ax1.bar(range(len(window_df)), window_df["return"] * 100, color=colors, alpha=0.7)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_xlabel("Window Number")
    ax1.set_ylabel("Return (%)")
    ax1.set_title("Per-Window Out-of-Sample Returns")
    ax1.grid(True, alpha=0.3, axis="y")
    
    ax2 = axes[1]
    ax2.bar(range(len(window_df)), window_df["sharpe"], color="steelblue", alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(window_df["sharpe"].mean(), color="red", linestyle="--", 
                label=f"Mean: {window_df['sharpe'].mean():.2f}")
    ax2.set_xlabel("Window Number")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Per-Window Out-of-Sample Sharpe Ratios")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "06_wfo_window_performance.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {REPORTS_DIR / '06_wfo_window_performance.png'}")
    
    # =========================================================================
    # 4. Summary Report
    # =========================================================================
    print("\n[4/4] Generating summary report...")
    
    n_trades = window_df["trades"].sum()
    win_rate = (window_df["return"] > 0).mean()
    
    summary = f"""
Walk-Forward Optimization Analysis Report
==========================================
Date Range: 2020-01-01 to 2020-12-31
Timeframe: 5-minute bars
Training Window: {train_days} days
Test Window: {test_days} days

TRIPLET: EUR/USD, GBP/USD, EUR/GBP

WFO CONFIGURATION
-----------------
Training Size: {train_size:,} bars ({train_days} days)
Test Size: {test_size:,} bars ({test_days} days)
Total Windows: {len(window_results)}

OUT-OF-SAMPLE PERFORMANCE (Aggregated)
--------------------------------------
Total Return: {total_return:.2%}
Annualized Volatility: {ann_vol:.2%}
Sharpe Ratio: {sharpe:.2f}
Max Drawdown: {max_dd:.2%}

PER-WINDOW STATISTICS
---------------------
Mean Window Return: {window_df['return'].mean():.2%}
Std Window Return: {window_df['return'].std():.2%}
Win Rate (windows): {win_rate:.1%}
Mean Window Sharpe: {window_df['sharpe'].mean():.2f}
Total Trades: {n_trades}

IMPORTANT NOTES
---------------
* These results use WALK-FORWARD OPTIMIZATION
* Cointegration/hedge ratios trained ONLY on past data
* NO data leakage - truly out-of-sample performance
* Transaction costs NOT included

Compare to original (leaked) backtest:
  Original Total Return: ~120% -> WFO: {total_return:.2%}
  Original Sharpe: ~12 -> WFO: {sharpe:.2f}

Generated Files:
  - reports/05_wfo_backtest_results.png
  - reports/06_wfo_window_performance.png
"""
    
    print(summary)
    
    with open(REPORTS_DIR / "wfo_summary_report.txt", "w") as f:
        f.write(summary)
    print(f"\nSaved: {REPORTS_DIR / 'wfo_summary_report.txt'}")
    
    print("\n" + "=" * 60)
    print("WFO Report generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
