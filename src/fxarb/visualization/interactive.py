"""
Interactive visualization widgets for Jupyter notebooks.
"""
from typing import Any


import ipywidgets as widgets
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

from fxarb.analysis.strategy import generate_signals
from fxarb.backtest.engine import Backtester
from fxarb.backtest.kelly import kelly_criterion


def visualize_interactive_equity(spread_data: pd.Series, zscore_data: pd.Series) -> widgets.VBox:
    """
    Interactive widget using persistent FigureWidget to strictly prevent stacking.
    
    Args:
        spread_data: Series of spread prices
        zscore_data: Series of Z-scores
        
    Returns:
        The widget container (VBox) to be displayed.
    """

    # 1. Pre-calculate Kelly Cache & Matrix
    # We calculate across the full range of widget options to ensure snappy interaction.
    # We also reproduce the specific "Analysis Matrix" requested for display.
    
    # Widget ranges
    cost_min, cost_max, cost_step = 0.0, 5.0, 0.25
    cost_values = [round(x * cost_step, 2) for x in range(int(cost_min/cost_step), int(cost_max/cost_step)+1)]
    
    # Z-Score options from existing logic
    percentiles = [0.95, 0.975, 0.99, 0.995, 0.999]
    # (Label, Value) tuples
    z_options = [(f'{p:.1%} ({norm.ppf(p):.2f}σ)', round(norm.ppf(p), 3)) for p in percentiles]
    z_values = [z[1] for z in z_options]
    
    kelly_cache = {} # Key: (cost, z_val) -> KellyResult
    
    # For the display table (subset of costs as per user snippet)
    display_costs = [0.0, 0.5, 1.0, 1.5, 2.0]
    results_matrix = pd.DataFrame(index=display_costs, columns=percentiles)
    results_matrix.index.name = 'Cost (bps)'
    results_matrix.columns.name = 'Percentile'

    # Output Widget for the Matrix Table

    
    # Pre-calculation message (printed to cell, not widget)
    print(f"Pre-calculating Kelly Matrix ({len(cost_values)} costs x {len(z_values)} thresholds)...")

    # Optimization: Signals depend ONLY on Z structure, not cost.
    # We can pre-calculate signal series for each Z.
    signals_map = {}
    for z in z_values:
        signals_map[z] = generate_signals(zscore_data, entry_threshold=z, exit_threshold=0.0)

    # Run Loop
    for cost in cost_values:
        for p_idx, z in enumerate(z_values):
            # 1. Backtest (1x)
            # Note: Backtester is fast enough for this loop (approx 100-200 iterations)
            bt = Backtester(
                initial_capital=100_000.0,
                transaction_cost_pips=cost * 1.0,
                leverage=1.0,
                compounding='geometric'
            )
            res = bt.run(spread_data, signals_map[z]['position'])
            
            # 2. Calculate Kelly
            if res.trades is not None and len(res.trades) > 5:
                k_res = kelly_criterion(res.trades)
                kelly_cache[(cost, z)] = k_res
                
                # Add to display matrix if this is a display cost
                if cost in display_costs:
                    p = percentiles[p_idx]
                    results_matrix.loc[cost, p] = f"{k_res.quarter_kelly:.1%}"
            else:
                kelly_cache[(cost, z)] = None
                if cost in display_costs:
                    p = percentiles[p_idx]
                    results_matrix.loc[cost, p] = "N/A"

    # Convert to HTML
    # We use basic Bootstrap classes which are often supported in Jupyter environments
    # to make the table look nicer.
    table_html = (
        "<h4>Suggested Quarter-Kelly Leverage (OOS Data)</h4>"
        + results_matrix.to_html(classes="table table-striped table-hover", border=0)
    )
    w_table = widgets.HTML(value=table_html)

    # 2. Widgets
    sizing_modes = ['Fixed Leverage', 'Full Kelly', 'Half Kelly', 'Quarter Kelly']
    w_mode = widgets.Dropdown(
        options=sizing_modes,
        value='Fixed Leverage',
        description='Sizing Mode:',
        layout=widgets.Layout(width='200px')
    )

    w_val = widgets.FloatSlider(value=10.0, min=1.0, max=50.0, step=1.0, description='Lev / Cap:')
    w_cost = widgets.FloatSlider(value=1.0, min=0.0, max=5.0, step=0.25, description='Cost (bps):')

    w_z = widgets.SelectionSlider(
        options=z_options,
        value=z_options[2][1], 
        description='Z-Entry:',
        continuous_update=False,
        layout=widgets.Layout(width='400px')
    )

    w_stats = widgets.HTML(
        value="Initializing...",
        placeholder="Stats will appear here",
        description="",
    )

    # 3. Figure Widget
    fig = go.FigureWidget()
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines',
        line=dict(width=1.5, color='#00cc96'),
        name='Equity'
    ))

    fig.update_layout(
        title='Strategy Reality Check',
        xaxis_title="Time",
        yaxis_title="Account Equity (Log Scale)",
        yaxis_type="log",
        template="plotly_dark",
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    def update_plot(change: Any = None) -> None:
        mode = w_mode.value
        val_setting = w_val.value 
        cost_bps = w_cost.value
        entry_z = w_z.value

        if mode == 'Fixed Leverage':
            w_val.description = 'Leverage (x):'
        else:
            w_val.description = 'Max Lev Cap:'

        # 1. Lookup Kelly from Cache
        k_res = kelly_cache.get((cost_bps, entry_z))
        
        # 2. Determine Leverage
        if mode == 'Fixed Leverage':
            target_leverage = val_setting
            kelly_est = 0.0
        else:
            if k_res is not None:
                if mode == 'Full Kelly':
                    raw_lev = k_res.full_kelly
                elif mode == 'Half Kelly':
                    raw_lev = k_res.half_kelly
                else: 
                    raw_lev = k_res.quarter_kelly
                
                kelly_est = max(0.0, raw_lev)
                
                if kelly_est > 0:
                    # Positive Edge: Use Kelly, capped by slider
                    target_leverage = min(kelly_est, val_setting)
                else:
                    # Negative Edge: Kelly says 0.
                    # Fallback to slider value to "visualize the downslope" (show the loss)
                    target_leverage = val_setting
            else:
                # No trades / Insufficient data
                target_leverage = val_setting
                kelly_est = 0.0

        # 3. Final Backtest (for Equity Curve)
        # We still need to run this to get the equity curve for the plot
        # But we skip the pre-calculation steps
        sigs = signals_map[entry_z] # Re-use pre-calc signals
        
        bt_final = Backtester(
            initial_capital=100_000.0,
            transaction_cost_pips=cost_bps * 1.0,
            leverage=target_leverage,
            compounding='geometric'
        )
        res = bt_final.run(spread_data, sigs['position'])

        with fig.batch_update():
            if res.equity_curve is not None and not res.equity_curve.empty:
                fig.data[0].x = res.equity_curve.index
                fig.data[0].y = res.equity_curve.values
            else:
                fig.data[0].x = []
                fig.data[0].y = []
                
            fig.data[0].name = f'Equity ({target_leverage:.2f}x)'
            
            title_text = (
                f'<b>Strategy Reality Check</b><br>'
                f'Mode: {mode} | Applied Lev: {target_leverage:.2f}x'
            )
            if mode != 'Fixed Leverage':
                 title_text += f' (Calc: {kelly_est:.2f}x)'
                 if kelly_est <= 0.001:
                     title_text += " [Negative Edge: Using Cap]"
            
            title_text += f' | Cost: {cost_bps} bps | Z: {entry_z:.2f}'
            fig.layout.title.text = title_text

        stats_html = (
            f'<div style="line-height: 1.5; font-family: monospace">'
            f'<b>CAGR:</b> {res.cagr:.1%}&nbsp;&nbsp;|&nbsp;&nbsp;'
            f'<b>Sharpe:</b> {res.sharpe:.2f}&nbsp;&nbsp;|&nbsp;&nbsp;'
            f'<b>Max DD:</b> {res.max_drawdown:.2%}&nbsp;&nbsp;|&nbsp;&nbsp;'
            f'<b>Trades:</b> {res.n_trades}'
            f'</div>'
        )
        w_stats.value = stats_html

    w_mode.observe(update_plot, names='value')
    w_val.observe(update_plot, names='value')
    w_cost.observe(update_plot, names='value')
    w_z.observe(update_plot, names='value')

    # Final Layout
    ui = widgets.VBox([
        widgets.HBox([w_mode, w_val]),
        widgets.HBox([w_z, w_cost]),
        w_stats,
        fig,
        w_table # Append HTML table at the bottom
    ])
    
    update_plot()
    
    return ui
