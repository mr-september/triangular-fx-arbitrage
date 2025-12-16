import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy.stats import norm
import plotly.offline as py_offline
from fxarb.backtest.engine import Backtester
from fxarb.backtest.kelly import kelly_criterion
from fxarb.analysis.strategy import generate_signals

class NaNSafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if np.isnan(obj): return None
            if np.isinf(obj): return None
        return super().default(obj)

def generate_portable_widget(spread_data: pd.Series, zscore_data: pd.Series) -> str:
    print("Generating Portable Widget (Pre-computing vectors)...")
    
    # --- 1. CONFIGURATION ---
    resample_rule = '1h'
    cost_min, cost_max, cost_step = 0.0, 5.0, 0.25
    cost_values = [round(x * cost_step, 2) for x in range(int(cost_min/cost_step), int(cost_max/cost_step)+1)]
    percentiles = [0.95, 0.975, 0.99, 0.995, 0.999]
    z_options = [(f'{p:.1%} ({norm.ppf(p):.2f}σ)', round(norm.ppf(p), 3)) for p in percentiles]
    z_values = [z[1] for z in z_options]
    
    # --- 2. PRE-COMPUTATION ---
    kelly_matrix = {} 
    
    signals_map = {}
    for z in z_values:
        signals_map[z] = generate_signals(zscore_data, entry_threshold=z, exit_threshold=0.0)

    for cost in cost_values:
        for z in z_values:
            bt = Backtester(transaction_cost_pips=cost, leverage=1.0)
            res = bt.run(spread_data, signals_map[z]['position'])
            
            k_data = {'full': 0.0, 'half': 0.0, 'quarter': 0.0}
            if res.trades is not None and len(res.trades) > 5:
                try:
                    k = kelly_criterion(res.trades)
                    k_data = {
                        'full': round(k.full_kelly, 2),
                        'half': round(k.half_kelly, 2),
                        'quarter': round(k.quarter_kelly, 2)
                    }
                except:
                    pass 
            
            key = f"{cost:.2f}_{z:.3f}"
            kelly_matrix[key] = k_data

    vectors = {}
    for z in z_values:
        pos = signals_map[z]['position']
        z_key = f"{z:.3f}"
        
        # Performance vectors (Raw PnL before Lev/Cost)
        lag = 1
        pos_lagged = pos.shift(lag).fillna(0)
        price_changes = spread_data.diff().fillna(0)
        
        raw_pnl = pos_lagged * price_changes 
        turnover = pos_lagged.diff().fillna(0).abs()
        
        df_temp = pd.DataFrame({'ret': raw_pnl, 'to': turnover})
        
        if len(df_temp) > 0:
            df_resampled = df_temp.resample(resample_rule).sum().dropna()
            times = df_resampled.index.strftime('%Y-%m-%d %H:%M').tolist()
            rets = df_resampled['ret'].fillna(0.0).tolist()
            tovr = df_resampled['to'].fillna(0.0).tolist()
        else:
            times, rets, tovr = [], [], []
        
        vectors[z_key] = {
            'times': times,
            'returns': rets,
            'turnover': tovr
        }

    # --- 3. HTML/JS GENERATION ---
    print("Embedding PlotlyJS (Offline)...")
    plotly_js_code = py_offline.get_plotlyjs()
    
    data_obj = {
        'kelly': kelly_matrix,
        'vectors': vectors,
        'defaults': {
            'cost': 1.0,
            'z': z_values[2], 
            'lev': 10.0,
            'mode': 'Fixed Leverage'
        },
        'options': {
            'z_values': z_values,
            'z_labels': [z[0] for z in z_options]
        }
    }
    
    data_json = json.dumps(data_obj, cls=NaNSafeEncoder)
    
    html_template = f"""
    <div id="fxarb-widget-root" style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; background: white;">
        <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; align-items: center;">
            <div>
                <label><b>Sizing Mode:</b></label><br>
                <select id="w-mode" style="width: 150px; padding: 5px;">
                    <option>Fixed Leverage</option>
                    <option>Full Kelly</option>
                    <option>Half Kelly</option>
                    <option>Quarter Kelly</option>
                </select>
            </div>
             <div>
                <label id="lbl-lev"><b>Leverage (x):</b> <span id="val-lev">10.0</span></label><br>
                <input type="range" id="w-lev" min="1" max="50" step="1" value="10" style="width: 150px;">
            </div>
            <div>
                <label><b>Entry Threshold:</b></label><br>
                <select id="w-z" style="width: 200px; padding: 5px;"></select>
            </div>
            <div>
                <label><b>Cost (bps):</b> <span id="val-cost">1.0</span></label><br>
                <input type="range" id="w-cost" min="0" max="5" step="0.25" value="1.0" style="width: 150px;">
            </div>
        </div>

        <div id="w-error" style="color: red; display: none; margin-bottom: 10px; padding: 10px; background: #ffe6e6; border: 1px solid red;"></div>
        <div id="w-stats" style="margin-bottom: 10px; font-family: monospace; background: #f8f9fa; padding: 8px; border-radius: 4px;">Initializing...</div>
        <div id="w-chart" style="width: 100%; height: 500px; border: 1px solid #eee;"></div>

        <script type="text/javascript">
        (function(){{
            var _oldDefine = window.define;
            window.define = undefined;
            {plotly_js_code}
            window.define = _oldDefine;
        }})();
        </script>

        <script>
        (function() {{
            const elError = document.getElementById('w-error');
            const elStats = document.getElementById('w-stats');
            const elChart = document.getElementById('w-chart');

            function showError(msg) {{
                elError.innerText = "Widget Error: " + msg;
                elError.style.display = 'block';
                console.error(msg);
            }}

            function withPlotly(callback) {{
                if (typeof window.Plotly !== 'undefined') callback(window.Plotly);
                else if (typeof require !== 'undefined') {{
                    require(['plotly'], function(p) {{ callback(p); }}, function() {{
                        if (window.Plotly) callback(window.Plotly);
                        else showError("Plotly not found.");
                    }});
                }} else showError("Plotly not found.");
            }}

            try {{
                const DATA = {data_json};
                const elMode = document.getElementById('w-mode');
                const elLev = document.getElementById('w-lev');
                const elLevLabel = document.getElementById('lbl-lev');
                const elLevVal = document.getElementById('val-lev');
                const elZ = document.getElementById('w-z');
                const elCost = document.getElementById('w-cost');
                const elCostVal = document.getElementById('val-cost');
                
                DATA.options.z_values.forEach((val, idx) => {{
                    let opt = document.createElement('option');
                    opt.value = val;
                    opt.text = DATA.options.z_labels[idx];
                    if (Math.abs(val - DATA.defaults.z) < 0.0001) opt.selected = true;
                    elZ.appendChild(opt);
                }});

                let state = {{
                    mode: DATA.defaults.mode,
                    lev: DATA.defaults.lev,
                    z: DATA.defaults.z,
                    cost: DATA.defaults.cost
                }};

                withPlotly(function(Plotly) {{
                    
                    function calculateEquity() {{
                        let zKey = state.z.toFixed(3); 
                        let vec = DATA.vectors[zKey];
                        
                        if (!vec) {{
                           const keys = Object.keys(DATA.vectors);
                           const match = keys.find(k => Math.abs(parseFloat(k) - state.z) < 0.001);
                           if (match) vec = DATA.vectors[match];
                        }}

                        if (!vec || !vec.returns || vec.returns.length === 0) 
                             return {{ x:[], y:[], finalLev: 0, kellyEst: 0, isSimulated: false}};

                        let targetLev = state.lev;
                        let kellyEst = 0.0;
                        let isSimulated = false; // Flag for Negative Edge visualization
                        
                        if (state.mode !== 'Fixed Leverage') {{
                            const kKey = state.cost.toFixed(2) + "_" + state.z.toFixed(3);
                            const kData = DATA.kelly[kKey];
                            
                            if (kData) {{
                                if (state.mode === 'Full Kelly') kellyEst = kData.full;
                                else if (state.mode === 'Half Kelly') kellyEst = kData.half;
                                else kellyEst = kData.quarter;
                            }}
                            
                            // LOGIC CHANGE: Visualization of Negative Edge
                            if (kellyEst > 0.0001) {{
                                // Positive Edge: Use Kelly (Capped by Slider)
                                targetLev = Math.min(kellyEst, state.lev);
                                isSimulated = false;
                            }} else {{
                                // Negative Edge:
                                // "Educate me" mode: Use Slider Cap to visualize the loss
                                targetLev = state.lev;
                                isSimulated = true;
                            }}
                        }}

                        const costFactor = state.cost * 0.0001;
                        const equity = [100000]; 
                        
                        for(let i=0; i<vec.returns.length; i++) {{
                            const r = vec.returns[i] || 0;
                            const t = vec.turnover[i] || 0;
                            
                            // Correct Scaled Calculation
                            const netRet = targetLev * (r - (t * costFactor));
                            
                            const newEq = equity[i] * (1 + netRet);
                            equity.push(newEq);
                        }}
                        
                        return {{
                            x: ['start', ...vec.times],
                            y: equity,
                            finalLev: targetLev,
                            kellyEst: kellyEst,
                            mode: state.mode,
                            isSimulated: isSimulated
                        }};
                    }}

                    function update() {{
                        try {{
                            elLevVal.innerText = state.lev;
                            elCostVal.innerText = state.cost;
                            
                            elLevLabel.innerHTML = (state.mode === 'Fixed Leverage') ? 
                                `<b>Leverage (x):</b> <span id="val-lev">${{state.lev}}</span>` : 
                                `<b>Max Lev Cap:</b> <span id="val-lev">${{state.lev}}</span>`;

                            const res = calculateEquity();
                            
                            if (res.x.length === 0) {{
                                elStats.innerText = "No data available.";
                                return;
                            }}

                            const totalRet = (res.y[res.y.length-1] / 100000) - 1;
                            
                            let levMsg = "";
                            let color = "#000000";

                            if (res.mode === 'Fixed Leverage') {{
                                levMsg = `Applied Lev: ${{ res.finalLev.toFixed(2) }}x`;
                            }} else {{
                                if (res.isSimulated) {{
                                    // WARNING MODE
                                    levMsg = `
                                        <span style="color: #d62728; font-weight: bold;">
                                        ⚠ NEGATIVE EDGE (Kelly ≤ 0). Simulating at ${{res.finalLev}}x leverage.
                                        </span>
                                    `;
                                }} else {{
                                    levMsg = `Applied Lev: ${{ res.finalLev.toFixed(2) }}x (Kelly Calc: ${{res.kellyEst.toFixed(2)}}x)`;
                                }}
                            }}

                            const msg = `
                                <b>Total Return:</b> ${{ (totalRet*100).toFixed(2) }}% &nbsp;|&nbsp;
                                ${{levMsg}}
                            `;
                            elStats.innerHTML = msg;

                            // Chart Color Logic
                            // Green if profitable
                            // Red if unprofitable
                            // Orange/Dashed if Simulated? (Simplest is just Red line for loss)
                            const traceColor = totalRet >= 0 ? '#00cc96' : '#ef553b';

                            const trace = {{
                                x: res.x,
                                y: res.y,
                                type: 'scatter',
                                mode: 'lines',
                                line: {{ color: traceColor, width: 2 }},
                                name: 'Equity'
                            }};
                            
                            const layout = {{
                                title: `Portable Strategy (1H Resampled)`,
                                yaxis: {{ type: 'log', title: 'Equity (Log Scale)' }},
                                xaxis: {{ title: 'Date' }},
                                margin: {{ t: 40, b: 40, l: 60, r: 20 }},
                                template: 'plotly_white',
                                height: 500
                            }};

                            Plotly.react(elChart, [trace], layout);
                            
                        }} catch (e) {{
                            showError(e.message);
                        }}
                    }}

                    elMode.addEventListener('change', (e) => {{ state.mode = e.target.value; update(); }});
                    elLev.addEventListener('input', (e) => {{ state.lev = parseFloat(e.target.value); update(); }});
                    elCost.addEventListener('input', (e) => {{ state.cost = parseFloat(e.target.value); update(); }});
                    elZ.addEventListener('change', (e) => {{ state.z = parseFloat(e.target.value); update(); }});

                    update();
                }});

            }} catch(err) {{
                showError("Init Error: " + err.message);
            }}

        }})();
        </script>
    </div>
    """
    return html_template