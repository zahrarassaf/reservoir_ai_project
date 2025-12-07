# src/visualizer.py
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import json

class Visualizer:
    def __init__(self):
        self.figures = []
        
    def create_dashboard(self, results, dataset_id):
        try:
            os.makedirs('visualizations', exist_ok=True)
            
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Production Forecast', 'Economic Metrics',
                              'Cash Flow Analysis', 'Decline Curve Analysis',
                              'Scenario Analysis', 'Reservoir Properties'),
                specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                       [{'type': 'scatter'}, {'type': 'bar'}],
                       [{'type': 'bar'}, {'type': 'bar'}]],
                vertical_spacing=0.12,
                horizontal_spacing=0.15
            )
            
            if 'production_forecast' in results:
                forecast = results['production_forecast']
                if 'time' in forecast and 'field_rate' in forecast:
                    time = forecast['time']
                    rate = forecast['field_rate']
                    
                    if len(time) > 0 and len(rate) > 0:
                        fig.add_trace(
                            go.Scatter(x=time, y=rate, mode='lines', 
                                     name='Production Rate', line=dict(color='blue', width=2)),
                            row=1, col=1
                        )
                        fig.update_xaxes(title_text="Time (days)", row=1, col=1)
                        fig.update_yaxes(title_text="Rate (bbl/day)", row=1, col=1)
            
            if 'economic_evaluation' in results:
                econ = results['economic_evaluation']
                
                metrics = ['NPV ($M)', 'IRR (%)', 'ROI (%)', 'Break-even ($/bbl)']
                npv_m = econ.get('npv', 0) / 1e6
                irr = econ.get('irr', 0)
                roi = econ.get('roi', 0)
                break_even = econ.get('break_even_price', 0)
                values = [npv_m, irr, roi, break_even]
                
                colors = ['green' if npv_m > 0 else 'red', 
                         'green' if irr > 0 else 'red',
                         'green' if roi > 0 else 'red',
                         'orange']
                
                fig.add_trace(
                    go.Bar(x=metrics, y=values, name='Metrics',
                          marker_color=colors,
                          text=[f'{v:.1f}' for v in values],
                          textposition='auto'),
                    row=1, col=2
                )
                fig.update_yaxes(title_text="Value", row=1, col=2)
            
            if 'economic_evaluation' in results:
                econ = results['economic_evaluation']
                if 'cash_flows' in econ:
                    cash_flows = econ['cash_flows']
                    years = list(range(len(cash_flows)))
                    
                    fig.add_trace(
                        go.Bar(x=years, y=cash_flows, name='Cash Flow',
                              marker_color=['red' if cf < 0 else 'green' for cf in cash_flows],
                              text=[f'${cf/1e6:.1f}M' for cf in cash_flows],
                              textposition='auto'),
                        row=2, col=1
                    )
                    fig.update_xaxes(title_text="Year", row=2, col=1)
                    fig.update_yaxes(title_text="Cash Flow ($)", row=2, col=1)
            
            if 'decline_analysis' in results:
                decline = results['decline_analysis']
                if decline:
                    well_names = list(decline.keys())[:5]
                    qi_values = [decline[wn].get('qi', 0) for wn in well_names]
                    di_values = [decline[wn].get('di', 0) * 100 for wn in well_names]
                    
                    fig.add_trace(
                        go.Bar(x=well_names, y=qi_values, name='Initial Rate (qi)',
                              marker_color='blue'),
                        row=2, col=2
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=well_names, y=di_values, mode='lines+markers',
                                  name='Decline Rate (di)', line=dict(color='red', width=2),
                                  yaxis='y2'),
                        row=2, col=2
                    )
                    
                    fig.update_yaxes(title_text="qi (bpd)", row=2, col=2)
                    fig.update_yaxes(title_text="di (%/day)", row=2, col=2, secondary_y=True)
            
            if 'uncertainty_analysis' in results:
                uncertainty = results['uncertainty_analysis']
                if 'scenario_analysis' in uncertainty:
                    scenarios = ['Low Case', 'Base Case', 'High Case']
                    npvs = [uncertainty['scenario_analysis']['low_case']['npv']/1e6,
                           uncertainty['scenario_analysis']['base_case']['npv']/1e6,
                           uncertainty['scenario_analysis']['high_case']['npv']/1e6]
                    
                    fig.add_trace(
                        go.Bar(x=scenarios, y=npvs, name='NPV by Scenario',
                              marker_color=['red', 'gray', 'green'],
                              text=[f'${v:.1f}M' for v in npvs],
                              textposition='auto'),
                        row=3, col=1
                    )
                    fig.update_yaxes(title_text="NPV ($M)", row=3, col=1)
            
            if 'reservoir_properties' in results:
                props = results['reservoir_properties']
                prop_names = ['OOIP (MMbbl)', 'Recoverable (MMbbl)', 'Recovery Factor (%)']
                ooip_mm = props.get('original_oil_in_place', 0) / 1e6
                recoverable_mm = props.get('recoverable_oil', 0) / 1e6
                rf_pct = props.get('recovery_factor', 0) * 100
                prop_values = [ooip_mm, recoverable_mm, rf_pct]
                
                fig.add_trace(
                    go.Bar(x=prop_names, y=prop_values, name='Reservoir Props',
                          marker_color=['blue', 'green', 'orange'],
                          text=[f'{v:.1f}' for v in prop_values],
                          textposition='auto'),
                    row=3, col=2
                )
                fig.update_yaxes(title_text="Value", row=3, col=2)
            
            fig.update_layout(
                height=1200,
                title_text=f"Reservoir Analysis Dashboard - {dataset_id[:20]}",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            output_file = f"visualizations/{dataset_id[:20]}_dashboard.html"
            fig.write_html(output_file)
            print(f"  Dashboard saved: {output_file}")
            
            return fig
            
        except Exception as e:
            print(f"  Dashboard creation failed: {e}")
            return None
    
    def plot_production_forecast(self, results, dataset_id):
        try:
            if 'production_forecast' in results:
                forecast = results['production_forecast']
                
                if 'time' in forecast and 'field_rate' in forecast:
                    time = forecast['time']
                    rate = forecast['field_rate']
                    cumulative = forecast.get('field_cumulative', np.cumsum(rate * 30.4))
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    ax1.plot(time, rate, 'b-', linewidth=2, label='Production Rate')
                    ax1.fill_between(time, 0, rate, alpha=0.3, color='blue')
                    ax1.set_xlabel('Time (days)', fontsize=12)
                    ax1.set_ylabel('Production Rate (bbl/day)', fontsize=12)
                    ax1.set_title(f'Production Rate Forecast - {dataset_id[:20]}', 
                                fontsize=14, fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()
                    
                    ax2.plot(time, cumulative, 'g-', linewidth=2, label='Cumulative Production')
                    ax2.fill_between(time, 0, cumulative, alpha=0.3, color='green')
                    ax2.set_xlabel('Time (days)', fontsize=12)
                    ax2.set_ylabel('Cumulative Production (bbl)', fontsize=12)
                    ax2.set_title('Cumulative Production', fontsize=14, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                    
                    plt.tight_layout()
                    os.makedirs('visualizations', exist_ok=True)
                    output_file = f'visualizations/{dataset_id[:20]}_production.png'
                    plt.savefig(output_file, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  Production plot saved: {output_file}")
                
        except Exception as e:
            print(f"  Production plot failed: {e}")
    
    def plot_economic_results(self, results, dataset_id):
        try:
            if 'economic_evaluation' in results:
                econ = results['economic_evaluation']
                
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.flatten()
                
                if 'cash_flows' in econ:
                    cash_flows = econ['cash_flows']
                    years = list(range(len(cash_flows)))
                    
                    colors = ['red' if cf < 0 else 'green' for cf in cash_flows]
                    axes[0].bar(years, cash_flows, color=colors)
                    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                    axes[0].set_xlabel('Year', fontsize=12)
                    axes[0].set_ylabel('Cash Flow ($)', fontsize=12)
                    axes[0].set_title('Annual Cash Flows', fontsize=14, fontweight='bold')
                    axes[0].grid(True, alpha=0.3)
                    
                    for i, cf in enumerate(cash_flows):
                        axes[0].text(i, cf, f'${cf/1e6:.1f}M', 
                                   ha='center', va='bottom' if cf >= 0 else 'top', 
                                   fontsize=9, fontweight='bold')
                
                metrics = ['NPV', 'IRR', 'ROI']
                npv_m = econ.get('npv', 0) / 1e6
                irr = econ.get('irr', 0)
                roi = econ.get('roi', 0)
                values = [npv_m, irr, roi]
                colors = ['green' if val > 0 else 'red' for val in values]
                
                axes[1].bar(metrics, values, color=colors)
                axes[1].set_ylabel('Value', fontsize=12)
                axes[1].set_title('Key Economic Metrics', fontsize=14, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                
                for i, v in enumerate(values):
                    axes[1].text(i, v, f'{v:.1f}', ha='center', 
                               va='bottom' if v >= 0 else 'top', 
                               fontweight='bold', fontsize=11)
                
                if 'uncertainty_analysis' in results:
                    uncertainty = results['uncertainty_analysis']
                    if 'scenario_analysis' in uncertainty:
                        scenarios = ['Low', 'Base', 'High']
                        npvs = [uncertainty['scenario_analysis']['low_case']['npv']/1e6,
                               uncertainty['scenario_analysis']['base_case']['npv']/1e6,
                               uncertainty['scenario_analysis']['high_case']['npv']/1e6]
                        
                        axes[2].bar(scenarios, npvs, color=['red', 'gray', 'green'])
                        axes[2].set_xlabel('Scenario', fontsize=12)
                        axes[2].set_ylabel('NPV ($M)', fontsize=12)
                        axes[2].set_title('Scenario Analysis', fontsize=14, fontweight='bold')
                        axes[2].grid(True, alpha=0.3)
                        
                        for i, v in enumerate(npvs):
                            axes[2].text(i, v, f'${v:.1f}M', ha='center', 
                                       va='bottom' if v >= 0 else 'top', 
                                       fontweight='bold')
                
                if 'key_performance_indicators' in results:
                    kpis = results['key_performance_indicators']
                    kpi_names = ['Prod/Well', 'Revenue/bbl', 'OPEX/bbl', 'Netback/bbl', 'Capex/bbl']
                    kpi_values = [kpis.get('production_per_well', 0) / 1000,
                                 kpis.get('revenue_per_bbl', 0),
                                 kpis.get('opex_per_bbl', 0),
                                 kpis.get('netback_per_bbl', 0),
                                 kpis.get('capex_per_bbl', 0)]
                    
                    axes[3].bar(kpi_names, kpi_values, color=['blue', 'green', 'red', 'purple', 'orange'])
                    axes[3].set_ylabel('Value ($/bbl or MMBbl)', fontsize=12)
                    axes[3].set_title('Key Performance Indicators', fontsize=14, fontweight='bold')
                    axes[3].grid(True, alpha=0.3)
                    axes[3].tick_params(axis='x', rotation=45)
                
                if 'reservoir_properties' in results:
                    props = results['reservoir_properties']
                    prop_names = ['OOIP', 'Recoverable', 'RF (%)']
                    ooip_mm = props.get('original_oil_in_place', 0) / 1e6
                    recoverable_mm = props.get('recoverable_oil', 0) / 1e6
                    rf_pct = props.get('recovery_factor', 0) * 100
                    prop_values = [ooip_mm, recoverable_mm, rf_pct]
                    
                    axes[4].bar(prop_names, prop_values, color=['blue', 'green', 'orange'])
                    axes[4].set_ylabel('Value (MMbbl or %)', fontsize=12)
                    axes[4].set_title('Reservoir Properties', fontsize=14, fontweight='bold')
                    axes[4].grid(True, alpha=0.3)
                
                if 'decline_analysis' in results:
                    decline = results['decline_analysis']
                    if decline:
                        well_names = list(decline.keys())[:5]
                        qi_values = [decline[wn].get('qi', 0) for wn in well_names]
                        
                        axes[5].bar(well_names, qi_values, color='steelblue')
                        axes[5].set_xlabel('Well Name', fontsize=12)
                        axes[5].set_ylabel('Initial Rate (bpd)', fontsize=12)
                        axes[5].set_title('Well Initial Rates', fontsize=14, fontweight='bold')
                        axes[5].grid(True, alpha=0.3)
                        axes[5].tick_params(axis='x', rotation=45)
                
                plt.suptitle(f'Economic Analysis - {dataset_id[:20]}', fontsize=16, fontweight='bold', y=1.02)
                plt.tight_layout()
                os.makedirs('visualizations', exist_ok=True)
                output_file = f'visualizations/{dataset_id[:20]}_economics.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Economics plot saved: {output_file}")
                
        except Exception as e:
            print(f"  Economic plot failed: {e}")
    
    def save_results_json(self, results, dataset_id):
        try:
            os.makedirs('results', exist_ok=True)
            output_file = f"results/{dataset_id[:20]}_results.json"
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"  JSON results saved: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"  JSON save failed: {e}")
            return None
