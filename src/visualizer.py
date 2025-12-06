# src/visualizer.py
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

class Visualizer:
    def create_dashboard(self, results, dataset_id):
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Production Forecast', 'Economic Metrics',
                              'Cash Flow Analysis', 'Decline Curve Analysis'),
                specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                       [{'type': 'scatter'}, {'type': 'scatter'}]]
            )
            
            if 'production_forecast' in results:
                forecast = results['production_forecast']
                if 'time' in forecast and 'field_rate' in forecast:
                    time = forecast['time']
                    rate = forecast['field_rate']
                    
                    if len(time) > 0 and len(rate) > 0:
                        fig.add_trace(
                            go.Scatter(x=time, y=rate, mode='lines', name='Production Rate'),
                            row=1, col=1
                        )
            
            if 'economic_evaluation' in results:
                econ = results['economic_evaluation']
                metrics = ['NPV ($M)', 'IRR (%)', 'ROI (%)']
                npv_m = econ.get('npv', 0) / 1e6
                irr = econ.get('irr', 0)
                roi = econ.get('roi', 0)
                values = [npv_m, irr, roi]
                
                colors = ['green' if npv_m > 0 else 'red', 
                         'green' if irr > 0 else 'red',
                         'green' if roi > 0 else 'red']
                
                fig.add_trace(
                    go.Bar(x=metrics, y=values, name='Economic Metrics',
                          marker_color=colors),
                    row=1, col=2
                )
            
            if 'economic_evaluation' in results:
                econ = results['economic_evaluation']
                if 'cash_flows' in econ:
                    cash_flows = econ['cash_flows']
                    years = list(range(len(cash_flows)))
                    
                    fig.add_trace(
                        go.Scatter(x=years, y=cash_flows, mode='lines+markers', 
                                  name='Cash Flow', line=dict(color='blue')),
                        row=2, col=1
                    )
            
            if 'decline_analysis' in results:
                decline = results['decline_analysis']
                if decline:
                    well_names = list(decline.keys())[:3]
                    qi_values = [decline[wn].get('qi', 0) for wn in well_names]
                    
                    fig.add_trace(
                        go.Bar(x=well_names, y=qi_values, name='Initial Rate (qi)',
                              marker_color='orange'),
                        row=2, col=2
                    )
            
            fig.update_layout(
                height=800, 
                title_text=f"Reservoir Analysis - {dataset_id[:20]}",
                showlegend=True
            )
            
            os.makedirs('visualizations', exist_ok=True)
            fig.write_html(f"visualizations/{dataset_id[:20]}_dashboard.html")
            print(f"  Dashboard saved: visualizations/{dataset_id[:20]}_dashboard.html")
            
        except Exception as e:
            print(f"  Dashboard creation failed: {e}")
    
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
                    ax1.set_title(f'Production Rate Forecast - {dataset_id[:20]}', fontsize=14, fontweight='bold')
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
                    plt.savefig(f'visualizations/{dataset_id[:20]}_production.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  Production plot saved: visualizations/{dataset_id[:20]}_production.png")
                
        except Exception as e:
            print(f"  Production plot failed: {e}")
    
    def plot_economic_results(self, results, dataset_id):
        try:
            if 'economic_evaluation' in results:
                econ = results['economic_evaluation']
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
                
                if 'cash_flows' in econ:
                    cash_flows = econ['cash_flows']
                    years = list(range(len(cash_flows)))
                    
                    ax1.bar(years, cash_flows, color=['red' if cf < 0 else 'green' for cf in cash_flows])
                    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                    ax1.set_xlabel('Year', fontsize=12)
                    ax1.set_ylabel('Cash Flow ($)', fontsize=12)
                    ax1.set_title('Annual Cash Flows', fontsize=14, fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                
                metrics = ['NPV', 'IRR', 'ROI']
                npv_m = econ.get('npv', 0) / 1e6
                irr = econ.get('irr', 0)
                roi = econ.get('roi', 0)
                values = [npv_m, irr, roi]
                colors = ['green' if val > 0 else 'red' for val in values]
                
                ax2.bar(metrics, values, color=colors)
                ax2.set_ylabel('Value', fontsize=12)
                ax2.set_title('Key Economic Metrics', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                for i, v in enumerate(values):
                    ax2.text(i, v, f'{v:.1f}', ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')
                
                if 'uncertainty_analysis' in results:
                    uncertainty = results['uncertainty_analysis']
                    if 'scenario_analysis' in uncertainty:
                        scenarios = ['Low', 'Base', 'High']
                        npvs = [uncertainty['scenario_analysis']['low_case']['npv']/1e6,
                               uncertainty['scenario_analysis']['base_case']['npv']/1e6,
                               uncertainty['scenario_analysis']['high_case']['npv']/1e6]
                        
                        ax3.bar(scenarios, npvs, color=['red', 'gray', 'green'])
                        ax3.set_xlabel('Scenario', fontsize=12)
                        ax3.set_ylabel('NPV ($M)', fontsize=12)
                        ax3.set_title('Scenario Analysis', fontsize=14, fontweight='bold')
                        ax3.grid(True, alpha=0.3)
                
                if 'key_performance_indicators' in results:
                    kpis = results['key_performance_indicators']
                    kpi_names = ['Production/Well', 'Revenue/bbl', 'OPEX/bbl', 'Netback/bbl']
                    kpi_values = [kpis.get('production_per_well', 0),
                                 kpis.get('revenue_per_bbl', 0),
                                 kpis.get('opex_per_bbl', 0),
                                 kpis.get('netback_per_bbl', 0)]
                    
                    ax4.bar(kpi_names, kpi_values, color=['blue', 'green', 'red', 'purple'])
                    ax4.set_ylabel('Value ($/bbl)', fontsize=12)
                    ax4.set_title('Key Performance Indicators', fontsize=14, fontweight='bold')
                    ax4.grid(True, alpha=0.3)
                    ax4.tick_params(axis='x', rotation=45)
                
                plt.suptitle(f'Economic Analysis - {dataset_id[:20]}', fontsize=16, fontweight='bold', y=1.02)
                plt.tight_layout()
                os.makedirs('visualizations', exist_ok=True)
                plt.savefig(f'visualizations/{dataset_id[:20]}_economics.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Economics plot saved: visualizations/{dataset_id[:20]}_economics.png")
                
        except Exception as e:
            print(f"  Economic plot failed: {e}")
