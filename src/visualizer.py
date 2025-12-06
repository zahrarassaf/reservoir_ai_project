import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

mpl.rcParams['figure.figsize'] = [12, 8]
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['font.size'] = 10
sns.set_style("whitegrid")
sns.set_palette("husl")

class Visualizer:
    def __init__(self):
        self.figsize = (14, 10)
        self.colors = {
            'production': '#2E86AB',
            'revenue': '#A23B72',
            'cash_flow': '#F18F01',
            'npv': '#73AB84',
            'decline': '#C73E1D',
            'water': '#3A86FF',
            'gas': '#FF9B71'
        }
    
    def create_comprehensive_dashboard(self, results: Dict, dataset_id: str = "") -> go.Figure:
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Production Forecast', 'Economic Cash Flow', 'NPV Sensitivity',
                'Decline Curve Analysis', 'Cumulative Production', 'Reservoir Performance',
                'Economic Metrics', 'Uncertainty Analysis', 'Key Indicators'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'table'}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        prod_forecast = results.get('production_forecast', {})
        econ_results = results.get('economic_evaluation', {})
        decline_results = results.get('decline_analysis', {})
        uncertainty = results.get('uncertainty_analysis', {})
        kpis = results.get('key_performance_indicators', {})
        
        # 1. Production Forecast
        if 'field_rate' in prod_forecast:
            time_years = prod_forecast['time'] / 365
            fig.add_trace(
                go.Scatter(
                    x=time_years,
                    y=prod_forecast['field_rate'],
                    mode='lines',
                    name='Field Rate',
                    line=dict(color=self.colors['production'], width=2),
                    fill='tozeroy'
                ),
                row=1, col=1
            )
        
        # 2. Economic Cash Flow
        if 'annual_cash_flows' in econ_results:
            years = list(range(1, len(econ_results['annual_cash_flows']) + 1))
            fig.add_trace(
                go.Bar(
                    x=years,
                    y=econ_results['annual_cash_flows'],
                    name='Annual Cash Flow',
                    marker_color=self.colors['cash_flow']
                ),
                row=1, col=2
            )
        
        # 3. NPV Sensitivity
        if 'tornado_analysis' in uncertainty:
            tornado_data = uncertainty['tornado_analysis']
            variables = [d['variable'] for d in tornado_data]
            low_impacts = [d['low_impact'] / 1e6 for d in tornado_data]
            high_impacts = [d['high_impact'] / 1e6 for d in tornado_data]
            
            fig.add_trace(
                go.Bar(
                    y=variables,
                    x=low_impacts,
                    name='Low Impact',
                    orientation='h',
                    marker_color='#FF6B6B'
                ),
                row=1, col=3
            )
            
            fig.add_trace(
                go.Bar(
                    y=variables,
                    x=high_impacts,
                    name='High Impact',
                    orientation='h',
                    marker_color='#4ECDC4'
                ),
                row=1, col=3
            )
        
        # 4. Decline Curve Analysis
        if decline_results:
            sample_wells = list(decline_results.keys())[:3]
            for i, well_name in enumerate(sample_wells):
                params = decline_results[well_name]
                time_range = np.linspace(0, 3650, 100)
                
                if params.get('method') == 'exponential':
                    rates = params['qi'] * np.exp(-params['di'] * time_range)
                else:
                    rates = params['qi'] / (1 + params['di'] * time_range)
                
                fig.add_trace(
                    go.Scatter(
                        x=time_range / 365,
                        y=rates,
                        mode='lines',
                        name=f'{well_name}',
                        line=dict(width=1.5),
                        showlegend=True if i == 0 else False
                    ),
                    row=2, col=1
                )
        
        # 5. Cumulative Production
        if 'field_cumulative' in prod_forecast:
            time_years = prod_forecast['time'] / 365
            fig.add_trace(
                go.Scatter(
                    x=time_years,
                    y=prod_forecast['field_cumulative'] / 1e6,
                    mode='lines',
                    name='Cumulative',
                    line=dict(color=self.colors['production'], width=3),
                    fill='tozeroy'
                ),
                row=2, col=2
            )
        
        # 6. Economic Metrics
        metrics = ['NPV', 'IRR', 'ROI', 'Payback']
        values = [
            econ_results.get('npv', 0) / 1e6,
            econ_results.get('irr', 0),
            econ_results.get('roi', 0),
            min(econ_results.get('payback_period', 99), 20)
        ]
        
        colors = ['#73AB84', '#A23B72', '#F18F01', '#2E86AB']
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                marker_color=colors,
                text=[f'${values[0]:.1f}M' if i == 0 else f'{v:.1f}%' if i < 3 else f'{v:.1f}y' 
                     for i, v in enumerate(values)],
                textposition='auto'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Reservoir Simulation Dashboard - {dataset_id[:20]}",
            height=1000,
            showlegend=True,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Years", row=1, col=1)
        fig.update_yaxes(title_text="Rate (bpd)", row=1, col=1)
        
        fig.update_xaxes(title_text="Year", row=1, col=2)
        fig.update_yaxes(title_text="Cash Flow ($M)", row=1, col=2)
        
        fig.update_xaxes(title_text="NPV Impact ($M)", row=1, col=3)
        fig.update_yaxes(title_text="Variable", row=1, col=3)
        
        fig.update_xaxes(title_text="Years", row=2, col=1)
        fig.update_yaxes(title_text="Rate (bpd)", row=2, col=1)
        
        fig.update_xaxes(title_text="Years", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative (MMbbl)", row=2, col=2)
        
        fig.update_xaxes(title_text="Metric", row=3, col=1)
        fig.update_yaxes(title_text="Value", row=3, col=1)
        
        return fig
    
    def plot_production_profiles(self, production_forecast: Dict, top_n_wells: int = 5):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Field production rate
        if 'field_rate' in production_forecast:
            time_years = production_forecast['time'] / 365
            axes[0, 0].plot(time_years, production_forecast['field_rate'], 
                          color=self.colors['production'], linewidth=2)
            axes[0, 0].set_title('Field Production Rate')
            axes[0, 0].set_xlabel('Years')
            axes[0, 0].set_ylabel('Rate (bpd)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].fill_between(time_years, production_forecast['field_rate'], 
                                   alpha=0.2, color=self.colors['production'])
        
        # Annual production
        if 'annual_production' in production_forecast:
            years = np.arange(1, len(production_forecast['annual_production']) + 1)
            axes[0, 1].bar(years, production_forecast['annual_production'] / 1e6, 
                          color=self.colors['production'], alpha=0.7)
            axes[0, 1].set_title('Annual Production')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Production (MMbbl)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative production
        if 'field_cumulative' in production_forecast:
            time_years = production_forecast['time'] / 365
            axes[1, 0].plot(time_years, production_forecast['field_cumulative'] / 1e6, 
                          color=self.colors['production'], linewidth=2)
            axes[1, 0].set_title('Cumulative Production')
            axes[1, 0].set_xlabel('Years')
            axes[1, 0].set_ylabel('Cumulative (MMbbl)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].fill_between(time_years, production_forecast['field_cumulative'] / 1e6, 
                                   alpha=0.2, color=self.colors['production'])
        
        # Well forecasts
        if 'well_forecasts' in production_forecast:
            well_forecasts = production_forecast['well_forecasts']
            top_wells = sorted(well_forecasts.keys(), 
                             key=lambda x: well_forecasts[x]['eur'], 
                             reverse=True)[:top_n_wells]
            
            for well_name in top_wells:
                forecast = well_forecasts[well_name]
                time_years = forecast['time'] / 365
                axes[1, 1].plot(time_years, forecast['rate'], 
                              label=well_name, alpha=0.7, linewidth=1.5)
            
            axes[1, 1].set_title(f'Top {top_n_wells} Wells')
            axes[1, 1].set_xlabel('Years')
            axes[1, 1].set_ylabel('Rate (bpd)')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend(fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_economic_results(self, economic_results: Dict):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Cash flow waterfall
        if 'cash_flows' in economic_results:
            cash_flows = economic_results['cash_flows']
            years = list(range(len(cash_flows)))
            
            colors = ['#FF6B6B' if cf < 0 else '#4ECDC4' for cf in cash_flows]
            axes[0, 0].bar(years, np.array(cash_flows) / 1e6, color=colors, alpha=0.7)
            axes[0, 0].set_title('Cash Flow Timeline')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Cash Flow ($M)')
            axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[0, 0].grid(True, alpha=0.3)
        
        # NPV sensitivity
        if 'tornado_analysis' in economic_results.get('uncertainty_analysis', {}):
            tornado_data = economic_results['uncertainty_analysis']['tornado_analysis']
            variables = [d['variable'].replace('_', ' ').title() for d in tornado_data]
            low_impacts = [d['low_impact'] / 1e6 for d in tornado_data]
            high_impacts = [d['high_impact'] / 1e6 for d in tornado_data]
            
            y_pos = np.arange(len(variables))
            axes[0, 1].barh(y_pos - 0.2, low_impacts, 0.4, 
                           label='Low Case', color='#FF6B6B', alpha=0.7)
            axes[0, 1].barh(y_pos + 0.2, high_impacts, 0.4, 
                           label='High Case', color='#4ECDC4', alpha=0.7)
            axes[0, 1].set_yticks(y_pos)
            axes[0, 1].set_yticklabels(variables)
            axes[0, 1].set_title('NPV Sensitivity Analysis')
            axes[0, 1].set_xlabel('NPV Impact ($M)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Economic metrics radar chart
        metrics = ['NPV', 'IRR', 'ROI', 'Payback', 'Break-even']
        values = [
            min(economic_results.get('npv', 0) / 1e6, 100),
            min(economic_results.get('irr', 0), 50),
            min(economic_results.get('roi', 0), 200),
            min(economic_results.get('payback_period', 99), 10),
            min(economic_results.get('break_even_price', 0) / 10, 10)
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        axes[1, 0] = plt.subplot(2, 2, 3, projection='polar')
        axes[1, 0].plot(angles, values, 'o-', linewidth=2, color=self.colors['npv'])
        axes[1, 0].fill(angles, values, alpha=0.25, color=self.colors['npv'])
        axes[1, 0].set_thetagrids(np.degrees(angles[:-1]), metrics)
        axes[1, 0].set_title('Economic Performance Radar')
        axes[1, 0].grid(True)
        
        # Scenario analysis
        if 'scenario_analysis' in economic_results.get('uncertainty_analysis', {}):
            scenarios = economic_results['uncertainty_analysis']['scenario_analysis']
            scenario_names = list(scenarios.keys())
            npv_values = [scenarios[name]['npv'] / 1e6 for name in scenario_names]
            
            colors = ['#FF6B6B', '#F18F01', '#4ECDC4']
            axes[1, 1].bar(scenario_names, npv_values, color=colors, alpha=0.7)
            axes[1, 1].set_title('Scenario Analysis')
            axes[1, 1].set_xlabel('Scenario')
            axes[1, 1].set_ylabel('NPV ($M)')
            axes[1, 1].grid(True, alpha=0.3)
            
            for i, v in enumerate(npv_values):
                axes[1, 1].text(i, v + 0.1, f'${v:.1f}M', ha='center')
        
        plt.tight_layout()
        return fig
    
    def plot_reservoir_performance(self, reservoir_props: Dict, decline_curves: Dict):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Recovery factor distribution
        if 'recovery_factor' in reservoir_props:
            rf = reservoir_props['recovery_factor'] * 100
            axes[0, 0].bar(['Actual'], [rf], color=self.colors['production'], alpha=0.7)
            axes[0, 0].axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Typical RF')
            axes[0, 0].axhline(y=40, color='green', linestyle='--', alpha=0.5, label='Good RF')
            axes[0, 0].set_title('Recovery Factor')
            axes[0, 0].set_ylabel('Recovery (%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Decline parameters distribution
        if decline_curves:
            qi_values = [d['qi'] for d in decline_curves.values()]
            di_values = [d['di'] * 1000 for d in decline_curves.values()]
            
            axes[0, 1].hist(qi_values, bins=15, alpha=0.7, color=self.colors['decline'])
            axes[0, 1].set_title('Initial Rate Distribution')
            axes[0, 1].set_xlabel('Initial Rate (bpd)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Decline rate vs initial rate
        if decline_curves:
            qi_values = [d['qi'] for d in decline_curves.values()]
            di_values = [d['di'] * 1000 for d in decline_curves.values()]
            
            scatter = axes[1, 0].scatter(qi_values, di_values, 
                                        c=[d.get('r2', 0) for d in decline_curves.values()],
                                        cmap='viridis', s=100, alpha=0.7)
            axes[1, 0].set_title('Decline Characteristics')
            axes[1, 0].set_xlabel('Initial Rate (bpd)')
            axes[1, 0].set_ylabel('Decline Rate (per 1000 days)')
            axes[1, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 0], label='R²')
        
        # Reserve distribution
        if decline_curves and 'well_forecasts' in reservoir_props.get('production_forecast', {}):
            eur_values = [f['eur'] / 1e6 for f in 
                         reservoir_props['production_forecast']['well_forecasts'].values()]
            
            axes[1, 1].hist(eur_values, bins=15, alpha=0.7, color=self.colors['gas'])
            axes[1, 1].set_title('EUR Distribution per Well')
            axes[1, 1].set_xlabel('EUR (MMbbl)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_dashboard(self, fig: go.Figure, filename: str = "dashboard.html"):
        fig.write_html(filename)
        logger.info(f"Dashboard saved to {filename}")
    
    def save_matplotlib_figure(self, fig, filename: str = "figure.png"):
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Figure saved to {filename}")
