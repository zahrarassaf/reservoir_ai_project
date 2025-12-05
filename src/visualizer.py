"""
Reservoir Simulation Visualizer
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


class ReservoirVisualizer:
    """Create visualizations for reservoir simulation results"""
    
    def __init__(self, data, results):
        """
        Initialize visualizer
        
        Parameters
        ----------
        data : ReservoirData
            Input data
        results : Dict
            Simulation results
        """
        self.data = data
        self.results = results
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    def create_dashboard(self, save_path: Optional[str] = None):
        """Create comprehensive dashboard"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        
        # Production forecast
        self._plot_production_forecast(axes[0, 0])
        
        # Pressure profile
        self._plot_pressure_profile(axes[0, 1])
        
        # Economic analysis
        self._plot_economic_analysis(axes[0, 2])
        
        # Decline curves
        self._plot_decline_curves(axes[1, 0])
        
        # Petrophysical properties
        if not self.data.petrophysical.empty:
            self._plot_petrophysical(axes[1, 1])
        
        # Material balance
        self._plot_material_balance(axes[1, 2])
        
        # Sensitivity analysis
        self._plot_sensitivity(axes[2, 0])
        
        # Well production comparison
        self._plot_well_comparison(axes[2, 1])
        
        # Recovery factor
        self._plot_recovery_factor(axes[2, 2])
        
        plt.suptitle('Reservoir Simulation Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dashboard saved to {save_path}")
        
        plt.show()
    
    def _plot_production_forecast(self, ax):
        """Plot production forecast"""
        if 'production_forecast' not in self.results:
            ax.text(0.5, 0.5, 'No production forecast data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Production Forecast')
            return
        
        forecast = self.results['production_forecast']
        time = np.array(forecast['time'])
        total_prod = np.array(forecast['total_production'])
        
        # Find historical/forecast boundary
        hist_end = len(self.data.time) if hasattr(self.data, 'time') else 0
        
        # Plot
        time_years = time / 365
        ax.plot(time_years[:hist_end], total_prod[:hist_end], 
               'b-', linewidth=2, label='Historical')
        ax.plot(time_years[hist_end:], total_prod[hist_end:], 
               'r--', linewidth=2, label='Forecast')
        
        # Add vertical line at forecast start
        if hist_end < len(time_years):
            ax.axvline(x=time_years[hist_end], color='k', 
                      linestyle=':', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Time (Years)')
        ax.set_ylabel('Production Rate (bbl/day)')
        ax.set_title('Production Forecast')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_pressure_profile(self, ax):
        """Plot pressure profile"""
        if 'pressure_forecast' not in self.results:
            ax.text(0.5, 0.5, 'No pressure data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Pressure Profile')
            return
        
        pressure_data = self.results['pressure_forecast']
        time = np.array(pressure_data['time'])
        pressure = np.array(pressure_data['pressure'])
        
        # Find historical/forecast boundary
        hist_end = len(self.data.time) if hasattr(self.data, 'time') else 0
        
        # Plot
        time_years = time / 365
        ax.plot(time_years[:hist_end], pressure[:hist_end], 
               'g-', linewidth=2, label='Historical')
        ax.plot(time_years[hist_end:], pressure[hist_end:], 
               'm--', linewidth=2, label='Forecast')
        
        ax.set_xlabel('Time (Years)')
        ax.set_ylabel('Pressure (psi)')
        ax.set_title('Pressure Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_economic_analysis(self, ax):
        """Plot economic analysis"""
        if 'economic_analysis' not in self.results:
            ax.text(0.5, 0.5, 'No economic data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Economic Analysis')
            return
        
        econ = self.results['economic_analysis']
        cash_flow = np.array(econ['cumulative_cash_flow'])
        
        if 'production_forecast' in self.results:
            time = np.array(self.results['production_forecast']['time'])
            time_years = time / 365
            
            ax.plot(time_years, cash_flow / 1e6, 'b-', linewidth=2)
            
            # Add NPV line
            npv = econ.get('npv', 0)
            ax.axhline(y=npv/1e6, color='r', linestyle='--', 
                      label=f'NPV: ${npv/1e6:.1f}M')
            
            # Add payback period
            payback = econ.get('payback_period')
            if payback:
                ax.axvline(x=payback, color='g', linestyle=':', 
                          label=f'Payback: {payback:.1f} years')
        
        ax.set_xlabel('Time (Years)')
        ax.set_ylabel('Cumulative Cash Flow (Million USD)')
        ax.set_title('Economic Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_decline_curves(self, ax):
        """Plot decline curves"""
        if 'decline_analysis' not in self.results:
            ax.text(0.5, 0.5, 'No decline analysis data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Decline Curve Analysis')
            return
        
        decline_data = self.results['decline_analysis']
        
        for i, (well, data) in enumerate(list(decline_data.items())[:4]):  # Plot first 4 wells
            if 'exponential' in data:
                exp = data['exponential']
                ax.semilogy([i], [exp['initial_rate']], 'o', 
                           color=self.colors[i % len(self.colors)], 
                           label=f'{well}: D={exp["decline_rate"]:.3f}/yr')
        
        ax.set_xlabel('Well')
        ax.set_ylabel('Initial Rate (bbl/day) - Log Scale')
        ax.set_title('Decline Curve Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
    
    def _plot_petrophysical(self, ax):
        """Plot petrophysical properties"""
        if self.data.petrophysical.empty:
            ax.text(0.5, 0.5, 'No petrophysical data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Petrophysical Properties')
            return
        
        petro = self.data.petrophysical
        x = range(len(petro))
        width = 0.2
        
        properties = ['porosity', 'permeability', 'netthickness', 'watersaturation']
        colors = self.colors[:4]
        
        for i, (prop, color) in enumerate(zip(properties, colors)):
            if prop in petro.columns:
                positions = [pos - width*(1.5 - i) for pos in x]
                ax.bar(positions, petro[prop], width, 
                      label=prop.capitalize(), color=color, alpha=0.7)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Property Value')
        ax.set_title('Petrophysical Properties by Layer')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_material_balance(self, ax):
        """Plot material balance"""
        if 'material_balance' not in self.results:
            ax.text(0.5, 0.5, 'No material balance data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Material Balance')
            return
        
        mb = self.results['material_balance']
        
        if 'regression' in mb:
            reg = mb['regression']
            ax.text(0.5, 0.5, 
                   f'OOIP: {mb.get("ooip_stb", 0):,.0f} STB\n'
                   f'R²: {reg.get("r_squared", 0):.3f}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title('Material Balance Analysis')
        ax.axis('off')
    
    def _plot_sensitivity(self, ax):
        """Plot sensitivity analysis"""
        if 'sensitivity_analysis' not in self.results:
            ax.text(0.5, 0.5, 'No sensitivity data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Sensitivity Analysis')
            return
        
        sensitivity = self.results['sensitivity_analysis']
        
        # Get parameters (exclude 'key_parameters')
        params = [p for p in sensitivity.keys() if p != 'key_parameters']
        
        if len(params) > 0:
            param_names = params[:4]  # Show first 4 parameters
            base_npv = sensitivity[param_names[0]]['tornado_data']['base'] / 1e6
            
            low_vals = []
            high_vals = []
            
            for param in param_names:
                data = sensitivity[param]['tornado_data']
                low_vals.append(data['low'] / 1e6 - base_npv)
                high_vals.append(data['high'] / 1e6 - base_npv)
            
            y_pos = np.arange(len(param_names))
            
            ax.barh(y_pos, low_vals, height=0.4, color='red', alpha=0.6, label='Low')
            ax.barh(y_pos, high_vals, height=0.4, color='green', alpha=0.6, label='High')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([p.replace('_', ' ').title() for p in param_names])
            ax.set_xlabel('NPV Change (Million USD)')
            ax.set_title('Sensitivity Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_well_comparison(self, ax):
        """Plot well production comparison"""
        if not self.data.production.empty:
            # Plot first 4 wells
            wells_to_plot = self.data.production.columns[:4]
            
            for i, well in enumerate(wells_to_plot):
                production = self.data.production[well].values[:365]  # First year
                ax.plot(np.arange(len(production)), production, 
                       color=self.colors[i % len(self.colors)], 
                       label=well, alpha=0.7)
            
            ax.set_xlabel('Days')
            ax.set_ylabel('Production Rate (bbl/day)')
            ax.set_title('Well Production Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_recovery_factor(self, ax):
        """Plot recovery factor"""
        if ('production_forecast' in self.results and 
            'material_balance' in self.results):
            
            prod = self.results['production_forecast']
            mb = self.results['material_balance']
            
            cumulative_prod = np.array(prod['cumulative_production'])
            ooip = mb.get('ooip_stb', 1)
            
            if ooip > 0:
                recovery_factor = cumulative_prod / ooip * 100
                time_years = np.array(prod['time']) / 365
                
                ax.plot(time_years, recovery_factor, 'b-', linewidth=2)
                
                final_rf = recovery_factor[-1]
                ax.annotate(f'Final RF: {final_rf:.1f}%',
                           xy=(0.05, 0.95), xycoords='axes fraction',
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_xlabel('Time (Years)')
                ax.set_ylabel('Recovery Factor (%)')
                ax.set_title('Recovery Factor Profile')
                ax.grid(True, alpha=0.3)
                return
        
        ax.text(0.5, 0.5, 'No recovery factor data', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Recovery Factor')
    
    def create_interactive_dashboard(self, save_path: Optional[str] = None):
        """Create interactive Plotly dashboard"""
        if 'production_forecast' not in self.results:
            logger.warning("No forecast data for interactive dashboard")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Production Forecast', 'Pressure Profile', 
                          'Economic Analysis', 'Decline Curves', 
                          'Recovery Factor', 'Sensitivity Analysis'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Production forecast
        forecast = self.results['production_forecast']
        time = np.array(forecast['time'])
        total_prod = np.array(forecast['total_production'])
        hist_end = len(self.data.time)
        
        time_years = time / 365
        
        fig.add_trace(
            go.Scatter(x=time_years[:hist_end], y=total_prod[:hist_end],
                      mode='lines', name='Historical',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time_years[hist_end:], y=total_prod[hist_end:],
                      mode='lines', name='Forecast',
                      line=dict(color='red', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Pressure profile
        if 'pressure_forecast' in self.results:
            pressure_data = self.results['pressure_forecast']
            pressure = np.array(pressure_data['pressure'])
            
            fig.add_trace(
                go.Scatter(x=time_years, y=pressure,
                          mode='lines', name='Pressure',
                          line=dict(color='green', width=2)),
                row=1, col=2
            )
        
        # Economic analysis
        if 'economic_analysis' in self.results:
            econ = self.results['economic_analysis']
            cash_flow = np.array(econ['cumulative_cash_flow'])
            
            fig.add_trace(
                go.Scatter(x=time_years, y=cash_flow / 1e6,
                          mode='lines', name='Cash Flow',
                          line=dict(color='purple', width=2)),
                row=1, col=3
            )
        
        # Update layout
        fig.update_layout(
            title_text='Reservoir Simulation Dashboard',
            title_font_size=20,
            showlegend=True,
            height=800,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text='Time (Years)', row=1, col=1)
        fig.update_yaxes(title_text='Production (bbl/day)', row=1, col=1)
        
        fig.update_xaxes(title_text='Time (Years)', row=1, col=2)
        fig.update_yaxes(title_text='Pressure (psi)', row=1, col=2)
        
        fig.update_xaxes(title_text='Time (Years)', row=1, col=3)
        fig.update_yaxes(title_text='Cash Flow (Million USD)', row=1, col=3)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        fig.show()
