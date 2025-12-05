"""
Reservoir Simulation Visualization

This module provides comprehensive visualization capabilities for
reservoir simulation results including production profiles, pressure
analysis, economic indicators, and sensitivity studies.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


class ReservoirVisualizer:
    """
    Advanced visualization toolkit for reservoir simulation results.
    
    Features:
    - Production profile plots
    - Pressure analysis charts
    - Economic indicator dashboards
    - Sensitivity analysis visualizations
    - Interactive 3D plots
    - Professional report-ready figures
    """
    
    def __init__(self, data: Dict, results: Dict):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        data : Dict
            Input reservoir data
        results : Dict
            Simulation results
        """
        self.data = data
        self.results = results
        self.historical_end = len(data.get('time', []))
        
        # Set professional plotting style
        self._set_plotting_style()
        logger.info("ReservoirVisualizer initialized")
    
    def _set_plotting_style(self):
        """Set professional plotting style."""
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
    
    def create_comprehensive_dashboard(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization dashboard.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the dashboard figure
        """
        logger.info("Creating comprehensive visualization dashboard")
        
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Production Profile
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_production_profile(ax1)
        
        # Pressure Profile
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_pressure_profile(ax2)
        
        # Injection Profile
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_injection_profile(ax3)
        
        # Economic Analysis
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_economic_analysis(ax4)
        
        # Decline Curve Analysis
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_decline_analysis(ax5)
        
        # Petrophysical Properties
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_petrophysical_distribution(ax6)
        
        # Correlation Matrix
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_correlation_matrix(ax7)
        
        # Material Balance
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_material_balance(ax8)
        
        # Recovery Factor
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_recovery_factor(ax9)
        
        # Sensitivity Analysis
        ax10 = fig.add_subplot(gs[3, :])
        self._plot_sensitivity_analysis(ax10)
        
        fig.suptitle('Advanced Reservoir Simulation Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dashboard saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def _plot_production_profile(self, ax):
        """Plot production profile with historical and forecast data."""
        production = self.results['production']
        time_years = self.results['time'] / 365
        
        total_production = production.sum(axis=1)
        
        ax.plot(time_years[:self.historical_end], 
                total_production[:self.historical_end],
                'b-', linewidth=2, label='Historical', alpha=0.8)
        
        ax.plot(time_years[self.historical_end:], 
                total_production[self.historical_end:],
                'r--', linewidth=2, label='Forecast', alpha=0.8)
        
        ax.axvline(x=time_years[self.historical_end], 
                  color='k', linestyle=':', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Time (Years)')
        ax.set_ylabel('Total Production Rate (bbl/day)')
        ax.set_title('Production Profile')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add annotation for peak production
        peak_idx = np.argmax(total_production)
        ax.annotate(f'Peak: {total_production[peak_idx]:.0f} bbl/day',
                   xy=(time_years[peak_idx], total_production[peak_idx]),
                   xytext=(time_years[peak_idx] + 0.5, total_production[peak_idx] * 0.9),
                   arrowprops=dict(arrowstyle='->', color='gray'))
    
    def _plot_pressure_profile(self, ax):
        """Plot reservoir pressure profile."""
        pressure = self.results['pressure']
        time_years = self.results['time'] / 365
        
        ax.plot(time_years[:self.historical_end], 
                pressure[:self.historical_end],
                'b-', linewidth=2, label='Historical', alpha=0.8)
        
        ax.plot(time_years[self.historical_end:], 
                pressure[self.historical_end:],
                'r--', linewidth=2, label='Forecast', alpha=0.8)
        
        ax.axvline(x=time_years[self.historical_end], 
                  color='k', linestyle=':', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Time (Years)')
        ax.set_ylabel('Reservoir Pressure (psi)')
        ax.set_title('Pressure Profile')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add pressure depletion annotation
        pressure_depletion = ((pressure[0] - pressure[-1]) / pressure[0]) * 100
        ax.annotate(f'Depletion: {pressure_depletion:.1f}%',
                   xy=(0.05, 0.05), xycoords='axes fraction',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_injection_profile(self, ax):
        """Plot injection profile."""
        if 'injection' in self.results and self.results['injection'].size > 0:
            injection = self.results['injection']
            time_years = self.results['time'] / 365
            
            total_injection = injection.sum(axis=1)
            
            ax.plot(time_years[:self.historical_end], 
                    total_injection[:self.historical_end],
                    'g-', linewidth=2, label='Historical', alpha=0.8)
            
            ax.plot(time_years[self.historical_end:], 
                    total_injection[self.historical_end:],
                    'm--', linewidth=2, label='Forecast', alpha=0.8)
            
            ax.axvline(x=time_years[self.historical_end], 
                      color='k', linestyle=':', alpha=0.5, linewidth=1)
            
            ax.set_xlabel('Time (Years)')
            ax.set_ylabel('Total Injection Rate (bbl/day)')
            ax.set_title('Injection Profile')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Injection Data Available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Injection Profile')
    
    def _plot_economic_analysis(self, ax):
        """Plot economic analysis results."""
        economics = self.results['economics']
        time_years = self.results['time'] / 365
        
        cumulative_cash_flow = np.array(economics['cumulative_cash_flow_usd'])
        
        ax.plot(time_years, cumulative_cash_flow / 1e6, 
                'b-', linewidth=2, label='Cumulative Cash Flow')
        
        # Add NPV line
        ax.axhline(y=economics['npv_usd'] / 1e6, 
                  color='r', linestyle='--', 
                  label=f'NPV: ${economics["npv_usd"]/1e6:.1f}M')
        
        # Add payback period if available
        if economics['payback_period_years']:
            ax.axvline(x=economics['payback_period_years'], 
                      color='g', linestyle=':', 
                      label=f'Payback: {economics["payback_period_years"]:.1f} years')
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('Time (Years)')
        ax.set_ylabel('Cash Flow (Million USD)')
        ax.set_title('Economic Analysis')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_decline_analysis(self, ax):
        """Plot decline curve analysis."""
        production = self.results['production']
        time_years = self.results['time'] / 365
        
        # Select first well for analysis
        well_idx = 0
        if production.shape[1] > 0:
            well_production = production[:, well_idx]
            
            # Only positive production rates
            valid_mask = well_production > 0
            valid_time = time_years[valid_mask]
            valid_prod = well_production[valid_mask]
            
            if len(valid_time) > 10:
                ax.semilogy(valid_time, valid_prod, 'bo', 
                          alpha=0.5, markersize=3, label='Well Production')
                
                # Fit exponential decline
                mask_early = valid_time < valid_time[min(100, len(valid_time)-1)]
                if np.sum(mask_early) > 5:
                    log_prod = np.log(valid_prod[mask_early])
                    coeffs = np.polyfit(valid_time[mask_early], log_prod, 1)
                    
                    fit_line = np.exp(coeffs[1]) * np.exp(coeffs[0] * valid_time)
                    ax.semilogy(valid_time, fit_line, 'r-', linewidth=2, 
                              label=f'Exponential Fit\nDecline: {-coeffs[0]*365:.3f}/year')
        
        ax.set_xlabel('Time (Years)')
        ax.set_ylabel('Production Rate (bbl/day) - Log Scale')
        ax.set_title(f'Decline Curve Analysis (Well {well_idx+1})')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
    
    def _plot_petrophysical_distribution(self, ax):
        """Plot petrophysical property distribution."""
        if 'petrophysical' in self.data:
            petrophysical = self.data['petrophysical']
            
            x = range(len(petrophysical))
            width = 0.2
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            ax.bar([i - width*1.5 for i in x], 
                   petrophysical['Porosity'], 
                   width, label='Porosity', alpha=0.8, color=colors[0])
            
            ax.bar([i - width/2 for i in x], 
                   petrophysical['Permeability'] / 100, 
                   width, label='Permeability/100', alpha=0.8, color=colors[1])
            
            ax.bar([i + width/2 for i in x], 
                   petrophysical['NetThickness'], 
                   width, label='Net Thickness', alpha=0.8, color=colors[2])
            
            ax.bar([i + width*1.5 for i in x], 
                   petrophysical['WaterSaturation'], 
                   width, label='Water Saturation', alpha=0.8, color=colors[3])
            
            ax.set_xlabel('Layer')
            ax.set_ylabel('Property Value')
            ax.set_title('Petrophysical Properties by Layer')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Petrophysical Data Available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Petrophysical Properties')
    
    def _plot_correlation_matrix(self, ax):
        """Plot correlation matrix."""
        if 'petrophysical' in self.data:
            petrophysical = self.data['petrophysical']
            corr_matrix = petrophysical.corr()
            
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_matrix.columns)
            
            # Add correlation values
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
                    ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha='center', va='center', 
                           color=text_color, fontsize=9)
            
            ax.set_title('Petrophysical Parameter Correlations')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'No Data for Correlation Matrix', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Correlation Matrix')
    
    def _plot_material_balance(self, ax):
        """Plot material balance analysis."""
        if 'pressure' in self.data and 'production' in self.data:
            production_total = self.data['production'].sum(axis=1).values
            pressure = self.data['pressure']
            
            # Calculate cumulative production
            time = self.data['time']
            cumulative_prod = np.zeros(len(time))
            
            for i in range(1, len(time)):
                dt = time[i] - time[i-1]
                cumulative_prod[i] = cumulative_prod[i-1] + production_total[i] * dt
            
            ax.plot(cumulative_prod / 1e6, pressure, 
                    'b-o', markersize=3, linewidth=1, label='Production')
            
            # Add injection if available
            if 'injection' in self.data:
                injection_total = self.data['injection'].sum(axis=1).values
                cumulative_inj = np.zeros(len(time))
                
                for i in range(1, len(time)):
                    dt = time[i] - time[i-1]
                    cumulative_inj[i] = cumulative_inj[i-1] + injection_total[i] * dt
                
                ax.plot(cumulative_inj / 1e6, pressure, 
                        'g-s', markersize=3, linewidth=1, label='Injection')
            
            ax.set_xlabel('Cumulative Volume (Million bbl)')
            ax.set_ylabel('Pressure (psi)')
            ax.set_title('Material Balance Analysis')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient Data for Material Balance', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Material Balance Analysis')
    
    def _plot_recovery_factor(self, ax):
        """Plot recovery factor over time."""
        recovery = self.results['recovery']
        time_years = self.results['time'] / 365
        
        recovery_factor = recovery['recovery_factor_percent']
        
        ax.plot(time_years, recovery_factor, 
                'b-', linewidth=2, label='Recovery Factor')
        
        ax.axvline(x=time_years[self.historical_end], 
                  color='k', linestyle=':', alpha=0.5, linewidth=1)
        
        # Add final recovery factor annotation
        final_rf = recovery_factor[-1]
        ax.annotate(f'Final RF: {final_rf:.1f}%', 
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Time (Years)')
        ax.set_ylabel('Recovery Factor (%)')
        ax.set_title('Recovery Factor Profile')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_sensitivity_analysis(self, ax):
        """Plot sensitivity analysis results."""
        sensitivity = self.results['sensitivity']
        
        if sensitivity:
            n_params = len(sensitivity)
            bar_width = 0.2
            index = np.arange(n_params)
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, (param_name, param_data) in enumerate(sensitivity.items()):
                npv_values = param_data['npv_usd']
                positions = index[i] + np.arange(len(npv_values)) * bar_width
                
                bars = ax.bar(positions, npv_values, bar_width,
                            label=param_name.replace('_', ' ').title(),
                            alpha=0.7, color=colors[i % len(colors)])
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'${height/1e6:.1f}M',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3), textcoords="offset points",
                              ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Parameters')
            ax.set_ylabel('NPV (USD)')
            ax.set_title('Sensitivity Analysis - NPV Impact')
            ax.set_xticks(index + bar_width * (len(sensitivity) - 1) / 2)
            ax.set_xticklabels([name.replace('_', ' ').title() 
                              for name in sensitivity.keys()], rotation=45)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No Sensitivity Analysis Available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Sensitivity Analysis')
    
    def create_interactive_dashboard(self, save_path: Optional[str] = None):
        """
        Create interactive dashboard using Plotly.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the interactive HTML dashboard
        """
        logger.info("Creating interactive dashboard")
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Production Profile', 'Pressure Profile', 'Injection Profile',
                          'Economic Analysis', 'Decline Analysis', 'Recovery Factor',
                          'Sensitivity Analysis', 'Material Balance', 'Correlation Matrix'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
            specs=[[{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'xy', 'colspan': 2}, None, {'type': 'heatmap'}]]
        )
        
        # Add traces for each subplot
        self._add_interactive_traces(fig)
        
        # Update layout
        fig.update_layout(
            title_text='Interactive Reservoir Simulation Dashboard',
            title_font_size=24,
            height=1200,
            showlegend=True,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        fig.show()
    
    def _add_interactive_traces(self, fig):
        """Add interactive traces to Plotly figure."""
        time_years = self.results['time'] / 365
        
        # Production Profile (1,1)
        production = self.results['production'].sum(axis=1)
        fig.add_trace(
            go.Scatter(x=time_years[:self.historical_end], 
                      y=production[:self.historical_end],
                      mode='lines', name='Historical Production',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_years[self.historical_end:], 
                      y=production[self.historical_end:],
                      mode='lines', name='Forecast Production',
                      line=dict(color='red', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Pressure Profile (1,2)
        pressure = self.results['pressure']
        fig.add_trace(
            go.Scatter(x=time_years[:self.historical_end], 
                      y=pressure[:self.historical_end],
                      mode='lines', name='Historical Pressure',
                      line=dict(color='green', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=time_years[self.historical_end:], 
                      y=pressure[self.historical_end:],
                      mode='lines', name='Forecast Pressure',
                      line=dict(color='orange', width=2, dash='dash')),
            row=1, col=2
        )
        
        # Update axes labels
        fig.update_xaxes(title_text='Time (Years)', row=1, col=1)
        fig.update_yaxes(title_text='Production Rate (bbl/day)', row=1, col=1)
        fig.update_xaxes(title_text='Time (Years)', row=1, col=2)
        fig.update_yaxes(title_text='Pressure (psi)', row=1, col=2)
        
        # Add more traces for other subplots...
        # This is a simplified version - implement full version as needed
    
    def export_individual_plots(self, output_dir: str = './outputs/plots/'):
        """
        Export individual plots as separate files.
        
        Parameters
        ----------
        output_dir : str
            Directory to save plot files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Exporting individual plots to {output_dir}")
        
        # Production profile
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_production_profile(ax)
        plt.savefig(os.path.join(output_dir, 'production_profile.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Pressure profile
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_pressure_profile(ax)
        plt.savefig(os.path.join(output_dir, 'pressure_profile.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Economic analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_economic_analysis(ax)
        plt.savefig(os.path.join(output_dir, 'economic_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Individual plots exported successfully")
