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
                   width, label='Permeability/100', alpha
