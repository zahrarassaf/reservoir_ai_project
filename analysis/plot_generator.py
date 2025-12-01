"""
Generate plots and visualizations from simulation results
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PlotGenerator:
    """Generate visualization plots from simulation results"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "plots"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'oil': '#FF6B6B',
            'gas': '#4ECDC4',
            'water': '#556270',
            'total': '#C44D58'
        }
        
    def create_production_profile(self, summary_data: pd.DataFrame) -> str:
        """Create production profile plot"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Oil production
        if 'FOPR' in summary_data.columns and 'DATE' in summary_data.columns:
            axes[0, 0].plot(summary_data['DATE'], summary_data['FOPR'], 
                          color=self.colors['oil'], linewidth=2)
            axes[0, 0].set_title('Oil Production Rate', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('STB/Day', fontsize=10)
            axes[0, 0].grid(True, alpha=0.3)
            
        # Gas production
        if 'FGPR' in summary_data.columns:
            axes[0, 1].plot(summary_data['DATE'], summary_data['FGPR'], 
                          color=self.colors['gas'], linewidth=2)
            axes[0, 1].set_title('Gas Production Rate', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('MSCF/Day', fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)
            
        # Water production
        if 'FWPR' in summary_data.columns:
            axes[1, 0].plot(summary_data['DATE'], summary_data['FWPR'], 
                          color=self.colors['water'], linewidth=2)
            axes[1, 0].set_title('Water Production Rate', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('STB/Day', fontsize=10)
            axes[1, 0].set_xlabel('Time', fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)
            
        # Cumulative oil
        if 'FOPT' in summary_data.columns:
            axes[1, 1].plot(summary_data['DATE'], summary_data['FOPT'], 
                          color=self.colors['total'], linewidth=2)
            axes[1, 1].set_title('Cumulative Oil Production', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('STB', fontsize=10)
            axes[1, 1].set_xlabel('Time', fontsize=10)
            axes[1, 1].grid(True, alpha=0.3)
            
        plt.suptitle('SPE9 Production Profile', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = self.output_dir / "production_profile.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Production profile saved to: {output_path}")
        return str(output_path)
        
    def create_well_performance_chart(self, summary_data: pd.DataFrame) -> str:
        """Create well performance comparison chart"""
        # Extract well production data
        well_oil_cols = [col for col in summary_data.columns if 'WOPR' in col]
        if not well_oil_cols:
            return ""
            
        # Get average production per well
        avg_production = summary_data[well_oil_cols].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar chart
        bars = ax.bar(range(len(avg_production)), avg_production.values, 
                     color=plt.cm.viridis(np.linspace(0, 1, len(avg_production))))
        
        # Add labels
        ax.set_xlabel('Well', fontsize=11)
        ax.set_ylabel('Average Oil Rate (STB/Day)', fontsize=11)
        ax.set_title('Well Performance Comparison', fontsize=13, fontweight='bold')
        
        # Format x-axis
        well_names = [name.replace('WOPR:', '').replace('WOPR', '') for name in avg_production.index]
        ax.set_xticks(range(len(well_names)))
        ax.set_xticklabels(well_names, rotation=45, ha='right', fontsize=9)
        
        # Add value labels on bars
        for i, v in enumerate(avg_production.values):
            ax.text(i, v + max(avg_production.values) * 0.01, 
                   f'{v:,.0f}', ha='center', va='bottom', fontsize=8)
            
        plt.tight_layout()
        
        output_path = self.output_dir / "well_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Well performance chart saved to: {output_path}")
        return str(output_path)
        
    def create_water_cut_plot(self, summary_data: pd.DataFrame) -> str:
        """Create water cut development plot"""
        if 'FWCT' not in summary_data.columns or 'DATE' not in summary_data.columns:
            return ""
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(summary_data['DATE'], summary_data['FWCT'] * 100, 
                color='#3498DB', linewidth=2.5)
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Water Cut (%)', fontsize=11)
        ax.set_title('Water Cut Development', fontsize=13, fontweight='bold')
        
        # Add grid and styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Fill under curve
        ax.fill_between(summary_data['DATE'], 0, summary_data['FWCT'] * 100,
                       alpha=0.2, color='#3498DB')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "water_cut.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Water cut plot saved to: {output_path}")
        return str(output_path)
        
    def create_recovery_factor_plot(self, summary_data: pd.DataFrame, ooip: float = 7.758e7) -> str:
        """Create recovery factor plot"""
        if 'FOPT' not in summary_data.columns or 'DATE' not in summary_data.columns:
            return ""
            
        recovery_factor = (summary_data['FOPT'] / ooip) * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(summary_data['DATE'], recovery_factor, 
                color='#9B59B6', linewidth=2.5, label='Oil Recovery Factor')
        
        # Add milestone markers
        milestones = [5, 10, 15, 20]  # Percent recovery milestones
        for milestone in milestones:
            if recovery_factor.max() >= milestone:
                idx = (recovery_factor >= milestone).idxmax()
                if idx:
                    ax.plot(summary_data['DATE'].iloc[idx], milestone, 
                           'o', color='#E74C3C', markersize=8)
                    ax.text(summary_data['DATE'].iloc[idx], milestone + 0.5,
                           f'{milestone}%', ha='center', fontsize=9, fontweight='bold')
                    
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Recovery Factor (%)', fontsize=11)
        ax.set_title('Oil Recovery Factor Development', fontsize=13, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "recovery_factor.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Recovery factor plot saved to: {output_path}")
        return str(output_path)
        
    def create_all_plots(self, summary_data: pd.DataFrame) -> Dict[str, str]:
        """Create all standard plots"""
        plots = {}
        
        plot_methods = [
            ('production_profile', self.create_production_profile),
            ('well_performance', self.create_well_performance_chart),
            ('water_cut', self.create_water_cut_plot),
            ('recovery_factor', self.create_recovery_factor_plot),
        ]
        
        for plot_name, plot_method in plot_methods:
            try:
                plot_path = plot_method(summary_data)
                if plot_path:
                    plots[plot_name] = plot_path
            except Exception as e:
                logger.error(f"Failed to create {plot_name}: {e}")
                
        logger.info(f"Created {len(plots)} plots")
        return plots
