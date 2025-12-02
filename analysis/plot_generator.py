"""
Generate plots and visualizations from simulation results
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

logger = logging.getLogger(__name__)


class PlotGenerator:
    """Generate visualization plots from simulation results"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'oil': '#FF6B6B',
            'gas': '#4ECDC4',
            'water': '#556270',
            'total': '#C44D58',
            'injector': '#3498DB',
            'producer': '#2ECC71'
        }
        
        logger.info(f"📊 Plot generator initialized. Output directory: {self.output_dir}")
        
    def create_production_profile(self, summary_data: pd.DataFrame) -> str:
        """Create production profile plot"""
        if summary_data.empty or 'DATE' not in summary_data.columns:
            logger.warning("⚠️ No data available for production profile plot")
            return ""
            
        logger.info("🖼️ Creating production profile plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SPE9 Production Profile', fontsize=16, fontweight='bold', y=1.02)
        
        try:
            # Oil production
            if 'FOPR' in summary_data.columns:
                axes[0, 0].plot(summary_data['DATE'], summary_data['FOPR'], 
                              color=self.colors['oil'], linewidth=2, label='Oil Rate')
                axes[0, 0].fill_between(summary_data['DATE'], 0, summary_data['FOPR'],
                                       color=self.colors['oil'], alpha=0.2)
                axes[0, 0].set_title('Oil Production Rate', fontsize=12, fontweight='bold')
                axes[0, 0].set_ylabel('STB/Day', fontsize=10)
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend(loc='best')
                
                # Add average line
                avg_oil = summary_data['FOPR'].mean()
                axes[0, 0].axhline(y=avg_oil, color='red', linestyle='--', alpha=0.5, 
                                  label=f'Avg: {avg_oil:.0f} STB/D')
                axes[0, 0].legend(loc='best')
            
            # Gas production
            if 'FGPR' in summary_data.columns:
                axes[0, 1].plot(summary_data['DATE'], summary_data['FGPR'], 
                              color=self.colors['gas'], linewidth=2, label='Gas Rate')
                axes[0, 1].fill_between(summary_data['DATE'], 0, summary_data['FGPR'],
                                       color=self.colors['gas'], alpha=0.2)
                axes[0, 1].set_title('Gas Production Rate', fontsize=12, fontweight='bold')
                axes[0, 1].set_ylabel('MSCF/Day', fontsize=10)
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend(loc='best')
            
            # Water production
            if 'FWPR' in summary_data.columns:
                axes[1, 0].plot(summary_data['DATE'], summary_data['FWPR'], 
                              color=self.colors['water'], linewidth=2, label='Water Rate')
                axes[1, 0].fill_between(summary_data['DATE'], 0, summary_data['FWPR'],
                                       color=self.colors['water'], alpha=0.2)
                axes[1, 0].set_title('Water Production Rate', fontsize=12, fontweight='bold')
                axes[1, 0].set_ylabel('STB/Day', fontsize=10)
                axes[1, 0].set_xlabel('Time', fontsize=10)
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend(loc='best')
            
            # Cumulative oil
            if 'FOPT' in summary_data.columns:
                axes[1, 1].plot(summary_data['DATE'], summary_data['FOPT'], 
                              color=self.colors['total'], linewidth=2, label='Cumulative Oil')
                axes[1, 1].set_title('Cumulative Oil Production', fontsize=12, fontweight='bold')
                axes[1, 1].set_ylabel('STB', fontsize=10)
                axes[1, 1].set_xlabel('Time', fontsize=10)
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend(loc='best')
                
                # Add final value annotation
                final_oil = summary_data['FOPT'].iloc[-1]
                axes[1, 1].annotate(f'Final: {final_oil:,.0f} STB',
                                   xy=(summary_data['DATE'].iloc[-1], final_oil),
                                   xytext=(10, 10), textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            output_path = self.output_dir / "production_profile.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"✅ Production profile saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error creating production profile: {e}")
            plt.close(fig)
            return ""
        
    def create_well_performance_chart(self, summary_data: pd.DataFrame) -> str:
        """Create well performance comparison chart"""
        # Extract well production data
        well_oil_cols = [col for col in summary_data.columns if 'WOPR' in col]
        if not well_oil_cols:
            logger.warning("⚠️ No well production data found for performance chart")
            return ""
            
        logger.info(f"🖼️ Creating well performance chart for {len(well_oil_cols)} wells...")
        
        # Get average production per well
        avg_production = summary_data[well_oil_cols].mean().sort_values(ascending=False)
        
        # Limit to top 15 wells for readability
        if len(avg_production) > 15:
            avg_production = avg_production.head(15)
            
        fig, ax = plt.subplots(figsize=(14, 6))
        
        try:
            # Create bar chart with gradient colors
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(avg_production)))
            bars = ax.bar(range(len(avg_production)), avg_production.values, color=colors)
            
            # Add labels
            ax.set_xlabel('Well Name', fontsize=11)
            ax.set_ylabel('Average Oil Rate (STB/Day)', fontsize=11)
            ax.set_title('Top Well Performance Comparison', fontsize=13, fontweight='bold')
            
            # Format x-axis
            well_names = []
            for name in avg_production.index:
                if ':' in name:
                    well_names.append(name.split(':')[1])
                else:
                    well_names.append(name.replace('WOPR', ''))
                    
            ax.set_xticks(range(len(well_names)))
            ax.set_xticklabels(well_names, rotation=45, ha='right', fontsize=9)
            
            # Add value labels on bars
            for i, (bar, v) in enumerate(zip(bars, avg_production.values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(avg_production.values)*0.01,
                       f'{v:,.0f}', ha='center', va='bottom', fontsize=8)
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add summary statistics
            total_avg = avg_production.mean()
            ax.axhline(y=total_avg, color='red', linestyle='--', alpha=0.7, 
                      label=f'Average: {total_avg:,.0f} STB/D')
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            
            output_path = self.output_dir / "well_performance.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"✅ Well performance chart saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error creating well performance chart: {e}")
            plt.close(fig)
            return ""
        
    def create_water_cut_plot(self, summary_data: pd.DataFrame) -> str:
        """Create water cut development plot"""
        if 'FWCT' not in summary_data.columns or 'DATE' not in summary_data.columns:
            logger.warning("⚠️ No water cut data available for plot")
            return ""
            
        logger.info("🖼️ Creating water cut development plot...")
        
              # Line 198 should be:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Or for multi-line clarity:
        fig, ax = plt.subplots(
            figsize=(10, 6),
            constrained_layout=True
        )
