"""
Plot Generator - Fixed Version
Handles all common errors
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np

logger = logging.getLogger(__name__)

class PlotGenerator:
    """Generate plots for simulation results."""
    
    def __init__(self, results: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None):
        """
        Initialize plot generator.
        
        Args:
            results: Simulation results
            metrics: Performance metrics (optional)
        """
        self.results = results if results else {}
        self.metrics = metrics if metrics else {}
        
        logger.info("📈 Plot Generator initialized")
    
    def create_pressure_plot(self):
        """Create pressure distribution plot."""
        try:
            import matplotlib.pyplot as plt
            
            if 'pressure' not in self.results:
                logger.warning("No pressure data for plotting")
                return None
            
            pressure_data = self.results['pressure']
            
            # Handle different data formats
            if isinstance(pressure_data, list):
                if len(pressure_data) > 0:
                    # Flatten if nested
                    if isinstance(pressure_data[0], list):
                        flat_data = []
                        for sublist in pressure_data:
                            if isinstance(sublist, list):
                                flat_data.extend(sublist)
                            else:
                                flat_data.append(sublist)
                        plot_data = flat_data
                    else:
                        plot_data = pressure_data
                else:
                    logger.warning("Empty pressure data")
                    return None
            else:
                logger.warning(f"Unexpected pressure data type: {type(pressure_data)}")
                return None
            
            # Limit data for plotting
            max_points = 1000
            if len(plot_data) > max_points:
                plot_data = plot_data[:max_points]
                logger.debug(f"Truncated pressure data to {max_points} points")
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(plot_data, 'b-', alpha=0.7, linewidth=1)
            ax.set_xlabel('Data Point Index', fontsize=12)
            ax.set_ylabel('Pressure', fontsize=12)
            ax.set_title('Reservoir Pressure Distribution', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add statistics if available
            stats_text = f"Points: {len(plot_data)}"
            if len(plot_data) > 0:
                stats_text += f"\nMean: {np.mean(plot_data):.2f}"
                stats_text += f"\nStd: {np.std(plot_data):.2f}"
            
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            logger.info("✅ Created pressure plot")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pressure plot: {e}")
            return None
    
    def create_production_plot(self):
        """Create production history plot."""
        try:
            import matplotlib.pyplot as plt
            
            if 'production' not in self.results:
                logger.warning("No production data for plotting")
                return None
            
            prod_data = self.results['production']
            if not isinstance(prod_data, dict):
                logger.warning(f"Production data is not a dict: {type(prod_data)}")
                return None
            
            # Collect available phases
            phases_to_plot = []
            phase_data = {}
            
            for phase in ['oil', 'water', 'gas']:
                if phase in prod_data:
                    data = prod_data[phase]
                    if isinstance(data, list) and len(data) > 0:
                        phases_to_plot.append(phase)
                        phase_data[phase] = data
            
            if not phases_to_plot:
                logger.warning("No valid production phase data")
                return None
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = {'oil': 'green', 'water': 'blue', 'gas': 'red'}
            labels = {'oil': 'Oil', 'water': 'Water', 'gas': 'Gas'}
            
            for phase in phases_to_plot:
                data = phase_data[phase]
                # Limit data points
                max_points = 500
                if len(data) > max_points:
                    data = data[:max_points]
                
                ax.plot(data, color=colors.get(phase, 'black'),
                       label=labels.get(phase, phase),
                       linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('Production Rate', fontsize=12)
            ax.set_title('Production History', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add total production if metrics available
            if self.metrics:
                metrics_text = "Total Production:\n"
                for phase in phases_to_plot:
                    key = f'total_{phase}_produced'
                    if key in self.metrics:
                        metrics_text += f"{phase}: {self.metrics[key]:.0f}\n"
                
                ax.text(0.02, 0.98, metrics_text,
                       transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            logger.info(f"✅ Created production plot with {len(phases_to_plot)} phases")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating production plot: {e}")
            return None
    
    def create_saturation_plot(self):
        """Create fluid saturation plot."""
        try:
            import matplotlib.pyplot as plt
            
            # Look for saturation data
            sat_keys = ['saturation_oil', 'saturation_water', 'saturation_gas']
            available_sat = {}
            
            for key in sat_keys:
                if key in self.results:
                    data = self.results[key]
                    if isinstance(data, list) and len(data) > 0:
                        available_sat[key] = data
            
            if not available_sat:
                logger.warning("No saturation data for plotting")
                return None
            
            # Create plot
            fig, axes = plt.subplots(1, len(available_sat), figsize=(5*len(available_sat), 6))
            if len(available_sat) == 1:
                axes = [axes]
            
            titles = {
                'saturation_oil': 'Oil Saturation',
                'saturation_water': 'Water Saturation', 
                'saturation_gas': 'Gas Saturation'
            }
            
            for idx, (key, data) in enumerate(available_sat.items()):
                ax = axes[idx]
                
                # Take first time step if 2D
                if isinstance(data[0], list):
                    plot_data = data[0] if len(data) > 0 else []
                else:
                    plot_data = data
                
                if plot_data:
                    # Limit points
                    max_points = 500
                    if len(plot_data) > max_points:
                        plot_data = plot_data[:max_points]
                    
                    ax.plot(plot_data, 'purple', alpha=0.7, linewidth=2)
                    ax.set_xlabel('Cell Index')
                    ax.set_ylabel('Saturation')
                    ax.set_title(titles.get(key, key), fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    if plot_data:
                        ax.text(0.02, 0.98, f"Mean: {np.mean(plot_data):.3f}",
                               transform=ax.transAxes,
                               verticalalignment='top')
            
            plt.suptitle('Fluid Saturation Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            logger.info(f"✅ Created saturation plot with {len(available_sat)} fluids")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating saturation plot: {e}")
            return None
    
    def create_metrics_summary_plot(self):
        """Create summary plot of key metrics."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.metrics:
                logger.warning("No metrics for summary plot")
                return None
            
            # Select numeric metrics for bar plot
            numeric_metrics = {}
            for key, value in self.metrics.items():
                if isinstance(value, (int, float)) and not key.endswith('_count'):
                    # Shorten long keys for display
                    display_key = key.replace('_', ' ').title()
                    numeric_metrics[display_key] = float(value)
            
            if len(numeric_metrics) < 2:
                logger.warning("Not enough numeric metrics for summary plot")
                return None
            
            # Take top metrics
            top_metrics = dict(list(numeric_metrics.items())[:8])
            
            # Create bar plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            keys = list(top_metrics.keys())
            values = list(top_metrics.values())
            
            bars = ax.bar(range(len(keys)), values, color='steelblue', alpha=0.8)
            ax.set_xlabel('Metrics', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Key Performance Metrics', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom')
            
            plt.tight_layout()
            logger.info(f"✅ Created metrics summary plot with {len(top_metrics)} metrics")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating metrics plot: {e}")
            return None
    
    # Aliases for backward compatibility
    def plot_pressure_distribution(self):
        return self.create_pressure_plot()
    
    def plot_production_history(self):
        return self.create_production_plot()
    
    def plot_saturation(self):
        return self.create_saturation_plot()
    
    def plot_metrics(self):
        return self.create_metrics_summary_plot()
    
    def generate_plots(self):
        """Generate all available plots."""
        plots = {}
        
        plot_methods = [
            ('pressure', self.create_pressure_plot),
            ('production', self.create_production_plot),
            ('saturation', self.create_saturation_plot),
            ('metrics', self.create_metrics_summary_plot)
        ]
        
        for name, method in plot_methods:
            try:
                plot = method()
                if plot:
                    plots[name] = plot
                    logger.info(f"Generated {name} plot")
            except Exception as e:
                logger.warning(f"Failed to generate {name} plot: {e}")
        
        return {'figures': list(plots.values()), 'count': len(plots)}
