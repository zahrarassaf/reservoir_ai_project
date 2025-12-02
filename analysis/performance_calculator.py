"""
Calculate reservoir performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceCalculator:
    """Calculate comprehensive performance metrics"""
    
    def __init__(self, summary_data: pd.DataFrame):
        self.data = summary_data
        self.metrics: Dict[str, float] = {}
        
    def calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate all performance metrics"""
        logger.info("📊 Calculating comprehensive performance metrics...")
        
        try:
            self._calculate_production_metrics()
            self._calculate_economic_metrics()
            self._calculate_reservoir_metrics()
            self._calculate_well_metrics()
            
            logger.info(f"✅ Calculated {len(self.metrics)} performance metrics")
            return self.metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics: {e}")
            return {}
        
    def _calculate_production_metrics(self) -> None:
        """Calculate production-related metrics"""
        if self.data.empty:
            logger.warning("⚠️ No data available for production metrics")
            return
            
        logger.info("  Calculating production metrics...")
        
        # Cumulative production
        if 'FOPT' in self.data.columns:
            self.metrics['cumulative_oil'] = float(self.data['FOPT'].iloc[-1])
        if 'FGPT' in self.data.columns:
            self.metrics['cumulative_gas'] = float(self.data['FGPT'].iloc[-1])
        if 'FWPT' in self.data.columns:
            self.metrics['cumulative_water'] = float(self.data['FWPT'].iloc[-1])
        
        # Peak rates
        if 'FOPR' in self.data.columns:
            self.metrics['peak_oil_rate'] = float(self.data['FOPR'].max())
            self.metrics['avg_oil_rate'] = float(self.data['FOPR'].mean())
        if 'FGPR' in self.data.columns:
            self.metrics['peak_gas_rate'] = float(self.data['FGPR'].max())
            self.metrics['avg_gas_rate'] = float(self.data['FGPR'].mean())
        if 'FWPR' in self.data.columns:
            self.metrics['peak_water_rate'] = float(self.data['FWPR'].max())
        
        # Final rates
        if not self.data.empty:
            final_data = self.data.iloc[-1]
            if 'FOPR' in final_data:
                self.metrics['final_oil_rate'] = float(final_data['FOPR'])
            if 'FWCT' in final_data:
                self.metrics['final_water_cut'] = float(final_data['FWCT'])
            if 'FGOR' in final_data:
                self.metrics['final_gor'] = float(final_data['FGOR'])
                
        logger.info(f"    ✓ Production metrics: {len([k for k in self.metrics if 'oil' in k or 'gas' in k or 'water' in k])} calculated")
        
    def _calculate_economic_metrics(self) -> None:
        """Calculate economic indicators"""
        if 'cumulative_oil' not in self.metrics:
            return
            
        logger.info("  Calculating economic metrics...")
        
        # Price assumptions (example values - can be configured)
        oil_price = 60.0  # $/bbl
        gas_price = 3.0   # $/MCF
        water_disposal_cost = 2.0  # $/bbl
        operating_cost = 20.0  # $/bbl
        
        # Revenue calculations
        oil_revenue = self.metrics['cumulative_oil'] * oil_price
        gas_revenue = self.metrics.get('cumulative_gas', 0) * gas_price / 1000  # Convert MSCF to MCF
        water_cost = self.metrics.get('cumulative_water', 0) * water_disposal_cost
        operating_cost_total = self.metrics['cumulative_oil'] * operating_cost
        
        self.metrics['gross_revenue'] = oil_revenue + gas_revenue
        self.metrics['net_revenue'] = self.metrics['gross_revenue'] - water_cost - operating_cost_total
        
        if self.metrics['cumulative_oil'] > 0:
            self.metrics['revenue_per_barrel'] = self.metrics['net_revenue'] / self.metrics['cumulative_oil']
        else:
            self.metrics['revenue_per_barrel'] = 0
            
        logger.info("    ✓ Economic metrics calculated")
        
    def _calculate_reservoir_metrics(self) -> None:
        """Calculate reservoir engineering metrics"""
        if 'cumulative_oil' not in self.metrics:
            return
            
        logger.info("  Calculating reservoir metrics...")
        
        # From SPE9 specification
        initial_oil_in_place = 7.758e7  # STB
        initial_gas_in_place = 1.0e8    # MSCF (approximate)
        
        # Recovery factors
        self.metrics['oil_recovery_factor'] = (self.metrics['cumulative_oil'] / initial_oil_in_place) * 100
        if 'cumulative_gas' in self.metrics:
            self.metrics['gas_recovery_factor'] = (self.metrics['cumulative_gas'] / initial_gas_in_place) * 100
        
        # Decline analysis (simplified)
        if 'FOPR' in self.data.columns and len(self.data) > 12:
            oil_rates = self.data['FOPR'].values
            if oil_rates[-1] > 0 and oil_rates[0] > 0 and len(oil_rates) > 1:
                try:
                    decline_rate = (oil_rates[0] / oil_rates[-1]) ** (1/len(oil_rates)) - 1
                    self.metrics['annual_decline_rate'] = decline_rate * 365 * 100  # Convert to annual percentage
                except:
                    self.metrics['annual_decline_rate'] = 0
                    
        logger.info(f"    ✓ Reservoir metrics: Recovery factor = {self.metrics.get('oil_recovery_factor', 0):.2f}%")
        
    def _calculate_well_metrics(self) -> None:
        """Calculate well performance metrics"""
        # Extract well production data
        well_oil_cols = [col for col in self.data.columns if 'WOPR' in col]
        
        if not well_oil_cols:
            logger.warning("    ⚠️ No well production data found for metrics")
            return
            
        logger.info(f"  Calculating well metrics from {len(well_oil_cols)} wells...")
        
        # Average well productivity
        avg_well_oil_rate = self.data[well_oil_cols].mean().mean()
        self.metrics['avg_well_oil_productivity'] = float(avg_well_oil_rate)
        
        # Well efficiency (variation)
        if avg_well_oil_rate > 0:
            well_variation = self.data[well_oil_cols].std().mean() / avg_well_oil_rate
            self.metrics['well_productivity_variation'] = float(well_variation)
        else:
            self.metrics['well_productivity_variation'] = 0
            
        # Top 3 producers
        well_avg_rates = self.data[well_oil_cols].mean().sort_values(ascending=False)
        if len(well_avg_rates) >= 3:
            for i in range(3):
                well_name = well_avg_rates.index[i].replace('WOPR:', '').replace('WOPR', '')
                self.metrics[f'top_producer_{i+1}'] = float(well_avg_rates.iloc[i])
                self.metrics[f'top_producer_{i+1}_name'] = well_name
                
        logger.info(f"    ✓ Well metrics: Average productivity = {avg_well_oil_rate:.0f} STB/D")
        
    def generate_detailed_report(self) -> pd.DataFrame:
        """Generate detailed metrics report as DataFrame"""
        if not self.metrics:
            self.calculate_all_metrics()
            
        metrics_df = pd.DataFrame.from_dict(self.metrics, orient='index', columns=['Value'])
        metrics_df['Unit'] = self._get_metric_units()
        metrics_df['Description'] = self._get_metric_descriptions()
        metrics_df['Category'] = self._get_metric_categories()
        
        # Sort by category
        metrics_df = metrics_df.sort_values('Category')
        
        return metrics_df
        
    def _get_metric_units(self) -> pd.Series:
        """Get units for each metric"""
        units_map = {
            'cumulative_oil': 'STB',
            'cumulative_gas': 'MSCF',
            'cumulative_water': 'STB',
            'peak_oil_rate': 'STB/D',
            'peak_gas_rate': 'MSCF/D',
            'peak_water_rate': 'STB/D',
            'avg_oil_rate': 'STB/D',
            'avg_gas_rate': 'MSCF/D',
            'final_oil_rate': 'STB/D',
            'final_water_cut': 'fraction',
            'final_gor': 'SCF/STB',
            'oil_recovery_factor': '%',
            'gas_recovery_factor': '%',
            'annual_decline_rate': '%/year',
            'gross_revenue': 'USD',
            'net_revenue': 'USD',
            'revenue_per_barrel': 'USD/STB',
            'avg_well_oil_productivity': 'STB/D/well',
            'well_productivity_variation': 'fraction',
        }
        
        # Create series with same index as metrics
        return pd.Series({k: units_map.get(k, '') for k in self.metrics.keys()})
        
    def _get_metric_descriptions(self) -> pd.Series:
        """Get descriptions for each metric"""
        descriptions_map = {
            'cumulative_oil': 'Total oil produced over simulation period',
            'cumulative_gas': 'Total gas produced over simulation period',
            'cumulative_water': 'Total water produced over simulation period',
            'peak_oil_rate': 'Maximum daily oil production rate',
            'peak_gas_rate': 'Maximum daily gas production rate',
            'peak_water_rate': 'Maximum daily water production rate',
            'avg_oil_rate': 'Average daily oil production rate',
            'avg_gas_rate': 'Average daily gas production rate',
            'final_oil_rate': 'Oil production rate at end of simulation',
            'final_water_cut': 'Water cut (water/oil ratio) at end of simulation',
            'final_gor': 'Gas-oil ratio at end of simulation',
            'oil_recovery_factor': 'Percentage of original oil in place recovered',
            'gas_recovery_factor': 'Percentage of original gas in place recovered',
            'annual_decline_rate': 'Estimated annual production decline rate',
            'gross_revenue': 'Total revenue from oil and gas sales',
            'net_revenue': 'Revenue after water disposal and operating costs',
            'revenue_per_barrel': 'Net revenue per barrel of oil produced',
            'avg_well_oil_productivity': 'Average oil production per well',
            'well_productivity_variation': 'Variation in well productivity (std/mean)',
        }
        
        # Add descriptions for top producers
        for i in range(1, 4):
            if f'top_producer_{i}' in self.metrics:
                descriptions_map[f'top_producer_{i}'] = f'Production rate of {i}rd top producer'
            if f'top_producer_{i}_name' in self.metrics:
                descriptions_map[f'top_producer_{i}_name'] = f'Name of {i}rd top producer'
        
        return pd.Series({k: descriptions_map.get(k, '') for k in self.metrics.keys()})
        
    def _get_metric_categories(self) -> pd.Series:
        """Get categories for each metric"""
        categories = {}
        
        for metric in self.metrics.keys():
            if any(word in metric for word in ['oil', 'gas', 'water', 'rate', 'production']):
                categories[metric] = 'Production'
            elif any(word in metric for word in ['revenue', 'cost', 'economic']):
                categories[metric] = 'Economic'
            elif any(word in metric for word in ['recovery', 'decline', 'reservoir']):
                categories[metric] = 'Reservoir'
            elif any(word in metric for word in ['well', 'producer', 'productivity']):
                categories[metric] = 'Well Performance'
            else:
                categories[metric] = 'Other'
                
        return pd.Series(categories)
        
    def save_metrics_to_csv(self, filepath: str = "results/analysis_results/detailed_metrics.csv") -> None:
        """Save metrics to CSV file"""
        metrics_df = self.generate_detailed_report()
        
        # Ensure directory exists
        from pathlib import Path
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metrics_df.to_csv(output_path)
        logger.info(f"📄 Detailed metrics saved to: {output_path}")
