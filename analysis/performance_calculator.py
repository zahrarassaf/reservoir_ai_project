"""
Calculate reservoir performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceCalculator:
    """Calculate comprehensive performance metrics"""
    
    def __init__(self, summary_data: pd.DataFrame):
        self.data = summary_data
        self.metrics = {}
        
    def calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate all performance metrics"""
        self._calculate_production_metrics()
        self._calculate_economic_metrics()
        self._calculate_reservoir_metrics()
        self._calculate_well_metrics()
        
        logger.info(f"Calculated {len(self.metrics)} performance metrics")
        return self.metrics
        
    def _calculate_production_metrics(self) -> None:
        """Calculate production-related metrics"""
        if self.data.empty:
            return
            
        # Cumulative production
        self.metrics['cumulative_oil'] = self.data['FOPT'].max()
        self.metrics['cumulative_gas'] = self.data['FGPT'].max()
        self.metrics['cumulative_water'] = self.data['FWPT'].max()
        
        # Peak rates
        self.metrics['peak_oil_rate'] = self.data['FOPR'].max()
        self.metrics['peak_gas_rate'] = self.data['FGPR'].max()
        self.metrics['peak_water_rate'] = self.data['FWPR'].max()
        
        # Average rates
        self.metrics['avg_oil_rate'] = self.data['FOPR'].mean()
        self.metrics['avg_gas_rate'] = self.data['FGPR'].mean()
        
        # Final rates
        final_data = self.data.iloc[-1]
        self.metrics['final_oil_rate'] = final_data['FOPR']
        self.metrics['final_water_cut'] = final_data.get('FWCT', 0)
        self.metrics['final_gor'] = final_data.get('FGOR', 0)
        
    def _calculate_economic_metrics(self) -> None:
        """Calculate economic indicators"""
        # Price assumptions (example values)
        oil_price = 60.0  # $/bbl
        gas_price = 3.0   # $/MCF
        water_disposal_cost = 2.0  # $/bbl
        operating_cost = 20.0  # $/bbl
        
        # Revenue calculations
        oil_revenue = self.metrics['cumulative_oil'] * oil_price
        gas_revenue = self.metrics['cumulative_gas'] * gas_price / 1000  # Convert MSCF to MCF
        water_cost = self.metrics['cumulative_water'] * water_disposal_cost
        operating_cost_total = self.metrics['cumulative_oil'] * operating_cost
        
        self.metrics['gross_revenue'] = oil_revenue + gas_revenue
        self.metrics['net_revenue'] = self.metrics['gross_revenue'] - water_cost - operating_cost_total
        self.metrics['revenue_per_barrel'] = self.metrics['net_revenue'] / self.metrics['cumulative_oil'] if self.metrics['cumulative_oil'] > 0 else 0
        
    def _calculate_reservoir_metrics(self) -> None:
        """Calculate reservoir engineering metrics"""
        # From SPE9 specification
        initial_oil_in_place = 7.758e7  # STB
        initial_gas_in_place = 1.0e8    # MSCF (approximate)
        
        # Recovery factors
        self.metrics['oil_recovery_factor'] = (self.metrics['cumulative_oil'] / initial_oil_in_place) * 100
        self.metrics['gas_recovery_factor'] = (self.metrics['cumulative_gas'] / initial_gas_in_place) * 100
        
        # Decline analysis (simplified)
        if len(self.data) > 12:
            oil_rates = self.data['FOPR'].values
            if oil_rates[-1] > 0 and oil_rates[0] > 0:
                decline_rate = (oil_rates[0] / oil_rates[-1]) ** (1/len(oil_rates)) - 1
                self.metrics['annual_decline_rate'] = decline_rate * 365 * 100  # Convert to annual percentage
                
    def _calculate_well_metrics(self) -> None:
        """Calculate well performance metrics"""
        # Extract well production data
        well_oil_cols = [col for col in self.data.columns if 'WOPR' in col]
        well_gas_cols = [col for col in self.data.columns if 'WGPR' in col]
        
        if well_oil_cols:
            # Average well productivity
            avg_well_oil_rate = self.data[well_oil_cols].mean().mean()
            self.metrics['avg_well_oil_productivity'] = avg_well_oil_rate
            
            # Well efficiency (variation)
            well_variation = self.data[well_oil_cols].std().mean() / avg_well_oil_rate if avg_well_oil_rate > 0 else 0
            self.metrics['well_productivity_variation'] = well_variation
            
    def generate_detailed_report(self) -> pd.DataFrame:
        """Generate detailed metrics report as DataFrame"""
        metrics_df = pd.DataFrame.from_dict(self.metrics, orient='index', columns=['Value'])
        metrics_df['Unit'] = self._get_metric_units()
        metrics_df['Description'] = self._get_metric_descriptions()
        
        return metrics_df
        
    def _get_metric_units(self) -> Dict[str, str]:
        """Get units for each metric"""
        units = {
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
            'well_productivity_variation': 'fraction'
        }
        
        # Create series with same index as metrics
        return pd.Series({k: units.get(k, '') for k in self.metrics.keys()})
        
    def _get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions for each metric"""
        descriptions = {
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
            'well_productivity_variation': 'Variation in well productivity (std/mean)'
        }
        
        return pd.Series({k: descriptions.get(k, '') for k in self.metrics.keys()})
