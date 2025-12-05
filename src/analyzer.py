"""
Reservoir Data Analyzer
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class ReservoirAnalyzer:
    """Analyze reservoir data"""
    
    def __init__(self, data):
        """
        Initialize analyzer
        
        Parameters
        ----------
        data : ReservoirData
            Reservoir data
        """
        self.data = data
    
    def analyze_production(self) -> Dict:
        """Analyze production data"""
        logger.info("Analyzing production data")
        
        if self.data.production.empty:
            return {}
        
        results = {}
        
        # Basic statistics
        production_df = self.data.production
        total_production = production_df.sum(axis=1)
        
        results['total_production'] = {
            'mean': float(total_production.mean()),
            'std': float(total_production.std()),
            'min': float(total_production.min()),
            'max': float(total_production.max()),
            'sum': float(total_production.sum())
        }
        
        # Well statistics
        well_stats = {}
        for well in production_df.columns:
            rates = production_df[well].values
            valid_rates = rates[rates > 0]
            
            if len(valid_rates) > 0:
                well_stats[well] = {
                    'mean': float(np.mean(valid_rates)),
                    'std': float(np.std(valid_rates)),
                    'max': float(np.max(valid_rates)),
                    'min': float(np.min(valid_rates)),
                    'data_points': int(len(valid_rates))
                }
        
        results['well_statistics'] = well_stats
        
        # Decline rate calculation
        decline_rates = []
        for well in production_df.columns:
            rates = production_df[well].values
            valid_mask = rates > 0
            
            if np.sum(valid_mask) > 10:
                try:
                    # Fit exponential decline
                    valid_rates = rates[valid_mask]
                    valid_times = self.data.time[valid_mask]
                    
                    if len(valid_rates) > 100:
                        log_rates = np.log(valid_rates[:100])
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            valid_times[:100], log_rates
                        )
                        
                        decline_rate = -slope * 365  # Annualized
                        decline_rates.append(decline_rate)
                        
                except Exception as e:
                    logger.warning(f"Decline analysis failed for {well}: {e}")
        
        if decline_rates:
            results['decline_statistics'] = {
                'mean': float(np.mean(decline_rates)),
                'std': float(np.std(decline_rates)),
                'min': float(np.min(decline_rates)),
                'max': float(np.max(decline_rates)),
                'count': len(decline_rates)
            }
        
        return results
    
    def analyze_pressure(self) -> Dict:
        """Analyze pressure data"""
        logger.info("Analyzing pressure data")
        
        if len(self.data.pressure) == 0:
            return {}
        
        pressure = self.data.pressure
        
        # Remove NaN values
        pressure_clean = pressure[~np.isnan(pressure)]
        
        if len(pressure_clean) < 2:
            return {}
        
        # Basic statistics
        results = {
            'statistics': {
                'mean': float(np.mean(pressure_clean)),
                'std': float(np.std(pressure_clean)),
                'min': float(np.min(pressure_clean)),
                'max': float(np.max(pressure_clean)),
                'initial': float(pressure_clean[0]),
                'final': float(pressure_clean[-1])
            }
        }
        
        # Pressure trend
        if len(pressure_clean) > 10:
            time_indices = np.arange(len(pressure_clean))
            
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    time_indices, pressure_clean
                )
                
                results['trend'] = {
                    'slope': float(slope),  # psi per day
                    'intercept': float(intercept),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'annual_decline': float(slope * 365)  # psi per year
                }
                
            except Exception as e:
                logger.warning(f"Pressure trend analysis failed: {e}")
        
        return results
    
    def analyze_petrophysical(self) -> Dict:
        """Analyze petrophysical data"""
        logger.info("Analyzing petrophysical data")
        
        if self.data.petrophysical.empty:
            return {}
        
        petro_df = self.data.petrophysical
        
        results = {}
        
        # Basic statistics for each property
        for column in petro_df.columns:
            if column in petro_df:
                data = petro_df[column].values
                data_clean = data[~np.isnan(data)]
                
                if len(data_clean) > 0:
                    results[column] = {
                        'mean': float(np.mean(data_clean)),
                        'std': float(np.std(data_clean)),
                        'min': float(np.min(data_clean)),
                        'max': float(np.max(data_clean)),
                        'median': float(np.median(data_clean))
                    }
        
        # Calculate kh product if both permeability and thickness are available
        if 'permeability' in petro_df.columns and 'netthickness' in petro_df.columns:
            kh = (petro_df['permeability'] * petro_df['netthickness']).sum()
            results['kh_product'] = {
                'total_md_ft': float(kh),
                'average_md_ft': float(kh / len(petro_df))
            }
        
        return results
    
    def calculate_ooip(self, area_acres: float = 1000, 
                       boi: float = 1.2) -> Dict:
        """
        Calculate Original Oil In Place (OOIP)
        
        Parameters
        ----------
        area_acres : float
            Reservoir area in acres
        boi : float
            Initial oil formation volume factor (rb/stb)
            
        Returns
        -------
        Dict
            OOIP calculation results
        """
        logger.info("Calculating OOIP")
        
        if self.data.petrophysical.empty:
            return {}
        
        petro_df = self.data.petrophysical
        
        # Check required columns
        required_cols = ['porosity', 'netthickness', 'watersaturation']
        missing_cols = [col for col in required_cols if col not in petro_df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for OOIP calculation: {missing_cols}")
            return {}
        
        # Calculate average properties
        phi_avg = petro_df['porosity'].mean()
        h_avg = petro_df['netthickness'].mean()
        sw_avg = petro_df['watersaturation'].mean()
        
        # OOIP calculation (simplified)
        # OOIP (STB) = (7758 * A * h * φ * (1 - Sw)) / Boi
        ooip = (7758 * area_acres * h_avg * phi_avg * (1 - sw_avg)) / boi
        
        results = {
            'ooip_stb': float(ooip),
            'parameters': {
                'area_acres': area_acres,
                'boi_rb_per_stb': boi,
                'average_porosity': phi_avg,
                'average_thickness_ft': h_avg,
                'average_water_saturation': sw_avg
            },
            'calculation': {
                'formula': 'OOIP = (7758 * A * h * φ * (1 - Sw)) / Boi',
                'units': {
                    'area': 'acres',
                    'thickness': 'ft',
                    'ooip': 'STB'
                }
            }
        }
        
        logger.info(f"OOIP calculation: {ooip:,.0f} STB")
        
        return results
    
    def perform_comprehensive_analysis(self) -> Dict:
        """Perform comprehensive data analysis"""
        logger.info("Performing comprehensive analysis")
        
        results = {
            'production_analysis': self.analyze_production(),
            'pressure_analysis': self.analyze_pressure(),
            'petrophysical_analysis': self.analyze_petrophysical(),
            'ooip_calculation': self.calculate_ooip(),
            'summary': {}
        }
        
        # Create summary
        summary = {}
        
        # Production summary
        prod_stats = results['production_analysis'].get('total_production', {})
        if prod_stats:
            summary['peak_production'] = prod_stats.get('max', 0)
            summary['average_production'] = prod_stats.get('mean', 0)
        
        # Pressure summary
        press_stats = results['pressure_analysis'].get('statistics', {})
        if press_stats:
            summary['initial_pressure'] = press_stats.get('initial', 0)
            summary['pressure_drop'] = press_stats.get('initial', 0) - press_stats.get('final', 0)
        
        # OOIP summary
        ooip_calc = results['ooip_calculation']
        if ooip_calc:
            summary['ooip'] = ooip_calc.get('ooip_stb', 0)
        
        results['summary'] = summary
        
        return results
