"""
Exploratory Data Analysis for Reservoir Data

This module provides comprehensive data analysis tools for reservoir
engineering datasets including statistical analysis, trend detection,
and data quality assessment.
"""

import pandas as pd
import numpy as np
from scipy import stats, signal
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for EDA analysis"""
    confidence_level: float = 0.95
    trend_window: int = 30
    outlier_threshold: float = 3.0
    seasonal_period: int = 365


class ReservoirEDA:
    """
    Comprehensive Exploratory Data Analysis for reservoir data.
    
    Features:
    - Statistical summary
    - Trend analysis
    - Seasonality detection
    - Outlier detection
    - Correlation analysis
    - Data quality metrics
    """
    
    def __init__(self, data: Dict, config: Optional[AnalysisConfig] = None):
        """
        Initialize EDA analyzer.
        
        Parameters
        ----------
        data : Dict
            Reservoir data dictionary
        config : AnalysisConfig, optional
            Analysis configuration
        """
        self.data = data
        self.config = config or AnalysisConfig()
        self.results = {}
        logger.info("ReservoirEDA initialized")
    
    def perform_comprehensive_analysis(self) -> Dict:
        """
        Perform complete EDA analysis.
        
        Returns
        -------
        Dict
            Dictionary containing all analysis results
        """
        logger.info("Starting comprehensive EDA analysis")
        
        analysis_results = {
            'basic_statistics': self._basic_statistics(),
            'production_analysis': self._production_analysis(),
            'pressure_analysis': self._pressure_analysis(),
            'material_balance': self._material_balance_analysis(),
            'decline_analysis': self._decline_curve_analysis(),
            'petrophysical_analysis': self._petrophysical_analysis(),
            'correlation_analysis': self._correlation_analysis(),
            'data_quality': self._data_quality_assessment(),
            'trend_analysis': self._trend_analysis(),
            'seasonality_analysis': self._seasonality_analysis()
        }
        
        self.results = analysis_results
        logger.info("EDA analysis completed")
        return analysis_results
    
    def _basic_statistics(self) -> Dict:
        """Calculate basic statistical parameters."""
        stats_dict = {}
        
        for key, df in self.data.items():
            if isinstance(df, pd.DataFrame):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    desc = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75])
                    
                    # Additional statistics
                    skewness = df[numeric_cols].skew()
                    kurtosis = df[numeric_cols].kurtosis()
                    cv = (df[numeric_cols].std() / df[numeric_cols].mean()) * 100
                    
                    stats_dict[key] = {
                        'descriptive': desc.to_dict(),
                        'skewness': skewness.to_dict(),
                        'kurtosis': kurtosis.to_dict(),
                        'coefficient_of_variation': cv.to_dict()
                    }
        
        return stats_dict
    
    def _production_analysis(self) -> Dict:
        """Analyze production data."""
        if 'production' not in self.data:
            return {}
        
        prod_df = self.data['production']
        analysis = {}
        
        # Calculate total production
        total_production = prod_df.sum(axis=1)
        
        # Decline rate analysis
        decline_rates = []
        eur_estimates = []
        
        for column in prod_df.columns:
            rates = prod_df[column].values
            mask = rates > 0
            
            if np.sum(mask) > 10:
                # Exponential decline fit
                log_rates = np.log(rates[mask])
                time_subset = self.data['time'][mask]
                
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        time_subset[:100], log_rates[:100]
                    )
                    
                    decline_rate = -slope * 365  # Annualized
                    decline_rates.append(decline_rate)
                    
                    # EUR estimate (simplified)
                    q_initial = np.exp(intercept)
                    eur = q_initial / (decline_rate/365) if decline_rate > 0 else 0
                    eur_estimates.append(eur)
                    
                except:
                    continue
        
        analysis['total_production'] = {
            'mean': float(total_production.mean()),
            'std': float(total_production.std()),
            'max': float(total_production.max()),
            'min': float(total_production.min())
        }
        
        if decline_rates:
            analysis['decline_analysis'] = {
                'mean_decline_rate': float(np.mean(decline_rates)),
                'std_decline_rate': float(np.std(decline_rates)),
                'total_eur': float(np.sum(eur_estimates))
            }
        
        return analysis
    
    def _pressure_analysis(self) -> Dict:
        """Analyze reservoir pressure data."""
        if 'pressure' not in self.data:
            return {}
        
        pressure = self.data['pressure']
        analysis = {}
        
        # Pressure trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            self.data['time'], pressure
        )
        
        # Pressure volatility
        pressure_diff = np.diff(pressure)
        pressure_volatility = np.std(pressure_diff)
        
        # Pressure distribution
        pressure_stats = {
            'mean': float(np.mean(pressure)),
            'std': float(np.std(pressure)),
            'min': float(np.min(pressure)),
            'max': float(np.max(pressure)),
            'median': float(np.median(pressure))
        }
        
        analysis['pressure_trend'] = {
            'slope_psi_per_year': float(slope * 365),
            'intercept_psi': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value)
        }
        
        analysis['pressure_statistics'] = pressure_stats
        analysis['pressure_volatility'] = {
            'daily_std_psi': float(pressure_volatility)
        }
        
        return analysis
    
    def _material_balance_analysis(self) -> Dict:
        """Perform material balance analysis."""
        if 'production' not in self.data or 'injection' not in self.data:
            return {}
        
        total_production = self.data['production'].sum(axis=1)
        total_injection = self.data['injection'].sum(axis=1)
        
        # Cumulative volumes
        time = self.data['time']
        cumulative_prod = np.trapz(total_production, time)
        cumulative_inj = np.trapz(total_injection, time)
        
        # Voidage replacement ratio
        vrr = cumulative_inj / cumulative_prod if cumulative_prod > 0 else 0
        
        return {
            'cumulative_production_bbl': float(cumulative_prod),
            'cumulative_injection_bbl': float(cumulative_inj),
            'voidage_replacement_ratio': float(vrr),
            'net_voidage_bbl': float(cumulative_prod - cumulative_inj)
        }
    
    def _decline_curve_analysis(self) -> Dict:
        """Perform decline curve analysis using Arps equation."""
        if 'production' not in self.data:
            return {}
        
        analysis = {}
        prod_df = self.data['production']
        
        for i, well in enumerate(prod_df.columns[:3]):  # Analyze first 3 wells
            rates = prod_df[well].values
            valid_idx = rates > 0
            
            if np.sum(valid_idx) > 50:
                q = rates[valid_idx]
                t = self.data['time'][valid_idx]
                
                try:
                    # Exponential decline fit
                    log_q = np.log(q[:100])
                    coeffs = np.polyfit(t[:100], log_q, 1)
                    
                    qi = np.exp(coeffs[1])
                    di = -coeffs[0]
                    
                    analysis[well] = {
                        'initial_rate_bbl_per_day': float(qi),
                        'decline_rate_per_year': float(di * 365),
                        'fit_quality': 'good' if np.std(log_q - np.polyval(coeffs, t[:100])) < 0.1 else 'moderate'
                    }
                except:
                    analysis[well] = {'fit_quality': 'failed'}
        
        return analysis
    
    def _petrophysical_analysis(self) -> Dict:
        """Analyze petrophysical data."""
        if 'petrophysical' not in self.data:
            return {}
        
        petro_df = self.data['petrophysical']
        analysis = {}
        
        # Basic statistics for each property
        for column in petro_df.columns:
            data = petro_df[column].values
            analysis[column] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'median': float(np.median(data))
            }
        
        # Calculate kh product (permeability-thickness)
        if 'Permeability' in petro_df.columns and 'NetThickness' in petro_df.columns:
            kh = (petro_df['Permeability'] * petro_df['NetThickness']).sum()
            analysis['kh_product'] = {
                'total_md_ft': float(kh),
                'average_md_ft': float(kh / len(petro_df))
            }
        
        # Estimate OOIP (simplified)
        if all(col in petro_df.columns for col in ['Porosity', 'NetThickness', 'WaterSaturation']):
            area = 1000  # acres (example)
            boi = 1.2  # rb/stb
            
            h_avg = petro_df['NetThickness'].mean()
            phi_avg = petro_df['Porosity'].mean()
            sw_avg = petro_df['WaterSaturation'].mean()
            
            ooip = (7758 * area * h_avg * phi_avg * (1 - sw_avg)) / boi
            analysis['ooip_estimate'] = {
                'stb': float(ooip),
                'assumptions': {
                    'area_acres': 1000,
                    'boi_rb_per_stb': 1.2
                }
            }
        
        return analysis
    
    def _correlation_analysis(self) -> Dict:
        """Perform correlation analysis."""
        correlations = {}
        
        # Petrophysical correlations
        if 'petrophysical' in self.data:
            petro_corr = self.data['petrophysical'].corr()
            correlations['petrophysical'] = petro_corr.to_dict()
        
        # Production correlations
        if 'production' in self.data:
            prod_corr = self.data['production'].corr()
            correlations['production'] = prod_corr.to_dict()
        
        return correlations
    
    def _data_quality_assessment(self) -> Dict:
        """Assess data quality."""
        quality = {}
        
        for key, df in self.data.items():
            if isinstance(df, pd.DataFrame):
                missing = df.isnull().sum().sum()
                total = df.size
                missing_pct = (missing / total) * 100 if total > 0 else 0
                
                quality[key] = {
                    'total_values': int(total),
                    'missing_values': int(missing),
                    'missing_percentage': float(missing_pct),
                    'duplicates': int(df.duplicated().sum()),
                    'data_types': str(df.dtypes.to_dict())
                }
        
        return quality
    
    def _trend_analysis(self) -> Dict:
        """Perform trend analysis on time series data."""
        trends = {}
        
        # Production trend
        if 'production' in self.data:
            total_prod = self.data['production'].sum(axis=1)
            
            # Moving average
            window = self.config.trend_window
            moving_avg = pd.Series(total_prod).rolling(window=window, center=True).mean()
            
            # Trend decomposition
            detrended = signal.detrend(total_prod)
            
            trends['production'] = {
                'moving_average_window': window,
                'trend_strength': float(1 - (np.var(detrended) / np.var(total_prod))),
                'is_stationary': self._check_stationarity(total_prod)
            }
        
        return trends
    
    def _seasonality_analysis(self) -> Dict:
        """Analyze seasonality in time series data."""
        seasonality = {}
        
        if 'production' in self.data and len(self.data['time']) > 365:
            total_prod = self.data['production'].sum(axis=1)
            
            # Check for annual seasonality
            period = self.config.seasonal_period
            if len(total_prod) >= period * 2:
                try:
                    # Simple autocorrelation at seasonal lag
                    acf = self._autocorrelation(total_prod, lag=period)
                    seasonality['production'] = {
                        'seasonal_period_days': period,
                        'autocorrelation_at_lag': float(acf),
                        'has_seasonality': abs(acf) > 0.3
                    }
                except:
                    pass
        
        return seasonality
    
    def _check_stationarity(self, series: np.ndarray, p_value: float = 0.05) -> bool:
        """Check if time series is stationary using Augmented Dickey-Fuller test."""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series)
            return result[1] < p_value
        except:
            return False
    
    def _autocorrelation(self, series: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at specified lag."""
        if len(series) < lag + 1:
            return 0.0
        
        x = series[:-lag]
        y = series[lag:]
        
        if len(x) > 1 and len(y) > 1:
            return np.corrcoef(x, y)[0, 1]
        return 0.0
    
    def generate_report(self) -> pd.DataFrame:
        """Generate comprehensive report dataframe."""
        report_data = []
        
        for analysis_type, results in self.results.items():
            if isinstance(results, dict):
                # Flatten nested dictionaries for reporting
                for key, value in results.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            report_data.append({
                                'analysis_type': analysis_type,
                                'metric': f"{key}.{subkey}",
                                'value': str(subvalue)
                            })
                    else:
                        report_data.append({
                            'analysis_type': analysis_type,
                            'metric': key,
                            'value': str(value)
                        })
        
        return pd.DataFrame(report_data)
