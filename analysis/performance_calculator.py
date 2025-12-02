"""
Performance Calculator Module - FIXED VERSION
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

class PerformanceCalculator:
    """Calculate performance metrics from simulation results."""
    
    def __init__(self, simulation_results=None):
        """Initialize calculator."""
        # FIX: Handle DataFrame truth value ambiguity
        if simulation_results is not None and not simulation_results.empty:
            self.results = simulation_results
        else:
            self.results = {}
    
    def calculate_metrics(self):
        """Calculate basic performance metrics."""
        if not hasattr(self, 'results') or not self.results:
            return self._empty_metrics()
        
        # Handle both dict and DataFrame
        if isinstance(self.results, dict):
            return self._calculate_from_dict()
        elif isinstance(self.results, pd.DataFrame):
            return self._calculate_from_dataframe()
        else:
            return self._empty_metrics()
    
    def _calculate_from_dict(self):
        """Calculate metrics from dictionary results."""
        metrics = {}
        
        # Extract production data
        prod = self.results.get('production', {})
        if prod:
            metrics['total_oil_produced'] = float(np.sum(prod.get('oil', [0])))
            metrics['total_water_produced'] = float(np.sum(prod.get('water', [0])))
            metrics['total_gas_produced'] = float(np.sum(prod.get('gas', [0])))
        
        # Extract injection data
        inj = self.results.get('injection', {})
        if inj:
            metrics['total_water_injected'] = float(np.sum(inj.get('water', [0])))
        
        # Well count
        metrics['well_count'] = len(self.results.get('wells', []))
        
        # Time steps
        time_series = self.results.get('time_series', {})
        metrics['simulation_days'] = len(time_series.get('time_steps', []))
        
        # Pressure
        reservoir_state = self.results.get('reservoir_state', {})
        avg_pressure = reservoir_state.get('average_pressure', [])
        if avg_pressure:
            metrics['average_pressure'] = float(np.mean(avg_pressure))
            metrics['final_pressure'] = float(avg_pressure[-1] if avg_pressure else 0)
        
        return metrics
    
    def _calculate_from_dataframe(self):
        """Calculate metrics from DataFrame."""
        df = self.results
        
        if df.empty:
            return self._empty_metrics()
        
        metrics = {}
        
        # Check for required columns
        required_columns = ['FOPR', 'FOPT', 'FGPR', 'FGPT', 'FWPR', 'FWPT']
        available_columns = [col for col in required_columns if col in df.columns]
        
        for col in available_columns:
            if col in df.columns:
                if col.endswith('R'):  # Rate columns
                    metrics[f'average_{col.lower()}'] = float(df[col].mean())
                elif col.endswith('T'):  # Total columns
                    metrics[f'final_{col.lower()}'] = float(df[col].iloc[-1])
        
        # Calculate recovery factor if OOIP available
        if 'FOPT' in df.columns and 'OOIP' in self.results.get('metadata', {}):
            ooip = self.results['metadata']['OOIP']
            final_oil = df['FOPT'].iloc[-1]
            metrics['recovery_factor_percent'] = (final_oil / ooip * 100) if ooip > 0 else 0
        
        # Calculate water cut
        if 'FWPR' in df.columns and 'FOPR' in df.columns:
            final_fwpr = df['FWPR'].iloc[-1]
            final_fopr = df['FOPR'].iloc[-1]
            if final_fopr + final_fwpr > 0:
                metrics['final_water_cut_percent'] = (final_fwpr / (final_fopr + final_fwpr)) * 100
        
        return metrics
    
    def _empty_metrics(self):
        """Return empty metrics structure."""
        return {
            'total_oil_produced': 0.0,
            'total_water_produced': 0.0,
            'total_gas_produced': 0.0,
            'total_water_injected': 0.0,
            'well_count': 0,
            'simulation_days': 0,
            'average_pressure': 0.0,
            'final_pressure': 0.0
        }
    
    def calculate_all_metrics(self):
        """Alias for calculate_metrics for compatibility."""
        return self.calculate_metrics()
