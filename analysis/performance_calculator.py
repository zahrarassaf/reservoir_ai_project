"""
Performance Calculator - Final Fixed Version
Completely removes .empty() calls
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class PerformanceCalculator:
    """Calculate reservoir performance metrics."""
    
    def __init__(self, simulation_results: Dict[str, Any]):
        """Initialize with simulation results."""
        self.results = simulation_results if simulation_results else {}
        self.metrics = {}
        
        logger.info("📊 Performance Calculator initialized")
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all performance metrics - SAFE VERSION."""
        logger.info("Calculating performance metrics...")
        
        try:
            # SAFE CHECK: Instead of .empty, check dictionary
            if not self.results or len(self.results) == 0:
                logger.warning("No results data available for metrics calculation")
                return self._get_basic_metrics()
            
            # Calculate metrics safely
            metrics_calculated = 0
            
            # Production metrics
            prod_metrics = self._safe_calculate_production()
            if prod_metrics:
                self.metrics.update(prod_metrics)
                metrics_calculated += len(prod_metrics)
            
            # Injection metrics
            inj_metrics = self._safe_calculate_injection()
            if inj_metrics:
                self.metrics.update(inj_metrics)
                metrics_calculated += len(inj_metrics)
            
            # Recovery factors
            recovery_metrics = self._safe_calculate_recovery()
            if recovery_metrics:
                self.metrics.update(recovery_metrics)
                metrics_calculated += len(recovery_metrics)
            
            # Economic metrics
            economic_metrics = self._safe_calculate_economic()
            if economic_metrics:
                self.metrics.update(economic_metrics)
                metrics_calculated += len(economic_metrics)
            
            logger.info(f"✅ Successfully calculated {metrics_calculated} metrics")
            
            # Add metadata
            self.metrics['calculation_status'] = 'success'
            self.metrics['metrics_count'] = metrics_calculated
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error in calculate_all_metrics: {e}")
            return self._get_basic_metrics()
    
    def _safe_calculate_production(self) -> Dict[str, float]:
        """Safely calculate production metrics."""
        metrics = {}
        
        try:
            if 'production' not in self.results:
                return metrics
            
            prod_data = self.results['production']
            if not isinstance(prod_data, dict):
                return metrics
            
            for phase in ['oil', 'water', 'gas']:
                if phase in prod_data:
                    phase_data = prod_data[phase]
                    if isinstance(phase_data, list) and len(phase_data) > 0:
                        try:
                            rates = np.array(phase_data, dtype=float)
                            metrics[f'total_{phase}_produced'] = float(np.sum(rates))
                            metrics[f'average_{phase}_rate'] = float(np.mean(rates))
                            metrics[f'max_{phase}_rate'] = float(np.max(rates)) if len(rates) > 0 else 0.0
                        except (ValueError, TypeError):
                            continue
        
        except Exception as e:
            logger.debug(f"Error in production calculation: {e}")
        
        return metrics
    
    def _safe_calculate_injection(self) -> Dict[str, float]:
        """Safely calculate injection metrics."""
        metrics = {}
        
        try:
            if 'injection' not in self.results:
                return metrics
            
            inj_data = self.results['injection']
            if not isinstance(inj_data, dict):
                return metrics
            
            if 'water' in inj_data:
                water_data = inj_data['water']
                if isinstance(water_data, list) and len(water_data) > 0:
                    try:
                        rates = np.array(water_data, dtype=float)
                        metrics['total_water_injected'] = float(np.sum(rates))
                        metrics['average_injection_rate'] = float(np.mean(rates))
                        metrics['max_injection_rate'] = float(np.max(rates)) if len(rates) > 0 else 0.0
                    except (ValueError, TypeError):
                        pass
        
        except Exception as e:
            logger.debug(f"Error in injection calculation: {e}")
        
        return metrics
    
    def _safe_calculate_recovery(self) -> Dict[str, float]:
        """Safely calculate recovery factors."""
        metrics = {}
        
        try:
            # Estimate OOIP from grid size
            grid_dims = self.results.get('grid_dimensions', (24, 25, 15))
            if isinstance(grid_dims, (tuple, list)) and len(grid_dims) == 3:
                total_cells = grid_dims[0] * grid_dims[1] * grid_dims[2]
                ooip_estimate = total_cells * 1000  # Simplified estimation
                metrics['estimated_ooip'] = float(ooip_estimate)
            
            # Calculate recovery factor if we have oil production
            total_oil = self.metrics.get('total_oil_produced', 0)
            ooip = metrics.get('estimated_ooip', 1_000_000)  # Default fallback
            
            if ooip > 0:
                recovery_factor = total_oil / ooip
                metrics['oil_recovery_factor'] = float(recovery_factor)
        
        except Exception as e:
            logger.debug(f"Error in recovery calculation: {e}")
        
        return metrics
    
    def _safe_calculate_economic(self) -> Dict[str, float]:
        """Safely calculate economic metrics."""
        metrics = {}
        
        try:
            total_oil = self.metrics.get('total_oil_produced', 0)
            oil_price = 70.0  # USD per barrel
            
            metrics['gross_revenue_usd'] = float(total_oil * oil_price)
            metrics['net_present_value_usd'] = float(total_oil * oil_price * 0.7)
        
        except Exception as e:
            logger.debug(f"Error in economic calculation: {e}")
        
        return metrics
    
    def _get_basic_metrics(self) -> Dict[str, Any]:
        """Get basic metrics even if detailed calculation fails."""
        basic_metrics = {
            'calculation_status': 'basic',
            'simulation_data_available': list(self.results.keys()) if self.results else [],
            'well_count': len(self.results.get('wells', [])),
            'timesteps': len(self.results.get('time_steps', []))
        }
        
        # Try to get any production data
        if 'production' in self.results:
            prod = self.results['production']
            if isinstance(prod, dict):
                for phase in ['oil', 'water', 'gas']:
                    if phase in prod and isinstance(prod[phase], list):
                        basic_metrics[f'{phase}_data_points'] = len(prod[phase])
        
        return basic_metrics
    
    # For backward compatibility
    def calculate_metrics(self) -> Dict[str, Any]:
        """Alias for calculate_all_metrics."""
        return self.calculate_all_metrics()
