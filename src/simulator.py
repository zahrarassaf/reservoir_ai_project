# src/simulator.py - VERSION COMPATIBLE WITH NEW economics.py

import numpy as np
import logging
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class ReservoirSimulator:
    def __init__(self, data, economic_params=None):
        self.data = data
        self.economic_params = economic_params or {}
        self.results = {}
        
    def run_simulation(self, forecast_years=10, oil_price=75.0, operating_cost=18.0):
        """
        Wrapper function that uses the NEW economics.py module
        """
        logger.info("Starting comprehensive reservoir simulation")
        
        try:
            # Import the NEW economic simulator
            from .economics import ReservoirSimulator as EconomicSimulator
            from .economics import SimulationParameters
            
            # Create simulation parameters
            sim_params = SimulationParameters(
                forecast_years=forecast_years,
                oil_price=oil_price,
                operating_cost=operating_cost,
                discount_rate=self.economic_params.get('discount_rate', 0.10),
                capex_per_well=self.economic_params.get('capex_per_well', 1_000_000),
                fixed_annual_opex=self.economic_params.get('fixed_annual_opex', 500_000),
                tax_rate=self.economic_params.get('tax_rate', 0.30),
                royalty_rate=self.economic_params.get('royalty_rate', 0.125)
            )
            
            # Run the NEW economic simulation
            economic_simulator = EconomicSimulator(self.data, sim_params)
            results = economic_simulator.run_comprehensive_simulation()
            
            # Extract economic results
            economic_analysis = results.get('economic_analysis', {})
            
            # Format results for display
            self.results = {
                'npv': economic_analysis.get('npv', 0),
                'irr': economic_analysis.get('irr', 0),
                'roi': economic_analysis.get('roi', 0),
                'payback': economic_analysis.get('payback_period_years', None),
                'total_production': sum(self._extract_production_data()),
                'economic_details': economic_analysis
            }
            
            logger.info("Simulation completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'npv': 0,
                'irr': 0,
                'roi': 0,
                'payback': None,
                'total_production': 0,
                'error': str(e)
            }
    
    def _extract_production_data(self):
        """Extract production data from wells"""
        productions = []
        if hasattr(self.data, 'wells'):
            for well_name, well in self.data.wells.items():
                if hasattr(well, 'production_rates'):
                    rates = well.production_rates
                    if hasattr(well, 'time_points') and len(rates) > 0:
                        time_points = well.time_points
                        if len(time_points) >= 2:
                            # Simple integration
                            avg_rate = np.mean(rates)
                            time_span = time_points[-1] - time_points[0]
                            productions.append(avg_rate * time_span)
        return productions
    
    def get_summary(self):
        """Get simulation summary"""
        return self.results
