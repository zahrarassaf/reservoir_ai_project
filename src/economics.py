"""
Economic Analysis for Reservoir Simulation
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EconomicAnalyzer:
    """Perform economic analysis for reservoir projects"""
    
    def __init__(self, parameters: Optional[Dict] = None):
        """
        Initialize economic analyzer
        
        Parameters
        ----------
        parameters : Dict, optional
            Economic parameters
        """
        self.parameters = parameters or self.default_parameters()
    
    def default_parameters(self) -> Dict:
        """Get default economic parameters"""
        return {
            'oil_price': 75.0,  # USD/bbl
            'operating_cost': 18.0,  # USD/bbl
            'capital_cost': 5000000.0,  # USD (initial)
            'discount_rate': 0.12,
            'tax_rate': 0.30,
            'project_life': 20,  # years
            'royalty_rate': 0.125,
            'abandonment_cost': 1000000.0  # USD
        }
    
    def calculate_npv(self, cash_flows: List[float], 
                     time_periods: List[float]) -> float:
        """
        Calculate Net Present Value
        
        Parameters
        ----------
        cash_flows : List[float]
            Annual cash flows
        time_periods : List[float]
            Time periods in years
            
        Returns
        -------
        float
            NPV in USD
        """
        discount_rate = self.parameters['discount_rate']
        npv = 0.0
        
        for cf, t in zip(cash_flows
