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
        
        for cf, t in zip(cash_flows, time_periods):
            discount_factor = 1.0 / ((1.0 + discount_rate) ** t)
            npv += cf * discount_factor
        
        return npv
    
    def calculate_irr(self, cash_flows: List[float], 
                     time_periods: List[float]) -> float:
        """
        Calculate Internal Rate of Return
        
        Parameters
        ----------
        cash_flows : List[float]
            Annual cash flows
        time_periods : List[float]
            Time periods in years
            
        Returns
        -------
        float
            IRR as decimal
        """
        from scipy import optimize
        
        def npv_func(rate):
            npv_val = 0.0
            for cf, t in zip(cash_flows, time_periods):
                discount_factor = 1.0 / ((1.0 + rate) ** t)
                npv_val += cf * discount_factor
            return npv_val
        
        try:
            irr = optimize.newton(npv_func, 0.1, maxiter=1000)
            return max(0, irr)  # Ensure non-negative
        except:
            return 0.0
    
    def calculate_payback_period(self, cash_flows: List[float],
                                time_periods: List[float]) -> float:
        """
        Calculate payback period
        
        Parameters
        ----------
        cash_flows : List[float]
            Annual cash flows
        time_periods : List[float]
            Time periods in years
            
        Returns
        -------
        float
            Payback period in years
        """
        cumulative_cf = 0.0
        for cf, t in zip(cash_flows, time_periods):
            cumulative_cf += cf
            if cumulative_cf >= 0:
                return t
        
        return float('inf')  # Never pays back
    
    def calculate_roi(self, initial_investment: float,
                     net_profit: float) -> float:
        """
        Calculate Return on Investment
        
        Parameters
        ----------
        initial_investment : float
            Initial investment in USD
        net_profit : float
            Net profit in USD
            
        Returns
        -------
        float
            ROI as decimal
        """
        if initial_investment == 0:
            return 0.0
        
        return net_profit / initial_investment
    
    def analyze_production_scenario(self, production_profile: List[float],
                                   time_years: List[float]) -> Dict:
        """
        Analyze economic scenario for production profile
        
        Parameters
        ----------
        production_profile : List[float]
            Annual production in bbl
        time_years : List[float]
            Time in years
            
        Returns
        -------
        Dict
            Economic analysis results
        """
        oil_price = self.parameters['oil_price']
        operating_cost = self.parameters['operating_cost']
        discount_rate = self.parameters['discount_rate']
        tax_rate = self.parameters['tax_rate']
        
        # Calculate annual cash flows
        cash_flows = []
        revenues = []
        costs = []
        
        for i, production in enumerate(production_profile):
            # Revenue
            revenue = production * oil_price
            
            # Operating costs
            opex = production * operating_cost
            
            # Taxable income
            taxable_income = revenue - opex
            
            # Tax
            tax = max(0, taxable_income) * tax_rate
            
            # Net cash flow
            net_cf = taxable_income - tax
            
            # Add initial capital cost in year 0
            if i == 0:
                net_cf -= self.parameters['capital_cost']
            
            # Add abandonment cost in last year
            if i == len(production_profile) - 1:
                net_cf -= self.parameters['abandonment_cost']
            
            cash_flows.append(net_cf)
            revenues.append(revenue)
            costs.append(opex)
        
        # Calculate metrics
        npv = self.calculate_npv(cash_flows, time_years)
        irr = self.calculate_irr(cash_flows, time_years)
        payback = self.calculate_payback_period(cash_flows, time_years)
        
        total_revenue = sum(revenues)
        total_cost = sum(costs)
        net_profit = total_revenue - total_cost - self.parameters['capital_cost']
        roi = self.calculate_roi(self.parameters['capital_cost'], net_profit)
        
        results = {
            'npv_usd': float(npv),
            'irr': float(irr),
            'payback_period_years': float(payback) if payback != float('inf') else None,
            'roi': float(roi),
            'total_revenue_usd': float(total_revenue),
            'total_cost_usd': float(total_cost),
            'net_profit_usd': float(net_profit),
            'cash_flows': [float(cf) for cf in cash_flows],
            'revenues': [float(r) for r in revenues],
            'costs': [float(c) for c in costs],
            'parameters': self.parameters.copy()
        }
        
        logger.info(f"Economic analysis: NPV = ${npv/1e6:.2f}M, IRR = {irr*100:.1f}%")
        
        return results
    
    def perform_sensitivity_analysis(self, base_production: List[float],
                                    base_time: List[float]) -> Dict:
        """
        Perform sensitivity analysis
        
        Parameters
        ----------
        base_production : List[float]
            Base case production profile
        base_time : List[float]
            Time periods
            
        Returns
        -------
        Dict
            Sensitivity analysis results
        """
        # Get base case NPV
        base_results = self.analyze_production_scenario(base_production, base_time)
        base_npv = base_results['npv_usd']
        
        # Sensitivity parameters
        sensitivities = {
            'oil_price': [60, 70, 75, 80, 90],
            'operating_cost': [12, 15, 18, 21, 24],
            'discount_rate': [0.08, 0.10, 0.12, 0.14, 0.16],
            'capital_cost': [3e6, 4e6, 5e6, 6e6, 7e6]
        }
        
        results = {}
        
        for param_name, values in sensitivities.items():
            npv_values = []
            
            for value in values:
                # Save original parameter
                original_value = self.parameters[param_name]
                
                # Update parameter
                self.parameters[param_name] = value
                
                # Recalculate NPV
                scenario_results = self.analyze_production_scenario(base_production, base_time)
                npv_values.append(scenario_results['npv_usd'])
                
                # Restore original parameter
                self.parameters[param_name] = original_value
            
            # Calculate sensitivity index
            sensitivity_index = (max(npv_values) - min(npv_values)) / base_npv
            
            results[param_name] = {
                'values': values,
                'npv_values': npv_values,
                'sensitivity_index': sensitivity_index,
                'base_value': self.parameters[param_name]
            }
        
        # Restore original parameters
        self.parameters = self.default_parameters()
        
        # Identify key parameters
        key_params = []
        for param_name, data in results.items():
            if data['sensitivity_index'] > 0.3:
                key_params.append(param_name)
        
        results['key_parameters'] = key_params
        
        return results
