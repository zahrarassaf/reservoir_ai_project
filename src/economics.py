import numpy as np
from numpy_financial import irr as npf_irr
from scipy.optimize import newton
import logging

logger = logging.getLogger(__name__)

class EconomicAnalyzer:
    def __init__(self, config=None):
        self.config = config or {}
        self.capex = self.config.get('capex', 50000000)  # 50M USD
        self.discount_rate = self.config.get('discount_rate', 0.10)  # 10%
        self.fixed_opex = self.config.get('fixed_opex', 2000000)  # 2M USD/year
        self.tax_rate = self.config.get('tax_rate', 0.30)  # 30%
        self.royalty_rate = self.config.get('royalty_rate', 0.125)  # 12.5%
        self.abandonment_cost = self.config.get('abandonment_cost', 5000000)  # 5M USD
        
    def calculate_project_economics(self, production_forecast, oil_price, opex_per_bbl, 
                                   forecast_years=10):
        """
        Calculate comprehensive project economics
        
        Parameters:
        -----------
        production_forecast : list
            Annual production forecast in barrels
        oil_price : float
            Oil price in USD/bbl
        opex_per_bbl : float
            Operating cost in USD/bbl
        forecast_years : int
            Forecast period in years
        """
        if len(production_forecast) != forecast_years:
            raise ValueError(f"Production forecast must have {forecast_years} years")
        
        # 1. Calculate annual cash flows
        cash_flows = self._calculate_cash_flows(
            production_forecast, oil_price, opex_per_bbl, forecast_years
        )
        
        # 2. Calculate financial metrics
        npv = self._calculate_npv(cash_flows)
        irr = self._calculate_irr(cash_flows)
        roi = self._calculate_roi(cash_flows)
        payback = self._calculate_payback_period(cash_flows)
        profitability_index = self._calculate_profitability_index(cash_flows)
        
        # 3. Calculate annual metrics
        annual_metrics = self._calculate_annual_metrics(
            production_forecast, cash_flows, oil_price, opex_per_bbl
        )
        
        return {
            'npv': npv,  # in millions USD
            'irr': irr,  # percentage
            'roi': roi,  # percentage
            'payback_period': payback,  # years
            'profitability_index': profitability_index,
            'cash_flows': cash_flows,  # annual cash flows
            'annual_metrics': annual_metrics,
            'break_even_price': self._calculate_break_even_price(cash_flows, production_forecast),
            'unit_development_cost': self._calculate_udc(cash_flows, sum(production_forecast))
        }
    
    def _calculate_cash_flows(self, production_forecast, oil_price, opex_per_bbl, years):
        """Calculate detailed annual cash flows"""
        cash_flows = []
        
        # Year 0: Capital expenditures (negative)
        cash_flows.append(-self.capex)
        
        # Production years
        for year in range(years):
            annual_production = production_forecast[year]
            
            # Gross revenue
            gross_revenue = annual_production * oil_price
            
            # Royalty payments (government share)
            royalty = gross_revenue * self.royalty_rate
            
            # Operating costs
            variable_opex = annual_production * opex_per_bbl
            total_opex = variable_opex + self.fixed_opex
            
            # Depreciation (straight-line over 10 years)
            depreciation = self.capex / 10 if year < 10 else 0
            
            # Earnings before interest and tax (EBIT)
            ebit = gross_revenue - royalty - total_opex - depreciation
            
            # Tax calculation
            taxable_income = max(0, ebit)  # No tax on losses
            tax = taxable_income * self.tax_rate
            
            # Net operating cash flow
            # Add back depreciation (non-cash expense)
            operating_cf = ebit - tax + depreciation
            
            cash_flows.append(operating_cf)
        
        # Last year: Abandonment cost
        cash_flows[-1] -= self.abandonment_cost
        
        return cash_flows
    
    def _calculate_npv(self, cash_flows):
        """Calculate Net Present Value with proper discounting"""
        npv = 0.0
        for t, cf in enumerate(cash_flows):
            discount_factor = 1 / ((1 + self.discount_rate) ** t)
            npv += cf * discount_factor
        
        # Convert to millions for readability
        return npv / 1_000_000
    
    def _calculate_irr(self, cash_flows):
        """Calculate Internal Rate of Return"""
        try:
            # Use numpy_financial IRR with fallback
            irr_value = npf_irr(cash_flows)
            if irr_value is None:
                return self._approximate_irr(cash_flows)
            
            # Check if IRR is realistic
            if abs(irr_value) > 10:  # Unrealistically high
                return self._approximate_irr(cash_flows)
                
            return irr_value * 100  # Convert to percentage
        except Exception as e:
            logger.warning(f"IRR calculation failed: {e}")
            return self._approximate_irr(cash_flows)
    
    def _approximate_irr(self, cash_flows):
        """Fallback IRR calculation using secant method"""
        if len(cash_flows) < 2:
            return 0.0
        
        def npv_func(rate):
            return sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows))
        
        try:
            # Try to find IRR between -0.9 and 10
            solution = newton(npv_func, x0=0.1, maxiter=100)
            if -0.9 < solution < 10:
                return solution * 100
        except:
            pass
        
        return 0.0
    
    def _calculate_roi(self, cash_flows):
        """Return on Investment calculation"""
        if len(cash_flows) < 2:
            return 0.0
        
        initial_investment = abs(cash_flows[0])
        if initial_investment == 0:
            return 0.0
        
        total_return = sum(cf for cf in cash_flows[1:] if cf > 0)
        roi_percentage = (total_return / initial_investment) * 100
        
        return roi_percentage
    
    def _calculate_payback_period(self, cash_flows):
        """Calculate discounted payback period"""
        if len(cash_flows) < 2:
            return None
        
        cumulative_pv = 0
        initial_investment = abs(cash_flows[0])
        
        for t, cf in enumerate(cash_flows):
            if t == 0:
                continue
            
            discount_factor = 1 / ((1 + self.discount_rate) ** t)
            discounted_cf = cf * discount_factor
            cumulative_pv += discounted_cf
            
            if cumulative_pv >= initial_investment:
                # Linear interpolation for partial years
                if t == 1:
                    return 1.0
                previous_pv = cumulative_pv - discounted_cf
                remaining = initial_investment - previous_pv
                fractional_year = remaining / discounted_cf
                return t - 1 + fractional_year
        
        return None  # Never pays back
    
    def _calculate_profitability_index(self, cash_flows):
        """Profitability Index = PV of future cash flows / Initial investment"""
        if len(cash_flows) < 2 or cash_flows[0] >= 0:
            return 0.0
        
        initial_investment = abs(cash_flows[0])
        pv_future_cash = sum(
            cf / ((1 + self.discount_rate) ** t) 
            for t, cf in enumerate(cash_flows[1:], 1)
        )
        
        return pv_future_cash / initial_investment
    
    def _calculate_annual_metrics(self, production, cash_flows, oil_price, opex_per_bbl):
        """Calculate detailed annual metrics"""
        metrics = []
        
        for year in range(len(production)):
            annual_metrics = {
                'year': year + 1,
                'production_bbl': production[year],
                'revenue_usd': production[year] * oil_price,
                'opex_usd': production[year] * opex_per_bbl + self.fixed_opex,
                'cash_flow_usd': cash_flows[year + 1] if year + 1 < len(cash_flows) else 0,
                'cumulative_cash_flow_usd': sum(cash_flows[1:year+2]),
                'unit_technical_cost': (production[year] * opex_per_bbl + self.fixed_opex) / 
                                      production[year] if production[year] > 0 else 0
            }
            metrics.append(annual_metrics)
        
        return metrics
    
    def _calculate_break_even_price(self, cash_flows, production_forecast):
        """Calculate minimum oil price for NPV = 0"""
        total_production = sum(production_forecast)
        if total_production == 0:
            return 0.0
        
        # Simple approximation: Price where revenue covers all costs
        total_costs = sum(abs(cf) for cf in cash_flows if cf < 0)
        break_even = total_costs / total_production
        
        return break_even
    
    def _calculate_udc(self, cash_flows, total_reserves):
        """Calculate Unit Development Cost (USD/bbl)"""
        if total_reserves == 0:
            return 0.0
        
        total_capex = abs(cash_flows[0])
        return total_capex / total_reserves
    
    def sensitivity_analysis(self, production_forecast, base_oil_price, base_opex, 
                           variables=['oil_price', 'opex', 'capex']):
        """Perform sensitivity analysis on key variables"""
        results = {}
        
        # Base case
        base_results = self.calculate_project_economics(
            production_forecast, base_oil_price, base_opex
        )
        base_npv = base_results['npv']
        
        # Sensitivity ranges (±20%)
        for variable in variables:
            if variable == 'oil_price':
                variations = [base_oil_price * (1 + d) for d in [-0.2, -0.1, 0, 0.1, 0.2]]
                npv_changes = []
                for price in variations:
                    results_var = self.calculate_project_economics(
                        production_forecast, price, base_opex
                    )
                    npv_changes.append((price, results_var['npv']))
                results[variable] = npv_changes
                
            elif variable == 'opex':
                variations = [base_opex * (1 + d) for d in [-0.2, -0.1, 0, 0.1, 0.2]]
                npv_changes = []
                for opex in variations:
                    results_var = self.calculate_project_economics(
                        production_forecast, base_oil_price, opex
                    )
                    npv_changes.append((opex, results_var['npv']))
                results[variable] = npv_changes
                
            elif variable == 'capex':
                variations = [self.capex * (1 + d) for d in [-0.2, -0.1, 0, 0.1, 0.2]]
                npv_changes = []
                for capex in variations:
                    original_capex = self.capex
                    self.capex = capex
                    results_var = self.calculate_project_economics(
                        production_forecast, base_oil_price, base_opex
                    )
                    npv_changes.append((capex, results_var['npv']))
                    self.capex = original_capex
                results[variable] = npv_changes
        
        return {
            'base_case': base_results,
            'sensitivity': results,
            'tornado_data': self._prepare_tornado_data(results, base_npv)
        }
    
    def _prepare_tornado_data(self, sensitivity_results, base_npv):
        """Prepare data for tornado chart"""
        tornado_data = []
        
        for variable, variations in sensitivity_results.items():
            if variations:
                min_npv = min(v[1] for v in variations)
                max_npv = max(v[1] for v in variations)
                tornado_data.append({
                    'variable': variable,
                    'min_impact': min_npv - base_npv,
                    'max_impact': max_npv - base_npv,
                    'range': abs(max_npv - min_npv)
                })
        
        # Sort by impact range
        tornado_data.sort(key=lambda x: x['range'], reverse=True)
        return tornado_data
