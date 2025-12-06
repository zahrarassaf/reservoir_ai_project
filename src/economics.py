import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Any
from numpy_financial import irr as npf_irr, npv as npf_npv
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

@dataclass
class SimulationParameters:
    forecast_years: int = 10
    oil_price: float = 75.0
    operating_cost: float = 18.0
    discount_rate: float = 0.10
    capex_per_well: float = 1_000_000
    fixed_annual_opex: float = 500_000
    tax_rate: float = 0.30
    royalty_rate: float = 0.125

class ReservoirSimulator:
    def __init__(self, data: Dict, params: SimulationParameters):
        self.data = data
        self.params = params
        self.results = {}
        
        if 'wells' not in self.data:
            self.data['wells'] = {}
    
    def run_comprehensive_simulation(self) -> Dict:
        logger.info("Starting comprehensive reservoir simulation")
        
        try:
            decline_results = self._analyze_production_history()
            
            forecast_results = self._generate_production_forecast(decline_results)
            
            economic_results = self._perform_economic_analysis(forecast_results)
            
            self.results = {
                'decline_analysis': decline_results,
                'production_forecast': forecast_results,
                'economic_analysis': economic_results,
                'simulation_parameters': {
                    'forecast_years': self.params.forecast_years,
                    'oil_price': self.params.oil_price,
                    'operating_cost': self.params.operating_cost,
                    'discount_rate': self.params.discount_rate
                }
            }
            
            logger.info("Simulation completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return self._get_empty_results()
    
    def _analyze_production_history(self) -> Dict:
        decline_results = {}
        
        for well_name, well in self.data.get('wells', {}).items():
            try:
                if hasattr(well, 'production_rates') and hasattr(well, 'time_points'):
                    rates = well.production_rates
                    times = well.time_points
                    
                    if len(rates) >= 3 and len(times) >= 3:
                        result = self._fit_decline_curve(times, rates)
                        if result:
                            decline_results[well_name] = result
                            logger.info(f"Decline analysis for {well_name}: qi={result['qi']:.1f}, di={result['di']:.4f}, R²={result.get('r_squared', 0):.3f}")
            except:
                continue
        
        return decline_results
    
    def _fit_decline_curve(self, time_data, rate_data):
        if len(time_data) < 3 or len(rate_data) < 3:
            return None
        
        valid_mask = (rate_data > 0) & (~np.isnan(rate_data))
        if np.sum(valid_mask) < 3:
            return None
        
        time_valid = time_data[valid_mask]
        rate_valid = rate_data[valid_mask]
        t_norm = time_valid - time_valid[0]
        
        try:
            def exp_func(t, qi, di):
                return qi * np.exp(-di * t)
            
            p0 = [rate_valid[0], 0.001]
            bounds = ([0, 1e-6], [np.inf, 1])
            
            popt, pcov = curve_fit(exp_func, t_norm, rate_valid, p0=p0, bounds=bounds, maxfev=5000)
            
            qi_fit, di_fit = popt
            qi_fit = max(qi_fit, 1)
            di_fit = max(min(di_fit, 0.5), 1e-6)
            
            rate_fit = exp_func(t_norm, qi_fit, di_fit)
            
            ss_res = np.sum((rate_valid - rate_fit) ** 2)
            ss_tot = np.sum((rate_valid - np.mean(rate_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'qi': qi_fit,
                'di': di_fit,
                'b': 0,
                'r_squared': r_squared,
                'method': 'exponential'
            }
        except:
            return None
    
    def _generate_production_forecast(self, decline_results: Dict) -> Dict:
        forecast_days = self.params.forecast_years * 365
        time_step = 30
        forecast_results = {}
        
        for well_name, params in decline_results.items():
            well = self.data['wells'].get(well_name)
            if not well or not hasattr(well, 'time_points'):
                continue
            
            historical_time = well.time_points - well.time_points[0]
            forecast_start = historical_time[-1] if len(historical_time) > 0 else 0
            
            forecast_time = np.arange(
                forecast_start,
                forecast_start + forecast_days + time_step,
                time_step
            )
            
            qi = params['qi']
            di = params['di']
            
            if params['method'] == 'exponential':
                forecast_rates = qi * np.exp(-di * forecast_time)
            else:
                forecast_rates = qi * np.exp(-di * forecast_time)
            
            economic_limit = 20.0
            forecast_rates = np.where(forecast_rates < economic_limit, 0, forecast_rates)
            
            if forecast_rates.size > 1:
                non_zero_idx = forecast_rates > 0
                if np.any(non_zero_idx):
                    eur = np.trapz(forecast_rates[non_zero_idx], forecast_time[non_zero_idx])
                else:
                    eur = 0
            else:
                eur = 0
            
            forecast_results[well_name] = {
                'forecast_time': forecast_time,
                'forecast_rates': forecast_rates,
                'eur': eur,
                'decline_parameters': params
            }
        
        return forecast_results
    
    def _perform_economic_analysis(self, forecast_results: Dict) -> Dict:
        if not forecast_results:
            return self._get_empty_economic_results()
        
        num_wells = len(forecast_results)
        
        well_cost = self.params.capex_per_well
        facilities_cost = 1_000_000
        infrastructure_cost = 500_000
        engineering_contingency = 0.20 * (num_wells * well_cost + facilities_cost + infrastructure_cost)
        
        initial_investment = (num_wells * well_cost + facilities_cost + infrastructure_cost + engineering_contingency)
        max_capex = 15_000_000
        initial_investment = min(initial_investment, max_capex)
        
        annual_production = self._calculate_annual_production(forecast_results)
        
        cash_flows = self._calculate_cash_flows(annual_production, initial_investment)
        
        economic_metrics = self._calculate_economic_metrics(cash_flows, annual_production)
        
        total_reserves = np.sum(annual_production)
        economic_metrics.update({
            'unit_development_cost': initial_investment / total_reserves if total_reserves > 0 else 0,
            'break_even_price': self._calculate_break_even_price(cash_flows, annual_production),
            'total_reserves': total_reserves,
            'peak_production': np.max(annual_production) if len(annual_production) > 0 else 0,
            'annual_production': annual_production.tolist(),
            'initial_investment': initial_investment
        })
        
        return economic_metrics
    
    def _calculate_annual_production(self, forecast_results: Dict) -> np.ndarray:
        forecast_years = self.params.forecast_years
        annual_production = np.zeros(forecast_years)
        
        for well_name, forecast in forecast_results.items():
            days = forecast['forecast_time']
            rates = forecast['forecast_rates']
            
            if len(days) < 2:
                continue
            
            for year in range(forecast_years):
                start_day = year * 365
                end_day = min((year + 1) * 365, days[-1])
                
                year_mask = (days >= start_day) & (days <= end_day)
                if not np.any(year_mask):
                    continue
                
                year_days = days[year_mask]
                year_rates = rates[year_mask]
                
                if len(year_days) > 1:
                    production = np.trapz(year_rates, year_days)
                else:
                    production = year_rates[0] * min(365, end_day - start_day)
                
                annual_production[year] += production
        
        return annual_production
    
    def _calculate_cash_flows(self, annual_production: np.ndarray, initial_investment: float) -> List[float]:
        forecast_years = len(annual_production)
        cash_flows = [-initial_investment]
        
        for year in range(forecast_years):
            annual_revenue = annual_production[year] * self.params.oil_price
            
            royalty = annual_revenue * self.params.royalty_rate
            
            variable_opex = annual_production[year] * self.params.operating_cost
            fixed_opex = self.params.fixed_annual_opex
            
            if year < 7:
                depreciation = initial_investment / 7
            else:
                depreciation = 0
            
            annual_opex = variable_opex + fixed_opex
            
            ebit = annual_revenue - royalty - annual_opex - depreciation
            
            tax = max(0, ebit) * self.params.tax_rate
            
            annual_cash_flow = ebit - tax + depreciation
            
            cash_flows.append(annual_cash_flow)
        
        abandonment_cost = 0.05 * initial_investment
        cash_flows[-1] -= abandonment_cost
        
        return cash_flows
    
    def _calculate_economic_metrics(self, cash_flows: List[float], annual_production: np.ndarray) -> Dict:
        npv_value = self._calculate_npv(cash_flows, self.params.discount_rate)
        
        irr_value = self._calculate_irr(cash_flows)
        
        initial_investment = abs(cash_flows[0]) if cash_flows[0] < 0 else 0
        total_positive_cash = sum(cf for cf in cash_flows[1:] if cf > 0)
        roi_value = (total_positive_cash / initial_investment) * 100 if initial_investment > 0 else 0
        
        payback_years = self._calculate_payback(cash_flows)
        
        total_revenue = np.sum(annual_production) * self.params.oil_price
        total_variable_opex = np.sum(annual_production) * self.params.operating_cost
        total_fixed_opex = len(annual_production) * self.params.fixed_annual_opex
        total_opex = total_variable_opex + total_fixed_opex
        
        gross_profit = total_revenue - total_opex
        net_profit = sum(cash_flows)
        
        return {
            'npv': npv_value / 1_000_000,
            'irr': irr_value * 100,
            'roi': roi_value,
            'total_revenue': total_revenue,
            'total_opex': total_opex,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'payback_period_years': payback_years
        }
    
    def _calculate_npv(self, cash_flows: List[float], discount_rate: float) -> float:
        try:
            return npf_npv(discount_rate, cash_flows)
        except:
            npv = 0.0
            for t, cf in enumerate(cash_flows):
                denominator = (1 + discount_rate) ** t
                npv += cf / denominator
            return npv
    
    def _calculate_irr(self, cash_flows: List[float]) -> float:
        try:
            trimmed = [cf for cf in cash_flows if abs(cf) > 1e-10]
            if len(trimmed) < 2:
                return 0.0
            
            irr_value = npf_irr(trimmed)
            
            if irr_value is None or np.isnan(irr_value):
                return 0.0
            
            if irr_value < -0.9 or irr_value > 10:
                return 0.0
            
            return irr_value
            
        except:
            return 0.0
    
    def _calculate_payback(self, cash_flows: List[float]) -> float:
        if len(cash_flows) < 2:
            return float('inf')
        
        initial_investment = abs(cash_flows[0]) if cash_flows[0] < 0 else 0
        
        if initial_investment <= 0:
            return 0.0
        
        cumulative = 0.0
        for year, cf in enumerate(cash_flows[1:], start=1):
            cumulative += cf
            
            if cumulative >= initial_investment:
                prev_cumulative = cumulative - cf
                cf_in_year = cf
                
                if cf_in_year > 0:
                    fraction = (initial_investment - prev_cumulative) / cf_in_year
                    return year - 1 + fraction
                else:
                    return year - 1
        
        return float('inf')
    
    def _calculate_break_even_price(self, cash_flows: List[float], annual_production: np.ndarray) -> float:
        total_production = np.sum(annual_production)
        if total_production <= 0:
            return 0.0
        
        total_costs = sum(abs(cf) for cf in cash_flows if cf < 0)
        return total_costs / total_production
    
    def _get_empty_economic_results(self) -> Dict:
        return {
            'npv': 0.0,
            'irr': 0.0,
            'roi': 0.0,
            'total_revenue': 0.0,
            'total_opex': 0.0,
            'gross_profit': 0.0,
            'net_profit': 0.0,
            'payback_period_years': float('inf'),
            'initial_investment': 0.0,
            'unit_development_cost': 0.0,
            'break_even_price': 0.0,
            'total_reserves': 0.0,
            'peak_production': 0.0,
            'annual_production': []
        }
    
    def _get_empty_results(self) -> Dict:
        return {
            'decline_analysis': {},
            'production_forecast': {},
            'economic_analysis': self._get_empty_economic_results(),
            'simulation_parameters': {}
        }
