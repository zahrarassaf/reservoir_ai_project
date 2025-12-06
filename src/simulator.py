import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
from scipy import integrate, interpolate, optimize
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class ReservoirProperties:
    porosity: float
    permeability: float
    thickness: float
    area: float
    compressibility: float
    initial_pressure: float
    temperature: float
    water_saturation: float
    oil_viscosity: float
    formation_volume_factor: float

@dataclass
class SimulationParameters:
    forecast_years: int = 3
    oil_price: float = 75.0
    operating_cost: float = 18.0
    discount_rate: float = 0.12
    time_step_days: int = 30
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    minimum_production_rate: float = 10.0
    economic_limit: float = 5.0
    initial_investment_per_well: float = 3000000.0
    annual_fixed_costs: float = 1000000.0

class MaterialBalance:
    
    @staticmethod
    def calculate_oip(area: float, thickness: float, porosity: float, 
                     sw: float, boi: float) -> float:
        if area <= 0 or thickness <= 0 or porosity <= 0 or boi <= 0:
            return 0.0
        return 7758 * area * thickness * porosity * (1 - sw) / boi
    
    @staticmethod
    def calculate_cumulative_production(production_rates: np.ndarray, 
                                       time_points: np.ndarray) -> float:
        if len(production_rates) < 2 or len(time_points) < 2:
            if len(time_points) > 0 and len(production_rates) > 0:
                avg_rate = np.mean(production_rates)
                time_span = time_points[-1] - time_points[0]
                return avg_rate * time_span
            return 0.0
        
        min_len = min(len(production_rates), len(time_points))
        return np.trapz(production_rates[:min_len], time_points[:min_len])
    
    @staticmethod
    def solve_material_balance(oip: float, n_p: float, bo: float, boi: float,
                              ce: float, delta_p: float) -> float:
        if oip <= 0:
            return 0.0
            
        f = n_p * bo
        eo = (bo - boi) + boi * ce * delta_p
        
        return f / eo if abs(eo) > 1e-10 else oip

class DeclineCurveAnalysis:
    
    @staticmethod
    def hyperbolic_decline(qi: float, di: float, b: float, t: np.ndarray) -> np.ndarray:
        if b <= 0 or di <= 0:
            return qi * np.exp(-di * t)
        
        safe_b = min(max(b, 0.1), 1.9)
        safe_di = min(max(di, 1e-6), 1.0)
        
        denominator = 1 + safe_b * safe_di * t
        denominator = np.clip(denominator, 1e-10, None)
        
        return qi / denominator ** (1/safe_b)
    
    @staticmethod
    def exponential_decline(qi: float, di: float, t: np.ndarray) -> np.ndarray:
        safe_di = min(max(di, 1e-6), 1.0)
        return qi * np.exp(-safe_di * t)
    
    @staticmethod
    def harmonic_decline(qi: float, di: float, t: np.ndarray) -> np.ndarray:
        safe_di = min(max(di, 1e-6), 1.0)
        denominator = 1 + safe_di * t
        denominator = np.clip(denominator, 1e-10, None)
        return qi / denominator
    
    @staticmethod
    def fit_decline_curve(time: np.ndarray, rate: np.ndarray, 
                         method: str = 'hyperbolic') -> Dict:
        if len(time) < 3 or len(rate) < 3:
            return {}
        
        valid_mask = (rate > 0) & (~np.isnan(rate)) & (~np.isinf(rate))
        if np.sum(valid_mask) < 3:
            return {}
        
        time_valid = time[valid_mask]
        rate_valid = rate[valid_mask]
        
        t_norm = time_valid - time_valid[0]
        
        if method == 'exponential':
            try:
                log_rate = np.log(rate_valid)
                
                if len(t_norm) < 2:
                    return {}
                
                A = np.vstack([t_norm, np.ones_like(t_norm)]).T
                m, c = np.linalg.lstsq(A, log_rate, rcond=None)[0]
                
                qi_fit = np.exp(c)
                di_fit = max(-m, 1e-6)
                
                rate_fit = qi_fit * np.exp(-di_fit * t_norm)
                
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
            except Exception as e:
                logger.warning(f"Exponential fit failed: {e}")
                return {}
        
        elif method == 'hyperbolic':
            try:
                def hyperbolic_func(t, qi, di, b):
                    b_safe = max(min(b, 1.9), 0.1)
                    di_safe = max(min(di, 0.5), 1e-6)
                    denom = 1 + b_safe * di_safe * t
                    denom = np.clip(denom, 1e-10, None)
                    return qi / denom ** (1/b_safe)
                
                qi_guess = rate_valid[0]
                if len(rate_valid) > 1:
                    rate_ratio = rate_valid[-1] / rate_valid[0]
                    if rate_ratio > 0 and t_norm[-1] > 0:
                        di_guess = -np.log(rate_ratio) / t_norm[-1]
                        di_guess = max(min(di_guess, 0.5), 0.001)
                    else:
                        di_guess = 0.01
                else:
                    di_guess = 0.01
                
                b_guess = 0.8
                
                bounds = (
                    [max(qi_guess * 0.1, 1), 1e-6, 0.1],
                    [qi_guess * 10, 1.0, 2.0]
                )
                
                popt, pcov = optimize.curve_fit(
                    hyperbolic_func,
                    t_norm,
                    rate_valid,
                    p0=[qi_guess, di_guess, b_guess],
                    bounds=bounds,
                    maxfev=5000,
                    method='trf'
                )
                
                qi_fit, di_fit, b_fit = popt
                
                qi_fit = max(qi_fit, 1)
                di_fit = max(min(di_fit, 0.5), 1e-6)
                b_fit = max(min(b_fit, 1.9), 0.1)
                
                rate_fit = hyperbolic_func(t_norm, qi_fit, di_fit, b_fit)
                
                ss_res = np.sum((rate_valid - rate_fit) ** 2)
                ss_tot = np.sum((rate_valid - np.mean(rate_valid)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                if r_squared < 0.7:
                    logger.warning(f"Poor hyperbolic fit R²={r_squared:.3f}")
                    return DeclineCurveAnalysis.fit_decline_curve(time, rate, 'exponential')
                
                return {
                    'qi': qi_fit,
                    'di': di_fit,
                    'b': b_fit,
                    'r_squared': r_squared,
                    'method': 'hyperbolic'
                }
                
            except Exception as e:
                logger.warning(f"Hyperbolic fit failed, trying exponential: {e}")
                return DeclineCurveAnalysis.fit_decline_curve(time, rate, 'exponential')
        
        return {}

class ReservoirSimulator:
    
    def __init__(self, data, params: SimulationParameters):
        self.data = data
        self.params = params
        self.results = {}
        self.well_analysis = {}
        self._initialize_well_data()
    
    def _initialize_well_data(self):
        if not hasattr(self.data, 'wells'):
            self.data.wells = {}
    
    def _get_parameters_summary(self) -> Dict:
        return {
            'forecast_years': self.params.forecast_years,
            'oil_price': self.params.oil_price,
            'operating_cost': self.params.operating_cost,
            'discount_rate': self.params.discount_rate,
            'time_step_days': self.params.time_step_days,
            'max_iterations': self.params.max_iterations,
            'convergence_tolerance': self.params.convergence_tolerance,
            'initial_investment_per_well': self.params.initial_investment_per_well,
            'annual_fixed_costs': self.params.annual_fixed_costs
        }
    
    def _generate_error_results(self) -> Dict:
        return {
            'error': 'Simulation failed due to insufficient data',
            'material_balance': {},
            'decline_analysis': {},
            'production_forecast': {},
            'economic_analysis': {
                'npv': 0,
                'irr': 0,
                'roi': 0,
                'total_revenue': 0,
                'total_opex': 0,
                'gross_profit': 0,
                'net_profit': 0,
                'payback_period_years': float('inf'),
                'initial_investment': 0
            },
            'simulation_parameters': self._get_parameters_summary()
        }
    
    def run_comprehensive_simulation(self) -> Dict:
        logger.info("Starting comprehensive reservoir simulation")
        
        if not self._validate_data():
            logger.error("Invalid input data")
            return self._generate_error_results()
        
        material_balance_results = self._perform_material_balance()
        
        decline_analysis_results = self._perform_decline_analysis()
        
        forecast_results = self._generate_production_forecast(
            material_balance_results,
            decline_analysis_results
        )
        
        economic_results = self._perform_economic_analysis(forecast_results)
        
        self.results = {
            'material_balance': material_balance_results,
            'decline_analysis': decline_analysis_results,
            'production_forecast': forecast_results,
            'economic_analysis': economic_results,
            'simulation_parameters': self._get_parameters_summary()
        }
        
        logger.info("Simulation completed successfully")
        return self.results
    
    def _validate_data(self) -> bool:
        if not self.data.wells or len(self.data.wells) == 0:
            logger.error("No well data available")
            return False
        
        valid_wells = 0
        for well_name, well in self.data.wells.items():
            if (hasattr(well, 'time_points') and hasattr(well, 'production_rates')):
                time_points = well.time_points if hasattr(well.time_points, '__len__') else []
                production_rates = well.production_rates if hasattr(well.production_rates, '__len__') else []
                
                if (len(time_points) >= 3 and len(production_rates) >= 3 and 
                    len(time_points) == len(production_rates)):
                    if np.any(production_rates > 0):
                        valid_wells += 1
        
        if valid_wells == 0:
            logger.error("No valid production data available")
            return False
        
        logger.info(f"Validated {valid_wells} wells with production data")
        return True
    
    def _perform_material_balance(self) -> Dict:
        results = {}
        mb = MaterialBalance()
        
        for well_name, well in self.data.wells.items():
            if not hasattr(well, 'time_points') or not hasattr(well, 'production_rates'):
                continue
            
            time_points = well.time_points
            production_rates = well.production_rates
            
            if len(time_points) < 2 or len(production_rates) < 2:
                continue
            
            min_len = min(len(time_points), len(production_rates))
            time_points = time_points[:min_len]
            production_rates = production_rates[:min_len]
            
            cum_production = mb.calculate_cumulative_production(
                production_rates,
                time_points
            )
            
            area = 40.0
            thickness = 50.0
            porosity = 0.15
            sw = 0.25
            boi = 1.2
            
            oip = mb.calculate_oip(area, thickness, porosity, sw, boi)
            
            recovery_factor = (cum_production / oip) if oip > 0 else 0
            
            results[well_name] = {
                'cumulative_production': cum_production,
                'estimated_oip': oip,
                'recovery_factor': min(recovery_factor, 1.0),
                'production_days': time_points[-1] - time_points[0] if len(time_points) > 0 else 0,
                'peak_rate': np.max(production_rates) if len(production_rates) > 0 else 0
            }
        
        return results
    
    def _perform_decline_analysis(self) -> Dict:
        results = {}
        dca = DeclineCurveAnalysis()
        
        for well_name, well in self.data.wells.items():
            if not hasattr(well, 'time_points') or not hasattr(well, 'production_rates'):
                continue
            
            time_data = well.time_points
            rate_data = well.production_rates
            
            if len(time_data) < 3 or len(rate_data) < 3:
                continue
            
            min_len = min(len(time_data), len(rate_data))
            time_data = time_data[:min_len]
            rate_data = rate_data[:min_len]
            
            valid_mask = ~np.isnan(rate_data) & ~np.isinf(rate_data) & (rate_data > 0)
            if np.sum(valid_mask) < 3:
                continue
            
            time_valid = time_data[valid_mask]
            rate_valid = rate_data[valid_mask]
            
            hyp_result = dca.fit_decline_curve(
                time_valid, rate_valid, method='hyperbolic'
            )
            
            if not hyp_result:
                exp_result = dca.fit_decline_curve(
                    time_valid, rate_valid, method='exponential'
                )
                if exp_result:
                    results[well_name] = exp_result
            else:
                results[well_name] = hyp_result
            
            if well_name in results:
                result = results[well_name]
                logger.info(f"Decline analysis for {well_name}: "
                          f"qi={result['qi']:.1f}, "
                          f"di={result['di']:.4f}, "
                          f"R²={result.get('r_squared', 0):.3f}")
        
        return results
    
    def _generate_production_forecast(self, mb_results: Dict, 
                                    dca_results: Dict) -> Dict:
        forecast_days = self.params.forecast_years * 365
        dca = DeclineCurveAnalysis()
        
        forecast_data = {}
        
        for well_name, well in self.data.wells.items():
            if well_name not in dca_results:
                continue
            
            if not hasattr(well, 'time_points'):
                continue
            
            params = dca_results[well_name]
            
            historical_time = well.time_points - well.time_points[0]
            
            forecast_start = historical_time[-1] if len(historical_time) > 0 else 0
            forecast_time = np.arange(
                forecast_start,
                forecast_start + forecast_days + self.params.time_step_days,
                self.params.time_step_days
            )
            
            if params['method'] == 'exponential':
                forecast_rates = dca.exponential_decline(
                    params['qi'], params['di'], forecast_time
                )
            elif params['method'] == 'hyperbolic':
                forecast_rates = dca.hyperbolic_decline(
                    params['qi'], params['di'], params['b'], forecast_time
                )
            else:
                forecast_rates = dca.exponential_decline(
                    params['qi'], params['di'], forecast_time
                )
            
            economic_limit = self.params.economic_limit
            forecast_rates = np.where(forecast_rates < economic_limit, 0, forecast_rates)
            
            if forecast_rates.size > 1:
                non_zero_idx = forecast_rates > 0
                if np.any(non_zero_idx):
                    eur = np.trapz(forecast_rates[non_zero_idx], 
                                 forecast_time[non_zero_idx])
                else:
                    eur = 0
            else:
                eur = 0
            
            forecast_data[well_name] = {
                'forecast_time': forecast_time,
                'forecast_rates': forecast_rates,
                'eur': eur,
                'decline_parameters': params,
                'economic_limit': economic_limit
            }
        
        return forecast_data
    
    def _perform_economic_analysis(self, forecast_results: Dict) -> Dict:
    """Perform realistic economic analysis on forecast results."""
    if not forecast_results:
        return {
            'npv': 0,
            'irr': 0,
            'roi': 0,
            'total_revenue': 0,
            'total_opex': 0,
            'gross_profit': 0,
            'net_profit': 0,
            'payback_period_years': float('inf'),
            'initial_investment': 0
        }
    
    # Calculate realistic initial investment
    num_wells = len(forecast_results)
    
    # More realistic investment costs (in USD)
    # - Drilling: $1-2M per well
    # - Completion: $0.5-1M per well
    # - Facilities: $0.5M per well
    # Total: ~$2-3M per well
    drilling_cost_per_well = 1_500_000  # USD
    completion_cost_per_well = 800_000   # USD
    facilities_cost = 2_000_000  # Fixed facility cost
    
    initial_investment = (num_wells * (drilling_cost_per_well + completion_cost_per_well) + 
                         facilities_cost)
    
    # Calculate annual production and revenues
    forecast_years = self.params.forecast_years
    annual_production = np.zeros(forecast_years)
    annual_revenues = np.zeros(forecast_years)
    annual_opex_costs = np.zeros(forecast_years)
    
    for well_name, forecast in forecast_results.items():
        days = forecast['forecast_time']
        rates = forecast['forecast_rates']
        
        if len(days) < 2:
            continue
        
        # Calculate annual production for each well
        for year in range(forecast_years):
            start_day = year * 365
            end_day = min((year + 1) * 365, days[-1])
            
            # Find indices for this year
            year_mask = (days >= start_day) & (days <= end_day)
            if not np.any(year_mask):
                continue
            
            year_days = days[year_mask]
            year_rates = rates[year_mask]
            
            if len(year_days) > 1:
                # Calculate production using trapezoidal integration
                production = np.trapz(year_rates, year_days)
            else:
                production = year_rates[0] * min(365, end_day - start_day)
            
            annual_production[year] += production
    
    # Calculate revenues and costs
    total_revenue = 0
    total_opex = 0
    
    for year in range(forecast_years):
        # Revenue from oil sales
        revenue = annual_production[year] * self.params.oil_price
        annual_revenues[year] = revenue
        
        # Operating costs (per barrel + fixed annual costs)
        variable_opex = annual_production[year] * self.params.operating_cost
        fixed_annual_opex = 500_000  # USD per year for maintenance, personnel, etc.
        annual_opex_costs[year] = variable_opex + fixed_annual_opex
        
        total_revenue += revenue
        total_opex += annual_opex_costs[year]
    
    # Calculate cash flows
    cash_flows = [-initial_investment]  # Year 0: initial investment
    
    for year in range(forecast_years):
        net_cash_flow = annual_revenues[year] - annual_opex_costs[year]
        cash_flows.append(net_cash_flow)
    
    # Calculate economic metrics
    npv_value = self._calculate_realistic_npv(cash_flows, self.params.discount_rate)
    irr_value = self._calculate_realistic_irr(cash_flows)
    
    # Calculate total profit and ROI
    total_profit = total_revenue - total_opex - initial_investment
    roi_value = (total_profit / initial_investment) * 100 if initial_investment > 0 else 0
    
    # Calculate payback period
    payback_years = self._calculate_realistic_payback(cash_flows)
    
    # Apply realistic constraints
    if irr_value > 1.0:  # Cap at 100% (1.0)
        irr_value = 1.0
    elif irr_value < -0.9:  # Cap at -90%
        irr_value = -0.9
    
    if roi_value > 300:  # Cap ROI at 300%
        roi_value = 300
    elif roi_value < -100:  # Cap ROI at -100%
        roi_value = -100
    
    return {
        'npv': npv_value / 1e6,  # Convert to millions
        'irr': irr_value * 100,  # Convert to percentage
        'roi': roi_value,
        'total_revenue': total_revenue,
        'total_opex': total_opex,
        'gross_profit': total_revenue - total_opex,
        'net_profit': total_profit,
        'payback_period_years': payback_years,
        'initial_investment': initial_investment
    }

def _calculate_realistic_npv(self, cash_flows: List[float], discount_rate: float) -> float:
    """Calculate NPV with realistic assumptions."""
    npv = 0.0
    for t, cf in enumerate(cash_flows):
        if cf != 0:
            denominator = (1 + discount_rate) ** t
            if denominator != 0:
                npv += cf / denominator
    
    # Round to avoid tiny negative/positive values
    if abs(npv) < 1000:
        npv = 0
    
    return npv

def _calculate_realistic_irr(self, cash_flows: List[float]) -> float:
    """Calculate IRR with realistic bounds and validation."""
    try:
        # Filter out trailing zeros
        cash_flows_array = np.array(cash_flows)
        non_zero_idx = np.where(cash_flows_array != 0)[0]
        
        if len(non_zero_idx) < 2:
            return 0.0
        
        # Check if there's at least one positive and one negative cash flow
        positive_flows = cash_flows_array[cash_flows_array > 0]
        negative_flows = cash_flows_array[cash_flows_array < 0]
        
        if len(positive_flows) == 0 or len(negative_flows) == 0:
            return 0.0
        
        # Try to find IRR using scipy's root finding
        def npv_func(rate):
            npv_val = 0.0
            for t, cf in enumerate(cash_flows):
                if cf != 0:
                    denominator = (1 + rate) ** t
                    if denominator != 0:
                        npv_val += cf / denominator
            return npv_val
        
        # Try different starting points
        test_points = [-0.5, -0.1, 0.0, 0.1, 0.5, 1.0]
        best_rate = 0.0
        min_error = float('inf')
        
        for start_rate in test_points:
            try:
                # Use secant method
                rate = start_rate
                for _ in range(50):
                    f_val = npv_func(rate)
                    f_prime = 0.0
                    
                    # Approximate derivative
                    epsilon = 1e-6
                    f_val_plus = npv_func(rate + epsilon)
                    f_prime = (f_val_plus - f_val) / epsilon if epsilon != 0 else 1
                    
                    if abs(f_prime) < 1e-12:
                        break
                    
                    rate_new = rate - f_val / f_prime
                    
                    # Keep rate within reasonable bounds
                    if rate_new < -0.9:
                        rate_new = -0.9
                    elif rate_new > 2.0:
                        rate_new = 2.0
                    
                    if abs(rate_new - rate) < 1e-8:
                        rate = rate_new
                        break
                    
                    rate = rate_new
                
                # Check if this is a better solution
                error = abs(npv_func(rate))
                if error < min_error and -0.9 < rate < 2.0:
                    min_error = error
                    best_rate = rate
                    
            except:
                continue
        
        if min_error < 1e-3:
            return best_rate
        else:
            return 0.0
            
    except Exception as e:
        logger.warning(f"IRR calculation failed: {e}")
        return 0.0

def _calculate_realistic_payback(self, cash_flows: List[float]) -> float:
    """Calculate realistic payback period."""
    if len(cash_flows) < 2:
        return float('inf')
    
    initial_investment = abs(cash_flows[0]) if cash_flows[0] < 0 else 0
    
    if initial_investment <= 0:
        return 0.0
    
    cumulative_cash_flow = 0.0
    for year, cf in enumerate(cash_flows[1:], start=1):
        cumulative_cash_flow += cf
        
        if cumulative_cash_flow >= initial_investment:
            # Linear interpolation for fractional year
            previous_cumulative = cumulative_cash_flow - cf
            cash_flow_in_year = cf
            
            if cash_flow_in_year > 0:
                fraction = (initial_investment - previous_cumulative) / cash_flow_in_year
                return year - 1 + fraction
            else:
                return year - 1
    
    return float('inf')
