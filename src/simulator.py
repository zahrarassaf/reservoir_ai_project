import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
from scipy import integrate, interpolate, optimize
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class ReservoirProperties:
    porosity: float
    permeability: float  # md
    thickness: float    # ft
    area: float         # acres
    compressibility: float  # psi^-1
    initial_pressure: float  # psia
    temperature: float  # °F
    water_saturation: float
    oil_viscosity: float  # cp
    formation_volume_factor: float  # rb/stb

@dataclass
class SimulationParameters:
    forecast_years: int = 3
    oil_price: float = 75.0  # USD/bbl
    operating_cost: float = 18.0  # USD/bbl
    discount_rate: float = 0.12  # 12% annual
    time_step_days: int = 30
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    minimum_production_rate: float = 10.0  # Minimum production rate STB/day
    economic_limit: float = 5.0  # Economic limit STB/day

class MaterialBalance:
    """
    Implement material balance equation for reservoir simulation.
    """
    
    @staticmethod
    def calculate_oip(area: float, thickness: float, porosity: float, 
                     sw: float, boi: float) -> float:
        """Calculate Original Oil In Place (STB)."""
        # 7758 = conversion factor (acre-ft to bbl)
        if area <= 0 or thickness <= 0 or porosity <= 0 or boi <= 0:
            return 0.0
        return 7758 * area * thickness * porosity * (1 - sw) / boi
    
    @staticmethod
    def calculate_cumulative_production(production_rates: np.ndarray, 
                                       time_points: np.ndarray) -> float:
        """Calculate cumulative production using trapezoidal integration."""
        if len(production_rates) < 2 or len(time_points) < 2:
            return production_rates[0] * (time_points[-1] - time_points[0]) if len(time_points) > 0 else 0.0
        
        # Ensure arrays are the same length
        min_len = min(len(production_rates), len(time_points))
        return np.trapz(production_rates[:min_len], time_points[:min_len])
    
    @staticmethod
    def solve_material_balance(oip: float, n_p: float, bo: float, boi: float,
                              ce: float, delta_p: float) -> float:
        """
        Solve simplified material balance equation:
        F = N * Eo + We
        
        Where:
        F = underground withdrawal
        N = original oil in place
        Eo = oil expansion term
        We = water influx
        """
        # Simplified for volumetric depletion drive
        if oip <= 0:
            return 0.0
            
        f = n_p * bo
        eo = (bo - boi) + boi * ce * delta_p
        
        return f / eo if abs(eo) > 1e-10 else oip

class DeclineCurveAnalysis:
    """
    Implement Arps decline curve analysis with robust fitting.
    """
    
    @staticmethod
    def hyperbolic_decline(qi: float, di: float, b: float, t: np.ndarray) -> np.ndarray:
        """
        Calculate production rate using hyperbolic decline.
        
        Args:
            qi: Initial production rate
            di: Initial decline rate
            b: Decline exponent (0 < b <= 2)
            t: Time array
        
        Returns:
            Production rates over time
        """
        if b <= 0 or di <= 0:
            # Fall back to exponential decline
            return qi * np.exp(-di * t)
        
        # Prevent division by zero and overflow
        safe_b = min(max(b, 0.1), 1.9)
        safe_di = min(max(di, 1e-6), 1.0)
        
        denominator = 1 + safe_b * safe_di * t
        # Clip denominator to prevent overflow
        denominator = np.clip(denominator, 1e-10, None)
        
        return qi / denominator ** (1/safe_b)
    
    @staticmethod
    def exponential_decline(qi: float, di: float, t: np.ndarray) -> np.ndarray:
        """Calculate production rate using exponential decline."""
        safe_di = min(max(di, 1e-6), 1.0)
        return qi * np.exp(-safe_di * t)
    
    @staticmethod
    def harmonic_decline(qi: float, di: float, t: np.ndarray) -> np.ndarray:
        """Calculate production rate using harmonic decline."""
        safe_di = min(max(di, 1e-6), 1.0)
        denominator = 1 + safe_di * t
        denominator = np.clip(denominator, 1e-10, None)
        return qi / denominator
    
    @staticmethod
    def fit_decline_curve(time: np.ndarray, rate: np.ndarray, 
                         method: str = 'hyperbolic') -> Dict:
        """
        Fit decline curve parameters to historical data.
        
        Returns:
            Dictionary with fitted parameters and R² value
        """
        if len(time) < 3 or len(rate) < 3:
            return {}
        
        # Remove any zero or negative rates
        valid_mask = (rate > 0) & (~np.isnan(rate)) & (~np.isinf(rate))
        if np.sum(valid_mask) < 3:
            return {}
        
        time_valid = time[valid_mask]
        rate_valid = rate[valid_mask]
        
        # Normalize time
        t_norm = time_valid - time_valid[0]
        
        if method == 'exponential':
            try:
                # Linear regression on log(q) vs t
                log_rate = np.log(rate_valid)
                
                if len(t_norm) < 2:
                    return {}
                
                A = np.vstack([t_norm, np.ones_like(t_norm)]).T
                m, c = np.linalg.lstsq(A, log_rate, rcond=None)[0]
                
                qi_fit = np.exp(c)
                di_fit = max(-m, 1e-6)  # Ensure positive decline
                
                # Calculate fitted rates
                rate_fit = qi_fit * np.exp(-di_fit * t_norm)
                
                # Calculate R²
                ss_res = np.sum((rate_valid - rate_fit) ** 2)
                ss_tot = np.sum((rate_valid - np.mean(rate_valid)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                return {
                    'qi': qi_fit,
                    'di': di_fit,
                    'b': 0,  # Exponential has b=0
                    'r_squared': r_squared,
                    'method': 'exponential'
                }
            except Exception as e:
                logger.warning(f"Exponential fit failed: {e}")
                return {}
        
        elif method == 'hyperbolic':
            try:
                # Use more robust fitting approach
                def hyperbolic_func(t, qi, di, b):
                    # Safe implementation
                    b_safe = max(min(b, 1.9), 0.1)
                    di_safe = max(min(di, 0.5), 1e-6)
                    denom = 1 + b_safe * di_safe * t
                    denom = np.clip(denom, 1e-10, None)
                    return qi / denom ** (1/b_safe)
                
                # Initial guess from data
                qi_guess = rate_valid[0]
                # Estimate decline rate from first and last points
                if len(rate_valid) > 1:
                    rate_ratio = rate_valid[-1] / rate_valid[0]
                    if rate_ratio > 0 and t_norm[-1] > 0:
                        di_guess = -np.log(rate_ratio) / t_norm[-1]
                        di_guess = max(min(di_guess, 0.5), 0.001)
                    else:
                        di_guess = 0.01
                else:
                    di_guess = 0.01
                
                b_guess = 0.8  # Reasonable starting point
                
                # Use bounds for stability
                bounds = (
                    [max(qi_guess * 0.1, 1), 1e-6, 0.1],  # Lower bounds
                    [qi_guess * 10, 1.0, 2.0]             # Upper bounds
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
                
                # Validate parameters
                qi_fit = max(qi_fit, 1)
                di_fit = max(min(di_fit, 0.5), 1e-6)
                b_fit = max(min(b_fit, 1.9), 0.1)
                
                # Calculate fitted rates
                rate_fit = hyperbolic_func(t_norm, qi_fit, di_fit, b_fit)
                
                # Calculate R²
                ss_res = np.sum((rate_valid - rate_fit) ** 2)
                ss_tot = np.sum((rate_valid - np.mean(rate_valid)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Ensure good fit
                if r_squared < 0.7:
                    logger.warning(f"Poor hyperbolic fit R²={r_squared:.3f}")
                    # Fall back to exponential
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
    """
    Main reservoir simulator implementing material balance and decline curve analysis.
    """
    
    def __init__(self, data, params: SimulationParameters):
        self.data = data
        self.params = params
        self.results = {}
        self.well_analysis = {}
        self._initialize_well_data()
    
    def _initialize_well_data(self):
        """Initialize and validate well data."""
        if not hasattr(self.data, 'wells'):
            self.data.wells = {}
    
    def _get_parameters_summary(self) -> Dict:
        """Get summary of simulation parameters."""
        return {
            'forecast_years': self.params.forecast_years,
            'oil_price': self.params.oil_price,
            'operating_cost': self.params.operating_cost,
            'discount_rate': self.params.discount_rate,
            'time_step_days': self.params.time_step_days,
            'max_iterations': self.params.max_iterations,
            'convergence_tolerance': self.params.convergence_tolerance
        }
    
    def _generate_error_results(self) -> Dict:
        """Generate error results when simulation fails."""
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
        """
        Run complete reservoir simulation including:
        1. Material balance analysis
        2. Decline curve analysis
        3. Production forecasting
        4. Economic analysis
        """
        logger.info("Starting comprehensive reservoir simulation")
        
        # Validate input data
        if not self._validate_data():
            logger.error("Invalid input data")
            return self._generate_error_results()
        
        # Step 1: Material balance for each well
        material_balance_results = self._perform_material_balance()
        
        # Step 2: Decline curve analysis
        decline_analysis_results = self._perform_decline_analysis()
        
        # Step 3: Generate forecast
        forecast_results = self._generate_production_forecast(
            material_balance_results,
            decline_analysis_results
        )
        
        # Step 4: Economic analysis
        economic_results = self._perform_economic_analysis(forecast_results)
        
        # Combine results
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
        """Validate input data for simulation."""
        if not self.data.wells or len(self.data.wells) == 0:
            logger.error("No well data available")
            return False
        
        valid_wells = 0
        for well_name, well in self.data.wells.items():
            # Check if well has time_points and production_rates attributes
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
        """Perform material balance analysis for each well."""
        results = {}
        mb = MaterialBalance()
        
        for well_name, well in self.data.wells.items():
            if not hasattr(well, 'time_points') or not hasattr(well, 'production_rates'):
                continue
            
            time_points = well.time_points
            production_rates = well.production_rates
            
            if len(time_points) < 2 or len(production_rates) < 2:
                continue
            
            # Ensure same length
            min_len = min(len(time_points), len(production_rates))
            time_points = time_points[:min_len]
            production_rates = production_rates[:min_len]
            
            # Calculate cumulative production
            cum_production = mb.calculate_cumulative_production(
                production_rates,
                time_points
            )
            
            # Estimate reservoir properties (from SPE9 typical values)
            area = 40.0  # acres (assumed)
            thickness = 50.0  # ft (assumed)
            porosity = 0.15  # fraction (assumed)
            sw = 0.25  # water saturation (assumed)
            boi = 1.2  # initial oil FVF (rb/stb)
            
            # Calculate OOIP
            oip = mb.calculate_oip(area, thickness, porosity, sw, boi)
            
            # Recovery factor
            recovery_factor = (cum_production / oip) if oip > 0 else 0
            
            results[well_name] = {
                'cumulative_production': cum_production,
                'estimated_oip': oip,
                'recovery_factor': min(recovery_factor, 1.0),  # Cap at 100%
                'production_days': time_points[-1] - time_points[0] if len(time_points) > 0 else 0,
                'peak_rate': np.max(production_rates) if len(production_rates) > 0 else 0
            }
        
        return results
    
    def _perform_decline_analysis(self) -> Dict:
        """Perform decline curve analysis for each well."""
        results = {}
        dca = DeclineCurveAnalysis()
        
        for well_name, well in self.data.wells.items():
            if not hasattr(well, 'time_points') or not hasattr(well, 'production_rates'):
                continue
            
            time_data = well.time_points
            rate_data = well.production_rates
            
            if len(time_data) < 3 or len(rate_data) < 3:
                continue
            
            # Ensure same length and valid data
            min_len = min(len(time_data), len(rate_data))
            time_data = time_data[:min_len]
            rate_data = rate_data[:min_len]
            
            # Remove any NaNs or infs
            valid_mask = ~np.isnan(rate_data) & ~np.isinf(rate_data) & (rate_data > 0)
            if np.sum(valid_mask) < 3:
                continue
            
            time_valid = time_data[valid_mask]
            rate_valid = rate_data[valid_mask]
            
            # Try hyperbolic fit first
            hyp_result = dca.fit_decline_curve(
                time_valid, rate_valid, method='hyperbolic'
            )
            
            # If hyperbolic fails, try exponential
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
        """Generate production forecast using decline curve parameters."""
        forecast_days = self.params.forecast_years * 365
        dca = DeclineCurveAnalysis()
        
        forecast_data = {}
        
        for well_name, well in self.data.wells.items():
            if well_name not in dca_results:
                continue
            
            if not hasattr(well, 'time_points'):
                continue
            
            params = dca_results[well_name]
            
            # Historical time
            historical_time = well.time_points - well.time_points[0]
            
            # Forecast time
            forecast_start = historical_time[-1] if len(historical_time) > 0 else 0
            forecast_time = np.arange(
                forecast_start,
                forecast_start + forecast_days + self.params.time_step_days,
                self.params.time_step_days
            )
            
            # Calculate forecast rates
            if params['method'] == 'exponential':
                forecast_rates = dca.exponential_decline(
                    params['qi'], params['di'], forecast_time
                )
            elif params['method'] == 'hyperbolic':
                forecast_rates = dca.hyperbolic_decline(
                    params['qi'], params['di'], params['b'], forecast_time
                )
            else:
                # Default to exponential
                forecast_rates = dca.exponential_decline(
                    params['qi'], params['di'], forecast_time
                )
            
            # Apply economic limit
            economic_limit = self.params.economic_limit
            forecast_rates = np.where(forecast_rates < economic_limit, 0, forecast_rates)
            
            # Calculate EUR (Estimated Ultimate Recovery)
            if forecast_rates.size > 1:
                # Find index where production goes below economic limit
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
        """Perform economic analysis on forecast results."""
        # Initial investment based on number of wells
        num_wells = len(forecast_results)
        initial_investment = num_wells * 2_500_000  # $2.5M per well
        
        total_revenue = 0
        total_opex = 0
        monthly_cash_flows = []
        
        # Group production by month for all wells
        all_months = set()
        monthly_production = {}
        
        for well_name, forecast in forecast_results.items():
            days = forecast['forecast_time']
            rates = forecast['forecast_rates']
            
            if len(days) < 2:
                continue
            
            # Convert to monthly production
            for i in range(len(days) - 1):
                month = int(days[i] / 30)  # Approximate month
                all_months.add(month)
                
                # Average production for this period
                avg_rate = 0.5 * (rates[i] + rates[i+1])
                time_interval = days[i+1] - days[i]
                production = avg_rate * time_interval
                
                if month not in monthly_production:
                    monthly_production[month] = 0
                monthly_production[month] += production
        
        # Calculate cash flows by month
        sorted_months = sorted(list(all_months))
        cash_flows = [-initial_investment]  # Initial investment
        
        for month in sorted_months:
            if month in monthly_production and monthly_production[month] > 0:
                production = monthly_production[month]
                revenue = production * self.params.oil_price
                opex = production * self.params.operating_cost
                
                total_revenue += revenue
                total_opex += opex
                
                monthly_cash_flow = revenue - opex
                cash_flows.append(monthly_cash_flow)
            else:
                cash_flows.append(0)
        
        # Calculate economic metrics
        if len(cash_flows) > 1:
            # NPV calculation with monthly discount rate
            monthly_discount_rate = (1 + self.params.discount_rate) ** (1/12) - 1
            npv_value = self._calculate_npv(cash_flows, monthly_discount_rate)
            
            # IRR calculation
            irr_value = self._calculate_irr(cash_flows)
            
            # ROI
            total_profit = total_revenue - total_opex - initial_investment
            roi_value = (total_profit / initial_investment) * 100 if initial_investment > 0 else 0
            
            # Payback period
            payback_years = self._calculate_payback(cash_flows)
        else:
            npv_value = 0
            irr_value = 0
            roi_value = 0
            payback_years = float('inf')
        
        return {
            'npv': npv_value / 1e6,  # Convert to millions
            'irr': irr_value * 100,  # Convert to percentage
            'roi': roi_value,
            'total_revenue': total_revenue,
            'total_opex': total_opex,
            'gross_profit': total_revenue - total_opex,
            'net_profit': total_revenue - total_opex - initial_investment,
            'payback_period_years': payback_years,
            'initial_investment': initial_investment
        }
    
    def _calculate_npv(self, cash_flows: List[float], discount_rate: float) -> float:
        """Calculate Net Present Value."""
        if discount_rate <= -1:  # Invalid discount rate
            return sum(cash_flows)
        
        npv = 0.0
        for t, cf in enumerate(cash_flows):
            if cf != 0:
                denominator = (1 + discount_rate) ** t
                if denominator != 0:
                    npv += cf / denominator
        return npv
    
    def _calculate_irr(self, cash_flows: List[float]) -> float:
        """Calculate Internal Rate of Return using Newton-Raphson method."""
        # Filter out trailing zeros
        cash_flows_array = np.array(cash_flows)
        non_zero_idx = np.where(cash_flows_array != 0)[0]
        
        if len(non_zero_idx) == 0:
            return 0.0
        
        last_non_zero = non_zero_idx[-1]
        trimmed_cash_flows = cash_flows[:last_non_zero + 1]
        
        # Check for valid IRR calculation
        if len(trimmed_cash_flows) < 2:
            return 0.0
        
        # Check for sign changes (necessary for IRR)
        signs = np.sign(trimmed_cash_flows)
        sign_changes = np.sum(np.diff(signs) != 0)
        
        if sign_changes < 1:
            return 0.0
        
        try:
            # Newton-Raphson method for IRR
            def npv_func(rate):
                npv = 0.0
                for t, cf in enumerate(trimmed_cash_flows):
                    npv += cf / ((1 + rate) ** t)
                return npv
            
            def npv_derivative(rate):
                deriv = 0.0
                for t, cf in enumerate(trimmed_cash_flows):
                    if t > 0:
                        deriv -= t * cf / ((1 + rate) ** (t + 1))
                return deriv
            
            # Initial guess
            rate_guess = 0.1  # 10% initial guess
            
            # Newton-Raphson iteration
            max_iter = 50
            tolerance = 1e-8
            
            for i in range(max_iter):
                f_val = npv_func(rate_guess)
                f_deriv = npv_derivative(rate_guess)
                
                if abs(f_deriv) < 1e-12:
                    break
                
                rate_new = rate_guess - f_val / f_deriv
                
                # Check convergence
                if abs(rate_new - rate_guess) < tolerance:
                    rate_guess = rate_new
                    break
                
                # Ensure rate stays within reasonable bounds
                if rate_new < -0.99:
                    rate_new = -0.9
                elif rate_new > 10:
                    rate_new = 1.0
                
                rate_guess = rate_new
            
            # Validate result
            if abs(rate_guess) > 10 or rate_guess <= -0.99:
                return 0.0
            
            return rate_guess
            
        except Exception as e:
            logger.warning(f"IRR calculation failed: {e}")
            return 0.0
    
    def _calculate_payback(self, cash_flows: List[float]) -> float:
        """Calculate payback period in years."""
        cumulative = 0
        for period, cf in enumerate(cash_flows):
            cumulative += cf
            if cumulative >= 0 and period > 0:
                return period / 12  # Convert months to years
        return float('inf')
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of simulation results."""
        if not self.results:
            return {}
        
        summary = {
            'total_wells': len(self.data.wells),
            'simulated_wells': len(self.results.get('decline_analysis', {})),
            'forecast_years': self.params.forecast_years,
            'economic_results': self.results.get('economic_analysis', {})
        }
        
        # Add well-level statistics
        decline_results = self.results.get('decline_analysis', {})
        if decline_results:
            qis = [r['qi'] for r in decline_results.values()]
            summary.update({
                'avg_initial_rate': np.mean(qis) if qis else 0,
                'min_initial_rate': np.min(qis) if qis else 0,
                'max_initial_rate': np.max(qis) if qis else 0
            })
        
        return summary
