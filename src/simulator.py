import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
from scipy import integrate, interpolate, optimize
from scipy.sparse import diags
import matplotlib.pyplot as plt

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
    discount_rate: float = 0.12  # 12%
    time_step_days: int = 30
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6

class MaterialBalance:
    """
    Implement material balance equation for reservoir simulation.
    """
    
    @staticmethod
    def calculate_oip(area: float, thickness: float, porosity: float, 
                     sw: float, boi: float) -> float:
        """Calculate Original Oil In Place (STB)."""
        # 7758 = conversion factor (acre-ft to bbl)
        return 7758 * area * thickness * porosity * (1 - sw) / boi
    
    @staticmethod
    def calculate_cumulative_production(production_rates: np.ndarray, 
                                       time_points: np.ndarray) -> float:
        """Calculate cumulative production using trapezoidal integration."""
        if len(production_rates) < 2:
            return 0.0
        return np.trapz(production_rates, time_points)
    
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
        f = n_p * bo
        eo = (bo - boi) + boi * ce * delta_p
        
        return f / eo if eo != 0 else oip

class DeclineCurveAnalysis:
    """
    Implement Arps decline curve analysis.
    """
    
    @staticmethod
    def hyperbolic_decline(qi: float, di: float, b: float, t: np.ndarray) -> np.ndarray:
        """
        Calculate production rate using hyperbolic decline.
        
        Args:
            qi: Initial production rate
            di: Initial decline rate
            b: Decline exponent (0 < b <= 1)
            t: Time array
        
        Returns:
            Production rates over time
        """
        return qi / (1 + b * di * t) ** (1/b)
    
    @staticmethod
    def exponential_decline(qi: float, di: float, t: np.ndarray) -> np.ndarray:
        """Calculate production rate using exponential decline."""
        return qi * np.exp(-di * t)
    
    @staticmethod
    def harmonic_decline(qi: float, di: float, t: np.ndarray) -> np.ndarray:
        """Calculate production rate using harmonic decline."""
        return qi / (1 + di * t)
    
    @staticmethod
    def fit_decline_curve(time: np.ndarray, rate: np.ndarray, 
                         method: str = 'hyperbolic') -> Dict:
        """
        Fit decline curve parameters to historical data.
        
        Returns:
            Dictionary with fitted parameters and R² value
        """
        if len(time) < 3:
            logger.warning("Insufficient data for decline curve fitting")
            return {}
        
        # Normalize time
        t_norm = time - time[0]
        
        if method == 'exponential':
            # Linear regression on log(q) vs t
            log_rate = np.log(rate[rate > 0])
            valid_idx = rate > 0
            t_valid = t_norm[valid_idx]
            
            if len(t_valid) < 2:
                return {}
            
            A = np.vstack([t_valid, np.ones_like(t_valid)]).T
            m, c = np.linalg.lstsq(A, log_rate, rcond=None)[0]
            
            qi_fit = np.exp(c)
            di_fit = -m
            
            # Calculate fitted rates
            rate_fit = qi_fit * np.exp(-di_fit * t_norm)
            
            # Calculate R²
            ss_res = np.sum((rate - rate_fit) ** 2)
            ss_tot = np.sum((rate - np.mean(rate)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'qi': qi_fit,
                'di': di_fit,
                'b': 0,  # Exponential has b=0
                'r_squared': r_squared,
                'method': 'exponential'
            }
        
        elif method == 'hyperbolic':
            # Non-linear least squares for hyperbolic
            def hyperbolic_func(t, qi, di, b):
                return qi / (1 + b * di * t) ** (1/b)
            
            # Initial guess
            qi_guess = rate[0]
            di_guess = 0.01
            b_guess = 0.8
            
            try:
                popt, _ = optimize.curve_fit(
                    hyperbolic_func,
                    t_norm,
                    rate,
                    p0=[qi_guess, di_guess, b_guess],
                    bounds=([0, 1e-6, 0.1], [np.inf, 1, 2])
                )
                
                qi_fit, di_fit, b_fit = popt
                rate_fit = hyperbolic_func(t_norm, *popt)
                
                ss_res = np.sum((rate - rate_fit) ** 2)
                ss_tot = np.sum((rate - np.mean(rate)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                return {
                    'qi': qi_fit,
                    'di': di_fit,
                    'b': b_fit,
                    'r_squared': r_squared,
                    'method': 'hyperbolic'
                }
                
            except Exception as e:
                logger.error(f"Hyperbolic fit failed: {e}")
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
        if not self.data.wells:
            logger.error("No well data available")
            return False
        
        valid_wells = 0
        for well_name, well in self.data.wells.items():
            if len(well.time_points) >= 3 and len(well.production_rates) >= 3:
                if np.any(well.production_rates > 0):
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
            if len(well.time_points) < 2:
                continue
            
            # Calculate cumulative production
            cum_production = mb.calculate_cumulative_production(
                well.production_rates,
                well.time_points
            )
            
            # Estimate reservoir properties (simplified)
            area = 40.0  # acres (assumed)
            thickness = 50.0  # ft (assumed)
            porosity = 0.15  # fraction (assumed)
            sw = 0.25  # water saturation (assumed)
            boi = 1.2  # initial oil FVF (rb/stb)
            
            # Calculate OOIP
            oip = mb.calculate_oip(area, thickness, porosity, sw, boi)
            
            # Recovery factor
            recovery_factor = cum_production / oip if oip > 0 else 0
            
            results[well_name] = {
                'cumulative_production': cum_production,
                'estimated_oip': oip,
                'recovery_factor': recovery_factor,
                'production_days': well.time_points[-1] - well.time_points[0]
            }
        
        return results
    
    def _perform_decline_analysis(self) -> Dict:
        """Perform decline curve analysis for each well."""
        results = {}
        dca = DeclineCurveAnalysis()
        
        for well_name, well in self.data.wells.items():
            if len(well.time_points) < 3:
                continue
            
            time_data = well.time_points - well.time_points[0]
            rate_data = well.production_rates
            
            # Try exponential fit first
            exp_result = dca.fit_decline_curve(
                well.time_points, rate_data, method='exponential'
            )
            
            # Try hyperbolic fit
            hyp_result = dca.fit_decline_curve(
                well.time_points, rate_data, method='hyperbolic'
            )
            
            # Select best fit based on R²
            best_result = {}
            if exp_result and exp_result.get('r_squared', 0) > 0.8:
                best_result = exp_result
            elif hyp_result and hyp_result.get('r_squared', 0) > 0.8:
                best_result = hyp_result
            elif exp_result:
                best_result = exp_result
            elif hyp_result:
                best_result = hyp_result
            
            if best_result:
                results[well_name] = best_result
                logger.info(f"Decline analysis for {well_name}: "
                          f"qi={best_result['qi']:.1f}, "
                          f"di={best_result['di']:.4f}, "
                          f"R²={best_result.get('r_squared', 0):.3f}")
        
        return results
    
    def _generate_production_forecast(self, mb_results: Dict, 
                                    dca_results: Dict) -> Dict:
        """Generate production forecast using decline curve parameters."""
        forecast_days = self.params.forecast_years * 365
        dca = DeclineCurveAnalysis()
        
        forecast_data = {}
        
        for well_name in self.data.wells.keys():
            if well_name not in dca_results:
                continue
            
            well = self.data.wells[well_name]
            params = dca_results[well_name]
            
            # Historical time
            historical_time = well.time_points - well.time_points[0]
            
            # Forecast time
            forecast_start = historical_time[-1]
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
            
            # Ensure rates are positive
            forecast_rates = np.maximum(0, forecast_rates)
            
            # Calculate EUR (Estimated Ultimate Recovery)
            if forecast_rates.size > 1:
                eur = np.trapz(forecast_rates, forecast_time)
            else:
                eur = 0
            
            forecast_data[well_name] = {
                'forecast_time': forecast_time,
                'forecast_rates': forecast_rates,
                'eur': eur,
                'decline_parameters': params
            }
        
        return forecast_data
    
    def _perform_economic_analysis(self, forecast_results: Dict) -> Dict:
        """Perform economic analysis on forecast results."""
        total_revenue = 0
        total_opex = 0
        cash_flows = []
        
        for well_name, forecast in forecast_results.items():
            # Calculate well revenue and costs
            daily_production = forecast['forecast_rates']
            days = forecast['forecast_time']
            
            if len(daily_production) < 2:
                continue
            
            # Calculate cumulative production
            time_intervals = np.diff(days)
            avg_rates = 0.5 * (daily_production[:-1] + daily_production[1:])
            monthly_production = avg_rates * time_intervals
            
            # Revenue and costs
            monthly_revenue = monthly_production * self.params.oil_price
            monthly_opex = monthly_production * self.params.operating_cost
            
            total_revenue += np.sum(monthly_revenue)
            total_opex += np.sum(monthly_opex)
            
            # Monthly cash flows
            monthly_cash_flow = monthly_revenue - monthly_opex
            cash_flows.extend(monthly_cash_flow.tolist())
        
        # Calculate economic metrics
        initial_investment = 50_000_000  # $50M
        cash_flows = [-initial_investment] + cash_flows
        
        # NPV calculation
        npv = self._calculate_npv(cash_flows, self.params.discount_rate/12)
        
        # IRR calculation
        irr = self._calculate_irr(cash_flows)
        
        # ROI
        total_profit = total_revenue - total_opex
        roi = (total_profit - initial_investment) / initial_investment
        
        # Payback period
        payback_years = self._calculate_payback(cash_flows)
        
        return {
            'npv': npv,
            'irr': irr,
            'roi': roi,
            'total_revenue': total_revenue,
            'total_opex': total_opex,
            'gross_profit': total_revenue - total_opex,
            'net_profit': total_revenue - total_opex - initial_investment,
            'payback_period_years': payback_years,
            'initial_investment': initial_investment
        }
    
    def _calculate_npv(self, cash_flows: List[float], discount_rate: float) -> float:
        """Calculate Net Present Value."""
        npv = 0
        for t, cf in enumerate(cash_flows):
            npv += cf / ((1 + discount_rate) ** t)
        return npv
    
    def _calculate_irr(self, cash_flows: List[float]) -> float:
        """Calculate Internal Rate of Return."""
        try:
            return np.irr(cash_flows)
        except:
            return 0.0
    
    def _calculate_payback(self, cash_flows: List[float]) -> float:
        """Calculate payback period in years."""
        cumulative = 0
        for period, cf in enumerate(cash_flows):
            cumulative += cf
            if cumulative >= 0:
                return period / 12  # Convert months to years
        return float('inf')
