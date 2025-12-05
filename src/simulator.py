"""
Reservoir Simulation Engine

This module implements advanced reservoir simulation capabilities including
material balance calculations, decline curve analysis, and production forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from scipy import optimize, stats
import warnings

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for reservoir simulation"""
    forecast_years: int = 2
    time_step_days: int = 1
    decline_model: str = "exponential"  # exponential, hyperbolic, harmonic
    material_balance_method: str = "tank"
    economic_parameters: Dict = None
    convergence_tolerance: float = 1e-6
    max_iterations: int = 1000
    
    def __post_init__(self):
        if self.economic_parameters is None:
            self.economic_parameters = {
                "oil_price_usd_per_bbl": 70.0,
                "operating_cost_usd_per_bbl": 15.0,
                "capital_cost_usd_per_year": 1e6,
                "discount_rate": 0.10,
                "tax_rate": 0.30
            }


@dataclass
class ReservoirParameters:
    """Reservoir physical parameters"""
    compressibility_psi_inv: float = 1e-5
    formation_volume_factor_rb_per_stb: float = 1.2
    viscosity_cp: float = 0.5
    total_compressibility_psi_inv: float = 1e-5
    initial_pressure_psi: float = 4500.0
    bubble_point_pressure_psi: float = 3000.0
    rock_compressibility_psi_inv: float = 3e-6
    water_compressibility_psi_inv: float = 3e-6


class ReservoirSimulator:
    """
    Advanced reservoir simulator with forecasting capabilities.
    
    Features:
    - Material balance calculations
    - Decline curve analysis (Arps models)
    - Pressure transient analysis
    - Production forecasting
    - Economic evaluation
    - Sensitivity analysis
    """
    
    def __init__(self, data: Dict, config: Optional[SimulationConfig] = None):
        """
        Initialize reservoir simulator.
        
        Parameters
        ----------
        data : Dict
            Reservoir data dictionary
        config : SimulationConfig, optional
            Simulation configuration
        """
        self.data = data
        self.config = config or SimulationConfig()
        self.reservoir_params = self._estimate_parameters()
        self.results = {}
        logger.info("ReservoirSimulator initialized")
    
    def run_simulation(self, forecast_years: Optional[int] = None) -> Dict:
        """
        Run complete reservoir simulation with forecasting.
        
        Parameters
        ----------
        forecast_years : int, optional
            Number of years to forecast
            
        Returns
        -------
        Dict
            Simulation results
        """
        forecast_years = forecast_years or self.config.forecast_years
        logger.info(f"Starting simulation with {forecast_years}-year forecast")
        
        # Prepare simulation time grid
        time_grid = self._create_time_grid(forecast_years)
        
        # Run simulation components
        production_simulation = self._simulate_production(time_grid)
        pressure_simulation = self._simulate_pressure(time_grid, production_simulation)
        injection_simulation = self._simulate_injection(time_grid)
        
        # Calculate derived metrics
        economic_analysis = self._economic_analysis(production_simulation, time_grid)
        recovery_analysis = self._recovery_analysis(production_simulation)
        sensitivity_analysis = self._sensitivity_analysis(production_simulation, time_grid)
        
        # Compile results
        self.results = {
            'time': time_grid,
            'production': production_simulation,
            'pressure': pressure_simulation,
            'injection': injection_simulation,
            'economics': economic_analysis,
            'recovery': recovery_analysis,
            'sensitivity': sensitivity_analysis,
            'parameters': self.reservoir_params.__dict__,
            'config': self.config.__dict__
        }
        
        logger.info("Simulation completed successfully")
        return self.results
    
    def _estimate_parameters(self) -> ReservoirParameters:
        """Estimate reservoir parameters from data."""
        params = ReservoirParameters()
        
        # Estimate compressibility from pressure and production data
        if 'pressure' in self.data and 'production' in self.data:
            pressure = self.data['pressure']
            production = self.data['production'].sum(axis=1).values
            
            if len(pressure) > 10 and len(production) > 10:
                pressure_diff = np.diff(pressure)
                production_avg = (production[:-1] + production[1:]) / 2
                
                valid_mask = (pressure_diff != 0) & (production_avg != 0)
                if np.sum(valid_mask) > 5:
                    # Simple compressibility estimate
                    ct_estimate = np.abs(np.mean(
                        production_avg[valid_mask] / pressure_diff[valid_mask]
                    )) / 1e6
                    
                    params.total_compressibility_psi_inv = max(1e-7, min(1e-4, ct_estimate))
        
        # Estimate initial pressure
        if 'pressure' in self.data:
            params.initial_pressure_psi = float(self.data['pressure'][0])
        
        return params
    
    def _create_time_grid(self, forecast_years: int) -> np.ndarray:
        """Create simulation time grid."""
        historical_time = self.data['time']
        forecast_days = forecast_years * 365
        
        forecast_time = np.linspace(
            historical_time[-1],
            historical_time[-1] + forecast_days,
            int(forecast_days / self.config.time_step_days)
        )
        
        return np.concatenate([historical_time, forecast_time])
    
    def _simulate_production(self, time_grid: np.ndarray) -> np.ndarray:
        """Simulate production using decline curve models."""
        n_wells = self.data.get('n_wells', 8)
        historical_end = len(self.data['time'])
        
        simulated = np.zeros((len(time_grid), n_wells))
        
        for i in range(n_wells):
            well_name = f'Well_{i+1}'
            if well_name in self.data['production'].columns:
                hist_data = self.data['production'][well_name].values
                
                # Fit decline curve to historical data
                decline_params = self._fit_decline_curve(hist_data, self.data['time'])
                
                # Simulate production
                for j, t in enumerate(time_grid):
                    if j < historical_end and j < len(hist_data):
                        simulated[j, i] = hist_data[j]
                    else:
                        dt = t - time_grid[historical_end - 1]
                        simulated[j, i] = self._decline_function(
                            dt, 
                            decline_params,
                            self.config.decline_model
                        )
        
        return simulated
    
    def _fit_decline_curve(self, rates: np.ndarray, time: np.ndarray) -> Dict:
        """Fit decline curve to production data."""
        valid_mask = rates > 0
        
        if np.sum(valid_mask) < 10:
            return {'qi': 0, 'di': 0, 'b': 0}
        
        q = rates[valid_mask]
        t = time[valid_mask]
        
        if self.config.decline_model == "exponential":
            # Exponential decline: q = qi * exp(-di * t)
            log_q = np.log(q[:100])
            coeffs = np.polyfit(t[:100], log_q, 1)
            
            return {
                'qi': np.exp(coeffs[1]),
                'di': -coeffs[0],
                'b': 0  # b-factor for hyperbolic
            }
        
        elif self.config.decline_model == "hyperbolic":
            # Hyperbolic decline: q = qi / (1 + b * di * t)^(1/b)
            try:
                def hyperbolic_func(t, qi, di, b):
                    return qi / (1 + b * di * t) ** (1/b)
                
                popt, _ = optimize.curve_fit(
                    hyperbolic_func,
                    t[:100],
                    q[:100],
                    p0=[q[0], 0.001, 0.5],
                    bounds=([0, 0, 0], [np.inf, 1, 2])
                )
                
                return {'qi': popt[0], 'di': popt[1], 'b': popt[2]}
            except:
                # Fallback to exponential
                return self._fit_decline_curve(rates, time)
        
        else:  # harmonic
            # Harmonic decline: q = qi / (1 + di * t)
            try:
                def harmonic_func(t, qi, di):
                    return qi / (1 + di * t)
                
                popt, _ = optimize.curve_fit(
                    harmonic_func,
                    t[:100],
                    q[:100],
                    p0=[q[0], 0.001]
                )
                
                return {'qi': popt[0], 'di': popt[1], 'b': 1}
            except:
                return {'qi': 0, 'di': 0, 'b': 0}
    
    def _decline_function(self, t: float, params: Dict, model: str) -> float:
        """Calculate production rate using decline model."""
        qi = params.get('qi', 0)
        di = params.get('di', 0)
        b = params.get('b', 0)
        
        if qi <= 0 or di <= 0:
            return 0.0
        
        if model == "exponential":
            return qi * np.exp(-di * t)
        elif model == "hyperbolic":
            if b == 0:
                return qi * np.exp(-di * t)
            return qi / (1 + b * di * t) ** (1/b)
        else:  # harmonic
            return qi / (1 + di * t)
    
    def _simulate_pressure(self, time_grid: np.ndarray, 
                          production: np.ndarray) -> np.ndarray:
        """Simulate reservoir pressure using material balance."""
        historical_end = len(self.data['time'])
        pressure = np.zeros(len(time_grid))
        
        # Set historical pressure
        if 'pressure' in self.data:
            hist_len = min(len(self.data['pressure']), historical_end)
            pressure[:hist_len] = self.data['pressure'][:hist_len]
        
        # Material balance calculation for forecast period
        if self.config.material_balance_method == "tank":
            pressure = self._tank_model_pressure(
                time_grid, production, pressure, historical_end
            )
        
        return pressure
    
    def _tank_model_pressure(self, time_grid: np.ndarray, 
                            production: np.ndarray, 
                            pressure: np.ndarray, 
                            historical_end: int) -> np.ndarray:
        """Calculate pressure using tank model material balance."""
        ct = self.reservoir_params.total_compressibility_psi_inv
        B = self.reservoir_params.formation_volume_factor_rb_per_stb
        
        production_total = production.sum(axis=1)
        
        for i in range(historical_end, len(time_grid)):
            if i == 0:
                continue
            
            dt = time_grid[i] - time_grid[i-1]
            
            # Cumulative production since forecast start
            cumulative_since_forecast = np.trapz(
                production_total[historical_end:i], 
                time_grid[historical_end:i]
            )
            
            # Pressure drop from material balance
            pressure_drop = (cumulative_since_forecast * B * ct) / 1e6
            
            pressure[i] = max(1000, pressure[historical_end-1] - pressure_drop)
        
        return pressure
    
    def _simulate_injection(self, time_grid: np.ndarray) -> np.ndarray:
        """Simulate injection rates."""
        if 'injection' not in self.data:
            return np.zeros((len(time_grid), 0))
        
        n_inj = len(self.data['injection'].columns)
        historical_end = len(self.data['time'])
        
        simulated = np.zeros((len(time_grid), n_inj))
        
        for i in range(n_inj):
            inj_name = f'Inj_{i+1}'
            if inj_name in self.data['injection'].columns:
                hist_data = self.data['injection'][inj_name].values
                
                for j, t in enumerate(time_grid):
                    if j < historical_end and j < len(hist_data):
                        simulated[j, i] = hist_data[j]
                    else:
                        # Continue with average of last 30 days
                        if len(hist_data) > 30:
                            simulated[j, i] = np.mean(hist_data[-30:])
                        else:
                            simulated[j, i] = hist_data[-1] if len(hist_data) > 0 else 0
        
        return simulated
    
    def _economic_analysis(self, production: np.ndarray, 
                          time_grid: np.ndarray) -> Dict:
        """Perform economic analysis."""
        econ_params = self.config.economic_parameters
        
        oil_price = econ_params["oil_price_usd_per_bbl"]
        operating_cost = econ_params["operating_cost_usd_per_bbl"]
        capital_cost = econ_params["capital_cost_usd_per_year"]
        discount_rate = econ_params["discount_rate"]
        tax_rate = econ_params["tax_rate"]
        
        daily_production = production.sum(axis=1)
        daily_revenue = daily_proproduction * oil_price
        daily_operating_cost = daily_production * operating_cost
        
        # Calculate cash flows
        cash_flows = np.zeros(len(time_grid))
        cumulative_cash_flow = np.zeros(len(time_grid))
        
        for i in range(len(time_grid)):
            # Revenue
            revenue = daily_revenue[i] * (time_grid[i] - time_grid[i-1] if i > 0 else 1)
            
            # Costs
            operating_expense = daily_operating_cost[i] * (time_grid[i] - time_grid[i-1] if i > 0 else 1)
            
            # Annual capital costs
            capital_expense = 0
            if i % 365 == 0 and i > 0:
                capital_expense = capital_cost
            
            # Pre-tax cash flow
            pre_tax_cf = revenue - operating_expense - capital_expense
            
            # Tax
            tax = max(0, pre_tax_cf) * tax_rate
            
            # After-tax cash flow
            cash_flows[i] = pre_tax_cf - tax
            
            # Cumulative
            if i == 0:
                cumulative_cash_flow[i] = cash_flows[i]
            else:
                cumulative_cash_flow[i] = cumulative_cash_flow[i-1] + cash_flows[i]
        
        # Calculate NPV
        npv = self._calculate_npv(cash_flows, time_grid, discount_rate)
        
        # Calculate IRR
        irr = self._calculate_irr(cash_flows, time_grid)
        
        # Find payback period
        payback_period = None
        for i in range(len(cumulative_cash_flow)):
            if cumulative_cash_flow[i] > 0:
                payback_period = time_grid[i] / 365
                break
        
        return {
            'npv_usd': float(npv),
            'irr': float(irr),
            'payback_period_years': float(payback_period) if payback_period else None,
            'cumulative_cash_flow_usd': cumulative_cash_flow.tolist(),
            'daily_revenue_usd': daily_revenue.tolist(),
            'daily_cost_usd': daily_operating_cost.tolist(),
            'cash_flows_usd': cash_flows.tolist(),
            'parameters': econ_params
        }
    
    def _calculate_npv(self, cash_flows: np.ndarray, 
                      time_grid: np.ndarray, 
                      discount_rate: float) -> float:
        """Calculate Net Present Value."""
        npv = 0.0
        
        for i in range(len(cash_flows)):
            if i == 0:
                discount_factor = 1.0
            else:
                discount_factor = 1.0 / ((1.0 + discount_rate) ** (time_grid[i] / 365))
            
            npv += cash_flows[i] * discount_factor
        
        return npv
    
    def _calculate_irr(self, cash_flows: np.ndarray, 
                      time_grid: np.ndarray) -> float:
        """Calculate Internal Rate of Return."""
        try:
            def npv_func(rate):
                npv_val = 0.0
                for i in range(len(cash_flows)):
                    if i == 0:
                        discount_factor = 1.0
                    else:
                        discount_factor = 1.0 / ((1.0 + rate) ** (time_grid[i] / 365))
                    npv_val += cash_flows[i] * discount_factor
                return npv_val
            
            # Use Brent's method for root finding
            irr = optimize.brentq(npv_func, -0.5, 2.0, maxiter=1000)
            return irr
        
        except:
            return 0.0
    
    def _recovery_analysis(self, production: np.ndarray) -> Dict:
        """Analyze recovery factors."""
        total_production = production.sum(axis=1)
        cumulative_production = np.zeros(len(total_production))
        
        for i in range(1, len(total_production)):
            dt = self.results['time'][i] - self.results['time'][i-1]
            cumulative_production[i] = cumulative_production[i-1] + total_production[i] * dt
        
        # Estimate OOIP (simplified)
        if 'petrophysical' in self.data:
            petro_df = self.data['petrophysical']
            
            area = 1000  # acres
            boi = 1.2  # rb/stb
            
            h_avg = petro_df['NetThickness'].mean()
            phi_avg = petro_df['Porosity'].mean()
            sw_avg = petro_df['WaterSaturation'].mean()
            
            ooip = (7758 * area * h_avg * phi_avg * (1 - sw_avg)) / boi
        else:
            ooip = 50e6  # Default assumption
        
        recovery_factor = (cumulative_production / ooip) * 100
        
        return {
            'cumulative_production_bbl': cumulative_production.tolist(),
            'estimated_ooip_bbl': float(ooip),
            'recovery_factor_percent': recovery_factor.tolist(),
            'final_recovery_factor_percent': float(recovery_factor[-1])
        }
    
    def _sensitivity_analysis(self, production: np.ndarray, 
                             time_grid: np.ndarray) -> Dict:
        """Perform sensitivity analysis on key parameters."""
        base_npv = self.results['economics']['npv_usd']
        
        parameters = {
            'oil_price': [60, 70, 80, 90],
            'operating_cost': [10, 15, 20, 25],
            'discount_rate': [0.08, 0.10, 0.12, 0.15],
            'decline_rate_multiplier': [0.8, 1.0, 1.2, 1.5]
        }
        
        sensitivities = {}
        
        for param_name, values in parameters.items():
            npv_values = []
            
            for value in values:
                if param_name == 'oil_price':
                    modified_npv = base_npv * (value / 70)
                elif param_name == 'operating_cost':
                    modified_npv = base_npv * (1 - (value - 15) / 70 * 0.5)
                elif param_name == 'discount_rate':
                    modified_npv = base_npv * (0.10 / value)
                elif param_name == 'decline_rate_multiplier':
                    modified_npv = base_npv * (1.0 / value)
                else:
                    modified_npv = base_npv
                
                npv_values.append(modified_npv)
            
            sensitivities[param_name] = {
                'values': values,
                'npv_usd': npv_values,
                'sensitivity_index': (max(npv_values) - min(npv_values)) / base_npv
            }
        
        return sensitivities
    
    def get_summary(self) -> pd.DataFrame:
        """Get simulation summary as dataframe."""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        
        # Production summary
        total_production = self.results['production'].sum(axis=1)
        summary_data.append({
            'category': 'production',
            'metric': 'peak_rate_bbl_per_day',
            'value': float(np.max(total_production))
        })
        
        summary_data.append({
            'category': 'production',
            'metric': 'final_rate_bbl_per_day',
            'value': float(total_production[-1])
        })
        
        # Economic summary
        econ = self.results['economics']
        summary_data.append({
            'category': 'economics',
            'metric': 'npv_usd',
            'value': float(econ['npv_usd'])
        })
        
        summary_data.append({
            'category': 'economics',
            'metric': 'irr',
            'value': float(econ['irr'])
        })
        
        # Pressure summary
        pressure = self.results['pressure']
        summary_data.append({
            'category': 'pressure',
            'metric': 'initial_pressure_psi',
            'value': float(pressure[0])
        })
        
        summary_data.append({
            'category': 'pressure',
            'metric': 'final_pressure_psi',
            'value': float(pressure[-1])
        })
        
        # Recovery summary
        recovery = self.results['recovery']
        summary_data.append({
            'category': 'recovery',
            'metric': 'final_recovery_factor_percent',
            'value': float(recovery['final_recovery_factor_percent'])
        })
        
        return pd.DataFrame(summary_data)
