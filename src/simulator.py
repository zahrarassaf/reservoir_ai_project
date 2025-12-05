"""
Advanced Reservoir Simulator
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import optimize, stats
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class SimulationParameters:
    """Simulation parameters"""
    forecast_years: int = 3
    time_step_days: int = 1
    decline_model: str = "exponential"  # exponential, hyperbolic, harmonic
    economic_limit: float = 10.0  # bbl/day
    oil_price: float = 75.0  # USD/bbl
    operating_cost: float = 18.0  # USD/bbl
    discount_rate: float = 0.12
    compressibility: float = 1e-5  # psi^-1


class ReservoirSimulator:
    """
    Advanced reservoir simulator with material balance and decline curve analysis
    """
    
    def __init__(self, data, params: Optional[SimulationParameters] = None):
        """
        Initialize reservoir simulator
        
        Parameters
        ----------
        data : ReservoirData
            Reservoir data
        params : SimulationParameters, optional
            Simulation parameters
        """
        self.data = data
        self.params = params or SimulationParameters()
        self.results = {}
        logger.info("Reservoir simulator initialized")
    
    def run_comprehensive_simulation(self) -> Dict[str, Any]:
        """
        Run comprehensive reservoir simulation
        
        Returns
        -------
        Dict
            Complete simulation results
        """
        logger.info("Starting comprehensive reservoir simulation")
        
        # 1. Material balance analysis
        material_balance = self._perform_material_balance()
        
        # 2. Decline curve analysis
        decline_analysis = self._perform_decline_analysis()
        
        # 3. Production forecast
        production_forecast = self._forecast_production()
        
        # 4. Pressure simulation
        pressure_forecast = self._simulate_pressure(production_forecast)
        
        # 5. Economic analysis
        economic_analysis = self._perform_economic_analysis(production_forecast)
        
        # 6. Sensitivity analysis
        sensitivity_analysis = self._perform_sensitivity_analysis()
        
        # Compile results
        self.results = {
            'material_balance': material_balance,
            'decline_analysis': decline_analysis,
            'production_forecast': production_forecast,
            'pressure_forecast': pressure_forecast,
            'economic_analysis': economic_analysis,
            'sensitivity_analysis': sensitivity_analysis,
            'parameters': self.params,
            'timestamp': pd.Timestamp.now()
        }
        
        logger.info("Simulation completed successfully")
        return self.results
    
    def _perform_material_balance(self) -> Dict[str, Any]:
        """Perform material balance analysis"""
        logger.info("Performing material balance analysis")
        
        if self.data.production.empty or len(self.data.pressure) == 0:
            logger.warning("Insufficient data for material balance")
            return {}
        
        # Calculate total production
        total_production = self.data.production.sum(axis=1).values
        
        # Ensure arrays are same length
        min_len = min(len(total_production), len(self.data.pressure))
        production = total_production[:min_len]
        pressure = self.data.pressure[:min_len]
        
        # Calculate cumulative production
        time = self.data.time[:min_len]
        dt = np.diff(time, prepend=time[0])
        cumulative_prod = np.cumsum(production * dt)
        
        # Tank model material balance
        Pi = pressure[0]
        delta_P = Pi - pressure
        
        # Remove invalid data points
        valid_mask = (delta_P > 0) & (cumulative_prod > 0)
        
        if np.sum(valid_mask) < 5:
            logger.warning("Insufficient valid data for material balance")
            return {}
        
        # Linear regression: Np vs delta_P
        x = delta_P[valid_mask]
        y = cumulative_prod[valid_mask]
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Calculate OOIP
            Bo = 1.2  # Formation volume factor (rb/stb)
            Ct = self.params.compressibility  # Total compressibility
            
            if slope > 0 and Ct > 0:
                ooip = slope / (Bo * Ct)
            else:
                ooip = 0
            
            results = {
                'ooip_stb': float(ooip),
                'regression': {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'std_error': float(std_err)
                },
                'data_points': {
                    'total': int(min_len),
                    'valid': int(np.sum(valid_mask)),
                    'pressure_range': float(np.ptp(pressure)),
                    'production_range': float(np.ptp(cumulative_prod))
                }
            }
            
            logger.info(f"Material balance: OOIP = {ooip:,.0f} STB, R² = {r_value**2:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Material balance analysis failed: {e}")
            return {}
    
    def _perform_decline_analysis(self) -> Dict[str, Any]:
        """Perform decline curve analysis"""
        logger.info("Performing decline curve analysis")
        
        if self.data.production.empty:
            logger.warning("No production data for decline analysis")
            return {}
        
        results = {}
        
        for well in self.data.production.columns:
            rates = self.data.production[well].values
            valid_mask = rates > self.params.economic_limit
            
            if np.sum(valid_mask) < 10:
                logger.warning(f"Insufficient data for well {well}")
                continue
            
            q = rates[valid_mask]
            t = self.data.time[valid_mask]
            
            # Exponential decline fit
            try:
                log_q = np.log(q[:100])  # Use first 100 points for fitting
                t_fit = t[:100]
                
                # Linear regression on log-transformed data
                slope, intercept = np.polyfit(t_fit, log_q, 1)
                
                qi = np.exp(intercept)
                Di = -slope  # Daily decline rate
                
                # Calculate EUR
                if Di > 0:
                    t_el = (np.log(qi / self.params.economic_limit)) / Di
                    eur = qi / Di * (1 - np.exp(-Di * t_el))
                else:
                    eur = 0
                
                results[well] = {
                    'exponential': {
                        'initial_rate': float(qi),
                        'decline_rate': float(Di * 365),  # Annualized
                        'eur': float(eur),
                        'fit_quality': 'good' if np.std(log_q - (intercept + slope * t_fit)) < 0.2 else 'fair'
                    },
                    'statistics': {
                        'data_points': int(len(q)),
                        'initial_rate_actual': float(q[0]),
                        'final_rate': float(q[-1]),
                        'decline_percent': float((q[0] - q[-1]) / q[0] * 100) if q[0] > 0 else 0
                    }
                }
                
            except Exception as e:
                logger.error(f"Decline analysis failed for well {well}: {e}")
                continue
        
        return results
    
    def _forecast_production(self) -> Dict[str, Any]:
        """Forecast future production"""
        logger.info("Forecasting production")
        
        if self.data.production.empty:
            logger.warning("No production data for forecasting")
            return {}
        
        # Create forecast time grid
        historical_end = len(self.data.time)
        forecast_days = self.params.forecast_years * 365
        forecast_time = np.arange(
            self.data.time[-1] + 1,
            self.data.time[-1] + forecast_days + 1
        )
        
        full_time = np.concatenate([self.data.time, forecast_time])
        
        # Forecast each well
        n_wells = len(self.data.production.columns)
        forecast_data = np.zeros((len(full_time), n_wells))
        
        decline_results = self._perform_decline_analysis()
        
        for i, well in enumerate(self.data.production.columns):
            # Historical data
            historical_rates = self.data.production[well].values
            
            # Fill historical period
            forecast_data[:historical_end, i] = historical_rates
            
            # Forecast future
            if well in decline_results:
                # Use decline curve parameters
                decline = decline_results[well]['exponential']
                qi = decline['initial_rate']
                Di = decline['decline_rate'] / 365  # Convert to daily
                
                for j in range(historical_end, len(full_time)):
                    dt = full_time[j] - full_time[historical_end - 1]
                    forecast_data[j, i] = qi * np.exp(-Di * dt)
                    forecast_data[j, i] = max(forecast_data[j, i], 0)
            else:
                # Simple exponential decline
                last_rate = historical_rates[-1] if len(historical_rates) > 0 else 0
                Di = 0.001  # Default decline
                
                for j in range(historical_end, len(full_time)):
                    dt = full_time[j] - full_time[historical_end - 1]
                    forecast_data[j, i] = last_rate * np.exp(-Di * dt)
                    forecast_data[j, i] = max(forecast_data[j, i], 0)
        
        # Calculate totals
        total_production = forecast_data.sum(axis=1)
        cumulative_production = np.cumsum(total_production)
        
        # Calculate statistics
        peak_production = np.max(total_production)
        final_production = total_production[-1]
        
        results = {
            'time': full_time.tolist(),
            'production': forecast_data.tolist(),
            'total_production': total_production.tolist(),
            'cumulative_production': cumulative_production.tolist(),
            'well_names': self.data.production.columns.tolist(),
            'statistics': {
                'peak_production': float(peak_production),
                'final_production': float(final_production),
                'total_cumulative': float(cumulative_production[-1]),
                'forecast_years': self.params.forecast_years
            }
        }
        
        logger.info(f"Production forecast: Peak = {peak_production:.0f} bbl/day, "
                   f"Final = {final_production:.0f} bbl/day")
        
        return results
    
    def _simulate_pressure(self, production_forecast: Dict) -> Dict[str, Any]:
        """Simulate reservoir pressure"""
        logger.info("Simulating reservoir pressure")
        
        if len(self.data.pressure) == 0:
            logger.warning("No pressure data for simulation")
            return {}
        
        time = np.array(production_forecast['time'])
        total_production = np.array(production_forecast['total_production'])
        
        # Initialize pressure array
        pressure = np.zeros(len(time))
        
        # Fill historical pressure
        hist_len = min(len(self.data.pressure), len(time))
        pressure[:hist_len] = self.data.pressure[:hist_len]
        
        # Forecast pressure using material balance
        Bo = 1.2  # rb/stb
        Ct = self.params.compressibility  # psi^-1
        
        # Calculate cumulative production since forecast start
        for i in range(hist_len, len(time)):
            if i == 0:
                continue
            
            dt = time[i] - time[i-1]
            
            # Cumulative production since historical end
            production_slice = total_production[hist_len:i]
            time_slice = time[hist_len:i]
            
            if len(production_slice) > 0:
                cumulative_since = np.trapz(production_slice, time_slice)
                
                # Pressure drop from material balance
                pressure_drop = cumulative_since * Bo * Ct / 1e6
                
                # Forecast pressure
                pressure[i] = max(1000, pressure[hist_len-1] - pressure_drop)
        
        results = {
            'time': time.tolist(),
            'pressure': pressure.tolist(),
            'statistics': {
                'initial_pressure': float(pressure[0]),
                'final_pressure': float(pressure[-1]),
                'pressure_drop': float(pressure[0] - pressure[-1]),
                'pressure_drop_percent': float((pressure[0] - pressure[-1]) / pressure[0] * 100)
            }
        }
        
        logger.info(f"Pressure simulation: Initial = {pressure[0]:.0f} psi, "
                   f"Final = {pressure[-1]:.0f} psi")
        
        return results
    
    def _perform_economic_analysis(self, production_forecast: Dict) -> Dict[str, Any]:
        """Perform economic analysis"""
        logger.info("Performing economic analysis")
        
        time = np.array(production_forecast['time'])
        total_production = np.array(production_forecast['total_production'])
        
        # Economic parameters
        oil_price = self.params.oil_price
        operating_cost = self.params.operating_cost
        discount_rate = self.params.discount_rate
        
        # Calculate daily revenue and costs
        dt = np.diff(time, prepend=time[0])
        daily_revenue = total_production * oil_price
        daily_opex = total_production * operating_cost
        
        # Calculate cash flows
        cash_flows = np.zeros(len(time))
        
        for i in range(len(time)):
            # Daily cash flow
            revenue = daily_revenue[i] * dt[i]
            opex = daily_opex[i] * dt[i]
            
            # Simple cash flow calculation
            cash_flows[i] = revenue - opex
        
        # Calculate NPV
        npv = 0.0
        for i in range(len(time)):
            if i == 0:
                discount_factor = 1.0
            else:
                discount_factor = 1.0 / ((1.0 + discount_rate) ** (time[i] / 365))
            npv += cash_flows[i] * discount_factor
        
        # Calculate cumulative cash flow
        cumulative_cf = np.cumsum(cash_flows)
        
        # Calculate payback period
        payback_period = None
        for i in range(len(cumulative_cf)):
            if cumulative_cf[i] > 0:
                payback_period = time[i] / 365
                break
        
        # Calculate IRR
        irr = self._calculate_irr(cash_flows, time)
        
        results = {
            'npv': float(npv),
            'irr': float(irr),
            'payback_period': float(payback_period) if payback_period else None,
            'cash_flows': cash_flows.tolist(),
            'cumulative_cash_flow': cumulative_cf.tolist(),
            'daily_revenue': daily_revenue.tolist(),
            'daily_opex': daily_opex.tolist(),
            'parameters': {
                'oil_price': oil_price,
                'operating_cost': operating_cost,
                'discount_rate': discount_rate
            },
            'summary': {
                'total_revenue': float(np.sum(daily_revenue * dt)),
                'total_opex': float(np.sum(daily_opex * dt)),
                'total_cash_flow': float(np.sum(cash_flows)),
                'profit_margin': float(
                    (np.sum(daily_revenue * dt) - np.sum(daily_opex * dt)) / 
                    np.sum(daily_revenue * dt) * 100
                ) if np.sum(daily_revenue * dt) > 0 else 0
            }
        }
        
        logger.info(f"Economic analysis: NPV = ${npv/1e6:.2f}M, IRR = {irr*100:.1f}%")
        
        return results
    
    def _calculate_irr(self, cash_flows: np.ndarray, time: np.ndarray) -> float:
        """Calculate Internal Rate of Return"""
        try:
            def npv_func(rate):
                npv_val = 0.0
                for i in range(len(cash_flows)):
                    if i == 0:
                        discount_factor = 1.0
                    else:
                        discount_factor = 1.0 / ((1.0 + rate) ** (time[i] / 365))
                    npv_val += cash_flows[i] * discount_factor
                return npv_val
            
            # Use Brent's method for root finding
            irr = optimize.brentq(npv_func, -0.5, 2.0, maxiter=1000)
            return max(0, irr)  # Ensure non-negative
            
        except Exception as e:
            logger.warning(f"IRR calculation failed: {e}")
            return 0.0
    
    def _perform_sensitivity_analysis(self) -> Dict[str, Any]:
        """Perform sensitivity analysis"""
        logger.info("Performing sensitivity analysis")
        
        # Base case NPV from economic analysis
        base_npv = self.results.get('economic_analysis', {}).get('npv', 1e6)
        
        # Sensitivity parameters
        parameters = {
            'oil_price': [60, 70, 75, 80, 90],
            'operating_cost': [12, 15, 18, 21, 24],
            'discount_rate': [0.08, 0.10, 0.12, 0.14, 0.16],
            'decline_rate': [0.8, 1.0, 1.2, 1.4, 1.6]
        }
        
        sensitivity_results = {}
        
        for param_name, values in parameters.items():
            npv_values = []
            
            for value in values:
                # Simplified sensitivity calculation
                if param_name == 'oil_price':
                    npv_adj = base_npv * (value / self.params.oil_price)
                elif param_name == 'operating_cost':
                    npv_adj = base_npv * (1 - (value - self.params.operating_cost) / self.params.oil_price * 0.7)
                elif param_name == 'discount_rate':
                    npv_adj = base_npv * (self.params.discount_rate / value)
                elif param_name == 'decline_rate':
                    npv_adj = base_npv * (1.0 / value)
                else:
                    npv_adj = base_npv
                
                npv_values.append(npv_adj)
            
            sensitivity_index = (max(npv_values) - min(npv_values)) / base_npv
            
            sensitivity_results[param_name] = {
                'values': values,
                'npv': [float(v) for v in npv_values],
                'sensitivity_index': float(sensitivity_index),
                'tornado_data': {
                    'low': float(min(npv_values)),
                    'base': float(base_npv),
                    'high': float(max(npv_values))
                }
            }
        
        # Identify key parameters
        key_params = []
        for param_name, data in sensitivity_results.items():
            if data['sensitivity_index'] > 0.3:
                key_params.append(param_name)
        
        sensitivity_results['key_parameters'] = key_params
        
        logger.info(f"Sensitivity analysis: Key parameters = {key_params}")
        
        return sensitivity_results
    
    def export_results(self, output_path: str = './outputs'):
        """Export simulation results"""
        import json
        import pickle
        from pathlib import Path
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Export as JSON
        json_path = output_dir / f'simulation_results_{timestamp}.json'
        
        def convert_for_json(obj):
            """Convert objects to JSON serializable format"""
            if isinstance(obj, (np.ndarray, pd.Series)):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, (pd.Timestamp, np.datetime64)):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            else:
                return obj
        
        json_results = convert_for_json(self.results)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Export as pickle
        pickle_path = output_dir / f'simulation_results_{timestamp}.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Export summary CSV
        summary_data = []
        
        # Economic summary
        econ = self.results.get('economic_analysis', {})
        if econ:
            summary_data.append({
                'category': 'economics',
                'metric': 'npv_usd',
                'value': econ.get('npv', 'N/A')
            })
            summary_data.append({
                'category': 'economics',
                'metric': 'irr_percent',
                'value': econ.get('irr', 'N/A') * 100 if econ.get('irr') else 'N/A'
            })
        
        # Production summary
        prod = self.results.get('production_forecast', {})
        if prod and 'statistics' in prod:
            stats = prod['statistics']
            summary_data.append({
                'category': 'production',
                'metric': 'peak_production_bbl_per_day',
                'value': stats.get('peak_production', 'N/A')
            })
            summary_data.append({
                'category': 'production',
                'metric': 'total_cumulative_bbl',
                'value': stats.get('total_cumulative', 'N/A')
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_path = output_dir / f'summary_{timestamp}.csv'
            summary_df.to_csv(csv_path, index=False)
        
        logger.info(f"Results exported to {output_dir}")
        
        return {
            'json': str(json_path),
            'pickle': str(pickle_path),
            'csv': str(csv_path) if summary_data else None
        }
