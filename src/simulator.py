"""
Advanced Reservoir Simulator
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
from scipy import optimize, stats
import warnings
import json
import os
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class SimulationParameters:
    forecast_years: int = 3
    time_step_days: int = 1
    decline_model: str = "exponential"
    economic_limit: float = 10.0
    oil_price: float = 75.0
    operating_cost: float = 18.0
    discount_rate: float = 0.12
    compressibility: float = 1e-5
    abandonment_pressure: float = 500.0
    initial_investment: float = 1000000.0
    
    def to_dict(self):
        return asdict(self)


class ReservoirSimulator:
    def __init__(self, data, params: Optional[SimulationParameters] = None):
        self.data = data
        self.params = params or SimulationParameters()
        self.results = {}
        logger.info("Reservoir simulator initialized")
    
    def run_comprehensive_simulation(self) -> Dict[str, Any]:
        logger.info("Starting comprehensive reservoir simulation")
        
        if not self.data.has_production_data:
            logger.warning("No production data available. Creating sample data.")
            self.data.create_sample_data()
        
        material_balance = self._perform_material_balance()
        decline_analysis = self._perform_decline_analysis()
        production_forecast = self._forecast_production()
        
        pressure_forecast = {}
        if production_forecast and 'time' in production_forecast:
            pressure_forecast = self._simulate_pressure(production_forecast)
        else:
            pressure_forecast = {
                'time': self.data.time.tolist() if hasattr(self.data, 'time') else [],
                'pressure': self.data.pressure.tolist() if hasattr(self.data, 'pressure') else [],
                'statistics': {}
            }
        
        economic_analysis = {}
        if production_forecast and 'time' in production_forecast:
            economic_analysis = self._perform_economic_analysis(production_forecast)
        else:
            economic_analysis = {
                'npv': 0.0,
                'irr': 0.0,
                'roi': 0.0,
                'payback_period': None,
                'cash_flows': [],
                'cumulative_cash_flow': [],
                'daily_revenue': [],
                'daily_opex': [],
                'parameters': {},
                'summary': {}
            }
        
        sensitivity_analysis = self._perform_sensitivity_analysis()
        
        default_production = {
            'time': self.data.time.tolist() if hasattr(self.data, 'time') else [],
            'production': self.data.production.values.tolist() if hasattr(self.data, 'production') and not self.data.production.empty else [],
            'total_production': [],
            'cumulative_production': [],
            'well_names': self.data.wells if hasattr(self.data, 'wells') else [],
            'statistics': {}
        }
        
        self.results = {
            'material_balance': material_balance,
            'decline_analysis': decline_analysis,
            'production_forecast': production_forecast if production_forecast and 'time' in production_forecast else default_production,
            'pressure_forecast': pressure_forecast,
            'economic_analysis': economic_analysis,
            'sensitivity_analysis': sensitivity_analysis,
            'parameters': self.params,
            'timestamp': pd.Timestamp.now()
        }
        
        logger.info("Simulation completed successfully")
        return self.results
    
    def _perform_material_balance(self) -> Dict[str, Any]:
        logger.info("Performing material balance analysis")
        
        if not self.data.has_production_data or not self.data.has_pressure_data:
            logger.warning("Insufficient data for material balance")
            return {}
        
        total_production = self.data.production.sum(axis=1).values
        
        min_len = min(len(total_production), len(self.data.pressure))
        if min_len < 5:
            logger.warning("Insufficient data points for material balance")
            return {}
        
        production = total_production[:min_len]
        pressure = self.data.pressure[:min_len]
        
        time = self.data.time[:min_len] if hasattr(self.data, 'time') else np.arange(min_len)
        dt = np.diff(time, prepend=time[0])
        cumulative_prod = np.cumsum(production * dt)
        
        Pi = pressure[0] if len(pressure) > 0 else 0
        delta_P = Pi - pressure
        
        valid_mask = (delta_P > 0) & (cumulative_prod > 0)
        
        if np.sum(valid_mask) < 5:
            logger.warning("Insufficient valid data for material balance")
            return {}
        
        x = delta_P[valid_mask]
        y = cumulative_prod[valid_mask]
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            Bo = 1.2
            Ct = self.params.compressibility
            
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
        logger.info("Performing decline curve analysis")
        
        if not self.data.has_production_data:
            logger.warning("No production data for decline analysis")
            return {}
        
        results = {}
        
        for well in self.data.production.columns:
            rates = self.data.production[well].values
            if len(rates) == 0:
                continue
            
            valid_mask = rates > self.params.economic_limit
            
            if np.sum(valid_mask) < 10:
                continue
            
            q = rates[valid_mask]
            t = np.arange(len(q))
            
            try:
                if len(q) > 100:
                    log_q = np.log(q[:100])
                    t_fit = t[:100]
                else:
                    log_q = np.log(q)
                    t_fit = t
                
                slope, intercept = np.polyfit(t_fit, log_q, 1)
                
                qi = np.exp(intercept)
                Di = -slope
                
                if Di > 0:
                    t_el = (np.log(qi / self.params.economic_limit)) / Di
                    eur = qi / Di * (1 - np.exp(-Di * t_el))
                else:
                    eur = 0
                
                results[well] = {
                    'exponential': {
                        'initial_rate': float(qi),
                        'decline_rate': float(Di * 365),
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
        logger.info("Forecasting production")
        
        if not self.data.has_production_data:
            logger.warning("No production data for forecasting")
            self.data.create_sample_data()
        
        if not hasattr(self.data, 'time') or len(self.data.time) == 0:
            self.data.time = np.arange(len(self.data.production))
        
        historical_end = len(self.data.time)
        forecast_days = self.params.forecast_years * 365
        
        if historical_end > 0:
            forecast_time = np.arange(
                self.data.time[-1] + 1,
                self.data.time[-1] + forecast_days + 1
            )
        else:
            forecast_time = np.arange(1, forecast_days + 1)
            historical_end = 0
        
        full_time = np.concatenate([self.data.time, forecast_time])
        
        n_wells = len(self.data.production.columns)
        forecast_data = np.zeros((len(full_time), n_wells))
        
        decline_results = self._perform_decline_analysis()
        
        for i, well in enumerate(self.data.production.columns):
            historical_rates = self.data.production[well].values
            if len(historical_rates) > 0:
                forecast_data[:historical_end, i] = historical_rates[:historical_end]
            else:
                qi = 500
                forecast_data[:historical_end, i] = qi * np.exp(-0.0005 * self.data.time[:historical_end])
            
            qi = 500
            Di = 0.0005
            
            if well in decline_results:
                decline = decline_results[well].get('exponential', {})
                qi = decline.get('initial_rate', 500)
                Di = decline.get('decline_rate', 0.0005 * 365) / 365
            
            for j in range(historical_end, len(full_time)):
                dt = full_time[j] - full_time[historical_end - 1] if historical_end > 0 else full_time[j]
                forecast_data[j, i] = max(0, qi * np.exp(-Di * dt))
        
        total_production = forecast_data.sum(axis=1)
        cumulative_production = np.cumsum(total_production)
        
        results = {
            'time': full_time.tolist(),
            'production': forecast_data.tolist(),
            'total_production': total_production.tolist(),
            'cumulative_production': cumulative_production.tolist(),
            'well_names': self.data.production.columns.tolist(),
            'statistics': {
                'peak_production': float(np.max(total_production)),
                'final_production': float(total_production[-1]),
                'total_cumulative': float(cumulative_production[-1]),
                'forecast_years': self.params.forecast_years
            }
        }
        
        logger.info(f"Production forecast: Peak = {np.max(total_production):.0f} bbl/day, "
                   f"Final = {total_production[-1]:.0f} bbl/day")
        
        return results
    
    def _simulate_pressure(self, production_forecast: Dict) -> Dict[str, Any]:
        logger.info("Simulating reservoir pressure")
        
        if not production_forecast or 'time' not in production_forecast:
            logger.warning("No forecast data for pressure simulation")
            return {
                'time': [],
                'pressure': [],
                'statistics': {}
            }
        
        time = np.array(production_forecast['time'])
        total_production = np.array(production_forecast['total_production'])
        
        pressure = np.zeros(len(time))
        
        if self.data.has_pressure_data and len(self.data.pressure) > 0:
            hist_len = min(len(self.data.pressure), len(time))
            pressure[:hist_len] = self.data.pressure[:hist_len]
        else:
            hist_len = 0
            pressure[0] = 4000
        
        Bo = 1.2
        Ct = self.params.compressibility
        
        for i in range(hist_len, len(time)):
            if i == 0:
                continue
            
            production_slice = total_production[hist_len:i]
            time_slice = time[hist_len:i]
            
            if len(production_slice) > 0:
                cumulative_since = np.trapz(production_slice, time_slice)
                pressure_drop = cumulative_since * Bo * Ct / 1e6
                base_pressure = pressure[hist_len-1] if hist_len > 0 else 4000
                pressure[i] = max(1000, base_pressure - pressure_drop)
        
        results = {
            'time': time.tolist(),
            'pressure': pressure.tolist(),
            'statistics': {
                'initial_pressure': float(pressure[0]),
                'final_pressure': float(pressure[-1]),
                'pressure_drop': float(pressure[0] - pressure[-1]),
                'pressure_drop_percent': float((pressure[0] - pressure[-1]) / pressure[0] * 100) if pressure[0] > 0 else 0
            }
        }
        
        logger.info(f"Pressure simulation: Initial = {pressure[0]:.0f} psi, "
                   f"Final = {pressure[-1]:.0f} psi")
        
        return results
    
    def _perform_economic_analysis(self, production_forecast: Dict) -> Dict[str, Any]:
        logger.info("Performing economic analysis")
        
        if not production_forecast or 'time' not in production_forecast:
            logger.warning("No forecast data for economic analysis")
            return {
                'npv': 0.0,
                'irr': 0.0,
                'roi': 0.0,
                'payback_period': None,
                'cash_flows': [],
                'cumulative_cash_flow': [],
                'daily_revenue': [],
                'daily_opex': [],
                'parameters': {},
                'summary': {}
            }
        
        time = np.array(production_forecast['time'])
        total_production = np.array(production_forecast['total_production'])
        
        oil_price = self.params.oil_price
        operating_cost = self.params.operating_cost
        discount_rate = self.params.discount_rate
        
        dt = np.diff(time, prepend=time[0])
        daily_revenue = total_production * oil_price
        daily_opex = total_production * operating_cost
        
        cash_flows = np.zeros(len(time))
        
        for i in range(len(time)):
            revenue = daily_revenue[i] * dt[i]
            opex = daily_opex[i] * dt[i]
            cash_flows[i] = revenue - opex
        
        cash_flows[0] -= self.params.initial_investment
        
        npv = 0.0
        for i in range(len(time)):
            if i == 0:
                discount_factor = 1.0
            else:
                discount_factor = 1.0 / ((1.0 + discount_rate) ** (time[i] / 365))
            npv += cash_flows[i] * discount_factor
        
        cumulative_cf = np.cumsum(cash_flows)
        
        payback_period = None
        for i in range(len(cumulative_cf)):
            if cumulative_cf[i] > 0:
                payback_period = time[i] / 365
                break
        
        irr = self._calculate_irr(cash_flows, time)
        
        roi = 0.0
        if self.params.initial_investment > 0:
            total_profit = np.sum(cash_flows)
            roi = (total_profit / self.params.initial_investment) * 100
        
        results = {
            'npv': float(npv),
            'irr': float(irr),
            'roi': float(roi),
            'payback_period': float(payback_period) if payback_period else None,
            'cash_flows': cash_flows.tolist(),
            'cumulative_cash_flow': cumulative_cf.tolist(),
            'daily_revenue': daily_revenue.tolist(),
            'daily_opex': daily_opex.tolist(),
            'parameters': {
                'oil_price': oil_price,
                'operating_cost': operating_cost,
                'discount_rate': discount_rate,
                'initial_investment': self.params.initial_investment
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
            
            npv_at_0 = npv_func(0)
            npv_at_1 = npv_func(1)
            
            if npv_at_0 * npv_at_1 < 0:
                irr = optimize.brentq(npv_func, 0, 1, maxiter=1000)
            elif npv_at_0 > 0 and npv_at_1 > 0:
                irr = 0.5
            else:
                irr = 0.0
                
            return max(0, irr)
            
        except Exception as e:
            logger.warning(f"IRR calculation failed: {e}")
            return 0.0
    
    def _perform_sensitivity_analysis(self) -> Dict[str, Any]:
        logger.info("Performing sensitivity analysis")
        
        base_npv = self.results.get('economic_analysis', {}).get('npv', 1e6)
        
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
            
            sensitivity_index = (max(npv_values) - min(npv_values)) / base_npv if base_npv != 0 else 0
            
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
        
        key_params = []
        for param_name, data in sensitivity_results.items():
            if data['sensitivity_index'] > 0.3:
                key_params.append(param_name)
        
        sensitivity_results['key_parameters'] = key_params
        
        logger.info(f"Sensitivity analysis: Key parameters = {key_params}")
        
        return sensitivity_results
    
    def export_results(self, output_dir: str) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = {}
        
        if not self.results:
            self.run_comprehensive_simulation()
        
        production = self.results.get('production_forecast', {})
        pressure = self.results.get('pressure_forecast', {})
        economics = self.results.get('economic_analysis', {})
        
        if production and 'time' in production:
            time_days = np.array(production['time'])
            time_years = time_days / 365.0
            total_production = np.array(production['total_production'])
        else:
            time_days = np.array([])
            time_years = np.array([])
            total_production = np.array([])
        
        if pressure and 'pressure' in pressure:
            pressure_values = np.array(pressure['pressure'])
        else:
            pressure_values = np.array([])
        
        if economics and 'cash_flows' in economics:
            cash_flows = np.array(economics['cash_flows'])
        else:
            cash_flows = np.array([])
        
        csv_path = os.path.join(output_dir, 'simulation_results.csv')
        
        if len(time_days) > 0:
            df = pd.DataFrame({
                'Time_Days': time_days,
                'Time_Years': time_years,
                'Oil_Rate_bbl/day': total_production,
                'Cumulative_Oil_bbl': np.cumsum(total_production) if len(total_production) > 0 else [],
                'Reservoir_Pressure_psi': pressure_values if len(pressure_values) > 0 else np.zeros(len(time_days)),
                'Cash_Flow_USD': cash_flows if len(cash_flows) > 0 else np.zeros(len(time_days)),
                'Cumulative_Cash_Flow_USD': np.cumsum(cash_flows) if len(cash_flows) > 0 else np.zeros(len(time_days))
            })
        else:
            df = pd.DataFrame({
                'Time_Days': [],
                'Time_Years': [],
                'Oil_Rate_bbl/day': [],
                'Cumulative_Oil_bbl': [],
                'Reservoir_Pressure_psi': [],
                'Cash_Flow_USD': [],
                'Cumulative_Cash_Flow_USD': []
            })
        
        df.to_csv(csv_path, index=False)
        output_files['csv'] = csv_path
        
        json_path = os.path.join(output_dir, 'simulation_results.json')
        
        json_results = {
            'parameters': self.params.to_dict(),
            'wells': production.get('well_names', []) if production else [],
            'forecast': {
                'time_days': time_days.tolist(),
                'time_years': time_years.tolist(),
                'oil_rate': total_production.tolist(),
                'cumulative_oil': np.cumsum(total_production).tolist() if len(total_production) > 0 else [],
                'well_production': production.get('production', []) if production else []
            },
            'economics': {
                'cash_flow': cash_flows.tolist(),
                'cumulative_cash_flow': np.cumsum(cash_flows).tolist() if len(cash_flows) > 0 else [],
                'npv': economics.get('npv', 0.0),
                'irr': economics.get('irr', 0.0),
                'roi': economics.get('roi', 0.0),
                'payback_period': economics.get('payback_period')
            },
            'pressure': {
                'time_days': time_days.tolist(),
                'pressure': pressure_values.tolist(),
                'initial_pressure': pressure.get('statistics', {}).get('initial_pressure', 0.0) if pressure else 0.0,
                'final_pressure': pressure.get('statistics', {}).get('final_pressure', 0.0) if pressure else 0.0
            },
            'summary': {
                'total_oil': float(np.sum(total_production)) if len(total_production) > 0 else 0.0,
                'peak_oil_rate': float(np.max(total_production)) if len(total_production) > 0 else 0.0,
                'final_pressure': float(pressure_values[-1]) if len(pressure_values) > 0 else 0.0,
                'simulation_days': int(len(time_days)),
                'total_npv': economics.get('npv', 0.0),
                'total_irr': economics.get('irr', 0.0)
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        output_files['json'] = json_path
        
        try:
            plots_path = os.path.join(output_dir, 'plots.png')
            self.create_visualizations(plots_path)
            output_files['plots'] = plots_path
        except Exception as e:
            logger.warning(f"Failed to create plots: {e}")
        
        try:
            report_path = os.path.join(output_dir, 'simulation_report.txt')
            with open(report_path, 'w') as f:
                f.write(self.generate_report())
            output_files['report'] = report_path
        except Exception as e:
            logger.warning(f"Failed to create report: {e}")
        
        return output_files
    
    def create_visualizations(self, output_path: str):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        production = self.results.get('production_forecast', {})
        pressure = self.results.get('pressure_forecast', {})
        economics = self.results.get('economic_analysis', {})
        
        if production and 'time' in production and 'total_production' in production:
            time_days = np.array(production['time'])
            time_years = time_days / 365.0
            total_production = np.array(production['total_production'])
            
            axes[0, 0].plot(time_years, total_production, 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Time (Years)')
            axes[0, 0].set_ylabel('Oil Rate (bbl/day)')
            axes[0, 0].set_title('Production Forecast')
            axes[0, 0].grid(True, alpha=0.3)
        
        if pressure and 'time' in pressure and 'pressure' in pressure:
            time_days = np.array(pressure['time'])
            time_years = time_days / 365.0
            pressure_values = np.array(pressure['pressure'])
            
            axes[0, 1].plot(time_years, pressure_values, 'r-', linewidth=2)
            axes[0, 1].set_xlabel('Time (Years)')
            axes[0, 1].set_ylabel('Pressure (psi)')
            axes[0, 1].set_title('Pressure Profile')
            axes[0, 1].grid(True, alpha=0.3)
        
        if economics and 'cash_flows' in economics:
            if production and 'time' in production:
                time_days = np.array(production['time'])
                time_years = time_days / 365.0
                cash_flows = np.array(economics['cash_flows'])
                
                axes[1, 0].plot(time_years, cash_flows / 1000, 'g-', linewidth=2)
                axes[1, 0].set_xlabel('Time (Years)')
                axes[1, 0].set_ylabel('Cash Flow ($K)')
                axes[1, 0].set_title('Economic Analysis')
                axes[1, 0].grid(True, alpha=0.3)
                
                cumulative_cf = np.cumsum(cash_flows)
                axes[1, 1].plot(time_years, cumulative_cf / 1000, 'purple', linewidth=2)
                axes[1, 1].set_xlabel('Time (Years)')
                axes[1, 1].set_ylabel('Cumulative Cash Flow ($K)')
                axes[1, 1].set_title('Cumulative Cash Flow')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self) -> str:
        if not self.results:
            return "No simulation results available."
        
        production = self.results.get('production_forecast', {})
        pressure = self.results.get('pressure_forecast', {})
        economics = self.results.get('economic_analysis', {})
        material = self.results.get('material_balance', {})
        
        report = "=" * 60 + "\n"
        report += "RESERVOIR SIMULATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        if production and 'statistics' in production:
            report += "1. PRODUCTION FORECAST\n"
            report += "-" * 40 + "\n"
            stats = production['statistics']
            report += f"Peak Production: {stats.get('peak_production', 0):.0f} bbl/day\n"
            report += f"Final Production: {stats.get('final_production', 0):.0f} bbl/day\n"
            report += f"Total Cumulative: {stats.get('total_cumulative', 0):,.0f} bbl\n"
            report += f"Forecast Period: {stats.get('forecast_years', 0)} years\n\n"
        
        if pressure and 'statistics' in pressure:
            report += "2. PRESSURE PROFILE\n"
            report += "-" * 40 + "\n"
            stats = pressure['statistics']
            report += f"Initial Pressure: {stats.get('initial_pressure', 0):.0f} psi\n"
            report += f"Final Pressure: {stats.get('final_pressure', 0):.0f} psi\n"
            report += f"Pressure Drop: {stats.get('pressure_drop', 0):.0f} psi\n"
            report += f"Pressure Drop %: {stats.get('pressure_drop_percent', 0):.1f}%\n\n"
        
        if material and 'ooip_stb' in material:
            report += "3. MATERIAL BALANCE\n"
            report += "-" * 40 + "\n"
            report += f"OOIP: {material.get('ooip_stb', 0):,.0f} STB\n"
            if 'regression' in material:
                report += f"R²: {material['regression'].get('r_squared', 0):.3f}\n"
            report += "\n"
        
        if economics:
            report += "4. ECONOMIC ANALYSIS\n"
            report += "-" * 40 + "\n"
            report += f"NPV: ${economics.get('npv', 0)/1000:.1f}K\n"
            report += f"IRR: {economics.get('irr', 0)*100:.1f}%\n"
            report += f"ROI: {economics.get('roi', 0):.1f}%\n"
            payback = economics.get('payback_period')
            if payback:
                report += f"Payback Period: {payback:.1f} years\n"
            
            if 'summary' in economics:
                summary = economics['summary']
                report += f"Total Revenue: ${summary.get('total_revenue', 0)/1000:.1f}K\n"
                report += f"Total OPEX: ${summary.get('total_opex', 0)/1000:.1f}K\n"
                report += f"Profit Margin: {summary.get('profit_margin', 0):.1f}%\n"
            report += "\n"
        
        report += "5. PARAMETERS\n"
        report += "-" * 40 + "\n"
        report += f"Oil Price: ${self.params.oil_price}/bbl\n"
        report += f"Operating Cost: ${self.params.operating_cost}/bbl\n"
        report += f"Discount Rate: {self.params.discount_rate*100:.1f}%\n"
        report += f"Initial Investment: ${self.params.initial_investment/1000:.0f}K\n\n"
        
        report += "=" * 60 + "\n"
        report += f"Report generated: {pd.Timestamp.now()}\n"
        report += "=" * 60
        
        return report
