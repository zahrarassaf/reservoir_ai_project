# src/economics.py
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from scipy.optimize import curve_fit, brentq
from scipy.stats import linregress
from numpy_financial import irr as npf_irr, npv as npf_npv
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class EconomicParameters:
    forecast_years: int = 15
    oil_price: float = 82.50
    gas_price: float = 3.50
    opex_per_bbl: float = 16.50
    fixed_opex: float = 2_500_000
    discount_rate: float = 0.095
    inflation_rate: float = 0.025
    tax_rate: float = 0.32
    royalty_rate: float = 0.125
    abandonment_cost: float = 5_000_000
    capex_per_producer: float = 3_500_000
    capex_per_injector: float = 2_800_000
    facilities_cost: float = 15_000_000
    contingency_rate: float = 0.15

@dataclass
class ReservoirProperties:
    original_oil_in_place: float = 0.0
    recoverable_oil: float = 0.0
    recovery_factor: float = 0.0
    drive_mechanism: str = "Solution Gas Drive"
    aquifer_strength: float = 0.0
    reservoir_volume: float = 0.0
    average_porosity: float = 0.15
    connate_water_saturation: float = 0.25
    formation_volume_factor: float = 1.2

@dataclass
class WellProductionData:
    time_points: np.ndarray
    oil_rate: np.ndarray
    gas_rate: Optional[np.ndarray] = None
    water_rate: Optional[np.ndarray] = None
    bottomhole_pressure: Optional[np.ndarray] = None
    well_type: str = "PRODUCER"

class DeclineCurveAnalysis:
    @staticmethod
    def hyperbolic_decline(t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
        epsilon = 1e-10
        b_safe = np.clip(b, 0.1, 1.9)
        di_safe = np.clip(di, 1e-6, 0.5)
        denominator = 1 + b_safe * di_safe * t
        denominator = np.maximum(denominator, epsilon)
        return qi / denominator ** (1 / b_safe)
    
    @staticmethod
    def exponential_decline(t: np.ndarray, qi: float, di: float) -> np.ndarray:
        di_safe = np.clip(di, 1e-6, 0.5)
        return qi * np.exp(-di_safe * t)
    
    @staticmethod
    def harmonic_decline(t: np.ndarray, qi: float, di: float) -> np.ndarray:
        di_safe = np.clip(di, 1e-6, 0.5)
        denominator = 1 + di_safe * t
        denominator = np.maximum(denominator, 1e-10)
        return qi / denominator
    
    @staticmethod
    def fit_decline_curve(time_series: np.ndarray, rate_series: np.ndarray) -> Optional[Dict]:
        if len(time_series) < 4 or len(rate_series) < 4:
            return None
        
        valid_mask = (rate_series > 0) & (~np.isnan(rate_series))
        if np.sum(valid_mask) < 4:
            return None
        
        time_normalized = time_series[valid_mask] - time_series[valid_mask][0]
        rates_valid = rate_series[valid_mask]
        
        models = [
            ('exponential', DeclineCurveAnalysis.exponential_decline),
            ('harmonic', DeclineCurveAnalysis.harmonic_decline),
        ]
        
        optimal_params = None
        best_r_squared = -np.inf
        
        for model_name, model_function in models:
            try:
                if model_name == 'exponential':
                    log_rates = np.log(rates_valid)
                    slope, intercept, r_value, _, _ = linregress(time_normalized, log_rates)
                    qi = np.exp(intercept)
                    di = max(-slope, 1e-6)
                    parameters = {'qi': qi, 'di': di, 'b': 0}
                else:
                    inv_rates = 1 / rates_valid
                    slope, intercept, r_value, _, _ = linregress(time_normalized, inv_rates)
                    qi = 1 / intercept if intercept != 0 else rates_valid[0]
                    di = slope / intercept if intercept != 0 else 1e-6
                    parameters = {'qi': qi, 'di': max(di, 1e-6), 'b': 1}
                
                predicted_rates = model_function(time_normalized, **{
                    k: v for k, v in parameters.items() 
                    if k in ['qi', 'di']
                })
                
                ss_residual = np.sum((rates_valid - predicted_rates) ** 2)
                ss_total = np.sum((rates_valid - np.mean(rates_valid)) ** 2)
                r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
                
                if r_squared > best_r_squared and r_squared > 0.7:
                    best_r_squared = r_squared
                    optimal_params = parameters
                    optimal_params['r2'] = r_squared
                    optimal_params['method'] = model_name
                    
            except Exception:
                continue
        
        if optimal_params is None:
            qi_guess = rates_valid[0]
            di_guess = 0.001
            optimal_params = {
                'qi': qi_guess,
                'di': di_guess,
                'b': 0,
                'r2': 0,
                'method': 'exponential'
            }
        
        return optimal_params

class ReservoirCharacterization:
    @staticmethod
    def estimate_original_oil_in_place(grid_dimensions: Tuple[int, int, int],
                                      cell_size: Tuple[float, float, float],
                                      porosity: float,
                                      water_saturation: float,
                                      formation_volume_factor: float) -> float:
        cells_x, cells_y, cells_z = grid_dimensions
        dx, dy, dz = cell_size
        
        bulk_volume = cells_x * dx * cells_y * dy * cells_z * dz
        pore_volume = bulk_volume * porosity
        hydrocarbon_pore_volume = pore_volume * (1 - water_saturation)
        
        return hydrocarbon_pore_volume / formation_volume_factor
    
    @staticmethod
    def estimate_recovery_factor(production_history: np.ndarray,
                                original_oil_in_place: float,
                                drive_mechanism: str) -> float:
        cumulative_production = np.sum(production_history)
        
        if original_oil_in_place <= 0:
            return 0.0
        
        base_recovery = cumulative_production / original_oil_in_place
        
        mechanism_factors = {
            "Solution Gas Drive": 0.15,
            "Water Drive": 0.35,
            "Gas Cap Drive": 0.25,
            "Combination Drive": 0.30
        }
        
        mechanism_factor = mechanism_factors.get(drive_mechanism, 0.20)
        
        return min(base_recovery * 1.5, mechanism_factor)

class ReservoirSimulator:
    def __init__(self, reservoir_data: Dict, 
                 economic_parameters: EconomicParameters = None):
        self.reservoir_data = reservoir_data
        self.economic_parameters = economic_parameters or EconomicParameters()
        self.results = {}
        self.decline_curves = {}
        self.reservoir_props = ReservoirProperties()
    
    def run_comprehensive_analysis(self) -> Dict:
        return self._execute_comprehensive_analysis()
    
    def execute_comprehensive_analysis(self) -> Dict:
        return self._execute_comprehensive_analysis()
    
    def _execute_comprehensive_analysis(self) -> Dict:
        logger.info("Starting advanced reservoir simulation")
        
        try:
            self._validate_input_data()
            
            self.reservoir_props = self._characterize_reservoir()
            
            self.decline_curves = self._perform_decline_analysis()
            
            production_forecast = self._generate_production_forecast()
            
            economic_results = self._perform_economic_evaluation(production_forecast)
            
            uncertainty_results = self._run_uncertainty_analysis(production_forecast)
            
            self.results = {
                'reservoir_properties': self._dict_from_dataclass(self.reservoir_props),
                'decline_analysis': self.decline_curves,
                'production_forecast': production_forecast,
                'economic_evaluation': economic_results,
                'uncertainty_analysis': uncertainty_results,
                'key_performance_indicators': self._calculate_kpis(economic_results, production_forecast)
            }
            
            logger.info("Simulation completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}", exc_info=True)
            return self._generate_error_results(str(e))
    
    def _validate_input_data(self):
        if 'wells' not in self.reservoir_data or not self.reservoir_data['wells']:
            raise ValueError("No well data available")
        
        valid_wells = 0
        for well_name, well in self.reservoir_data['wells'].items():
            if hasattr(well, 'oil_rate') and hasattr(well, 'time_points'):
                if len(well.oil_rate) >= 3 and len(well.time_points) >= 3:
                    valid_wells += 1
        
        if valid_wells < 3:
            raise ValueError(f"Insufficient valid well data: {valid_wells} wells")
        
        logger.info(f"Validated {valid_wells} wells with production data")
    
    def _characterize_reservoir(self) -> ReservoirProperties:
        wells = self.reservoir_data['wells']
        
        total_production = 0
        peak_rate = 0
        
        for well_name, well in wells.items():
            if hasattr(well, 'oil_rate'):
                rates = well.oil_rate
                if len(rates) > 0:
                    peak_rate = max(peak_rate, np.max(rates))
                    
                    if hasattr(well, 'time_points'):
                        times = well.time_points
                        if len(times) >= 2:
                            total_production += np.trapz(rates, times)
        
        avg_porosity = 0.15
        if 'grid' in self.reservoir_data and 'porosity' in self.reservoir_data['grid']:
            porosity_data = self.reservoir_data['grid']['porosity']
            if isinstance(porosity_data, np.ndarray) and len(porosity_data) > 0:
                avg_porosity = np.mean(porosity_data)
        
        grid_dims = self.reservoir_data['grid'].get('dimensions', (24, 25, 15))
        cell_count = grid_dims[0] * grid_dims[1] * grid_dims[2]
        
        avg_thickness = 50.0
        area_acres = 40.0
        boi = 1.2
        swi = 0.25
        
        ooip = 7758 * area_acres * avg_thickness * avg_porosity * (1 - swi) / boi
        
        recovery_factor = min(total_production / ooip if ooip > 0 else 0, 0.45)
        
        return ReservoirProperties(
            original_oil_in_place=ooip,
            recoverable_oil=ooip * recovery_factor,
            recovery_factor=recovery_factor,
            drive_mechanism="Solution Gas Drive" if recovery_factor < 0.25 else "Water Drive",
            aquifer_strength=min(recovery_factor / 0.3, 1.0),
            average_porosity=avg_porosity,
            connate_water_saturation=swi,
            formation_volume_factor=boi
        )
    
    def _perform_decline_analysis(self) -> Dict:
        decline_results = {}
        
        for well_name, well in self.reservoir_data['wells'].items():
            if hasattr(well, 'oil_rate') and hasattr(well, 'time_points'):
                rates = well.oil_rate
                times = well.time_points
                
                if len(rates) >= 4 and len(times) >= 4:
                    result = DeclineCurveAnalysis.fit_decline_curve(times, rates)
                    
                    if result and result.get('r2', 0) > 0.6:
                        decline_results[well_name] = result
                        
                        well_type = getattr(well, 'well_type', 'PRODUCER')
                        logger.info(f"{well_type} {well_name}: qi={result['qi']:.0f} bpd, "
                                  f"di={result['di']:.4f}, R²={result.get('r2', 0):.3f}")
        
        return decline_results
    
    def _generate_production_forecast(self) -> Dict:
        forecast_days = self.economic_parameters.forecast_years * 365
        monthly_steps = self.economic_parameters.forecast_years * 12
        
        time_forecast = np.linspace(0, forecast_days, monthly_steps)
        
        field_production = np.zeros(monthly_steps)
        well_forecasts = {}
        
        for well_name, decline_params in self.decline_curves.items():
            well = self.reservoir_data['wells'][well_name]
            well_type = getattr(well, 'well_type', 'PRODUCER')
            
            if well_type != 'PRODUCER':
                continue
            
            qi = decline_params['qi']
            di = decline_params['di']
            b = decline_params.get('b', 0)
            method = decline_params.get('method', 'exponential')
            
            if method == 'exponential':
                rates = DeclineCurveAnalysis.exponential_decline(time_forecast, qi, di)
            elif method == 'harmonic':
                rates = DeclineCurveAnalysis.harmonic_decline(time_forecast, qi, di)
            else:
                rates = DeclineCurveAnalysis.exponential_decline(time_forecast, qi, di)
            
            economic_limit = 50.0
            rates = np.where(rates < economic_limit, 0, rates)
            
            field_production += rates
            
            well_forecasts[well_name] = {
                'time': time_forecast,
                'rate': rates,
                'cumulative': np.cumsum(rates * 30.4),
                'eur': np.trapz(rates, time_forecast),
                'decline_params': decline_params
            }
        
        annual_production = self._convert_to_annual(field_production, monthly_steps)
        
        return {
            'time': time_forecast,
            'field_rate': field_production,
            'field_cumulative': np.cumsum(field_production * 30.4),
            'annual_production': annual_production,
            'well_forecasts': well_forecasts,
            'total_eur': np.trapz(field_production, time_forecast)
        }
    
    def _convert_to_annual(self, monthly_production: np.ndarray, total_months: int) -> np.ndarray:
        years = self.economic_parameters.forecast_years
        annual = np.zeros(years)
        
        months_per_year = 12
        for year in range(years):
            start_month = year * months_per_year
            end_month = min((year + 1) * months_per_year, total_months)
            if start_month < len(monthly_production):
                annual[year] = np.sum(monthly_production[start_month:end_month]) * 30.4
        
        return annual
    
    def _perform_economic_evaluation(self, production_forecast: Dict) -> Dict:
        annual_production = production_forecast['annual_production']
        years = len(annual_production)
        
        producers = sum(1 for w in self.reservoir_data['wells'].values() 
                       if getattr(w, 'well_type', 'PRODUCER') == 'PRODUCER')
        injectors = sum(1 for w in self.reservoir_data['wells'].values() 
                       if getattr(w, 'well_type', 'PRODUCER') == 'INJECTOR')
        
        capex_wells = (producers * self.economic_parameters.capex_per_producer + 
                      injectors * self.economic_parameters.capex_per_injector)
        capex_total = capex_wells + self.economic_parameters.facilities_cost
        capex_total *= (1 + self.economic_parameters.contingency_rate)
        
        cash_flows = [-capex_total]
        
        for year in range(years):
            oil_production = annual_production[year]
            
            oil_revenue = oil_production * self.economic_parameters.oil_price * (1 + self.economic_parameters.inflation_rate) ** year
            
            royalty = oil_revenue * self.economic_parameters.royalty_rate
            
            variable_opex = oil_production * self.economic_parameters.opex_per_bbl
            fixed_opex = self.economic_parameters.fixed_opex * (1 + self.economic_parameters.inflation_rate) ** year
            
            depreciation = capex_total / 10 if year < 10 else 0
            
            operating_cost = variable_opex + fixed_opex
            
            ebitda = oil_revenue - royalty - operating_cost
            
            tax = max(0, ebitda - depreciation) * self.economic_parameters.tax_rate
            
            net_cash_flow = ebitda - tax + depreciation
            
            cash_flows.append(net_cash_flow)
        
        if years > 0:
            cash_flows[-1] -= self.economic_parameters.abandonment_cost
        
        npv = self._calculate_npv(cash_flows, self.economic_parameters.discount_rate)
        irr = self._calculate_irr(cash_flows)
        
        total_revenue = sum(annual_production) * self.economic_parameters.oil_price
        total_opex = sum(annual_production) * self.economic_parameters.opex_per_bbl + years * self.economic_parameters.fixed_opex
        
        roi = (npv / capex_total) * 100 if capex_total > 0 else 0
        profit_index = -npv / capex_total if capex_total > 0 else 0
        
        payback = self._calculate_discounted_payback(cash_flows, self.economic_parameters.discount_rate)
        
        break_even_price = self._calculate_break_even_price(cash_flows, sum(annual_production))
        
        return {
            'capex': capex_total,
            'opex': total_opex,
            'revenue': total_revenue,
            'npv': npv,
            'irr': irr * 100,
            'roi': roi,
            'profitability_index': profit_index,
            'payback_period': payback,
            'break_even_price': break_even_price,
            'cash_flows': cash_flows,
            'annual_cash_flows': cash_flows[1:],
            'economic_limit': self._calculate_economic_limit(),
            'unit_development_cost': capex_total / sum(annual_production) if sum(annual_production) > 0 else 0
        }
    
    def _calculate_npv(self, cash_flows: List[float], discount_rate: float) -> float:
        try:
            return npf_npv(discount_rate, cash_flows)
        except:
            npv = 0.0
            for t, cf in enumerate(cash_flows):
                npv += cf / ((1 + discount_rate) ** t)
            return npv
    
    def _calculate_irr(self, cash_flows: List[float]) -> float:
        try:
            irr_value = npf_irr(cash_flows)
            if irr_value is None or np.isnan(irr_value) or irr_value < -0.9 or irr_value > 5:
                return self._calculate_irr_manual(cash_flows)
            return irr_value
        except:
            return self._calculate_irr_manual(cash_flows)
    
    def _calculate_irr_manual(self, cash_flows: List[float]) -> float:
        def npv_func(rate):
            return sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows))
        
        try:
            return brentq(npv_func, -0.5, 2.0, maxiter=1000)
        except:
            return 0.0
    
    def _calculate_discounted_payback(self, cash_flows: List[float], discount_rate: float) -> float:
        if len(cash_flows) < 2:
            return float('inf')
        
        initial_investment = abs(cash_flows[0])
        cumulative_pv = 0.0
        
        for year, cf in enumerate(cash_flows[1:], 1):
            discounted_cf = cf / ((1 + discount_rate) ** year)
            cumulative_pv += discounted_cf
            
            if cumulative_pv >= initial_investment:
                if year == 1:
                    return 1.0
                
                prev_cumulative = cumulative_pv - discounted_cf
                remaining = initial_investment - prev_cumulative
                fraction = remaining / discounted_cf
                return year - 1 + fraction
        
        return float('inf')
    
    def _calculate_break_even_price(self, cash_flows: List[float], total_production: float) -> float:
        if total_production <= 0:
            return 0.0
        
        total_cost = sum(abs(cf) for cf in cash_flows if cf < 0)
        return total_cost / total_production
    
    def _calculate_economic_limit(self) -> float:
        return (self.economic_parameters.fixed_opex / 365) / self.economic_parameters.opex_per_bbl
    
    def _run_uncertainty_analysis(self, production_forecast: Dict) -> Dict:
        base_npv = self.results.get('economic_evaluation', {}).get('npv', 0)
        
        scenarios = {
            'low_case': {'oil_price': -0.20, 'opex': 0.15, 'production': -0.15},
            'base_case': {'oil_price': 0.00, 'opex': 0.00, 'production': 0.00},
            'high_case': {'oil_price': 0.20, 'opex': -0.10, 'production': 0.15}
        }
        
        sensitivity = {}
        for case_name, variations in scenarios.items():
            modified_params = EconomicParameters(
                oil_price=self.economic_parameters.oil_price * (1 + variations['oil_price']),
                opex_per_bbl=self.economic_parameters.opex_per_bbl * (1 + variations['opex'])
            )
            
            modified_production = production_forecast['annual_production'] * (1 + variations['production'])
            
            modified_forecast = production_forecast.copy()
            modified_forecast['annual_production'] = modified_production
            
            economic_results = self._perform_economic_evaluation(modified_forecast)
            
            sensitivity[case_name] = {
                'npv': economic_results['npv'],
                'irr': economic_results['irr'],
                'key_assumptions': variations
            }
        
        tornado_data = []
        for variable in ['oil_price', 'production', 'opex']:
            low_npv = sensitivity['low_case']['npv'] if variable in ['oil_price', 'production'] else base_npv
            high_npv = sensitivity['high_case']['npv'] if variable in ['oil_price', 'production'] else base_npv
            
            tornado_data.append({
                'variable': variable,
                'low_impact': low_npv - base_npv,
                'high_impact': high_npv - base_npv,
                'swing': abs(high_npv - low_npv)
            })
        
        tornado_data.sort(key=lambda x: x['swing'], reverse=True)
        
        return {
            'scenario_analysis': sensitivity,
            'tornado_analysis': tornado_data,
            'base_case_npv': base_npv,
            'npv_range': (sensitivity['low_case']['npv'], sensitivity['high_case']['npv']),
            'confidence_interval': self._calculate_confidence_interval(sensitivity)
        }
    
    def _calculate_confidence_interval(self, sensitivity: Dict) -> Dict:
        npvs = [scenario['npv'] for scenario in sensitivity.values()]
        mean_npv = np.mean(npvs)
        std_npv = np.std(npvs)
        
        return {
            'mean': mean_npv,
            'std': std_npv,
            'p10': np.percentile(npvs, 10),
            'p50': np.percentile(npvs, 50),
            'p90': np.percentile(npvs, 90)
        }
    
    def _calculate_kpis(self, economic_results: Dict, production_forecast: Dict) -> Dict:
        total_production = sum(production_forecast['annual_production'])
        
        return {
            'production_per_well': total_production / len(self.reservoir_data['wells']) if self.reservoir_data['wells'] else 0,
            'revenue_per_bbl': economic_results['revenue'] / total_production if total_production > 0 else 0,
            'opex_per_bbl': economic_results['opex'] / total_production if total_production > 0 else 0,
            'netback_per_bbl': (economic_results['revenue'] - economic_results['opex']) / total_production if total_production > 0 else 0,
            'capex_per_bbl': economic_results['capex'] / total_production if total_production > 0 else 0,
            'reserve_replacement_ratio': 1.0,
            'production_decline_rate': self._calculate_decline_rate(production_forecast['annual_production'])
        }
    
    def _calculate_decline_rate(self, annual_production: np.ndarray) -> float:
        if len(annual_production) < 2:
            return 0.0
        
        decline_rates = []
        for i in range(1, len(annual_production)):
            if annual_production[i-1] > 0:
                decline = (annual_production[i-1] - annual_production[i]) / annual_production[i-1]
                decline_rates.append(decline)
        
        return np.mean(decline_rates) * 100 if decline_rates else 0.0
    
    def _dict_from_dataclass(self, obj):
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return {}
    
    def _generate_error_results(self, error_msg: str) -> Dict:
        return {
            'error': error_msg,
            'reservoir_properties': {},
            'decline_analysis': {},
            'production_forecast': {},
            'economic_evaluation': {
                'npv': 0,
                'irr': 0,
                'roi': 0,
                'payback_period': float('inf')
            },
            'uncertainty_analysis': {},
            'key_performance_indicators': {}
        }
