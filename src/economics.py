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
    """Comprehensive economic parameters for reservoir valuation."""
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
    """Physical properties of the reservoir."""
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
    """Container for well production time series."""
    time_points: np.ndarray
    oil_rate: np.ndarray
    gas_rate: Optional[np.ndarray] = None
    water_rate: Optional[np.ndarray] = None
    bottomhole_pressure: Optional[np.ndarray] = None
    well_type: str = "PRODUCER"

class DeclineCurveAnalysis:
    """Advanced decline curve analysis using Arps, exponential, and harmonic models."""
    
    @staticmethod
    def hyperbolic_decline(t: np.ndarray, initial_rate: float, 
                          initial_decline: float, b_factor: float) -> np.ndarray:
        """Arps hyperbolic decline model."""
        epsilon = 1e-10
        b_clamped = np.clip(b_factor, 0.1, 1.9)
        d_clamped = np.clip(initial_decline, 1e-6, 0.5)
        denominator = 1 + b_clamped * d_clamped * t
        denominator = np.maximum(denominator, epsilon)
        return initial_rate / denominator ** (1 / b_clamped)
    
    @staticmethod
    def exponential_decline(t: np.ndarray, initial_rate: float, 
                           decline_rate: float) -> np.ndarray:
        """Exponential decline model."""
        decline_rate_clamped = np.clip(decline_rate, 1e-6, 0.5)
        return initial_rate * np.exp(-decline_rate_clamped * t)
    
    @staticmethod
    def harmonic_decline(t: np.ndarray, initial_rate: float, 
                        decline_rate: float) -> np.ndarray:
        """Harmonic decline model."""
        decline_rate_clamped = np.clip(decline_rate, 1e-6, 0.5)
        denominator = 1 + decline_rate_clamped * t
        denominator = np.maximum(denominator, 1e-10)
        return initial_rate / denominator
    
    @staticmethod
    def fit_decline_curve(time_series: np.ndarray, rate_series: np.ndarray) -> Optional[Dict]:
        """Fit optimal decline curve to production data."""
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
                    initial_rate = np.exp(intercept)
                    decline_rate = max(-slope, 1e-6)
                    parameters = {'initial_rate': initial_rate, 
                                 'decline_rate': decline_rate, 'b_factor': 0}
                else:
                    inverse_rates = 1 / rates_valid
                    slope, intercept, r_value, _, _ = linregress(time_normalized, inverse_rates)
                    initial_rate = 1 / intercept if intercept != 0 else rates_valid[0]
                    decline_rate = slope / intercept if intercept != 0 else 1e-6
                    parameters = {'initial_rate': initial_rate, 
                                 'decline_rate': max(decline_rate, 1e-6), 'b_factor': 1}
                
                predicted_rates = model_function(time_normalized, **{
                    k: v for k, v in parameters.items() 
                    if k in ['initial_rate', 'decline_rate']
                })
                
                ss_residual = np.sum((rates_valid - predicted_rates) ** 2)
                ss_total = np.sum((rates_valid - np.mean(rates_valid)) ** 2)
                r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
                
                if r_squared > best_r_squared and r_squared > 0.7:
                    best_r_squared = r_squared
                    optimal_params = parameters
                    optimal_params['r_squared'] = r_squared
                    optimal_params['model_type'] = model_name
                    
            except Exception:
                continue
        
        if optimal_params is None:
            initial_rate_guess = rates_valid[0]
            decline_rate_guess = 0.001
            optimal_params = {
                'initial_rate': initial_rate_guess,
                'decline_rate': decline_rate_guess,
                'b_factor': 0,
                'r_squared': 0,
                'model_type': 'exponential'
            }
        
        return optimal_params

class ReservoirCharacterization:
    """Reservoir property estimation and characterization."""
    
    @staticmethod
    def estimate_original_oil_in_place(grid_dimensions: Tuple[int, int, int],
                                      cell_size: Tuple[float, float, float],
                                      porosity: float,
                                      water_saturation: float,
                                      formation_volume_factor: float) -> float:
        """Calculate OOIP using volumetric method."""
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
        """Estimate recovery factor based on production data and drive mechanism."""
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
    """Comprehensive reservoir simulation and economic evaluation system."""
    
    def __init__(self, reservoir_data: Dict, 
                 economic_parameters: EconomicParameters = None):
        self.reservoir_data = reservoir_data
        self.economic_parameters = economic_parameters or EconomicParameters()
        self.simulation_results = {}
        self.decline_curves = {}
        self.reservoir_properties = ReservoirProperties()
        
    def execute_comprehensive_analysis(self) -> Dict:
        """Execute full reservoir simulation workflow."""
        logger.info("Initiating comprehensive reservoir simulation")
        
        try:
            self._validate_input_data()
            
            self.reservoir_properties = self._characterize_reservoir()
            
            self.decline_curves = self._perform_decline_analysis()
            
            production_forecast = self._generate_production_forecast()
            
            economic_evaluation = self._perform_economic_analysis(production_forecast)
            
            uncertainty_analysis = self._execute_uncertainty_analysis(production_forecast)
            
            self.simulation_results = {
                'reservoir_properties': self._serialize_dataclass(self.reservoir_properties),
                'decline_analysis': self.decline_curves,
                'production_forecast': production_forecast,
                'economic_evaluation': economic_evaluation,
                'uncertainty_analysis': uncertainty_analysis,
                'performance_metrics': self._compute_performance_metrics(
                    economic_evaluation, production_forecast)
            }
            
            logger.info("Simulation completed successfully")
            return self.simulation_results
            
        except Exception as simulation_error:
            logger.error(f"Simulation terminated: {simulation_error}", exc_info=True)
            return self._generate_error_output(str(simulation_error))
    
    def _validate_input_data(self):
        """Validate input data structure and completeness."""
        if 'wells' not in self.reservoir_data or not self.reservoir_data['wells']:
            raise ValueError("Insufficient well data for analysis")
        
        validated_wells = 0
        for well_identifier, well_data in self.reservoir_data['wells'].items():
            if (hasattr(well_data, 'oil_rate') and hasattr(well_data, 'time_points') and
                len(well_data.oil_rate) >= 3 and len(well_data.time_points) >= 3):
                validated_wells += 1
        
        if validated_wells < 3:
            raise ValueError(f"Inadequate production data: {validated_wells} valid wells")
        
        logger.info(f"Data validation passed: {validated_wells} wells with production history")
    
    def _characterize_reservoir(self) -> ReservoirProperties:
        """Characterize reservoir properties from available data."""
        wells = self.reservoir_data['wells']
        
        total_cumulative_production = 0.0
        maximum_production_rate = 0.0
        
        for well_identifier, well_data in wells.items():
            if hasattr(well_data, 'oil_rate'):
                production_rates = well_data.oil_rate
                if len(production_rates) > 0:
                    maximum_production_rate = max(maximum_production_rate, 
                                                 np.max(production_rates))
                    
                    if hasattr(well_data, 'time_points'):
                        time_points = well_data.time_points
                        if len(time_points) >= 2:
                            total_cumulative_production += np.trapz(
                                production_rates, time_points)
        
        grid_info = self.reservoir_data.get('grid', {})
        
        average_porosity = 0.15
        if 'porosity' in grid_info:
            porosity_data = grid_info['porosity']
            if isinstance(porosity_data, np.ndarray) and len(porosity_data) > 0:
                average_porosity = np.mean(porosity_data)
        
        grid_dimensions = grid_info.get('dimensions', (24, 25, 15))
        grid_cell_count = grid_dimensions[0] * grid_dimensions[1] * grid_dimensions[2]
        
        average_net_pay = 50.0
        drainage_area = 40.0
        formation_volume_factor = 1.2
        connate_water_saturation = 0.25
        
        original_oil_in_place = (7758 * drainage_area * average_net_pay * 
                                average_porosity * (1 - connate_water_saturation) / 
                                formation_volume_factor)
        
        recovery_factor = ReservoirCharacterization.estimate_recovery_factor(
            total_cumulative_production, original_oil_in_place, "Solution Gas Drive")
        
        return ReservoirProperties(
            original_oil_in_place=original_oil_in_place,
            recoverable_oil=original_oil_in_place * recovery_factor,
            recovery_factor=recovery_factor,
            drive_mechanism="Solution Gas Drive" if recovery_factor < 0.25 else "Water Drive",
            aquifer_strength=min(recovery_factor / 0.3, 1.0),
            average_porosity=average_porosity,
            connate_water_saturation=connate_water_saturation,
            formation_volume_factor=formation_volume_factor
        )
    
    def _perform_decline_analysis(self) -> Dict:
        """Perform decline curve analysis for all wells."""
        decline_results = {}
        
        for well_identifier, well_data in self.reservoir_data['wells'].items():
            if hasattr(well_data, 'oil_rate') and hasattr(well_data, 'time_points'):
                production_rates = well_data.oil_rate
                time_points = well_data.time_points
                
                if len(production_rates) >= 4 and len(time_points) >= 4:
                    decline_parameters = DeclineCurveAnalysis.fit_decline_curve(
                        time_points, production_rates)
                    
                    if decline_parameters and decline_parameters.get('r_squared', 0) > 0.6:
                        decline_results[well_identifier] = decline_parameters
                        
                        well_classification = getattr(well_data, 'well_type', 'PRODUCER')
                        logger.info(
                            f"{well_classification} {well_identifier}: "
                            f"qi={decline_parameters['initial_rate']:.0f} bpd, "
                            f"di={decline_parameters['decline_rate']:.4f}, "
                            f"R²={decline_parameters.get('r_squared', 0):.3f}")
        
        return decline_results
    
    def _generate_production_forecast(self) -> Dict:
        """Generate production forecast based on decline curves."""
        forecast_duration_days = self.economic_parameters.forecast_years * 365
        monthly_intervals = self.economic_parameters.forecast_years * 12
        
        forecast_time = np.linspace(0, forecast_duration_days, monthly_intervals)
        
        field_production_profile = np.zeros(monthly_intervals)
        individual_well_forecasts = {}
        
        for well_identifier, decline_parameters in self.decline_curves.items():
            well_data = self.reservoir_data['wells'][well_identifier]
            well_classification = getattr(well_data, 'well_type', 'PRODUCER')
            
            if well_classification != 'PRODUCER':
                continue
            
            initial_rate = decline_parameters['initial_rate']
            decline_rate = decline_parameters['decline_rate']
            b_factor = decline_parameters.get('b_factor', 0)
            model_type = decline_parameters.get('model_type', 'exponential')
            
            if model_type == 'exponential':
                production_rates = DeclineCurveAnalysis.exponential_decline(
                    forecast_time, initial_rate, decline_rate)
            elif model_type == 'harmonic':
                production_rates = DeclineCurveAnalysis.harmonic_decline(
                    forecast_time, initial_rate, decline_rate)
            else:
                production_rates = DeclineCurveAnalysis.exponential_decline(
                    forecast_time, initial_rate, decline_rate)
            
            economic_limit_rate = 50.0
            production_rates = np.where(
                production_rates < economic_limit_rate, 0, production_rates)
            
            field_production_profile += production_rates
            
            individual_well_forecasts[well_identifier] = {
                'time': forecast_time,
                'production_rate': production_rates,
                'cumulative_production': np.cumsum(production_rates * 30.4),
                'estimated_ultimate_recovery': np.trapz(production_rates, forecast_time),
                'decline_parameters': decline_parameters
            }
        
        annual_production_profile = self._aggregate_annual_production(
            field_production_profile, monthly_intervals)
        
        return {
            'time_series': forecast_time,
            'field_production_rate': field_production_profile,
            'cumulative_field_production': np.cumsum(field_production_profile * 30.4),
            'annual_production': annual_production_profile,
            'well_forecasts': individual_well_forecasts,
            'total_estimated_ultimate_recovery': np.trapz(
                field_production_profile, forecast_time)
        }
    
    def _aggregate_annual_production(self, monthly_production: np.ndarray, 
                                    total_months: int) -> np.ndarray:
        """Convert monthly production to annual aggregates."""
        years = self.economic_parameters.forecast_years
        annual_production = np.zeros(years)
        
        months_per_year = 12
        for year_index in range(years):
            start_month = year_index * months_per_year
            end_month = min((year_index + 1) * months_per_year, total_months)
            if start_month < len(monthly_production):
                annual_production[year_index] = np.sum(
                    monthly_production[start_month:end_month]) * 30.4
        
        return annual_production
    
    def _perform_economic_analysis(self, production_forecast: Dict) -> Dict:
        """Perform detailed economic evaluation."""
        annual_production = production_forecast['annual_production']
        evaluation_years = len(annual_production)
        
        producer_count = sum(1 for well in self.reservoir_data['wells'].values() 
                           if getattr(well, 'well_type', 'PRODUCER') == 'PRODUCER')
        injector_count = sum(1 for well in self.reservoir_data['wells'].values() 
                           if getattr(well, 'well_type', 'PRODUCER') == 'INJECTOR')
        
        well_capital_expenditure = (
            producer_count * self.economic_parameters.capex_per_producer +
            injector_count * self.economic_parameters.capex_per_injector)
        total_capital_expenditure = (
            well_capital_expenditure + self.economic_parameters.facilities_cost)
        total_capital_expenditure *= (1 + self.economic_parameters.contingency_rate)
        
        cash_flow_series = [-total_capital_expenditure]
        
        for year_index in range(evaluation_years):
            annual_oil_production = annual_production[year_index]
            
            oil_revenue = (
                annual_oil_production * self.economic_parameters.oil_price *
                (1 + self.economic_parameters.inflation_rate) ** year_index)
            
            royalty_payment = oil_revenue * self.economic_parameters.royalty_rate
            
            variable_operating_cost = (
                annual_oil_production * self.economic_parameters.opex_per_bbl)
            fixed_operating_cost = (
                self.economic_parameters.fixed_opex *
                (1 + self.economic_parameters.inflation_rate) ** year_index)
            
            depreciation_expense = (
                total_capital_expenditure / 10 if year_index < 10 else 0)
            
            total_operating_cost = variable_operating_cost + fixed_operating_cost
            
            earnings_before_tax = (
                oil_revenue - royalty_payment - total_operating_cost)
            
            tax_liability = max(0, earnings_before_tax - depreciation_expense) * \
                           self.economic_parameters.tax_rate
            
            net_cash_flow = earnings_before_tax - tax_liability + depreciation_expense
            
            cash_flow_series.append(net_cash_flow)
        
        if evaluation_years > 0:
            cash_flow_series[-1] -= self.economic_parameters.abandonment_cost
        
        net_present_value = self._calculate_net_present_value(
            cash_flow_series, self.economic_parameters.discount_rate)
        internal_rate_of_return = self._calculate_internal_rate_of_return(cash_flow_series)
        
        total_revenue = sum(annual_production) * self.economic_parameters.oil_price
        total_operating_expenditure = (
            sum(annual_production) * self.economic_parameters.opex_per_bbl +
            evaluation_years * self.economic_parameters.fixed_opex)
        
        return_on_investment = (
            (net_present_value / total_capital_expenditure) * 100 
            if total_capital_expenditure > 0 else 0)
        profitability_index = (
            -net_present_value / total_capital_expenditure 
            if total_capital_expenditure > 0 else 0)
        
        discounted_payback = self._calculate_discounted_payback_period(
            cash_flow_series, self.economic_parameters.discount_rate)
        
        break_even_price = self._compute_break_even_price(
            cash_flow_series, sum(annual_production))
        
        return {
            'capital_expenditure': total_capital_expenditure,
            'operating_expenditure': total_operating_expenditure,
            'total_revenue': total_revenue,
            'net_present_value': net_present_value,
            'internal_rate_of_return': internal_rate_of_return * 100,
            'return_on_investment': return_on_investment,
            'profitability_index': profitability_index,
            'discounted_payback_period': discounted_payback,
            'break_even_price': break_even_price,
            'cash_flow_series': cash_flow_series,
            'annual_cash_flows': cash_flow_series[1:],
            'economic_limit_rate': self._determine_economic_limit(),
            'unit_development_cost': (
                total_capital_expenditure / sum(annual_production) 
                if sum(annual_production) > 0 else 0)
        }
    
    def _calculate_net_present_value(self, cash_flows: List[float], 
                                    discount_rate: float) -> float:
        """Calculate net present value."""
        try:
            return npf_npv(discount_rate, cash_flows)
        except Exception:
            present_value = 0.0
            for time_index, cash_flow in enumerate(cash_flows):
                present_value += cash_flow / ((1 + discount_rate) ** time_index)
            return present_value
    
    def _calculate_internal_rate_of_return(self, cash_flows: List[float]) -> float:
        """Calculate internal rate of return."""
        try:
            calculated_irr = npf_irr(cash_flows)
            if (calculated_irr is None or np.isnan(calculated_irr) or 
                calculated_irr < -0.9 or calculated_irr > 5):
                return self._compute_irr_numerically(cash_flows)
            return calculated_irr
        except Exception:
            return self._compute_irr_numerically(cash_flows)
    
    def _compute_irr_numerically(self, cash_flows: List[float]) -> float:
        """Numerical IRR calculation using Brent's method."""
        def present_value_function(rate):
            return sum(cf / ((1 + rate) ** idx) for idx, cf in enumerate(cash_flows))
        
        try:
            return brentq(present_value_function, -0.5, 2.0, maxiter=1000)
        except Exception:
            return 0.0
    
    def _calculate_discounted_payback_period(self, cash_flows: List[float], 
                                           discount_rate: float) -> float:
        """Calculate discounted payback period."""
        if len(cash_flows) < 2:
            return float('inf')
        
        initial_investment = abs(cash_flows[0])
        cumulative_present_value = 0.0
        
        for year, annual_cash_flow in enumerate(cash_flows[1:], 1):
            discounted_cash_flow = annual_cash_flow / ((1 + discount_rate) ** year)
            cumulative_present_value += discounted_cash_flow
            
            if cumulative_present_value >= initial_investment:
                if year == 1:
                    return 1.0
                
                previous_cumulative = cumulative_present_value - discounted_cash_flow
                remaining_investment = initial_investment - previous_cumulative
                fractional_year = remaining_investment / discounted_cash_flow
                return year - 1 + fractional_year
        
        return float('inf')
    
    def _compute_break_even_price(self, cash_flows: List[float], 
                                 total_production: float) -> float:
        """Calculate break-even oil price."""
        if total_production <= 0:
            return 0.0
        
        total_cost = sum(abs(cash_flow) for cash_flow in cash_flows if cash_flow < 0)
        return total_cost / total_production
    
    def _determine_economic_limit(self) -> float:
        """Determine economic limit production rate."""
        return (self.economic_parameters.fixed_opex / 365) / self.economic_parameters.opex_per_bbl
    
    def _execute_uncertainty_analysis(self, production_forecast: Dict) -> Dict:
        """Perform Monte Carlo uncertainty analysis."""
        base_net_present_value = self.simulation_results.get(
            'economic_evaluation', {}).get('net_present_value', 0)
        
        scenario_configurations = {
            'pessimistic_case': {
                'oil_price_variation': -0.20, 
                'operating_cost_variation': 0.15, 
                'production_variation': -0.15
            },
            'base_case': {
                'oil_price_variation': 0.00, 
                'operating_cost_variation': 0.00, 
                'production_variation': 0.00
            },
            'optimistic_case': {
                'oil_price_variation': 0.20, 
                'operating_cost_variation': -0.10, 
                'production_variation': 0.15
            }
        }
        
        scenario_analysis = {}
        for scenario_name, parameter_variations in scenario_configurations.items():
            modified_economic_parameters = EconomicParameters(
                oil_price=self.economic_parameters.oil_price * 
                         (1 + parameter_variations['oil_price_variation']),
                opex_per_bbl=self.economic_parameters.opex_per_bbl * 
                           (1 + parameter_variations['operating_cost_variation'])
            )
            
            modified_annual_production = (
                production_forecast['annual_production'] * 
                (1 + parameter_variations['production_variation']))
            
            modified_production_forecast = production_forecast.copy()
            modified_production_forecast['annual_production'] = modified_annual_production
            
            modified_economic_results = self._perform_economic_analysis(
                modified_production_forecast)
            
            scenario_analysis[scenario_name] = {
                'net_present_value': modified_economic_results['net_present_value'],
                'internal_rate_of_return': modified_economic_results['internal_rate_of_return'],
                'parameter_assumptions': parameter_variations
            }
        
        tornado_analysis = []
        for parameter in ['oil_price', 'production', 'opex']:
            low_net_present_value = (
                scenario_analysis['pessimistic_case']['net_present_value'] 
                if parameter in ['oil_price', 'production'] else base_net_present_value)
            high_net_present_value = (
                scenario_analysis['optimistic_case']['net_present_value'] 
                if parameter in ['oil_price', 'production'] else base_net_present_value)
            
            tornado_analysis.append({
                'parameter': parameter,
                'low_impact': low_net_present_value - base_net_present_value,
                'high_impact': high_net_present_value - base_net_present_value,
                'impact_range': abs(high_net_present_value - low_net_present_value)
            })
        
        tornado_analysis.sort(key=lambda x: x['impact_range'], reverse=True)
        
        return {
            'scenario_analysis': scenario_analysis,
            'tornado_analysis': tornado_analysis,
            'base_net_present_value': base_net_present_value,
            'net_present_value_range': (
                scenario_analysis['pessimistic_case']['net_present_value'], 
                scenario_analysis['optimistic_case']['net_present_value']),
            'confidence_intervals': self._compute_confidence_intervals(scenario_analysis)
        }
    
    def _compute_confidence_intervals(self, scenario_analysis: Dict) -> Dict:
        """Compute statistical confidence intervals."""
        net_present_values = [
            scenario['net_present_value'] for scenario in scenario_analysis.values()]
        
        return {
            'mean': np.mean(net_present_values),
            'standard_deviation': np.std(net_present_values),
            'percentile_10': np.percentile(net_present_values, 10),
            'percentile_50': np.percentile(net_present_values, 50),
            'percentile_90': np.percentile(net_present_values, 90)
        }
    
    def _compute_performance_metrics(self, economic_results: Dict, 
                                    production_forecast: Dict) -> Dict:
        """Compute key performance indicators."""
        total_production_volume = sum(production_forecast['annual_production'])
        
        return {
            'average_production_per_well': (
                total_production_volume / len(self.reservoir_data['wells']) 
                if self.reservoir_data['wells'] else 0),
            'revenue_per_barrel': (
                economic_results['total_revenue'] / total_production_volume 
                if total_production_volume > 0 else 0),
            'operating_cost_per_barrel': (
                economic_results['operating_expenditure'] / total_production_volume 
                if total_production_volume > 0 else 0),
            'netback_per_barrel': (
                (economic_results['total_revenue'] - economic_results['operating_expenditure']) / 
                total_production_volume if total_production_volume > 0 else 0),
            'capital_cost_per_barrel': (
                economic_results['capital_expenditure'] / total_production_volume 
                if total_production_volume > 0 else 0),
            'reserve_replacement_ratio': 1.0,
            'annual_production_decline_rate': self._calculate_annual_decline_rate(
                production_forecast['annual_production'])
        }
    
    def _calculate_annual_decline_rate(self, annual_production: np.ndarray) -> float:
        """Calculate annual production decline rate."""
        if len(annual_production) < 2:
            return 0.0
        
        annual_decline_rates = []
        for year_index in range(1, len(annual_production)):
            if annual_production[year_index - 1] > 0:
                decline_rate = ((annual_production[year_index - 1] - 
                               annual_production[year_index]) / 
                               annual_production[year_index - 1])
                annual_decline_rates.append(decline_rate)
        
        return np.mean(annual_decline_rates) * 100 if annual_decline_rates else 0.0
    
    def _serialize_dataclass(self, data_class_instance):
        """Convert dataclass instance to dictionary."""
        if hasattr(data_class_instance, '__dict__'):
            return {key: value for key, value in data_class_instance.__dict__.items() 
                   if not key.startswith('_')}
        return {}
    
    def _generate_error_output(self, error_message: str) -> Dict:
        """Generate error output structure."""
        return {
            'error': error_message,
            'reservoir_properties': {},
            'decline_analysis': {},
            'production_forecast': {},
            'economic_evaluation': {
                'net_present_value': 0,
                'internal_rate_of_return': 0,
                'return_on_investment': 0,
                'discounted_payback_period': float('inf')
            },
            'uncertainty_analysis': {},
            'performance_metrics': {}
        }
