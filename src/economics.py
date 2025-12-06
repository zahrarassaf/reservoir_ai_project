import numpy as np
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass
from .economics import ReservoirSimulator as EconomicSimulator, SimulationParameters

logger = logging.getLogger(__name__)

@dataclass
class WellData:
    name: str
    time_points: np.ndarray
    production_rates: np.ndarray
    well_type: str = "PRODUCER"
    location: Optional[tuple] = None
    completion_data: Optional[Dict] = None

@dataclass
class ReservoirData:
    wells: Dict[str, WellData]
    grid_dimensions: tuple = (24, 25, 15)
    porosity: float = 0.15
    permeability: float = 100.0
    initial_pressure: float = 3000.0
    temperature: float = 150.0
    water_saturation: float = 0.25
    oil_viscosity: float = 1.5
    formation_volume_factor: float = 1.2

class ReservoirSimulator:
    def __init__(self, reservoir_data: ReservoirData, economic_params: Optional[Dict] = None):
        self.reservoir_data = reservoir_data
        self.economic_params = economic_params or {}
        
        # Initialize results storage
        self.simulation_results = {}
        self.decline_analysis_results = {}
        self.production_forecast = {}
        self.economic_analysis_results = {}
        
        # Simulation state
        self.is_simulated = False
        self.simulation_time = None
        
        logger.info(f"Initialized ReservoirSimulator with {len(reservoir_data.wells)} wells")
    
    def run_simulation(self, forecast_years: int = 10, 
                      oil_price: float = 75.0, 
                      operating_cost: float = 18.0) -> Dict[str, Any]:
        """
        Run comprehensive reservoir simulation including decline analysis and economic evaluation
        
        Parameters:
        -----------
        forecast_years : int
            Number of years to forecast production
        oil_price : float
            Oil price in USD/bbl
        operating_cost : float
            Operating cost in USD/bbl
            
        Returns:
        --------
        Dict containing all simulation results
        """
        logger.info(f"Starting reservoir simulation for {forecast_years} years")
        logger.info(f"Economic parameters: Oil price=${oil_price}/bbl, Opex=${operating_cost}/bbl")
        
        try:
            # Step 1: Validate input data
            if not self._validate_input_data():
                logger.error("Invalid input data for simulation")
                return self._create_error_results("Invalid input data")
            
            # Step 2: Analyze production history
            self.decline_analysis_results = self._analyze_production_history()
            
            if not self.decline_analysis_results:
                logger.warning("No valid decline analysis results")
                return self._create_error_results("No valid decline analysis")
            
            # Step 3: Generate production forecast
            self.production_forecast = self._generate_production_forecast(forecast_years)
            
            # Step 4: Prepare data for economic analysis
            economic_input_data = self._prepare_economic_input_data()
            
            # Step 5: Run economic analysis
            self.economic_analysis_results = self._run_economic_analysis(
                economic_input_data, forecast_years, oil_price, operating_cost
            )
            
            # Step 6: Compile comprehensive results
            self.simulation_results = self._compile_simulation_results(
                forecast_years, oil_price, operating_cost
            )
            
            self.is_simulated = True
            logger.info("Reservoir simulation completed successfully")
            
            return self.simulation_results
            
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_error_results(f"Simulation error: {str(e)}")
    
    def _validate_input_data(self) -> bool:
        """Validate that the input data is sufficient for simulation"""
        if not self.reservoir_data.wells:
            logger.error("No well data available")
            return False
        
        valid_wells = 0
        for well_name, well in self.reservoir_data.wells.items():
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
    
    def _analyze_production_history(self) -> Dict[str, Dict]:
        """
        Analyze historical production data using decline curve analysis
        """
        logger.info("Analyzing production history with decline curve analysis")
        
        # Import here to avoid circular imports
        from .economics import DeclineCurveAnalysis
        
        dca = DeclineCurveAnalysis()
        decline_results = {}
        
        for well_name, well in self.reservoir_data.wells.items():
            if not hasattr(well, 'time_points') or not hasattr(well, 'production_rates'):
                continue
            
            time_data = well.time_points
            rate_data = well.production_rates
            
            if len(time_data) < 3 or len(rate_data) < 3:
                logger.warning(f"Insufficient data for well {well_name}")
                continue
            
            # Ensure arrays have same length
            min_len = min(len(time_data), len(rate_data))
            time_data = time_data[:min_len]
            rate_data = rate_data[:min_len]
            
            # Remove invalid data points
            valid_mask = ~np.isnan(rate_data) & ~np.isinf(rate_data) & (rate_data > 0)
            if np.sum(valid_mask) < 3:
                logger.warning(f"Not enough valid data points for well {well_name}")
                continue
            
            time_valid = time_data[valid_mask]
            rate_valid = rate_data[valid_mask]
            
            # Try hyperbolic fit first
            decline_result = dca.fit_decline_curve(time_valid, rate_valid, method='hyperbolic')
            
            if decline_result:
                decline_results[well_name] = decline_result
                logger.info(
                    f"Decline analysis for {well_name}: "
                    f"qi={decline_result['qi']:.1f}, "
                    f"di={decline_result['di']:.4f}, "
                    f"R²={decline_result.get('r_squared', 0):.3f}"
                )
            else:
                logger.warning(f"Could not fit decline curve for well {well_name}")
        
        logger.info(f"Completed decline analysis for {len(decline_results)} wells")
        return decline_results
    
    def _generate_production_forecast(self, forecast_years: int) -> Dict[str, Dict]:
        """
        Generate production forecast using decline curve parameters
        """
        logger.info(f"Generating {forecast_years}-year production forecast")
        
        from .economics import DeclineCurveAnalysis
        
        dca = DeclineCurveAnalysis()
        forecast_days = forecast_years * 365
        time_step_days = 30  # Monthly forecast
        
        forecast_results = {}
        
        for well_name, decline_params in self.decline_analysis_results.items():
            well = self.reservoir_data.wells[well_name]
            
            if not hasattr(well, 'time_points'):
                continue
            
            # Calculate forecast start time (end of historical data)
            historical_time = well.time_points - well.time_points[0]
            forecast_start = historical_time[-1] if len(historical_time) > 0 else 0
            
            # Create forecast time array
            forecast_time = np.arange(
                forecast_start,
                forecast_start + forecast_days + time_step_days,
                time_step_days
            )
            
            # Generate forecast rates based on decline type
            if decline_params['method'] == 'exponential':
                forecast_rates = dca.exponential_decline(
                    decline_params['qi'], 
                    decline_params['di'], 
                    forecast_time
                )
            elif decline_params['method'] == 'hyperbolic':
                forecast_rates = dca.hyperbolic_decline(
                    decline_params['qi'], 
                    decline_params['di'], 
                    decline_params['b'], 
                    forecast_time
                )
            else:
                # Default to exponential
                forecast_rates = dca.exponential_decline(
                    decline_params['qi'], 
                    decline_params['di'], 
                    forecast_time
                )
            
            # Apply economic limit
            economic_limit = 20.0  # Minimum economic production rate
            forecast_rates = np.where(forecast_rates < economic_limit, 0, forecast_rates)
            
            # Calculate EUR (Estimated Ultimate Recovery)
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
                'decline_parameters': decline_params,
                'economic_limit': economic_limit,
                'well_type': well.well_type if hasattr(well, 'well_type') else 'PRODUCER'
            }
        
        logger.info(f"Generated production forecast for {len(forecast_results)} wells")
        return forecast_results
    
    def _prepare_economic_input_data(self) -> Any:
        """
        Prepare data structure for economic analysis
        """
        class EconomicInputData:
            def __init__(self, wells):
                self.wells = wells
        
        # Create simplified well data for economic analysis
        economic_wells = {}
        
        for well_name, well in self.reservoir_data.wells.items():
            class WellData:
                def __init__(self, time_points, production_rates):
                    self.time_points = time_points
                    self.production_rates = production_rates
            
            economic_wells[well_name] = WellData(
                well.time_points,
                well.production_rates
            )
        
        return EconomicInputData(economic_wells)
    
    def _run_economic_analysis(self, economic_input_data, forecast_years: int,
                              oil_price: float, operating_cost: float) -> Dict[str, Any]:
        """
        Run comprehensive economic analysis
        """
        logger.info("Running economic analysis")
        
        # Create simulation parameters for economic analysis
        sim_params = SimulationParameters(
            forecast_years=forecast_years,
            oil_price=oil_price,
            operating_cost=operating_cost,
            discount_rate=self.economic_params.get('discount_rate', 0.10),
            capex_per_well=self.economic_params.get('capex_per_well', 1_000_000),
            fixed_annual_opex=self.economic_params.get('fixed_annual_opex', 500_000),
            tax_rate=self.economic_params.get('tax_rate', 0.30),
            royalty_rate=self.economic_params.get('royalty_rate', 0.125)
        )
        
        # Create and run economic simulator
        economic_simulator = EconomicSimulator(economic_input_data, sim_params)
        economic_results = economic_simulator.run_comprehensive_simulation()
        
        # Enhance results with production forecast data
        total_production_forecast = self._calculate_total_production_forecast()
        economic_results['total_forecast_production'] = total_production_forecast
        
        # Calculate annual production summary
        annual_production = self._calculate_annual_production_summary(forecast_years)
        economic_results['annual_production'] = annual_production
        
        logger.info(f"Economic analysis completed: NPV=${economic_results['economic_analysis']['npv']:.2f}M")
        
        return economic_results
    
    def _calculate_total_production_forecast(self) -> float:
        """Calculate total forecast production from all wells"""
        total_production = 0.0
        
        for well_name, forecast in self.production_forecast.items():
            if 'eur' in forecast:
                total_production += forecast['eur']
        
        return total_production
    
    def _calculate_annual_production_summary(self, forecast_years: int) -> List[float]:
        """Calculate annual production from forecast"""
        annual_production = np.zeros(forecast_years)
        
        for well_name, forecast in self.production_forecast.items():
            days = forecast['forecast_time']
            rates = forecast['forecast_rates']
            
            if len(days) < 2:
                continue
            
            # Skip injection wells
            well_type = forecast.get('well_type', 'PRODUCER')
            if 'INJE' in well_type.upper():
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
        
        return annual_production.tolist()
    
    def _compile_simulation_results(self, forecast_years: int, 
                                   oil_price: float, operating_cost: float) -> Dict[str, Any]:
        """
        Compile all simulation results into a comprehensive dictionary
        """
        # Calculate reservoir statistics
        reservoir_stats = self._calculate_reservoir_statistics()
        
        # Calculate well performance metrics
        well_performance = self._calculate_well_performance()
        
        # Extract economic results
        economic_results = self.economic_analysis_results.get('economic_analysis', {})
        
        # Compile comprehensive results
        comprehensive_results = {
            'simulation_summary': {
                'forecast_years': forecast_years,
                'oil_price_usd_per_bbl': oil_price,
                'operating_cost_usd_per_bbl': operating_cost,
                'total_wells': len(self.reservoir_data.wells),
                'successfully_simulated_wells': len(self.decline_analysis_results),
                'total_forecast_production_bbl': self._calculate_total_production_forecast(),
                'simulation_timestamp': pd.Timestamp.now().isoformat()
            },
            'reservoir_characteristics': {
                'grid_dimensions': self.reservoir_data.grid_dimensions,
                'average_porosity': self.reservoir_data.porosity,
                'average_permeability_md': self.reservoir_data.permeability,
                'initial_pressure_psi': self.reservoir_data.initial_pressure,
                'water_saturation': self.reservoir_data.water_saturation
            },
            'decline_analysis': self.decline_analysis_results,
            'production_forecast': self.production_forecast,
            'economic_analysis': economic_results,
            'well_performance': well_performance,
            'reservoir_statistics': reservoir_stats
        }
        
        return comprehensive_results
    
    def _calculate_reservoir_statistics(self) -> Dict[str, Any]:
        """Calculate reservoir-wide statistics"""
        from .economics import MaterialBalance
        
        mb = MaterialBalance()
        
        total_cumulative_production = 0.0
        peak_production_rate = 0.0
        
        for well_name, well in self.reservoir_data.wells.items():
            if hasattr(well, 'production_rates'):
                rates = well.production_rates
                if len(rates) > 0:
                    peak_rate = np.max(rates)
                    if peak_rate > peak_production_rate:
                        peak_production_rate = peak_rate
            
            # Calculate cumulative production if time points available
            if (hasattr(well, 'time_points') and hasattr(well, 'production_rates')):
                time_points = well.time_points
                production_rates = well.production_rates
                
                if len(time_points) >= 2 and len(production_rates) >= 2:
                    min_len = min(len(time_points), len(production_rates))
                    cum_prod = mb.calculate_cumulative_production(
                        production_rates[:min_len],
                        time_points[:min_len]
                    )
                    total_cumulative_production += cum_prod
        
        # Estimate OOIP (Original Oil in Place)
        area = 40.0  # acres
        thickness = 50.0  # feet
        porosity = self.reservoir_data.porosity
        sw = self.reservoir_data.water_saturation
        boi = self.reservoir_data.formation_volume_factor
        
        estimated_ooip = mb.calculate_oip(area, thickness, porosity, sw, boi)
        
        # Calculate recovery factor
        recovery_factor = (total_cumulative_production / estimated_ooip) if estimated_ooip > 0 else 0
        
        return {
            'total_cumulative_production_bbl': total_cumulative_production,
            'peak_production_rate_bpd': peak_production_rate,
            'estimated_ooip_bbl': estimated_ooip,
            'current_recovery_factor_percent': recovery_factor * 100,
            'number_of_producers': sum(1 for w in self.reservoir_data.wells.values() 
                                     if getattr(w, 'well_type', 'PRODUCER') == 'PRODUCER'),
            'number_of_injectors': sum(1 for w in self.reservoir_data.wells.values() 
                                     if getattr(w, 'well_type', 'PRODUCER') == 'INJECTOR')
        }
    
    def _calculate_well_performance(self) -> Dict[str, Dict]:
        """Calculate performance metrics for each well"""
        well_performance = {}
        
        for well_name, well in self.reservoir_data.wells.items():
            if not hasattr(well, 'production_rates'):
                continue
            
            rates = well.production_rates
            
            if len(rates) < 2:
                continue
            
            # Calculate basic metrics
            initial_rate = rates[0]
            final_rate = rates[-1]
            peak_rate = np.max(rates)
            avg_rate = np.mean(rates)
            
            # Calculate decline rate if enough data
            decline_rate = 0.0
            if initial_rate > 0 and final_rate > 0:
                if hasattr(well, 'time_points') and len(well.time_points) >= 2:
                    time_span = well.time_points[-1] - well.time_points[0]
                    if time_span > 0:
                        decline_rate = ((initial_rate - final_rate) / initial_rate) * 365 / time_span
            
            # Get decline parameters if available
            decline_params = self.decline_analysis_results.get(well_name, {})
            
            well_performance[well_name] = {
                'initial_rate_bpd': initial_rate,
                'final_rate_bpd': final_rate,
                'peak_rate_bpd': peak_rate,
                'average_rate_bpd': avg_rate,
                'annual_decline_rate_percent': decline_rate * 100,
                'decline_type': decline_params.get('method', 'unknown'),
                'decline_constant': decline_params.get('di', 0),
                'fit_quality_r2': decline_params.get('r_squared', 0),
                'well_type': getattr(well, 'well_type', 'PRODUCER')
            }
        
        return well_performance
    
    def _create_error_results(self, error_message: str) -> Dict[str, Any]:
        """Create error results when simulation fails"""
        return {
            'error': error_message,
            'simulation_summary': {
                'status': 'FAILED',
                'error_message': error_message,
                'total_wells': len(self.reservoir_data.wells),
                'successfully_simulated_wells': 0
            },
            'economic_analysis': {
                'npv': 0,
                'irr': 0,
                'roi': 0,
                'payback_period_years': float('inf'),
                'initial_investment': 0
            }
        }
    
    def get_summary_report(self) -> str:
        """Generate a text summary report of simulation results"""
        if not self.is_simulated:
            return "Simulation has not been run yet."
        
        summary = self.simulation_results.get('simulation_summary', {})
        economics = self.simulation_results.get('economic_analysis', {})
        reservoir = self.simulation_results.get('reservoir_statistics', {})
        
        report_lines = [
            "=" * 60,
            "RESERVOIR SIMULATION SUMMARY REPORT",
            "=" * 60,
            f"\nSimulation Overview:",
            f"  • Forecast Period: {summary.get('forecast_years', 0)} years",
            f"  • Oil Price: ${summary.get('oil_price_usd_per_bbl', 0):.2f}/bbl",
            f"  • Operating Cost: ${summary.get('operating_cost_usd_per_bbl', 0):.2f}/bbl",
            f"  • Total Wells: {summary.get('total_wells', 0)}",
            f"  • Successfully Simulated: {summary.get('successfully_simulated_wells', 0)}",
            
            f"\nReservoir Statistics:",
            f"  • Cumulative Production: {reservoir.get('total_cumulative_production_bbl', 0):,.0f} bbl",
            f"  • Peak Production Rate: {reservoir.get('peak_production_rate_bpd', 0):.1f} bpd",
            f"  • Estimated OOIP: {reservoir.get('estimated_ooip_bbl', 0):,.0f} bbl",
            f"  • Recovery Factor: {reservoir.get('current_recovery_factor_percent', 0):.1f}%",
            
            f"\nEconomic Results:",
            f"  • Net Present Value (NPV): ${economics.get('npv', 0):.2f}M",
            f"  • Internal Rate of Return (IRR): {economics.get('irr', 0):.1f}%",
            f"  • Return on Investment (ROI): {economics.get('roi', 0):.1f}%",
            f"  • Payback Period: {economics.get('payback_period_years', 'N/A')} years",
            f"  • Initial Investment: ${economics.get('initial_investment', 0)/1_000_000:.2f}M",
            
            f"\nProduction Forecast:",
            f"  • Total Forecast Production: {summary.get('total_forecast_production_bbl', 0):,.0f} bbl",
        ]
        
        # Add annual production if available
        annual_prod = self.simulation_results.get('economic_analysis', {}).get('annual_production', [])
        if annual_prod:
            report_lines.append(f"  • Peak Annual Production: {max(annual_prod):,.0f} bbl")
            report_lines.append(f"  • Average Annual Production: {np.mean(annual_prod):,.0f} bbl")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)
    
    def export_results(self, filename: str = "simulation_results.json"):
        """Export simulation results to JSON file"""
        import json
        
        if not self.is_simulated:
            logger.error("Cannot export results - simulation has not been run")
            return False
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.generic):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            exportable_results = convert_for_json(self.simulation_results)
            
            with open(filename, 'w') as f:
                json.dump(exportable_results, f, indent=2)
            
            logger.info(f"Results exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}")
            return False
    
    def run_sensitivity_analysis(self, base_params: Dict, 
                                variables: List[str] = ['oil_price', 'operating_cost', 'discount_rate'],
                                variations: List[float] = [-0.2, -0.1, 0, 0.1, 0.2]) -> Dict[str, Any]:
        """
        Run sensitivity analysis on key economic variables
        
        Parameters:
        -----------
        base_params : Dict
            Base economic parameters
        variables : List[str]
            Variables to analyze
        variations : List[float]
            Percentage variations to test
            
        Returns:
        --------
        Dict with sensitivity analysis results
        """
        logger.info(f"Running sensitivity analysis on {variables}")
        
        sensitivity_results = {}
        
        for variable in variables:
            variable_results = []
            
            for variation in variations:
                # Create modified parameters
                modified_params = base_params.copy()
                
                if variable in modified_params:
                    modified_params[variable] = modified_params[variable] * (1 + variation)
                
                # Run simulation with modified parameters
                modified_results = self.run_simulation(
                    forecast_years=modified_params.get('forecast_years', 10),
                    oil_price=modified_params.get('oil_price', 75.0),
                    operating_cost=modified_params.get('operating_cost', 18.0)
                )
                
                economic_results = modified_results.get('economic_analysis', {})
                
                variable_results.append({
                    'variation_percent': variation * 100,
                    'parameter_value': modified_params.get(variable),
                    'npv': economic_results.get('npv', 0),
                    'irr': economic_results.get('irr', 0),
                    'roi': economic_results.get('roi', 0)
                })
            
            sensitivity_results[variable] = variable_results
        
        # Calculate tornado chart data
        tornado_data = self._prepare_tornado_data(sensitivity_results)
        
        return {
            'sensitivity_analysis': sensitivity_results,
            'tornado_chart_data': tornado_data,
            'base_case': self.simulation_results.get('economic_analysis', {})
        }
    
    def _prepare_tornado_data(self, sensitivity_results: Dict) -> List[Dict]:
        """Prepare data for tornado chart visualization"""
        tornado_data = []
        
        for variable, variations in sensitivity_results.items():
            if not variations:
                continue
            
            npv_values = [result['npv'] for result in variations]
            base_npv = variations[len(variations)//2]['npv']  # Middle value is base case
            
            min_npv = min(npv_values)
            max_npv = max(npv_values)
            
            tornado_data.append({
                'variable': variable,
                'base_npv': base_npv,
                'min_npv': min_npv,
                'max_npv': max_npv,
                'impact_range': max_npv - min_npv
            })
        
        # Sort by impact range (descending)
        tornado_data.sort(key=lambda x: x['impact_range'], reverse=True)
        
        return tornado_data
