"""
Reservoir AI Simulation Framework - Professional Implementation
Main entry point for SPE9 reservoir simulation with AI components.
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import json
import yaml
from typing import Dict, Any, Optional

# Configure absolute imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
try:
    from data_parser.spe9_parser import SPE9ProjectParser
    from src.simulation_runner import ReservoirSimulationRunner
    from src.data_validator import DataValidator
    from src.results_processor import ResultsProcessor
    from analysis.performance_calculator import PerformanceCalculator
    from analysis.plot_generator import PlotGenerator
    HAS_ALL_MODULES = True
except ImportError as e:
    print(f"Import error: {e}")
    HAS_ALL_MODULES = False

def setup_logging() -> logging.Logger:
    """Setup professional logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"simulation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def load_configurations() -> Dict[str, Any]:
    """Load all configuration files."""
    config_dir = Path("config")
    configs = {}
    
    # Load simulation configuration
    sim_config_path = config_dir / "simulation_config.yaml"
    if sim_config_path.exists():
        with open(sim_config_path, 'r') as f:
            configs['simulation'] = yaml.safe_load(f)
    
    # Load grid parameters
    grid_config_path = config_dir / "grid_parameters.json"
    if grid_config_path.exists():
        with open(grid_config_path, 'r') as f:
            configs['grid'] = json.load(f)
    
    # Load well controls
    well_config_path = config_dir / "well_controls.json"
    if well_config_path.exists():
        with open(well_config_path, 'r') as f:
            configs['wells'] = json.load(f)
    
    return configs

def parse_real_data(logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Parse real SPE9 data files."""
    try:
        logger.info("Starting real SPE9 data parsing...")
        
        parser = SPE9ProjectParser(data_dir="data")
        parsed_data = parser.parse_all()
        
        # Validate data
        validator = DataValidator(parsed_data.get_simulation_data())
        if not validator.validate():
            logger.error("Data validation failed")
            return None
        
        # Export for simulation
        exported_files = parser.export_for_simulation("data/processed")
        
        logger.info("✅ Real SPE9 data parsed successfully")
        logger.info(f"Grid: {parsed_data.grid_dimensions}")
        logger.info(f"Wells: {len(parsed_data.wells)}")
        
        return parsed_data.get_simulation_data()
        
    except Exception as e:
        logger.error(f"Failed to parse real data: {e}", exc_info=True)
        return None

def run_comprehensive_simulation(data: Dict[str, Any], 
                                 configs: Dict[str, Any],
                                 logger: logging.Logger) -> bool:
    """Run comprehensive simulation pipeline."""
    
    try:
        # Initialize simulation runner
        logger.info("Initializing simulation runner...")
        runner = ReservoirSimulationRunner(
            reservoir_data=data,
            simulation_config=configs.get('simulation', {}),
            grid_config=configs.get('grid', {})
        )
        
        # Run simulation
        logger.info("Running reservoir simulation...")
        simulation_results = runner.run()
        
        if simulation_results is None:
            logger.error("Simulation failed to produce results")
            return False
        
        # Process results
        logger.info("Processing simulation results...")
        processor = ResultsProcessor(simulation_results)
        processed_results = processor.process_all()
        
        # Calculate performance metrics
        logger.info("Calculating performance metrics...")
        calculator = PerformanceCalculator(processed_results)
        metrics = calculator.calculate_comprehensive_metrics()
        
        # Save metrics
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        metrics_file = results_dir / f"metrics_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=float)
        
        logger.info(f"Performance metrics saved to: {metrics_file}")
        
        # Generate visualizations
        logger.info("Generating analysis plots...")
        plot_generator = PlotGenerator(processed_results, metrics)
        
        # Generate and save all plots
        plot_files = []
        plots_to_generate = [
            ('reservoir_properties', plot_generator.plot_reservoir_properties),
            ('well_performance', plot_generator.plot_well_performance),
            ('production_forecast', plot_generator.plot_production_forecast),
            ('saturation_distribution', plot_generator.plot_saturation_distribution),
            ('pressure_contour', plot_generator.plot_pressure_contour),
            ('recovery_factors', plot_generator.plot_recovery_factors)
        ]
        
        for plot_name, plot_func in plots_to_generate:
            try:
                fig = plot_func()
                if fig is not None:
                    plot_path = results_dir / f"{plot_name}_{datetime.now():%Y%m%d_%H%M%S}.png"
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plot_files.append(plot_path)
                    logger.info(f"Generated plot: {plot_path}")
            except Exception as e:
                logger.warning(f"Failed to generate {plot_name} plot: {e}")
        
        # Generate comprehensive report
        report_path = generate_final_report(processed_results, metrics, plot_files, logger)
        
        logger.info("=" * 60)
        logger.info("SIMULATION COMPLETED SUCCESSFULLY")
        logger.info(f"Results directory: {results_dir}")
        logger.info(f"Final report: {report_path}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Simulation pipeline failed: {e}", exc_info=True)
        return False

def generate_final_report(results: Dict[str, Any], 
                          metrics: Dict[str, Any],
                          plot_files: list,
                          logger: logging.Logger) -> Path:
    """Generate comprehensive final report."""
    
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    
    report_path = report_dir / f"simulation_report_{datetime.now():%Y%m%d_%H%M%S}.md"
    
    with open(report_path, 'w') as f:
        f.write("# Reservoir Simulation Report\n\n")
        f.write(f"**Generated:** {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- Total simulation time: {metrics.get('total_time', 'N/A')} days\n")
        f.write(f"- Final oil recovery: {metrics.get('recovery_factor', {}).get('oil', 0)*100:.1f}%\n")
        f.write(f"- Number of wells: {len(results.get('well_data', []))}\n\n")
        
        f.write("## Key Performance Indicators\n\n")
        f.write("| Metric | Value | Unit |\n")
        f.write("|--------|-------|------|\n")
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        f.write(f"| {key}.{subkey} | {subvalue:.4f} | - |\n")
            elif isinstance(value, (int, float)):
                f.write(f"| {key} | {value:.4f} | - |\n")
        
        f.write("\n## Generated Visualizations\n\n")
        for plot_file in plot_files:
            f.write(f"- `{plot_file.name}`\n")
        
        f.write("\n## Data Summary\n\n")
        f.write(f"- Grid dimensions: {results.get('grid_dimensions', 'N/A')}\n")
        f.write(f"- Timesteps simulated: {len(results.get('time_steps', []))}\n")
        f.write(f"- Final simulation date: {results.get('final_date', 'N/A')}\n")
    
    logger.info(f"Generated final report: {report_path}")
    return report_path

def main():
    """Main execution function."""
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("RESERVOIR AI SIMULATION FRAMEWORK - SPE9 REAL DATA")
    logger.info("=" * 70)
    
    # Check module availability
    if not HAS_ALL_MODULES:
        logger.error("Missing required modules. Please check imports.")
        return 1
    
    try:
        # Step 1: Load configurations
        logger.info("Step 1: Loading configurations...")
        configs = load_configurations()
        
        # Step 2: Parse real SPE9 data
        logger.info("Step 2: Parsing real SPE9 data...")
        simulation_data = parse_real_data(logger)
        
        if simulation_data is None:
            logger.error("Failed to parse simulation data. Exiting.")
            return 1
        
        # Step 3: Run simulation
        logger.info("Step 3: Running comprehensive simulation...")
        success = run_comprehensive_simulation(simulation_data, configs, logger)
        
        if success:
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
