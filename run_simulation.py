"""
Reservoir AI Simulation Framework - Main Entry Point
Professional-grade reservoir simulation with machine learning integration.
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure professional logging system."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'simulation_{datetime.now():%Y%m%d_%H%M%S}.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {log_level} level")
    
    return logger

# Import project modules after path configuration
try:
    from data_parser.spe9_parser import SPE9DataParser, create_spe9_parser
    HAS_SPE9_PARSER = True
except ImportError as e:
    logging.warning(f"SPE9 parser not available: {e}")
    HAS_SPE9_PARSER = False

from models.reservoir_model import ReservoirModel
from analysis.performance_calculator import PerformanceCalculator
from analysis.plot_generator import PlotGenerator

def validate_data_files(logger: logging.Logger) -> bool:
    """
    Validate existence of required data files.
    Returns True if validation passes or can be recovered.
    """
    required_files = [
        'data/well_locations.txt',
        'data/grid_tops.txt',
        'data/permeability.txt',
        'data/sgof_table.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"Missing {len(missing_files)} required data files")
        
        # Check if SPE9.DATA exists and we can parse it
        spe9_file = Path("data/SPE9.DATA")
        if spe9_file.exists() and HAS_SPE9_PARSER:
            logger.info("SPE9.DATA found. Attempting to extract required data...")
            try:
                parser = create_spe9_parser(str(spe9_file))
                exported_files = parser.export_to_project_format('data')
                
                if len(exported_files) >= 3:  # At least wells, permeability, tops
                    logger.info("Successfully extracted data from SPE9.DATA")
                    return True
                else:
                    logger.error("Insufficient data extracted from SPE9.DATA")
                    return False
            except Exception as e:
                logger.error(f"Failed to parse SPE9.DATA: {e}")
                return False
        else:
            logger.error("SPE9.DATA not found or parser unavailable")
            logger.error(f"Missing files: {missing_files}")
            return False
    
    logger.info("All required data files validated successfully")
    return True

def load_simulation_data(logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Load and prepare simulation data with robust error handling."""
    
    def safe_load_txt(filepath: str, delimiter: str = ',', skip_header: int = 0) -> np.ndarray:
        """Safely load text file with various formats."""
        try:
            if Path(filepath).exists():
                data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skip_header)
                logger.debug(f"Loaded {data.shape} from {filepath}")
                return data
            else:
                raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            # Return empty array to prevent crash
            return np.array([])
    
    try:
        # Load well data
        well_data = safe_load_txt('data/well_locations.txt')
        if well_data.size == 0:
            raise ValueError("No well data loaded")
        
        # Load grid data
        tops_data = safe_load_txt('data/grid_tops.txt')
        perm_data = safe_load_txt('data/permeability.txt')
        sgof_data = safe_load_txt('data/sgof_table.txt')
        
        # Validate data shapes
        if tops_data.size == 0:
            logger.warning("TOPS data is empty, using default grid")
            tops_data = np.random.uniform(8000, 8500, 100)
        
        if perm_data.size == 0:
            logger.warning("Permeability data is empty, using default values")
            perm_data = np.random.lognormal(mean=3.0, sigma=1.0, size=100)
        
        # Prepare data dictionary
        simulation_data = {
            'wells': well_data,
            'grid_tops': tops_data,
            'permeability': perm_data,
            'sgof_table': sgof_data,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'well_count': len(well_data),
                'grid_size': len(tops_data)
            }
        }
        
        logger.info(f"Loaded simulation data: {simulation_data['metadata']}")
        return simulation_data
        
    except Exception as e:
        logger.error(f"Critical error loading simulation data: {e}")
        return None

def run_simulation_pipeline(data: Dict[str, Any], logger: logging.Logger) -> bool:
    """Execute complete simulation pipeline."""
    
    try:
        logger.info("Initializing reservoir model...")
        
        # Initialize model
        reservoir_model = ReservoirModel(
            grid_tops=data['grid_tops'],
            permeability=data['permeability'],
            well_data=data['wells']
        )
        
        # Run simulation steps
        logger.info("Running reservoir simulation...")
        simulation_results = reservoir_model.run_simulation(
            time_steps=100,
            output_frequency=10
        )
        
        # Calculate performance metrics
        logger.info("Calculating performance metrics...")
        performance_calculator = PerformanceCalculator(simulation_results)
        metrics = performance_calculator.calculate_all_metrics()
        
        # Generate visualizations
        logger.info("Generating analysis plots...")
        plot_generator = PlotGenerator(simulation_results, metrics)
        
        # Create comprehensive plot suite
        plots = {
            'performance': plot_generator.plot_performance_metrics(),
            'reservoir': plot_generator.plot_reservoir_properties(),
            'wells': plot_generator.plot_well_performance(),
            'saturation': plot_generator.plot_saturation_distribution()
        }
        
        # Save plots
        for plot_name, fig in plots.items():
            if fig is not None:
                plot_path = f"results/{plot_name}_plot_{datetime.now():%Y%m%d_%H%M%S}.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved plot: {plot_path}")
        
        # Generate report
        report_path = f"results/simulation_report_{datetime.now():%Y%m%d_%H%M%S}.txt"
        with open(report_path, 'w') as f:
            f.write(f"Reservoir Simulation Report\n")
            f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"\nPerformance Metrics:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
        
        logger.info(f"Simulation completed successfully. Report: {report_path}")
        return True
        
    except Exception as e:
        logger.error(f"Simulation pipeline failed: {e}")
        return False

def main():
    """Main execution function with comprehensive error handling."""
    
    # Setup
    logger = setup_logging("INFO")
    logger.info("=" * 60)
    logger.info("Reservoir AI Simulation Framework - Starting")
    logger.info("=" * 60)
    
    try:
        # Validate data
        if not validate_data_files(logger):
            logger.error("Data validation failed. Exiting.")
            return 1
        
        # Load data
        simulation_data = load_simulation_data(logger)
        if simulation_data is None:
            logger.error("Failed to load simulation data. Exiting.")
            return 1
        
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Run simulation
        success = run_simulation_pipeline(simulation_data, logger)
        
        if success:
            logger.info("=" * 60)
            logger.info("SIMULATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            return 0
        else:
            logger.error("SIMULATION FAILED")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
