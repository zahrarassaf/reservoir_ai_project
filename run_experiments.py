#!/usr/bin/env python3
"""
Main script for running comprehensive reservoir computing experiments.
"""

import sys
import argparse
from pathlib import Path
import logging
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from experiments.runner import ExperimentRunner, ExperimentConfig


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_file: Path) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive reservoir computing experiments"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/spe9.yaml"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to data file (optional)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip hyperparameter optimization"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick experiment with reduced iterations"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config_dict = load_config(args.config)
        
        # Apply quick mode if requested
        if args.quick:
            if 'optimization_config' in config_dict:
                config_dict['optimization_config']['n_calls'] = 10
                config_dict['optimization_config']['n_initial_points'] = 5
            
            config_dict['data_config']['sequence_length'] = 10
            config_dict['data_config']['forecast_horizon'] = 5
        
        # Apply skip optimization if requested
        if args.skip_optimization:
            config_dict['optimize_hyperparameters'] = False
        
        # Create experiment configuration
        experiment_config = ExperimentConfig(config_dict)
        
        # Create and run experiment runner
        runner = ExperimentRunner(experiment_config)
        
        # Run all experiments
        results = runner.run_all_experiments(args.data)
        
        # Generate and save report
        report = runner.generate_report()
        print("\n" + report)
        
        if runner.results_dir:
            runner.save_report()
            logger.info(f"Complete results saved to: {runner.results_dir}")
        
        # Check overall performance
        if 'summary' in results:
            baseline_nse = results['summary'].get('baseline_performance', {}).get('nash_sutcliffe', 0)
            
            if baseline_nse < 0.5:
                logger.warning(f"Baseline performance is low (NSE: {baseline_nse:.3f}). Consider reviewing model configuration.")
            elif baseline_nse > 0.8:
                logger.info(f"Excellent baseline performance (NSE: {baseline_nse:.3f})")
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
