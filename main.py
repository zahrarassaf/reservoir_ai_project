#!/usr/bin/env python3
"""
PhD-Level Reservoir Simulation and Economic Analysis Framework
Author: Reservoir AI Research Team
License: MIT
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from src.core.physics_engine import BlackOilSimulator
from src.data.google_drive_client import SecureDriveClient
from src.economics.cash_flow import EconomicAnalyzer
from src.visualization.dashboard import ReservoirDashboard
from src.utils.logger import setup_logging
from src.utils.validator import ConfigValidator

class PhDReservoirSimulator:
    """Main orchestrator for PhD-level reservoir simulation."""
    
    def __init__(self, config_path: Path):
        self.config = self._load_config(config_path)
        self.logger = setup_logging(self.config['logging'])
        self.validator = ConfigValidator()
        
    def _load_config(self, config_path: Path) -> dict:
        """Load and validate configuration."""
        import yaml
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        self.validator.validate_config(config)
        
        return config
    
    def run_simulation(self, mode: str = "full") -> dict:
        """Execute complete simulation workflow."""
        self.logger.info(f"Starting PhD reservoir simulation in {mode} mode")
        
        results = {}
        
        try:
            # Phase 1: Data Acquisition
            if mode in ["full", "data"]:
                data_results = self._acquire_and_validate_data()
                results['data'] = data_results
            
            # Phase 2: Physics Simulation
            if mode in ["full", "physics"]:
                physics_results = self._run_physics_simulation(data_results)
                results['physics'] = physics_results
            
            # Phase 3: Economic Analysis
            if mode in ["full", "economics"]:
                economic_results = self._run_economic_analysis(physics_results)
                results['economics'] = economic_results
            
            # Phase 4: Visualization
            if mode in ["full", "visualization"]:
                visualization_results = self._generate_visualizations(results)
                results['visualization'] = visualization_results
            
            self.logger.info("Simulation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {str(e)}", exc_info=True)
            raise
        
        return results
    
    def _acquire_and_validate_data(self) -> dict:
        """Acquire data from Google Drive and validate."""
        self.logger.info("Acquiring data from Google Drive")
        
        # Initialize Google Drive client
        drive_client = SecureDriveClient(
            credentials_path=Path(self.config['google_drive']['credentials']),
            cache_dir=Path(self.config['paths']['cache'])
        )
        
        # Load SPE9 datasets
        spe_loader = SPEDatasetLoader(drive_client)
        datasets = spe_loader.load_spe9_benchmark()
        
        # Validate datasets
        validation_results = {}
        for name, dataset in datasets.items():
            validator = DataValidator(dataset)
            is_valid = validator.validate()
            
            validation_results[name] = {
                'dataset': dataset,
                'is_valid': is_valid,
                'validation_report': validator.get_report()
            }
        
        return validation_results
    
    def _run_physics_simulation(self, data_results: dict) -> dict:
        """Run physics-based reservoir simulation."""
        self.logger.info("Running physics simulation")
        
        simulation_results = {}
        
        for dataset_name, data_info in data_results.items():
            if not data_info['is_valid']:
                self.logger.warning(f"Skipping invalid dataset: {dataset_name}")
                continue
            
            dataset = data_info['dataset']
            
            # Initialize simulator
            simulator = BlackOilSimulator(
                grid=dataset['grid'],
                properties=dataset['properties'],
                pvt_data=dataset['pvt'],
                config=self.config['simulation']
            )
            
            # Run simulation
            simulation_result = simulator.run(
                time_steps=self.config['simulation']['time_steps'],
                output_frequency=self.config['simulation']['output_freq']
            )
            
            simulation_results[dataset_name] = {
                'simulator': simulator,
                'results': simulation_result,
                'performance': simulator.get_performance_metrics()
            }
        
        return simulation_results
    
    def _run_economic_analysis(self, physics_results: dict) -> dict:
        """Run comprehensive economic analysis."""
        self.logger.info("Running economic analysis")
        
        economic_results = {}
        
        # Initialize economic analyzer
        economic_config = self.config['economics']
        analyzer = EconomicAnalyzer(
            discount_rate=economic_config['discount_rate'],
            oil_price=economic_config['oil_price'],
            operating_cost=economic_config['operating_cost']
        )
        
        for dataset_name, physics_info in physics_results.items():
            production_profile = physics_info['results']['production']
            reservoir_properties = physics_info['simulator'].reservoir_properties
            
            # Calculate economic metrics
            economic_metrics = analyzer.analyze(
                production_profile=production_profile,
                capex=self.config['economics']['capex'],
                opex=self.config['economics']['opex'],
                tax_rate=self.config['economics']['tax_rate']
            )
            
            # Add risk analysis
            risk_analysis = analyzer.risk_assessment(
                economic_metrics,
                price_uncertainty=self.config['economics']['price_uncertainty'],
                cost_uncertainty=self.config['economics']['cost_uncertainty']
            )
            
            economic_results[dataset_name] = {
                'metrics': economic_metrics,
                'risk': risk_analysis,
                'sensitivity': analyzer.sensitivity_analysis(
                    economic_metrics,
                    parameters=self.config['economics']['sensitivity_params']
                )
            }
        
        return economic_results
    
    def _generate_visualizations(self, all_results: dict) -> dict:
        """Generate comprehensive visualizations."""
        self.logger.info("Generating visualizations")
        
        output_dir = Path(self.config['paths']['output'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dashboard = ReservoirDashboard(output_dir)
        
        visualization_results = {}
        
        # Generate dashboard for each dataset
        for dataset_name, results in all_results.get('physics', {}).items():
            viz_result = dashboard.create_dataset_dashboard(
                dataset_name=dataset_name,
                physics_results=results['results'],
                economic_results=all_results['economics'].get(dataset_name, {}),
                config=self.config['visualization']
            )
            
            visualization_results[dataset_name] = viz_result
        
        # Generate comparative dashboard
        comparative_dashboard = dashboard.create_comparative_dashboard(
            all_results=all_results
        )
        
        visualization_results['comparative'] = comparative_dashboard
        
        return visualization_results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PhD-Level Reservoir Simulation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/settings.yaml'),
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'data', 'physics', 'economics', 'visualization'],
        default='full',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(f'results/simulation_{datetime.now():%Y%m%d_%H%M%S}'),
        help='Output directory'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config with command-line arguments
        config_path = args.config
        
        simulator = PhDReservoirSimulator(config_path)
        results = simulator.run_simulation(mode=args.mode)
        
        # Save results
        import json
        results_file = args.output_dir / 'simulation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Simulation completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
        # Print summary
        if 'economics' in results:
            print("\n📊 Economic Summary:")
            print("-" * 50)
            for dataset, econ_data in results['economics'].items():
                npv = econ_data['metrics'].get('npv', 0)
                irr = econ_data['metrics'].get('irr', 0)
                print(f"{dataset:30} NPV: ${npv/1e6:8.2f}M  IRR: {irr:6.2f}%")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
