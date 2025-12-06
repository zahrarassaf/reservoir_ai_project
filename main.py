"""
Main entry point for Reservoir Simulation Project
"""

import sys
import os
import numpy as np
from typing import Dict, List, Optional
import tempfile
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import ReservoirData
from src.simulator import ReservoirSimulator, SimulationParameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handle data loading from various sources."""
    
    @staticmethod
    def download_from_drive(file_id: str, output_dir: str) -> Optional[str]:
        """Download file from Google Drive."""
        try:
            import gdown
            url = f'https://drive.google.com/uc?id={file_id}&export=download'
            output_path = os.path.join(output_dir, f'{file_id}.txt')
            
            logger.info(f"Downloading {file_id}...")
            gdown.download(url, output_path, quiet=False)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Downloaded: {os.path.getsize(output_path)} bytes")
                return output_path
            else:
                logger.error(f"Download failed for {file_id}")
                return None
                
        except ImportError:
            logger.error("gdown library not installed. Install with: pip install gdown")
            return None
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None
    
    @staticmethod
    def load_spe9_file(file_path: str) -> Optional[ReservoirData]:
        """Load and parse SPE9 data file."""
        try:
            data = ReservoirData()
            if data.load_spe9_file(file_path):
                return data
            else:
                logger.error(f"Failed to parse SPE9 data from {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def load_multiple_files(file_ids: List[str]) -> Dict[str, ReservoirData]:
        """Load multiple data files."""
        datasets = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for file_id in file_ids:
                logger.info(f"Processing file ID: {file_id}")
                
                file_path = DataLoader.download_from_drive(file_id, temp_dir)
                if not file_path:
                    continue
                
                data = DataLoader.load_spe9_file(file_path)
                if data:
                    datasets[file_id] = data
                    logger.info(f"Successfully loaded {file_id}")
                else:
                    logger.warning(f"Failed to load {file_id}")
        
        return datasets

def get_simulation_parameters() -> SimulationParameters:
    """Get simulation parameters from user input."""
    print("\n" + "="*50)
    print("SIMULATION PARAMETERS")
    print("="*50)
    
    try:
        forecast_input = input("Forecast years (default: 3): ").strip()
        forecast_years = int(forecast_input) if forecast_input else 3
        
        oil_input = input("Oil price USD/bbl (default: 75.0): ").strip()
        oil_price = float(oil_input) if oil_input else 75.0
        
        cost_input = input("Operating cost USD/bbl (default: 18.0): ").strip()
        operating_cost = float(cost_input) if cost_input else 18.0
        
        logger.info(f"Parameters set: Forecast={forecast_years} years, "
                   f"Oil price=${oil_price}/bbl, Opex=${operating_cost}/bbl")
        
        return SimulationParameters(
            forecast_years=forecast_years,
            oil_price=oil_price,
            operating_cost=operating_cost
        )
        
    except ValueError as e:
        logger.error(f"Invalid input: {e}. Using default values.")
        return SimulationParameters()

def run_simulation_for_dataset(data: ReservoirData, 
                              params: SimulationParameters,
                              dataset_name: str) -> Dict:
    """Run simulation for a single dataset."""
    print(f"\n{'='*60}")
    print(f"SIMULATING: {dataset_name}")
    print(f"{'='*60}")
    
    # Display data summary
    summary = data.summary()
    print(f"\nDATA SUMMARY:")
    print(f"  • Wells: {summary['wells']}")
    print(f"  • Grid: {summary['grid_dimensions']}")
    print(f"  • Production data: {'Available' if summary['has_production_data'] else 'Not available'}")
    
    if summary['has_production_data']:
        prod_range = summary.get('production_range', {})
        print(f"  • Max rate: {prod_range.get('max', 0):.1f} STB/day")
        print(f"  • Total production: {summary.get('total_production', 0):,.0f} bbl")
    
    # Run simulation
    simulator = ReservoirSimulator(data, params)
    results = simulator.run_comprehensive_simulation()
    
    # Display key results
    if 'economic_analysis' in results:
        econ = results['economic_analysis']
        print(f"\nECONOMIC RESULTS:")
        print(f"  • NPV: ${econ.get('npv', 0)/1e6:.2f}M")
        print(f"  • IRR: {econ.get('irr', 0)*100:.1f}%")
        print(f"  • ROI: {econ.get('roi', 0)*100:.1f}%")
        if 'payback_period_years' in econ:
            print(f"  • Payback: {econ.get('payback_period_years', 0):.1f} years")
    
    return results

def main():
    """Main entry point."""
    print("="*60)
    print("RESERVOIR SIMULATION PROJECT - PhD LEVEL")
    print("="*60)
    
    print("\nSelect data source:")
    print("1. Google Drive (6 SPE9 datasets)")
    print("2. Sample data (for testing)")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return
    
    if choice == '1':
        print("\n" + "="*60)
        print("GOOGLE DRIVE MODE - LOADING 6 SPE9 DATASETS")
        print("="*60)
        
        file_ids = [
            '13twFcFA35CKbI8neIzIt-D54dzDd1B-N',  # SPE9.DATA.txt
            '1n_auKzsDz5aHglQ4YvskjfHPK8ZuLBqC',
            '1bdyUFKx-FKPy7YOlq-E9Y4nupcrhOoXi',
            '1f0aJFS99ZBVkT8IXbKdZdVihbIZIpBwZ',
            '1sxq7sd4GSL-chE362k8wTLA_arehaD5U',
            '1ZwEswptUcexDn_kqm_q8qRcHYTl1WHq2'
        ]
        
        datasets = DataLoader.load_multiple_files(file_ids)
        
        if not datasets:
            print("\nERROR: Could not load any datasets.")
            print("Falling back to sample data...")
            data = ReservoirData()
            data.create_sample_data()
            datasets = {'sample': data}
    else:
        print("\n" + "="*60)
        print("SAMPLE DATA MODE")
        print("="*60)
        data = ReservoirData()
        data.create_sample_data()
        datasets = {'sample': data}
    
    print(f"\n✓ Loaded {len(datasets)} datasets")
    
    # Get simulation parameters
    params = get_simulation_parameters()
    
    # Run simulation for each dataset
    all_results = {}
    for dataset_name, data in datasets.items():
        results = run_simulation_for_dataset(data, params, dataset_name)
        all_results[dataset_name] = results
    
    # Compare results if multiple datasets
    if len(datasets) > 1:
        print("\n" + "="*60)
        print("DATASET COMPARISON")
        print("="*60)
        
        comparison_data = []
        for dataset_name, results in all_results.items():
            if 'economic_analysis' in results:
                econ = results['economic_analysis']
                comparison_data.append({
                    'Dataset': dataset_name,
                    'NPV ($M)': econ.get('npv', 0)/1e6,
                    'IRR (%)': econ.get('irr', 0)*100,
                    'ROI (%)': econ.get('roi', 0)*100,
                    'Payback (years)': econ.get('payback_period_years', 0)
                })
        
        # Display comparison table
        if comparison_data:
            import pandas as pd
            df = pd.DataFrame(comparison_data)
            print(f"\n{df.to_string(index=False)}")
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
