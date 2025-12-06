import os
import sys
from typing import List, Dict
import logging
import numpy as np
import pandas as pd
import gdown
import tempfile
from tabulate import tabulate

from src.data_loader import ReservoirData
from src.simulator import ReservoirSimulator, SimulationParameters

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_google_drive_files(file_ids: List[str]) -> Dict[str, str]:
    downloaded_files = {}
    
    for file_id in file_ids:
        try:
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, f"{file_id}.txt")
            
            logger.info(f"Downloading {file_id}...")
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}&export=download",
                output_path,
                quiet=False
            )
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"Downloaded: {file_size} bytes")
                downloaded_files[file_id] = output_path
            else:
                logger.warning(f"Failed to download {file_id}")
                
        except Exception as e:
            logger.error(f"Error downloading {file_id}: {e}")
    
    return downloaded_files

def run_simulation_for_dataset(file_path: str, dataset_name: str, 
                               forecast_years: int, oil_price: float, 
                               operating_cost: float) -> Dict:
    
    print(f"\n{'='*60}")
    print(f"SIMULATING: {dataset_name}")
    print(f"{'='*60}\n")
    
    try:
        # Create new ReservoirData instance for each dataset
        data_loader = ReservoirData()
        
        if not data_loader.load_spe9_file(file_path):
            print(f"Failed to load {dataset_name}")
            return None
            
        summary = data_loader.summary()
        
        print("DATA SUMMARY:")
        print(f"  • Wells: {summary['wells']}")
        print(f"  • Grid: {summary['grid_dimensions']}")
        print(f"  • Production data: {'Available' if summary['has_production_data'] else 'Not available'}")
        print(f"  • Max rate: {summary['production_range']['max']:.1f} STB/day")
        print(f"  • Total production: {summary['total_production']:,.0f} bbl")
        
        params = SimulationParameters(
            forecast_years=forecast_years,
            oil_price=oil_price,
            operating_cost=operating_cost,
            discount_rate=0.10,
            economic_limit=20.0
        )
        
        simulator = ReservoirSimulator(data_loader, params)
        results = simulator.run_comprehensive_simulation()
        
        economic = results['economic_analysis']
        
        print("\nECONOMIC RESULTS:")
        print(f"  • NPV: ${economic['npv']:.2f}M")
        print(f"  • IRR: {economic['irr']:.1f}%")
        print(f"  • ROI: {economic['roi']:.1f}%")
        if economic['payback_period_years'] == float('inf'):
            print(f"  • Payback: Never")
        else:
            print(f"  • Payback: {economic['payback_period_years']:.1f} years")
        
        return {
            'dataset': dataset_name,
            'npv': economic['npv'],
            'irr': economic['irr'],
            'roi': economic['roi'],
            'payback': economic['payback_period_years'],
            'wells': summary['wells'],
            'total_production': summary['total_production']
        }
        
    except Exception as e:
        logger.error(f"Error simulating {dataset_name}: {e}")
        return None

def main():
    print("=" * 60)
    print("RESERVOIR SIMULATION PROJECT - PhD LEVEL")
    print("=" * 60)
    print()
    
    print("Select data source:")
    print("1. Google Drive (6 SPE9 datasets)")
    print("2. Sample data (for testing)")
    print()
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        print()
        print("=" * 60)
        print("GOOGLE DRIVE MODE - LOADING 6 SPE9 DATASETS")
        print("=" * 60)
        
        file_ids = [
            "13twFcFA35CKbI8neIzIt-D54dzDd1B-N",
            "1n_auKzsDz5aHglQ4YvskjfHPK8ZuLBqC",
            "1bdyUFKx-FKPy7YOlq-E9Y4nupcrhOoXi",
            "1f0aJFS99ZBVkT8IXbKdZdVihbIZIpBwZ",
            "1sxq7sd4GSL-chE362k8wTLA_arehaD5U",
            "1ZwEswptUcexDn_kqm_q8qRcHYTl1WHq2"
        ]
        
        downloaded_files = download_google_drive_files(file_ids)
        
        if not downloaded_files:
            print("No files downloaded. Exiting.")
            return
        
        all_data = []  # This will store (file_id, file_path) tuples
        
        for file_id, file_path in downloaded_files.items():
            logger.info(f"Processing file ID: {file_id}")
            all_data.append((file_id, file_path))
            logger.info(f"Downloaded {file_id}")
        
        print(f"\n✓ Downloaded {len(all_data)} datasets")
        
    elif choice == "2":
        print()
        print("=" * 60)
        print("SAMPLE DATA MODE")
        print("=" * 60)
        
        # Create sample data
        data_loader = ReservoirData()
        data_loader.create_sample_data(n_wells=8, n_time_points=365*5)
        
        # Save to temp file
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "sample_data.txt")
        
        # We'll use a simple approach for sample data
        all_data = [("Sample Data", temp_file)]
        
        print(f"\n✓ Created sample data with 8 wells")
    
    else:
        print("Invalid choice. Exiting.")
        return
    
    print()
    print("=" * 50)
    print("SIMULATION PARAMETERS")
    print("=" * 50)
    
    forecast_input = input("Forecast years (default: 10): ").strip()
    if forecast_input:
        forecast_years = int(forecast_input)
    else:
        forecast_years = 10
    
    oil_price_input = input("Oil price USD/bbl (default: 75.0): ").strip()
    if oil_price_input:
        oil_price = float(oil_price_input)
    else:
        oil_price = 75.0
    
    operating_cost_input = input("Operating cost USD/bbl (default: 18.0): ").strip()
    if operating_cost_input:
        operating_cost = float(operating_cost_input)
    else:
        operating_cost = 18.0
    
    logger.info(f"Parameters set: Forecast={forecast_years} years, "
                f"Oil price=${oil_price}/bbl, Opex=${operating_cost}/bbl")
    
    results = []
    
    for dataset_name, file_path in all_data:
        try:
            result = run_simulation_for_dataset(
                file_path, dataset_name, forecast_years, oil_price, operating_cost
            )
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Error simulating {dataset_name}: {e}")
            continue
    
    if results:
        print()
        print("=" * 60)
        print("DATASET COMPARISON")
        print("=" * 60)
        print()
        
        table_data = []
        for result in results:
            payback_str = "Never" if result['payback'] == float('inf') else f"{result['payback']:.1f}"
            table_data.append([
                result['dataset'],
                f"${result['npv']:.2f}M",
                f"{result['irr']:.1f}%",
                f"{result['roi']:.1f}%",
                payback_str,
                result['wells'],
                f"{result['total_production']:,.0f}"
            ])
        
        headers = ["Dataset", "NPV", "IRR", "ROI", "Payback (years)", "Wells", "Total Production (bbl)"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print()
        print("=" * 60)
        print("SIMULATION COMPLETED")
        print("=" * 60)
        
        # Clean up temp files
        if choice == "1":
            for _, file_path in downloaded_files.items():
                try:
                    os.remove(file_path)
                    temp_dir = os.path.dirname(file_path)
                    os.rmdir(temp_dir)
                except:
                    pass
    else:
        print("\nNo simulation results available.")

if __name__ == "__main__":
    main()
