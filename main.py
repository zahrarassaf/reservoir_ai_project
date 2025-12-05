"""
Reservoir Simulation Project - Main Entry Point
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import ReservoirData
from src.simulator import ReservoirSimulator, SimulationParameters


def download_from_drive(file_id: str, output_dir: str) -> Optional[str]:
    """Download file from Google Drive"""
    try:
        import gdown
        url = f'https://drive.google.com/uc?id={file_id}'
        output_path = os.path.join(output_dir, f'{file_id}.csv')
        gdown.download(url, output_path, quiet=False)
        return output_path
    except Exception as e:
        print(f"Download error: {e}")
        return None


def load_google_drive_data() -> ReservoirData:
    """Load data from Google Drive"""
    file_ids = [
        '1ZwEswptUcexDn_kqm_q8qRcHYTl1WHq2',
        '1sxq7sd4GSL-chE362k8wTLA_arehaD5U',
        '1f0aJFS99ZBVkT8IXbKdZdVihbIZIpBwZ',
        '1bdyUFKx-FKPy7YOlq-E9Y4nupcrhOoXi',
        '1n_auKzsDz5aHglQ4YvskjfHPK8ZuLBqC',
        '13twFcFA35CKbI8neIzIt-D54dzDd1B-N'
    ]
    
    data = ReservoirData()
    success = False
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_id in file_ids:
            try:
                file_path = download_from_drive(file_id, temp_dir)
                if file_path and os.path.exists(file_path):
                    if data.load_csv(file_path):
                        success = True
                        print(f"✓ Successfully loaded data from Google Drive file")
                        break
            except Exception as e:
                print(f"Error processing file {file_id}: {e}")
                continue
    
    if not success:
        print("⚠️ Could not load Google Drive files. Creating sample data...")
        data.create_sample_data()
    
    return data


def get_simulation_parameters() -> SimulationParameters:
    """Get simulation parameters from user"""
    print("\n⚙️ Simulation Parameters:")
    
    try:
        forecast_years = input("Forecast years (default: 3): ").strip()
        forecast_years = int(forecast_years) if forecast_years else 3
        
        oil_price = input("Oil price USD/bbl (default: 75.0): ").strip()
        oil_price = float(oil_price) if oil_price else 75.0
        
        operating_cost = input("Operating cost USD/bbl (default: 18.0): ").strip()
        operating_cost = float(operating_cost) if operating_cost else 18.0
        
        return SimulationParameters(
            forecast_years=forecast_years,
            oil_price=oil_price,
            operating_cost=operating_cost
        )
    except ValueError:
        print("Invalid input. Using default values.")
        return SimulationParameters()


def main():
    print("=" * 60)
    print("RESERVOIR SIMULATION PROJECT - PhD LEVEL")
    print("=" * 60)
    print("\nSelect mode:")
    print("1. Google Drive (requires credentials.json)")
    print("2. Sample Data")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    print("\n" + "=" * 60)
    
    if choice == '1':
        print("🔗 Google Drive Mode")
        print("-" * 40)
        print("Loading data from Google Drive...")
        data = load_google_drive_data()
    else:
        print("📊 Sample Data Mode")
        print("-" * 40)
        print("Creating sample reservoir data...")
        data = ReservoirData()
        data.create_sample_data()
    
    summary = data.summary()
    print(f"\n📦 Data Summary:")
    print(f"   • Wells: {summary['wells']}")
    print(f"   • Time Points: {summary['time_points']}")
    print(f"   • Production Available: {'Yes' if data.has_production_data else 'No'}")
    print(f"   • Pressure Available: {'Yes' if data.has_pressure_data else 'No'}")
    
    params = get_simulation_parameters()
    
    print("\n🔬 Running data analysis...")
    
    simulator = ReservoirSimulator(data, params)
    
    print("\n⚡ Running reservoir simulation...")
    results = simulator.run_comprehensive_simulation()
    
    print("\n🎨 Creating visualizations...")
    
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    export_files = simulator.export_results(output_dir)
    
    print("\n💾 Exporting results...")
    print(f"✓ Results exported to: {output_dir}")
    for file_type, file_path in export_files.items():
        if os.path.exists(file_path):
            print(f"  • {file_type.upper()}: {file_path}")
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
