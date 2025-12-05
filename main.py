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
    try:
        import gdown
        url = f'https://drive.google.com/uc?id={file_id}&export=download'
        output_path = os.path.join(output_dir, f'{file_id}.txt')
        
        print(f"Downloading {file_id}...")
        gdown.download(url, output_path, quiet=False)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✓ Downloaded: {os.path.getsize(output_path)} bytes")
            return output_path
        else:
            print(f"✗ Download failed")
            return None
            
    except Exception as e:
        print(f"Download error: {e}")
        return None


def load_google_drive_data() -> ReservoirData:
    file_ids = [
        '13twFcFA35CKbI8neIzIt-D54dzDd1B-N',
        '1n_auKzsDz5aHglQ4YvskjfHPK8ZuLBqC',
        '1bdyUFKx-FKPy7YOlq-E9Y4nupcrhOoXi',
        '1f0aJFS99ZBVkT8IXbKdZdVihbIZIpBwZ',
        '1sxq7sd4GSL-chE362k8wTLA_arehaD5U',
        '1ZwEswptUcexDn_kqm_q8qRcHYTl1WHq2'
    ]
    
    data = ReservoirData()
    success = False
    
    print("\n" + "="*50)
    print("GOOGLE DRIVE DATA LOADING")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_id in file_ids:
            print(f"\nTrying file ID: {file_id}")
            
            file_path = download_from_drive(file_id, temp_dir)
            if not file_path:
                continue
            
            if data.load_txt_file(file_path):
                success = True
                print(f"\n✅ SUCCESS: Loaded data from {file_id}")
                break
            else:
                print(f"❌ FAILED: Could not parse {file_id}")
    
    if not success:
        print("\n⚠️  WARNING: Could not load any Google Drive files")
        print("Creating sample data instead...")
        data.create_sample_data()
    
    return data


def get_simulation_parameters() -> SimulationParameters:
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
        
        print(f"\n✓ Parameters set:")
        print(f"  - Forecast years: {forecast_years}")
        print(f"  - Oil price: ${oil_price}/bbl")
        print(f"  - Operating cost: ${operating_cost}/bbl")
        
        return SimulationParameters(
            forecast_years=forecast_years,
            oil_price=oil_price,
            operating_cost=operating_cost
        )
        
    except ValueError:
        print("\n⚠️  Invalid input. Using default values.")
        return SimulationParameters()


def main():
    print("="*60)
    print("RESERVOIR SIMULATION PROJECT - PhD LEVEL")
    print("="*60)
    
    print("\nSelect mode:")
    print("1. Google Drive (requires internet connection)")
    print("2. Sample Data (local simulation)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    print("\n" + "="*60)
    
    if choice == '1':
        print("🔗 GOOGLE DRIVE MODE")
        print("-" * 40)
        data = load_google_drive_data()
    else:
        print("📊 SAMPLE DATA MODE")
        print("-" * 40)
        data = ReservoirData()
        data.create_sample_data()
        print("✓ Sample data created")
    
    summary = data.summary()
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    print(f"• Wells: {summary['wells']}")
    print(f"• Time Points: {summary['time_points']}")
    print(f"• Production Available: {'Yes' if data.has_production_data else 'No'}")
    print(f"• Pressure Available: {'Yes' if data.has_pressure_data else 'No'}")
    
    if data.has_production_data:
        prod_stats = summary['production_range']
        print(f"• Production Range: {prod_stats['min']:.1f} to {prod_stats['max']:.1f}")
    
    if data.has_pressure_data:
        pres_stats = summary['pressure_range']
        print(f"• Pressure Range: {pres_stats['min']:.1f} to {pres_stats['max']:.1f}")
    
    params = get_simulation_parameters()
    
    print("\n" + "="*50)
    print("RUNNING SIMULATION")
    print("="*50)
    
    print("\n🔬 Running data analysis...")
    simulator = ReservoirSimulator(data, params)
    
    print("\n⚡ Running reservoir simulation...")
    results = simulator.run_comprehensive_simulation()
    
    print("\n🎨 Creating visualizations...")
    
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n💾 Exporting results...")
    export_files = simulator.export_results(output_dir)
    
    print("\n" + "="*50)
    print("EXPORT RESULTS")
    print("="*50)
    print(f"✓ Results exported to: {output_dir}")
    
    for file_type, file_path in export_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  • {file_type.upper():10} : {os.path.basename(file_path)} ({file_size:,} bytes)")
    
    print("\n" + "="*60)
    print("✅ SIMULATION COMPLETED SUCCESSFULLY")
    print("="*60)
    
    if 'economic_analysis' in results:
        econ = results['economic_analysis']
        print(f"\n📊 FINAL RESULTS:")
        print(f"  • NPV: ${econ.get('npv', 0)/1e6:.2f}M")
        print(f"  • IRR: {econ.get('irr', 0)*100:.1f}%")
        print(f"  • ROI: {econ.get('roi', 0):.1f}%")


if __name__ == "__main__":
    main()
