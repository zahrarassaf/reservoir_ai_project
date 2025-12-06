# main_simple.py - ULTRA SIMPLE VERSION

import sys
import os
import numpy as np

def main():
    print("=" * 60)
    print("RESERVOIR SIMULATION TEST")
    print("=" * 60)
    
    try:
        # Add current directory to Python path
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Import modules
        import src.data_loader
        import src.economics
        
        print("✓ Modules imported!")
        
        # Create instances
        data_loader = src.data_loader.DataLoader()
        print("✓ DataLoader created!")
        
        # Try to load sample data
        print("\nLoading sample data...")
        datasets = data_loader.load_sample_data()
        
        if datasets:
            print(f"✓ Loaded {len(datasets)} datasets")
            
            # Get first dataset
            dataset_id, reservoir_data = list(datasets.items())[0]
            print(f"\nTesting with dataset: {dataset_id}")
            print(f"Wells: {len(reservoir_data.get('wells', {}))}")
            
            # Create simulation parameters
            sim_params = src.economics.SimulationParameters(
                forecast_years=5,
                oil_price=75.0,
                operating_cost=18.0,
                discount_rate=0.10
            )
            print("✓ SimulationParameters created!")
            
            # Create simulator
            simulator = src.economics.ReservoirSimulator(reservoir_data, sim_params)
            print("✓ ReservoirSimulator created!")
            
            # Run simulation
            print("\nRunning simulation...")
            results = simulator.run_comprehensive_simulation()
            
            # Show results
            economic = results.get('economic_analysis', {})
            print(f"\nRESULTS:")
            print(f"NPV: ${economic.get('npv', 0):.2f}M")
            print(f"IRR: {economic.get('irr', 0):.1f}%")
            print(f"ROI: {economic.get('roi', 0):.1f}%")
            
            print("\n✓ TEST COMPLETED SUCCESSFULLY!")
            
        else:
            print("✗ No data loaded")
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
