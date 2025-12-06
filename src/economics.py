# main_final.py
import sys
import os

def main():
    print("=" * 60)
    print("RESERVOIR SIMULATION - FINAL TEST")
    print("=" * 60)
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Import directly from files
        import importlib.util
        
        # Load data_loader
        spec = importlib.util.spec_from_file_location("data_loader", "src/data_loader.py")
        data_loader_module = importlib.util.module_from_spec(spec)
        sys.modules["data_loader"] = data_loader_module
        spec.loader.exec_module(data_loader_module)
        
        # Load economics  
        spec = importlib.util.spec_from_file_location("economics", "src/economics.py")
        economics_module = importlib.util.module_from_spec(spec)
        sys.modules["economics"] = economics_module
        spec.loader.exec_module(economics_module)
        
        print("✓ Modules loaded successfully!")
        
        # Create instances
        DataLoader = data_loader_module.DataLoader
        ReservoirSimulator = economics_module.ReservoirSimulator
        SimulationParameters = economics_module.SimulationParameters
        
        data_loader = DataLoader()
        print("✓ DataLoader created")
        
        # Load sample data
        datasets = data_loader.load_sample_data()
        print(f"✓ Loaded {len(datasets)} datasets")
        
        # Get first dataset
        dataset_id, reservoir_data = list(datasets.items())[0]
        print(f"\nTesting dataset: {dataset_id}")
        print(f"Wells: {len(reservoir_data.get('wells', {}))}")
        
        # Create simulation parameters
        params = SimulationParameters(
            forecast_years=5,
            oil_price=75.0,
            operating_cost=18.0,
            discount_rate=0.10
        )
        print("✓ SimulationParameters created")
        
        # Create and run simulator
        simulator = ReservoirSimulator(reservoir_data, params)
        print("✓ ReservoirSimulator created")
        
        print("\nRunning simulation...")
        results = simulator.run_comprehensive_simulation()
        
        # Show results
        economic = results.get('economic_analysis', {})
        print(f"\nRESULTS:")
        print(f"NPV: ${economic.get('npv', 0):.2f}M")
        print(f"IRR: {economic.get('irr', 0):.1f}%")
        print(f"ROI: {economic.get('roi', 0):.1f}%")
        print(f"Payback: {economic.get('payback_period_years', 'N/A')} years")
        
        print("\n" + "=" * 60)
        print("✓ TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
