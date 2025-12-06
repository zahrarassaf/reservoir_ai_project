# main.py - SIMPLE WORKING VERSION

import sys
import os
import logging
import numpy as np
from prettytable import PrettyTable

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("RESERVOIR SIMULATION PROJECT - PhD LEVEL")
    print("=" * 60)
    
    try:
        # Import modules directly (bypass __init__.py)
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Import data_loader directly
        from src.data_loader import DataLoader
        
        # Import economics directly  
        from src.economics import ReservoirSimulator, SimulationParameters
        
        print("\nModules imported successfully!")
        
        # Initialize data loader
        data_loader = DataLoader()
        
        # Select data source
        print("\nSelect data source:")
        print("1. Google Drive (6 SPE9 datasets)")
        print("2. Sample data (for testing)")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            logger.info("Google Drive mode selected")
            datasets = data_loader.load_from_google_drive()
        else:
            logger.info("Sample data mode selected")
            datasets = data_loader.load_sample_data()
        
        if not datasets:
            logger.error("No datasets loaded")
            return
        
        print(f"\n✓ Loaded {len(datasets)} datasets")
        
        # Get simulation parameters
        print("\n" + "=" * 50)
        print("SIMULATION PARAMETERS")
        print("=" * 50)
        
        try:
            forecast_years = int(input(f"Forecast years (default: 10): ") or "10")
            oil_price = float(input(f"Oil price USD/bbl (default: 75.0): ") or "75.0")
            operating_cost = float(input(f"Operating cost USD/bbl (default: 18.0): ") or "18.0")
            discount_rate = float(input(f"Discount rate % (default: 10.0): ") or "10.0") / 100
        except ValueError:
            print("Invalid input, using defaults")
            forecast_years = 10
            oil_price = 75.0
            operating_cost = 18.0
            discount_rate = 0.10
        
        # Create simulation parameters
        sim_params = SimulationParameters(
            forecast_years=forecast_years,
            oil_price=oil_price,
            operating_cost=operating_cost,
            discount_rate=discount_rate,
            capex_per_well=1_000_000,
            fixed_annual_opex=500_000,
            tax_rate=0.30,
            royalty_rate=0.125
        )
        
        logger.info(f"Parameters: {forecast_years}y, Oil=${oil_price}/bbl, Opex=${operating_cost}/bbl")
        
        # Run simulations
        all_results = []
        
        for dataset_id, reservoir_data in datasets.items():
            print(f"\n{'='*60}")
            print(f"SIMULATING: {dataset_id[:30]}")
            print(f"{'='*60}")
            
            try:
                # Run simulation
                simulator = ReservoirSimulator(reservoir_data, sim_params)
                results = simulator.run_comprehensive_simulation()
                
                # Extract economic results
                economic_results = results.get('economic_analysis', {})
                
                # Display results
                wells_count = len(reservoir_data.get('wells', {}))
                print(f"\nDATA: {wells_count} wells")
                
                print(f"\nECONOMIC RESULTS:")
                print(f"  • NPV: ${economic_results.get('npv', 0):.2f}M")
                print(f"  • IRR: {economic_results.get('irr', 0):.1f}%")
                print(f"  • ROI: {economic_results.get('roi', 0):.1f}%")
                
                payback = economic_results.get('payback_period_years', None)
                if payback and payback != float('inf'):
                    print(f"  • Payback: {payback:.1f} years")
                else:
                    print(f"  • Payback: Never")
                
                # Store for comparison
                all_results.append({
                    'dataset_id': dataset_id,
                    'wells': wells_count,
                    'npv': economic_results.get('npv', 0),
                    'irr': economic_results.get('irr', 0),
                    'roi': economic_results.get('roi', 0),
                    'payback': payback
                })
                
            except Exception as e:
                logger.error(f"Simulation failed: {e}")
                all_results.append({
                    'dataset_id': dataset_id,
                    'error': str(e),
                    'npv': 0,
                    'irr': 0,
                    'roi': 0
                })
        
        # Display comparison table
        if all_results:
            print(f"\n{'='*60}")
            print("DATASET COMPARISON")
            print(f"{'='*60}")
            
            table = PrettyTable()
            table.field_names = ["Dataset", "Wells", "NPV ($M)", "IRR (%)", "ROI (%)", "Payback"]
            
            for result in all_results:
                if 'error' in result:
                    table.add_row([
                        result['dataset_id'][:15] + "...",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR"
                    ])
                else:
                    # Format NPV
                    npv = result['npv']
                    if npv >= 0:
                        npv_str = f"${npv:,.2f}M"
                    else:
                        npv_str = f"-${abs(npv):,.2f}M"
                    
                    # Format payback
                    payback = result['payback']
                    if payback and payback != float('inf'):
                        payback_str = f"{payback:.1f}y"
                    else:
                        payback_str = "Never"
                    
                    table.add_row([
                        result['dataset_id'][:15] + "...",
                        result['wells'],
                        npv_str,
                        f"{result['irr']:.1f}%",
                        f"{result['roi']:.1f}%",
                        payback_str
                    ])
            
            print(table)
            
            # Calculate statistics
            successful = [r for r in all_results if 'error' not in r]
            if successful:
                npvs = [r['npv'] for r in successful]
                print(f"\nSUMMARY:")
                print(f"  • Total simulations: {len(all_results)}")
                print(f"  • Successful: {len(successful)}")
                print(f"  • Average NPV: ${np.mean(npvs):.2f}M")
                print(f"  • Best NPV: ${np.max(npvs):.2f}M")
                print(f"  • Worst NPV: ${np.min(npvs):.2f}M")
        
        print(f"\n{'='*60}")
        print("SIMULATION COMPLETED")
        print(f"{'='*60}")
        
    except ImportError as e:
        print(f"\nERROR: Could not import required modules: {e}")
        print("Please check that:")
        print("1. src/data_loader.py exists and has DataLoader class")
        print("2. src/economics.py exists and has ReservoirSimulator and SimulationParameters")
        print("3. requirements.txt packages are installed")
        return 1
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        return 0
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
