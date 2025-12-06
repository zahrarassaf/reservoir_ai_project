import sys
import os
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("RESERVOIR SIMULATION PROJECT")
    print("=" * 60)
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        
        from src.data_loader import DataLoader
        from src.economics import ReservoirSimulator, SimulationParameters
        
        print("Modules imported successfully")
        
        data_loader = DataLoader()
        
        print("\nSelect data source:")
        print("1. Google Drive (6 SPE9 datasets)")
        print("2. Sample data (for testing)")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            logger.info("Loading from Google Drive")
            datasets = data_loader.load_from_google_drive()
        else:
            logger.info("Loading sample data")
            datasets = data_loader.load_sample_data()
        
        if not datasets:
            print("No datasets loaded")
            return
        
        print(f"Loaded {len(datasets)} datasets")
        
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
        
        params = SimulationParameters(
            forecast_years=forecast_years,
            oil_price=oil_price,
            operating_cost=operating_cost,
            discount_rate=discount_rate
        )
        
        all_results = []
        
        for dataset_id, reservoir_data in datasets.items():
            print(f"\n{'='*60}")
            print(f"SIMULATING: {dataset_id[:30]}")
            print(f"{'='*60}")
            
            try:
                wells_count = len(reservoir_data.get('wells', {}))
                print(f"\nData: {wells_count} wells")
                
                simulator = ReservoirSimulator(reservoir_data, params)
                results = simulator.run_comprehensive_simulation()
                
                economic = results.get('economic_analysis', {})
                
                print(f"\nECONOMIC RESULTS:")
                print(f"  NPV: ${economic.get('npv', 0):.2f}M")
                print(f"  IRR: {economic.get('irr', 0):.1f}%")
                print(f"  ROI: {economic.get('roi', 0):.1f}%")
                
                payback = economic.get('payback_period_years', None)
                if payback and payback != float('inf'):
                    print(f"  Payback: {payback:.1f} years")
                else:
                    print(f"  Payback: Never")
                
                all_results.append({
                    'dataset_id': dataset_id,
                    'wells': wells_count,
                    'npv': economic.get('npv', 0),
                    'irr': economic.get('irr', 0),
                    'roi': economic.get('roi', 0),
                    'payback': payback
                })
                
            except Exception as e:
                logger.error(f"Failed: {e}")
                all_results.append({
                    'dataset_id': dataset_id,
                    'error': str(e)
                })
        
        if all_results:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")
            
            for result in all_results:
                if 'error' in result:
                    print(f"{result['dataset_id'][:15]}...: ERROR")
                else:
                    npv = result['npv']
                    npv_str = f"${npv:+,.2f}M"
                    print(f"{result['dataset_id'][:15]}...: {npv_str}")
            
            successful = [r for r in all_results if 'error' not in r]
            if successful:
                npvs = [r['npv'] for r in successful]
                print(f"\nStatistics:")
                print(f"  Simulations: {len(all_results)}")
                print(f"  Successful: {len(successful)}")
                print(f"  Avg NPV: ${np.mean(npvs):.2f}M")
                print(f"  Min NPV: ${np.min(npvs):.2f}M")
                print(f"  Max NPV: ${np.max(npvs):.2f}M")
        
        print(f"\n{'='*60}")
        print("COMPLETED")
        print(f"{'='*60}")
        
    except ImportError as e:
        print(f"\nImport error: {e}")
        print("Check src/data_loader.py and src/economics.py")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
