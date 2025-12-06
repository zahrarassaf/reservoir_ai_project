# main.py - VERSION WITH FIXED IMPORTS
import sys
import os
import logging
from typing import Dict, List, Any
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import directly from modules, not from src package
try:
    from src.data_loader import DataLoader
    from src.economics import ReservoirSimulator, SimulationParameters
    # Try to import visualizer if exists
    try:
        from src.visualizer import Visualizer
        VISUALIZER_AVAILABLE = True
    except ImportError:
        VISUALIZER_AVAILABLE = False
        print("Visualizer not available, continuing without visualizations")
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import structure...")
    
    # Alternative: import modules directly
    import importlib.util
    import sys
    
    # Import data_loader
    spec = importlib.util.spec_from_file_location("data_loader", "src/data_loader.py")
    data_loader_module = importlib.util.module_from_spec(spec)
    sys.modules["data_loader"] = data_loader_module
    spec.loader.exec_module(data_loader_module)
    DataLoader = data_loader_module.DataLoader
    
    # Import economics
    spec = importlib.util.spec_from_file_location("economics", "src/economics.py")
    economics_module = importlib.util.module_from_spec(spec)
    sys.modules["economics"] = economics_module
    spec.loader.exec_module(economics_module)
    ReservoirSimulator = economics_module.ReservoirSimulator
    SimulationParameters = economics_module.SimulationParameters
    
    VISUALIZER_AVAILABLE = False

from prettytable import PrettyTable

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_simulation_parameters():
    """Get simulation parameters from user"""
    print("\n" + "=" * 50)
    print("SIMULATION PARAMETERS")
    print("=" * 50)
    
    try:
        forecast_years = int(input(f"Forecast years (default: 10): ") or "10")
        oil_price = float(input(f"Oil price USD/bbl (default: 75.0): ") or "75.0")
        operating_cost = float(input(f"Operating cost USD/bbl (default: 18.0): ") or "18.0")
        
        # Ask for discount rate
        discount_rate_input = input(f"Discount rate % (default: 10.0): ") or "10.0"
        discount_rate = float(discount_rate_input) / 100
        
        # Optional advanced parameters
        print("\n[Optional] Advanced parameters (press Enter for defaults):")
        try:
            capex_input = input(f"CAPEX per well $M (default: 1.0): ") or "1.0"
            capex_per_well = float(capex_input) * 1_000_000
        except:
            capex_per_well = 1_000_000
            
        try:
            fixed_opex_input = input(f"Fixed annual OPEX $M (default: 0.5): ") or "0.5"
            fixed_annual_opex = float(fixed_opex_input) * 1_000_000
        except:
            fixed_annual_opex = 500_000
            
        try:
            tax_rate_input = input(f"Tax rate % (default: 30.0): ") or "30.0"
            tax_rate = float(tax_rate_input) / 100
        except:
            tax_rate = 0.30
            
        try:
            royalty_rate_input = input(f"Royalty rate % (default: 12.5): ") or "12.5"
            royalty_rate = float(royalty_rate_input) / 100
        except:
            royalty_rate = 0.125
        
        return {
            'forecast_years': forecast_years,
            'oil_price': oil_price,
            'operating_cost': operating_cost,
            'discount_rate': discount_rate,
            'capex_per_well': capex_per_well,
            'fixed_annual_opex': fixed_annual_opex,
            'tax_rate': tax_rate,
            'royalty_rate': royalty_rate
        }
        
    except ValueError as e:
        print(f"Invalid input: {e}. Using defaults.")
        return {
            'forecast_years': 10,
            'oil_price': 75.0,
            'operating_cost': 18.0,
            'discount_rate': 0.10,
            'capex_per_well': 1_000_000,
            'fixed_annual_opex': 500_000,
            'tax_rate': 0.30,
            'royalty_rate': 0.125
        }

def run_single_simulation(dataset_id: str, reservoir_data: Dict, params: Dict) -> Dict:
    """Run simulation for a single dataset"""
    logger = logging.getLogger(__name__)
    
    print(f"\n{'='*60}")
    print(f"SIMULATING: {dataset_id[:30]}")
    print(f"{'='*60}")
    
    try:
        # Display data summary
        wells_count = len(reservoir_data.get('wells', {}))
        grid_dims = reservoir_data.get('grid_dimensions', 'N/A')
        
        print(f"\nDATA SUMMARY:")
        print(f"  • Wells: {wells_count}")
        print(f"  • Grid: {grid_dims}")
        
        # Try to get production summary
        total_production = 0
        max_rate = 0
        
        if 'wells' in reservoir_data:
            for well_name, well in reservoir_data['wells'].items():
                if hasattr(well, 'production_rates'):
                    rates = well.production_rates
                    if len(rates) > 0:
                        max_rate = max(max_rate, np.max(rates))
                        
                        # Calculate approximate production
                        if hasattr(well, 'time_points'):
                            time_points = well.time_points
                            if len(time_points) >= 2:
                                avg_rate = np.mean(rates)
                                time_span = time_points[-1] - time_points[0]
                                total_production += avg_rate * time_span
        
        print(f"  • Max rate: {max_rate:.1f} STB/day")
        print(f"  • Total production: {total_production:,.0f} bbl")
        
        # Create simulation parameters
        sim_params = SimulationParameters(
            forecast_years=params['forecast_years'],
            oil_price=params['oil_price'],
            operating_cost=params['operating_cost'],
            discount_rate=params['discount_rate'],
            capex_per_well=params['capex_per_well'],
            fixed_annual_opex=params['fixed_annual_opex'],
            tax_rate=params['tax_rate'],
            royalty_rate=params['royalty_rate']
        )
        
        # Run simulation
        logger.info(f"Starting economic simulation for {dataset_id}")
        simulator = ReservoirSimulator(reservoir_data, sim_params)
        results = simulator.run_comprehensive_simulation()
        
        # Extract and display economic results
        economic_results = results.get('economic_analysis', {})
        
        print(f"\nECONOMIC RESULTS:")
        
        # Format NPV
        npv_value = economic_results.get('npv', 0)
        npv_str = f"${npv_value:+.2f}M"
        print(f"  • NPV: {npv_str}")
        
        # Format IRR
        irr_value = economic_results.get('irr', 0)
        if irr_value > 100:  # Cap unrealistic IRR
            irr_value = 100
        print(f"  • IRR: {irr_value:.1f}%")
        
        # Format ROI
        roi_value = economic_results.get('roi', 0)
        if roi_value > 500:  # Cap unrealistic ROI
            roi_value = 500
        elif roi_value < -100:
            roi_value = -100
        print(f"  • ROI: {roi_value:.1f}%")
        
        # Format payback
        payback = economic_results.get('payback_period_years', None)
        if payback and payback != float('inf'):
            print(f"  • Payback: {payback:.1f} years")
        else:
            print(f"  • Payback: Never")
        
        # Store results with dataset info
        return {
            'dataset_id': dataset_id,
            'wells_count': wells_count,
            'total_production': total_production,
            'npv': npv_value,
            'irr': irr_value,
            'roi': roi_value,
            'payback': payback,
            'initial_investment': economic_results.get('initial_investment', 0),
            'total_revenue': economic_results.get('total_revenue', 0),
            'break_even_price': economic_results.get('break_even_price', 0),
            'full_results': results
        }
        
    except Exception as e:
        logger.error(f"Simulation failed for {dataset_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'dataset_id': dataset_id,
            'error': str(e),
            'npv': 0,
            'irr': 0,
            'roi': 0,
            'payback': None
        }

def display_comparison_table(results: List[Dict]):
    """Display comparison table of all simulations"""
    print(f"\n{'='*60}")
    print("DATASET COMPARISON")
    print(f"{'='*60}")
    
    table = PrettyTable()
    table.field_names = [
        "Dataset", "NPV ($M)", "IRR (%)", "ROI (%)", 
        "Payback", "Wells", "Production"
    ]
    
    table.align["Dataset"] = "l"
    table.align["NPV ($M)"] = "r"
    table.align["IRR (%)"] = "r"
    table.align["ROI (%)"] = "r"
    table.align["Payback"] = "r"
    table.align["Wells"] = "r"
    table.align["Production"] = "r"
    
    for result in results:
        if 'error' in result:
            table.add_row([
                result['dataset_id'][:15] + "...",
                "ERROR",
                "ERROR",
                "ERROR",
                "ERROR",
                "ERROR",
                "ERROR"
            ])
        else:
            # Format NPV with sign
            npv = result['npv']
            if npv >= 0:
                npv_str = f"${npv:,.2f}M"
            else:
                npv_str = f"-${abs(npv):,.2f}M"
            
            # Format IRR
            irr = min(result['irr'], 100)
            irr_str = f"{irr:.1f}%"
            
            # Format ROI
            roi = max(min(result['roi'], 500), -100)
            roi_str = f"{roi:.1f}%"
            
            # Format payback
            payback = result['payback']
            if payback and payback != float('inf'):
                payback_str = f"{payback:.1f}y"
            else:
                payback_str = "Never"
            
            # Format production
            production = result['total_production']
            if production >= 1_000_000:
                prod_str = f"{production/1_000_000:.1f}M"
            else:
                prod_str = f"{production:,.0f}"
            
            table.add_row([
                result['dataset_id'][:15] + "...",
                npv_str,
                irr_str,
                roi_str,
                payback_str,
                result['wells_count'],
                prod_str
            ])
    
    print(table)

def main():
    """Main function"""
    logger = setup_logging()
    
    try:
        print("=" * 60)
        print("RESERVOIR SIMULATION PROJECT - PhD LEVEL")
        print("=" * 60)
        
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
        params = get_simulation_parameters()
        
        logger.info(f"Parameters: {params['forecast_years']}y, "
                   f"Oil=${params['oil_price']}/bbl, "
                   f"Opex=${params['operating_cost']}/bbl, "
                   f"DR={params['discount_rate']*100:.1f}%")
        
        # Run simulations for all datasets
        all_results = []
        
        for dataset_id, reservoir_data in datasets.items():
            result = run_single_simulation(dataset_id, reservoir_data, params)
            all_results.append(result)
        
        # Display comparison
        display_comparison_table(all_results)
        
        # Summary statistics
        successful_results = [r for r in all_results if 'error' not in r]
        if successful_results:
            npvs = [r['npv'] for r in successful_results]
            profitable = sum(1 for npv in npvs if npv > 0)
            
            print(f"\nSUMMARY:")
            print(f"  • Total simulations: {len(all_results)}")
            print(f"  • Successful: {len(successful_results)}")
            print(f"  • Profitable projects: {profitable}")
            print(f"  • Average NPV: ${np.mean(npvs):.2f}M")
            print(f"  • Best NPV: ${np.max(npvs):.2f}M")
            print(f"  • Worst NPV: ${np.min(npvs):.2f}M")
        
        print(f"\n{'='*60}")
        print("SIMULATION COMPLETED")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
