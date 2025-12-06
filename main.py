import sys
import os
import logging
from typing import Dict, List, Any
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataLoader
from src.economics import ReservoirSimulator, SimulationParameters
from src.visualizer import Visualizer
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
        discount_rate = float(input(f"Discount rate % (default: 10.0): ") or "10.0") / 100
        
        # Optional advanced parameters
        print("\nAdvanced parameters (press Enter for defaults):")
        capex_per_well = float(input(f"CAPEX per well $M (default: 1.0): ") or "1.0") * 1_000_000
        fixed_opex = float(input(f"Fixed annual OPEX $M (default: 0.5): ") or "0.5") * 1_000_000
        tax_rate = float(input(f"Tax rate % (default: 30.0): ") or "30.0") / 100
        royalty_rate = float(input(f"Royalty rate % (default: 12.5): ") or "12.5") / 100
        
        return {
            'forecast_years': forecast_years,
            'oil_price': oil_price,
            'operating_cost': operating_cost,
            'discount_rate': discount_rate,
            'capex_per_well': capex_per_well,
            'fixed_annual_opex': fixed_opex,
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
    print(f"SIMULATING: {dataset_id}")
    print(f"{'='*60}")
    
    try:
        # Display data summary
        wells_count = len(reservoir_data.get('wells', {}))
        grid_dims = reservoir_data.get('grid_dimensions', 'N/A')
        
        print(f"\nDATA SUMMARY:")
        print(f"  • Wells: {wells_count}")
        print(f"  • Grid: {grid_dims}")
        
        if 'production_summary' in reservoir_data:
            prod = reservoir_data['production_summary']
            print(f"  • Max rate: {prod.get('max_rate', 0):.1f} STB/day")
            print(f"  • Total production: {prod.get('total_production', 0):,.0f} bbl")
        
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
        print(f"  • NPV: ${economic_results.get('npv', 0):.2f}M")
        
        irr_value = economic_results.get('irr', 0)
        if irr_value > 100:  # Cap unrealistic IRR
            irr_value = 100
        print(f"  • IRR: {irr_value:.1f}%")
        
        roi_value = economic_results.get('roi', 0)
        if roi_value > 500:  # Cap unrealistic ROI
            roi_value = 500
        elif roi_value < -100:
            roi_value = -100
        print(f"  • ROI: {roi_value:.1f}%")
        
        payback = economic_results.get('payback_period_years', None)
        if payback and payback != float('inf'):
            print(f"  • Payback: {payback:.1f} years")
        else:
            print(f"  • Payback: Never")
        
        # Additional metrics
        print(f"\nADDITIONAL METRICS:")
        print(f"  • Initial Investment: ${economic_results.get('initial_investment', 0)/1_000_000:.2f}M")
        print(f"  • Total Revenue: ${economic_results.get('total_revenue', 0)/1_000_000:.2f}M")
        print(f"  • Break-even Price: ${economic_results.get('break_even_price', 0):.2f}/bbl")
        
        # Store results with dataset info
        return {
            'dataset_id': dataset_id,
            'wells_count': wells_count,
            'total_production': reservoir_data.get('production_summary', {}).get('total_production', 0),
            'npv': economic_results.get('npv', 0),
            'irr': economic_results.get('irr', 0),
            'roi': economic_results.get('roi', 0),
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
        "Payback (years)", "Wells", "Production (bbl)"
    ]
    
    table.align["Dataset"] = "l"
    table.align["NPV ($M)"] = "r"
    table.align["IRR (%)"] = "r"
    table.align["ROI (%)"] = "r"
    table.align["Payback (years)"] = "r"
    table.align["Wells"] = "r"
    table.align["Production (bbl)"] = "r"
    
    for result in results:
        if 'error' in result:
            table.add_row([
                result['dataset_id'][:20] + "...",
                "ERROR",
                "ERROR",
                "ERROR",
                "ERROR",
                "ERROR",
                "ERROR"
            ])
        else:
            # Format NPV
            npv = result['npv']
            npv_str = f"${npv:,.2f}M"
            
            # Format IRR
            irr = min(result['irr'], 100)  # Cap at 100%
            irr_str = f"{irr:.1f}%"
            
            # Format ROI
            roi = max(min(result['roi'], 500), -100)  # Cap between -100% and 500%
            roi_str = f"{roi:.1f}%"
            
            # Format payback
            payback = result['payback']
            if payback and payback != float('inf'):
                payback_str = f"{payback:.1f}"
            else:
                payback_str = "Never"
            
            table.add_row([
                result['dataset_id'][:20] + "...",
                npv_str,
                irr_str,
                roi_str,
                payback_str,
                result['wells_count'],
                f"{result['total_production']:,.0f}"
            ])
    
    print(table)
    
    # Calculate and display statistics
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        npvs = [r['npv'] for r in successful_results]
        irrs = [r['irr'] for r in successful_results]
        
        print(f"\nSTATISTICS ({len(successful_results)} successful simulations):")
        print(f"  • Average NPV: ${np.mean(npvs):.2f}M")
        print(f"  • Minimum NPV: ${np.min(npvs):.2f}M")
        print(f"  • Maximum NPV: ${np.max(npvs):.2f}M")
        print(f"  • Average IRR: {np.mean(irrs):.1f}%")
        
        # Find best and worst projects
        best_idx = np.argmax(npvs)
        worst_idx = np.argmin(npvs)
        
        print(f"\nBEST PROJECT: {successful_results[best_idx]['dataset_id'][:20]}...")
        print(f"  • NPV: ${successful_results[best_idx]['npv']:.2f}M")
        print(f"  • IRR: {successful_results[best_idx]['irr']:.1f}%")
        
        print(f"\nWORST PROJECT: {successful_results[worst_idx]['dataset_id'][:20]}...")
        print(f"  • NPV: ${successful_results[worst_idx]['npv']:.2f}M")
        print(f"  • IRR: {successful_results[worst_idx]['irr']:.1f}%")

def generate_detailed_report(results: List[Dict], params: Dict):
    """Generate detailed report file"""
    try:
        import json
        from datetime import datetime
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'simulation_parameters': params,
            'datasets': []
        }
        
        for result in results:
            dataset_report = {
                'dataset_id': result['dataset_id'],
                'wells_count': result.get('wells_count', 0),
                'total_production': result.get('total_production', 0),
                'economic_metrics': {
                    'npv': result.get('npv', 0),
                    'irr': result.get('irr', 0),
                    'roi': result.get('roi', 0),
                    'payback_years': result.get('payback'),
                    'initial_investment': result.get('initial_investment', 0),
                    'total_revenue': result.get('total_revenue', 0),
                    'break_even_price': result.get('break_even_price', 0)
                }
            }
            
            if 'full_results' in result:
                # Include full results for the best project
                if result.get('npv', 0) == max([r.get('npv', 0) for r in results if 'error' not in r]):
                    dataset_report['full_simulation_results'] = result['full_results']
            
            report_data['datasets'].append(dataset_report)
        
        # Save report
        filename = f"simulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n✓ Detailed report saved to: {filename}")
        
    except Exception as e:
        print(f"\n⚠ Could not generate detailed report: {e}")

def run_sensitivity_analysis(best_result: Dict, params: Dict):
    """Run sensitivity analysis on the best project"""
    print(f"\n{'='*60}")
    print("SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    
    try:
        from src.economics import ReservoirSimulator, SimulationParameters
        
        dataset_id = best_result['dataset_id']
        reservoir_data = best_result['full_results']
        
        # Variables to test
        variables = ['oil_price', 'operating_cost', 'discount_rate']
        variations = [-0.2, -0.1, 0, 0.1, 0.2]
        
        print(f"\nAnalyzing sensitivity for: {dataset_id[:20]}...")
        print(f"Base NPV: ${best_result['npv']:.2f}M")
        
        sens_table = PrettyTable()
        sens_table.field_names = ["Variable", "-20%", "-10%", "Base", "+10%", "+20%"]
        
        for variable in variables:
            npv_values = []
            
            for variation in variations:
                # Create modified parameters
                modified_params = params.copy()
                if variable == 'oil_price':
                    modified_params[variable] = params['oil_price'] * (1 + variation)
                elif variable == 'operating_cost':
                    modified_params[variable] = params['operating_cost'] * (1 + variation)
                elif variable == 'discount_rate':
                    modified_params[variable] = params['discount_rate'] * (1 + variation)
                
                # Run simulation with modified parameters
                sim_params = SimulationParameters(
                    forecast_years=modified_params['forecast_years'],
                    oil_price=modified_params['oil_price'],
                    operating_cost=modified_params['operating_cost'],
                    discount_rate=modified_params['discount_rate'],
                    capex_per_well=modified_params['capex_per_well'],
                    fixed_annual_opex=modified_params['fixed_annual_opex'],
                    tax_rate=modified_params['tax_rate'],
                    royalty_rate=modified_params['royalty_rate']
                )
                
                simulator = ReservoirSimulator(reservoir_data, sim_params)
                results = simulator.run_comprehensive_simulation()
                npv = results.get('economic_analysis', {}).get('npv', 0)
                npv_values.append(f"${npv:.2f}M")
            
            sens_table.add_row([variable] + npv_values)
        
        print(sens_table)
        
    except Exception as e:
        print(f"\n⚠ Sensitivity analysis failed: {e}")

def main():
    """Main function"""
    logger = setup_logging()
    
    try:
        print("=" * 60)
        print("RESERVOIR SIMULATION PROJECT - PhD LEVEL")
        print("=" * 60)
        
        # Initialize components
        data_loader = DataLoader()
        visualizer = Visualizer() if 'Visualizer' in globals() or hasattr(__import__('src.visualizer'), 'Visualizer') else None
        
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
        
        logger.info(f"Parameters set: Forecast={params['forecast_years']} years, "
                   f"Oil price=${params['oil_price']}/bbl, "
                   f"Opex=${params['operating_cost']}/bbl, "
                   f"Discount rate={params['discount_rate']*100:.1f}%")
        
        # Run simulations for all datasets
        all_results = []
        
        for dataset_id, reservoir_data in datasets.items():
            result = run_single_simulation(dataset_id, reservoir_data, params)
            all_results.append(result)
        
        # Display comparison
        display_comparison_table(all_results)
        
        # Generate detailed report
        generate_detailed_report(all_results, params)
        
        # Run sensitivity analysis on best project
        successful_results = [r for r in all_results if 'error' not in r]
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['npv'])
            run_sensitivity_analysis(best_result, params)
        
        # Visualizations
        if visualizer and successful_results:
            try:
                # Plot comparison
                visualizer.plot_npv_comparison(successful_results)
                
                # Plot production forecast for best project
                best_full_results = best_result.get('full_results', {})
                if 'production_forecast' in best_full_results:
                    visualizer.plot_production_forecast(best_full_results['production_forecast'])
                
                print(f"\n✓ Visualizations generated")
            except Exception as e:
                print(f"\n⚠ Visualizations failed: {e}")
        
        print(f"\n{'='*60}")
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
        # Export results if visualizer available
        if visualizer:
            try:
                visualizer.export_results(all_results, "final_simulation_results.json")
                print(f"✓ Results exported to final_simulation_results.json")
            except:
                pass
        
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
