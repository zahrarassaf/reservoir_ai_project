import sys
import os
import json
import logging
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.economics import ReservoirSimulator, EconomicParameters
from src.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedReservoirSimulationProject:
    def __init__(self):
        self.data_loader = DataLoader()
        self.simulation_results = []
        self.econ_params = None
        
    def configure_economic_parameters(self) -> EconomicParameters:
        print("\n" + "="*50)
        print("ECONOMIC PARAMETERS CONFIGURATION")
        print("="*50)
        
        try:
            forecast_years_input = input("Forecast years [15]: ").strip()
            oil_price_input = input("Oil price (USD/bbl) [82.5]: ").strip()
            operating_cost_input = input("Operating cost (USD/bbl) [16.5]: ").strip()
            discount_rate_input = input("Discount rate (%) [9.5]: ").strip()
            
            forecast_years = int(forecast_years_input) if forecast_years_input else 15
            oil_price = float(oil_price_input) if oil_price_input else 82.5
            operating_cost = float(operating_cost_input) if operating_cost_input else 16.5
            discount_rate = (float(discount_rate_input) if discount_rate_input else 9.5) / 100.0
            
            print("\nAdvanced parameters (press Enter for defaults):")
            capex_input = input("CAPEX per producer ($M) [3.5]: ").strip()
            opex_input = input("Fixed annual OPEX ($M) [2.5]: ").strip()
            
            capex_per_producer = (float(capex_input) if capex_input else 3.5) * 1000000.0
            fixed_annual_opex = (float(opex_input) if opex_input else 2.5) * 1000000.0
            
            self.econ_params = EconomicParameters(
                forecast_years=forecast_years,
                oil_price=oil_price,
                opex_per_bbl=operating_cost,
                discount_rate=discount_rate,
                capex_per_producer=capex_per_producer,
                fixed_opex=fixed_annual_opex
            )
            
            logger.info(f"Parameters configured: {forecast_years}y forecast, ${oil_price}/bbl, {discount_rate*100:.1f}% DR")
            return self.econ_params
            
        except ValueError as e:
            print(f"Invalid input: {e}. Using default parameters.")
            return EconomicParameters()
    
    def load_google_drive_data(self) -> List[Dict]:
        print("\nLoading SPE9 datasets from Google Drive...")
        
        dataset_ids = [
            "13twFcFA35CKbI8neIzIt-D54dzDd1B-N",
            "1n_auKzsDz5aHglQ4YvskjfHPK8ZuLBqC",
            "1bdyUFKx-FKPy7YOlq-E9Y4nupcrhOoXi",
            "1f0aJFS99ZBVkT8IXbKdZdVihbIZIpBwZ",
            "1sxq7sd4GSL-chE362k8wTLA_arehaD5U",
            "1ZwEswptUcexDn_kqm_q8qRcHYTl1WHq2"
        ]
        
        datasets = []
        for dataset_id in dataset_ids:
            datasets.append({
                'id': dataset_id,
                'name': f'SPE9_Dataset_{dataset_ids.index(dataset_id)+1}',
                'type': 'google_drive'
            })
        
        print(f"Found {len(datasets)} datasets to analyze")
        return datasets
    
    def load_sample_data(self) -> List[Dict]:
        print("\nGenerating synthetic reservoir data for testing...")
        
        datasets = []
        for i in range(3):
            datasets.append({
                'id': f'sample_data_{i+1}',
                'name': f'Synthetic_Reservoir_{i+1}',
                'type': 'synthetic'
            })
        
        return datasets
    
    def load_custom_file(self) -> List[Dict]:
        file_path = input("\nEnter full path to data file: ").strip()
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []
        
        return [{
            'id': os.path.basename(file_path).split('.')[0],
            'name': os.path.basename(file_path),
            'path': file_path,
            'type': 'custom'
        }]
    
    def run_single_simulation(self, dataset: Dict) -> Dict:
        print("\n" + "="*60)
        print(f"DATASET: {dataset['id']}")
        print("="*60)
        
        try:
            if dataset['type'] == 'synthetic':
                self.data_loader._generate_synthetic_data()
                reservoir_data = self.data_loader.get_reservoir_data()
            elif dataset['type'] == 'google_drive':
                success = self.data_loader.load_google_drive_data(dataset['id'])
                if not success:
                    print("  Using synthetic data (download failed)")
                    self.data_loader._generate_synthetic_data()
                reservoir_data = self.data_loader.get_reservoir_data()
            elif dataset['type'] == 'custom':
                import pandas as pd
                df = pd.read_csv(dataset['path'])
                time_col = df.columns[0]
                rate_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                
                from src.economics import WellProductionData
                reservoir_data = {
                    'wells': {
                        'CUSTOM_WELL': WellProductionData(
                            time_points=df[time_col].values,
                            oil_rate=df[rate_col].values,
                            well_type='PRODUCER'
                        )
                    },
                    'grid': {
                        'dimensions': (24, 25, 15),
                        'porosity': np.random.uniform(0.15, 0.25, 100)
                    }
                }
            else:
                raise ValueError(f"Unknown dataset type: {dataset['type']}")
            
            simulator = ReservoirSimulator(reservoir_data, self.econ_params)
            results = simulator.run_comprehensive_analysis()
            
            if 'error' in results:
                raise ValueError(results['error'])
            
            wells_analyzed = len(results.get('decline_analysis', {}))
            
            print(f"\n  Wells analyzed: {wells_analyzed}")
            print(f"  Forecast period: {self.econ_params.forecast_years} years")
            print(f"\n  ECONOMIC RESULTS:")
            print(f"    NPV: ${results['economic_evaluation']['npv']/1_000_000:+.2f}M")
            print(f"    IRR: {results['economic_evaluation']['irr']:.1f}%")
            print(f"    ROI: {results['economic_evaluation']['roi']:.1f}%")
            payback = results['economic_evaluation']['payback_period']
            payback_str = f"{payback:.1f}" if payback < 100 else ">100"
            print(f"    Payback: {payback_str} years")
            print(f"    Break-even: ${results['economic_evaluation']['break_even_price']:.1f}/bbl")
            
            try:
                from src.visualizer import Visualizer
                visualizer = Visualizer()
                dataset_id_short = dataset['id'][:20]
                visualizer.create_dashboard(results, dataset_id_short)
                visualizer.plot_production_forecast(results, dataset_id_short)
                visualizer.plot_economic_results(results, dataset_id_short)
                print(f"  Visualizations generated for {dataset_id_short}")
            except Exception as viz_error:
                print(f"  Visualization failed: {viz_error}")
            
            return {
                'dataset_id': dataset['id'],
                'dataset_name': dataset['name'],
                'results': results,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Simulation failed for {dataset['id']}: {e}")
            print(f"  ERROR: {e}")
            
            return {
                'dataset_id': dataset['id'],
                'dataset_name': dataset['name'],
                'error': str(e),
                'success': False
            }
    
    def generate_comparison_table(self, simulation_results: List[Dict]):
        print("\n" + "="*80)
        print("DATASET COMPARISON")
        print("="*80)
        
        table_data = []
        for result in simulation_results:
            if result['success']:
                results = result['results']
                econ = results['economic_evaluation']
                wells = len(results.get('decline_analysis', {}))
                eur = results['production_forecast'].get('total_eur', 0) / 1_000_000
                payback = econ['payback_period']
                payback_str = f"{payback:.1f}" if payback < 100 else ">100"
                
                table_data.append({
                    'dataset': result['dataset_id'][:20],
                    'wells': wells,
                    'npv': econ['npv'] / 1_000_000,
                    'irr': econ['irr'],
                    'roi': econ['roi'],
                    'payback': payback_str,
                    'eur': eur,
                    'capex': econ['capex'] / 1_000_000
                })
            else:
                table_data.append({
                    'dataset': result['dataset_id'][:20],
                    'wells': 0,
                    'npv': 0.0,
                    'irr': 0.0,
                    'roi': 0.0,
                    'payback': ">100",
                    'eur': 0.0,
                    'capex': 0.0
                })
        
        print("+----------------------+-------+----------+---------+---------+-------------+-------------+------------+")
        print("| Dataset              | Wells | NPV ($M) | IRR (%) | ROI (%) | Payback (y) | EUR (MMbbl) | CAPEX ($M) |")
        print("+----------------------+-------+----------+---------+---------+-------------+-------------+------------+")
        
        for row in table_data:
            dataset = row['dataset'].ljust(20)
            wells = str(row['wells']).rjust(5)
            npv = f"{row['npv']:+.2f}".rjust(8)
            irr = f"{row['irr']:.1f}".rjust(7)
            roi = f"{row['roi']:.1f}".rjust(7)
            payback = str(row['payback']).rjust(11)
            eur = f"{row['eur']:.1f}".rjust(11)
            capex = f"{row['capex']:.1f}".rjust(10)
            
            print(f"| {dataset} | {wells} | {npv} | {irr} | {roi} | {payback} | {eur} | {capex} |")
        
        print("+----------------------+-------+----------+---------+---------+-------------+-------------+------------+")
        
        return table_data
    
    def generate_project_summary(self, simulation_results: List[Dict]):
        successful = [r for r in simulation_results if r['success']]
        if not successful:
            print("\nNo successful simulations to summarize.")
            return
        
        profitable = [r for r in successful if r['results']['economic_evaluation']['npv'] > 0]
        
        avg_npv = np.mean([r['results']['economic_evaluation']['npv'] for r in successful]) / 1_000_000
        avg_irr = np.mean([r['results']['economic_evaluation']['irr'] for r in successful])
        avg_roi = np.mean([r['results']['economic_evaluation']['roi'] for r in successful])
        
        best_project = max(successful, key=lambda x: x['results']['economic_evaluation']['npv'])
        worst_project = min(successful, key=lambda x: x['results']['economic_evaluation']['npv'])
        
        print("\n" + "="*80)
        print("PROJECT SUMMARY")
        print("="*80)
        print(f"  Total simulations: {len(simulation_results)}")
        print(f"  Successful: {len(successful)}")
        profitable_pct = (len(profitable)/len(successful)*100) if successful else 0
        print(f"  Profitable projects: {len(profitable)} ({profitable_pct:.0f}%)")
        print(f"  Average NPV: ${avg_npv:+.2f}M")
        print(f"  Average IRR: {avg_irr:.1f}%")
        print(f"  Average ROI: {avg_roi:.1f}%")
        
        if best_project:
            print(f"\n  BEST PROJECT: {best_project['dataset_id'][:20]}")
            print(f"    NPV: ${best_project['results']['economic_evaluation']['npv']/1_000_000:+.2f}M")
            print(f"    IRR: {best_project['results']['economic_evaluation']['irr']:.1f}%")
        
        if worst_project and worst_project != best_project:
            print(f"\n  WORST PROJECT: {worst_project['dataset_id'][:20]}")
            print(f"    NPV: ${worst_project['results']['economic_evaluation']['npv']/1_000_000:+.2f}M")
            print(f"    IRR: {worst_project['results']['economic_evaluation']['irr']:.1f}%")
    
    def save_comprehensive_report(self, simulation_results: List[Dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"simulation_report_{timestamp}.json"
        
        report_data = {
            'timestamp': timestamp,
            'economic_parameters': {
                'forecast_years': self.econ_params.forecast_years,
                'oil_price': self.econ_params.oil_price,
                'discount_rate': self.econ_params.discount_rate,
                'opex_per_bbl': self.econ_params.opex_per_bbl
            },
            'simulations': []
        }
        
        for result in simulation_results:
            sim_data = {
                'dataset_id': result['dataset_id'],
                'dataset_name': result['dataset_name'],
                'success': result['success']
            }
            
            if result['success']:
                results = result['results']
                sim_data.update({
                    'wells_analyzed': len(results.get('decline_analysis', {})),
                    'npv': float(results['economic_evaluation']['npv']),
                    'irr': float(results['economic_evaluation']['irr']),
                    'roi': float(results['economic_evaluation']['roi']),
                    'payback_period': float(results['economic_evaluation']['payback_period']),
                    'eur': float(results['production_forecast'].get('total_eur', 0))
                })
            else:
                sim_data['error'] = result.get('error', 'Unknown error')
            
            report_data['simulations'].append(sim_data)
        
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n✓ Comprehensive report saved to: {report_filename}")
        return report_filename
    
    def run(self):
        print("="*80)
        print("ADVANCED RESERVOIR SIMULATION PROJECT - PhD LEVEL")
        print("="*80)
        print("\nComprehensive reservoir characterization, decline analysis,")
        print("production forecasting, and economic evaluation system")
        print("="*80)
        
        print("\n" + "="*50)
        print("DATA SOURCE SELECTION")
        print("="*50)
        print("1. Google Drive - SPE9 Benchmark Datasets (6 datasets)")
        print("2. Sample Data - Synthetic reservoir for testing")
        print("3. Custom File - Load from local file")
        
        while True:
            try:
                option = input("\nSelect option (1-3): ").strip()
                if option in ['1', '2', '3']:
                    break
                print("Invalid option. Please enter 1, 2, or 3.")
            except KeyboardInterrupt:
                print("\n\nSimulation cancelled.")
                return
        
        if option == '1':
            datasets = self.load_google_drive_data()
        elif option == '2':
            datasets = self.load_sample_data()
        else:
            datasets = self.load_custom_file()
        
        if not datasets:
            print("No datasets available. Exiting.")
            return
        
        self.econ_params = self.configure_economic_parameters()
        
        print("\n" + "="*80)
        print(f"RUNNING SIMULATIONS - {len(datasets)} DATASETS")
        print("="*80)
        
        self.simulation_results = []
        for dataset in datasets:
            result = self.run_single_simulation(dataset)
            self.simulation_results.append(result)
        
        self.generate_comparison_table(self.simulation_results)
        
        report_file = self.save_comprehensive_report(self.simulation_results)
        
        self.generate_project_summary(self.simulation_results)
        
        print("\n" + "="*80)
        print("SIMULATION COMPLETED")
        print("="*80)

def main():
    try:
        project = AdvancedReservoirSimulationProject()
        project.run()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
