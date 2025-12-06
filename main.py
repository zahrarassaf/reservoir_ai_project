# main.py
import sys
import os
import json
import logging
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.economics import ReservoirSimulator, EconomicParameters
from src.data_loader import DataLoader
from src.visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedReservoirSimulationProject:
    def __init__(self):
        self.data_loader = DataLoader()
        self.visualizer = Visualizer()
        self.simulation_results = []
        self.econ_params = None
        
    def configure_economic_parameters(self) -> EconomicParameters:
        print("\n" + "="*50)
        print("ECONOMIC PARAMETERS CONFIGURATION")
        print("="*50)
        
        try:
            forecast_years = int(input("Forecast years [15]: ") or "15")
            oil_price = float(input("Oil price (USD/bbl) [82.5]: ") or "82.5")
            operating_cost = float(input("Operating cost (USD/bbl) [16.5]: ") or "16.5")
            discount_rate = float(input("Discount rate (%) [9.5]: ") or "9.5") / 100
            
            print("\nAdvanced parameters (press Enter for defaults):")
            capex_per_producer = float(input("CAPEX per producer ($M) [3.5]: ") or "3.5") * 1_000_000
            fixed_annual_opex = float(input("Fixed annual OPEX ($M) [2.5]: ") or "2.5") * 1_000_000
            
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
        for i, dataset_id in enumerate(dataset_ids, 1):
            dataset = {
                'id': dataset_id,
                'name': f'SPE9_Dataset_{i}',
                'path': f'downloaded_data/{dataset_id}.csv'
            }
            datasets.append(dataset)
            
            if not os.path.exists(dataset['path']):
                print(f"Dataset {dataset_id} not found locally.")
                return []
        
        return datasets
    
    def load_sample_data(self) -> List[Dict]:
        print("\nGenerating synthetic reservoir data for testing...")
        
        datasets = []
        for i in range(3):
            dataset = {
                'id': f'sample_data_{i+1}',
                'name': f'Synthetic_Reservoir_{i+1}',
                'type': 'synthetic'
            }
            datasets.append(dataset)
        
        return datasets
    
    def load_custom_file(self) -> List[Dict]:
        file_path = input("\nEnter full path to data file: ").strip()
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []
        
        dataset = {
            'id': os.path.basename(file_path).split('.')[0],
            'name': os.path.basename(file_path),
            'path': file_path
        }
        
        return [dataset]
    
    def run_single_simulation(self, dataset: Dict) -> Dict:
        print("\n" + "="*60)
        print(f"DATASET: {dataset['id']}")
        print("="*60)
        
        try:
            if dataset.get('type') == 'synthetic':
                self.data_loader.load_spe9_data("")
                reservoir_data = self.data_loader.get_reservoir_data()
            else:
                if not self.data_loader.load_spe9_data(dataset['path']):
                    raise ValueError(f"Failed to load data from {dataset['path']}")
                reservoir_data = self.data_loader.get_reservoir_data()
            
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
            print(f"    Payback: {results['economic_evaluation']['payback_period']:.1f} years")
            print(f"    Break-even: ${results['economic_evaluation']['break_even_price']:.1f}/bbl")
            
            dataset_id_short = dataset['id'][:20]
            self.visualizer.create_dashboard(results, dataset_id_short)
            self.visualizer.plot_production_forecast(results, dataset_id_short)
            self.visualizer.plot_economic_results(results, dataset_id_short)
            
            print(f"  Visualizations generated for {dataset_id_short}")
            
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
                
                table_data.append({
                    'dataset': result['dataset_id'][:20],
                    'wells': wells,
                    'npv': econ['npv'] / 1_000_000,
                    'irr': econ['irr'],
                    'roi': econ['roi'],
                    'payback': econ['payback_period'],
                    'eur': eur,
                    'capex': econ['capex'] / 1_000_000
                })
            else:
                table_data.append({
                    'dataset': result['dataset_id'][:20],
                    'wells': 0,
                    'npv': 0,
                    'irr': 0,
                    'roi': 0,
                    'payback': '>20',
                    'eur': 0,
                    'capex': 0
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
        profitable = [r for r in successful if r['results']['economic_evaluation']['npv'] > 0]
        
        avg_npv = np.mean([r['results']['economic_evaluation']['npv'] for r in successful]) / 1_000_000 if successful else 0
        avg_irr = np.mean([r['results']['economic_evaluation']['irr'] for r in successful]) if successful else 0
        avg_roi = np.mean([r['results']['economic_evaluation']['roi'] for r in successful]) if successful else 0
        
        best_project = None
        worst_project = None
        
        if profitable:
            best_project = max(profitable, key=lambda x: x['results']['economic_evaluation']['npv'])
            worst_project = min(profitable, key=lambda x: x['results']['economic_evaluation']['npv'])
        elif successful:
            best_project = max(successful, key=lambda x: x['results']['economic_evaluation']['npv'])
            worst_project = min(successful, key=lambda x: x['results']['economic_evaluation']['npv'])
        
        print("\n" + "="*80)
        print("PROJECT SUMMARY")
        print("="*80)
        print(f"  Total simulations: {len(simulation_results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Profitable projects: {len(profitable)} ({len(profitable)/len(successful)*100 if successful else 0:.0f}%)")
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
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
                    'npv': results['economic_evaluation']['npv'],
                    'irr': results['economic_evaluation']['irr'],
                    'roi': results['economic_evaluation']['roi'],
                    'payback_period': results['economic_evaluation']['payback_period'],
                    'eur': results['production_forecast'].get('total_eur', 0)
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
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print("="*80)

def main():
    project = AdvancedReservoirSimulationProject()
    project.run()

if __name__ == "__main__":
    main()
