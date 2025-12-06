import sys
import os
import logging
from datetime import datetime
import numpy as np
from prettytable import PrettyTable
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import DataLoader
from src.economics import ReservoirSimulator, EconomicParameters
from src.visualizer import Visualizer

class ReservoirSimulationProject:
    def __init__(self):
        self.data_loader = DataLoader()
        self.visualizer = Visualizer()
        self.results = []
        self.parameters = EconomicParameters()
        
    def run(self):
        self.display_header()
        
        try:
            datasets = self.select_data_source()
            
            if not datasets:
                logger.error("No datasets loaded")
                return
            
            self.configure_parameters()
            
            self.run_simulations(datasets)
            
            self.display_comparison()
            
            self.generate_reports()
            
            self.display_summary()
            
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user")
        except Exception as e:
            logger.error(f"Project execution failed: {e}", exc_info=True)
    
    def display_header(self):
        print("=" * 80)
        print("ADVANCED RESERVOIR SIMULATION PROJECT - PhD LEVEL")
        print("=" * 80)
        print("\nComprehensive reservoir characterization, decline analysis,")
        print("production forecasting, and economic evaluation system")
        print("=" * 80)
    
    def select_data_source(self):
        print("\n" + "=" * 50)
        print("DATA SOURCE SELECTION")
        print("=" * 50)
        print("1. Google Drive - SPE9 Benchmark Datasets (6 datasets)")
        print("2. Sample Data - Synthetic reservoir for testing")
        print("3. Custom File - Load from local file")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            logger.info("Loading SPE9 datasets from Google Drive")
            return self.data_loader.load_from_google_drive()
        elif choice == "2":
            logger.info("Loading sample data")
            return self.data_loader.load_sample_data()
        elif choice == "3":
            filepath = input("Enter file path: ").strip()
            return self.load_custom_file(filepath)
        else:
            print("Invalid choice, using sample data")
            return self.data_loader.load_sample_data()
    
    def load_custom_file(self, filepath):
        return {'custom_file': {'wells': {}, 'grid': {}}}
    
    def configure_parameters(self):
        print("\n" + "=" * 50)
        print("ECONOMIC PARAMETERS CONFIGURATION")
        print("=" * 50)
        
        try:
            self.parameters.forecast_years = int(
                input(f"Forecast years [{self.parameters.forecast_years}]: ") 
                or str(self.parameters.forecast_years)
            )
            
            self.parameters.oil_price = float(
                input(f"Oil price (USD/bbl) [{self.parameters.oil_price}]: ") 
                or str(self.parameters.oil_price)
            )
            
            self.parameters.opex_per_bbl = float(
                input(f"Operating cost (USD/bbl) [{self.parameters.opex_per_bbl}]: ") 
                or str(self.parameters.opex_per_bbl)
            )
            
            self.parameters.discount_rate = float(
                input(f"Discount rate (%) [{self.parameters.discount_rate*100:.1f}]: ") 
                or str(self.parameters.discount_rate * 100)
            ) / 100
            
            print("\nAdvanced parameters (press Enter for defaults):")
            
            self.parameters.capex_per_producer = float(
                input(f"CAPEX per producer ($M) [{self.parameters.capex_per_producer/1e6:.1f}]: ") 
                or str(self.parameters.capex_per_producer / 1e6)
            ) * 1e6
            
            self.parameters.fixed_opex = float(
                input(f"Fixed annual OPEX ($M) [{self.parameters.fixed_opex/1e6:.1f}]: ") 
                or str(self.parameters.fixed_opex / 1e6)
            ) * 1e6
            
            logger.info(f"Parameters configured: {self.parameters.forecast_years}y forecast, "
                       f"${self.parameters.oil_price}/bbl, {self.parameters.discount_rate*100:.1f}% DR")
            
        except ValueError as e:
            print(f"Invalid input: {e}. Using default parameters.")
            logger.warning(f"Parameter configuration error: {e}")
    
    def run_simulations(self, datasets):
        print(f"\n{'='*80}")
        print(f"RUNNING SIMULATIONS - {len(datasets)} DATASETS")
        print(f"{'='*80}")
        
        for dataset_id, data in datasets.items():
            print(f"\n{'='*60}")
            print(f"DATASET: {dataset_id[:40]}")
            print(f"{'='*60}")
            
            try:
                simulator = ReservoirSimulator(data, self.parameters)
                results = simulator.run_comprehensive_analysis()
                
                self.display_dataset_results(dataset_id, results)
                
                self.results.append({
                    'dataset_id': dataset_id,
                    'results': results,
                    'summary': self.extract_summary(results)
                })
                
                self.generate_visualizations(dataset_id, results)
                
            except Exception as e:
                logger.error(f"Simulation failed for {dataset_id}: {e}")
                print(f"  ERROR: {e}")
    
    def display_dataset_results(self, dataset_id, results):
        econ = results.get('economic_evaluation', {})
        prod = results.get('production_forecast', {})
        reservoir = results.get('reservoir_properties', {})
        
        wells = len(results.get('decline_analysis', {}))
        
        print(f"\n  Wells analyzed: {wells}")
        print(f"  Forecast period: {self.parameters.forecast_years} years")
        
        if reservoir.get('ooip', 0) > 0:
            print(f"  OOIP: {reservoir['ooip']/1e6:,.1f} MMbbl")
            print(f"  Recovery factor: {reservoir.get('recovery_factor', 0)*100:.1f}%")
        
        print(f"\n  ECONOMIC RESULTS:")
        print(f"    NPV: ${econ.get('npv', 0)/1e6:+,.2f}M")
        print(f"    IRR: {econ.get('irr', 0):.1f}%")
        print(f"    ROI: {econ.get('roi', 0):.1f}%")
        
        payback = econ.get('payback_period', float('inf'))
        if payback < 99:
            print(f"    Payback: {payback:.1f} years")
        else:
            print(f"    Payback: >{self.parameters.forecast_years} years")
        
        if 'total_eur' in prod:
            print(f"    EUR: {prod['total_eur']/1e6:,.1f} MMbbl")
        
        print(f"    Break-even: ${econ.get('break_even_price', 0):.1f}/bbl")
    
    def extract_summary(self, results):
        econ = results.get('economic_evaluation', {})
        prod = results.get('production_forecast', {})
        
        return {
            'npv': econ.get('npv', 0),
            'irr': econ.get('irr', 0),
            'roi': econ.get('roi', 0),
            'payback': econ.get('payback_period', float('inf')),
            'eur': prod.get('total_eur', 0),
            'capex': econ.get('capex', 0),
            'wells': len(results.get('decline_analysis', {}))
        }
    
    def generate_visualizations(self, dataset_id, results):
        try:
            dashboard = self.visualizer.create_comprehensive_dashboard(results, dataset_id)
            self.visualizer.save_dashboard(dashboard, f"{dataset_id[:20]}_dashboard.html")
            
            prod_fig = self.visualizer.plot_production_profiles(
                results.get('production_forecast', {})
            )
            self.visualizer.save_matplotlib_figure(
                prod_fig, f"{dataset_id[:20]}_production.png"
            )
            
            econ_fig = self.visualizer.plot_economic_results(
                results.get('economic_evaluation', {})
            )
            self.visualizer.save_matplotlib_figure(
                econ_fig, f"{dataset_id[:20]}_economics.png"
            )
            
            print(f"  Visualizations generated for {dataset_id[:20]}")
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
    
    def display_comparison(self):
        if not self.results:
            return
        
        print(f"\n{'='*80}")
        print("DATASET COMPARISON")
        print(f"{'='*80}")
        
        table = PrettyTable()
        table.field_names = [
            "Dataset", "Wells", "NPV ($M)", "IRR (%)", "ROI (%)", 
            "Payback (y)", "EUR (MMbbl)", "CAPEX ($M)"
        ]
        
        table.align["Dataset"] = "l"
        table.align["Wells"] = "r"
        table.align["NPV ($M)"] = "r"
        table.align["IRR (%)"] = "r"
        table.align["ROI (%)"] = "r"
        table.align["Payback (y)"] = "r"
        table.align["EUR (MMbbl)"] = "r"
        table.align["CAPEX ($M)"] = "r"
        
        for result in self.results:
            summary = result['summary']
            
            npv_str = f"{summary['npv']/1e6:+,.2f}"
            irr_str = f"{summary['irr']:.1f}"
            roi_str = f"{summary['roi']:.1f}"
            payback_str = f"{summary['payback']:.1f}" if summary['payback'] < 99 else ">20"
            eur_str = f"{summary['eur']/1e6:,.1f}"
            capex_str = f"{summary['capex']/1e6:.1f}"
            
            table.add_row([
                result['dataset_id'][:20],
                summary['wells'],
                npv_str,
                irr_str,
                roi_str,
                payback_str,
                eur_str,
                capex_str
            ])
        
        print(table)
    
    def generate_reports(self):
        if not self.results:
            return
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'parameters': self.parameters.__dict__,
            'datasets': []
        }
        
        for result in self.results:
            dataset_report = {
                'id': result['dataset_id'],
                'summary': result['summary'],
                'metrics': self.calculate_metrics(result['results'])
            }
            report_data['datasets'].append(dataset_report)
        
        filename = f"simulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n✓ Comprehensive report saved to: {filename}")
    
    def calculate_metrics(self, results):
        econ = results.get('economic_evaluation', {})
        prod = results.get('production_forecast', {})
        kpis = results.get('key_performance_indicators', {})
        
        return {
            'npv_per_well': econ.get('npv', 0) / max(len(results.get('decline_analysis', {})), 1),
            'eur_per_well': prod.get('total_eur', 0) / max(len(results.get('decline_analysis', {})), 1),
            'capex_per_bbl': econ.get('capex', 0) / prod.get('total_eur', 1),
            'opex_per_bbl': econ.get('opex', 0) / prod.get('total_eur', 1),
            'netback_per_bbl': (econ.get('revenue', 0) - econ.get('opex', 0)) / prod.get('total_eur', 1),
            **kpis
        }
    
    def display_summary(self):
        if not self.results:
            return
        
        successful = [r for r in self.results if 'error' not in r]
        if not successful:
            return
        
        npvs = [r['summary']['npv'] for r in successful]
        irrs = [r['summary']['irr'] for r in successful]
        rois = [r['summary']['roi'] for r in successful]
        
        profitable = sum(1 for npv in npvs if npv > 0)
        avg_npv = np.mean(npvs) / 1e6
        avg_irr = np.mean(irrs)
        avg_roi = np.mean(rois)
        
        best_idx = np.argmax(npvs)
        worst_idx = np.argmin(npvs)
        
        print(f"\n{'='*80}")
        print("PROJECT SUMMARY")
        print(f"{'='*80}")
        print(f"  Total simulations: {len(self.results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Profitable projects: {profitable} ({profitable/len(successful)*100:.0f}%)")
        print(f"  Average NPV: ${avg_npv:+,.2f}M")
        print(f"  Average IRR: {avg_irr:.1f}%")
        print(f"  Average ROI: {avg_roi:.1f}%")
        
        if len(successful) >= 2:
            print(f"\n  BEST PROJECT: {successful[best_idx]['dataset_id'][:20]}")
            print(f"    NPV: ${successful[best_idx]['summary']['npv']/1e6:+,.2f}M")
            print(f"    IRR: {successful[best_idx]['summary']['irr']:.1f}%")
            
            print(f"\n  WORST PROJECT: {successful[worst_idx]['dataset_id'][:20]}")
            print(f"    NPV: ${successful[worst_idx]['summary']['npv']/1e6:+,.2f}M")
            print(f"    IRR: {successful[worst_idx]['summary']['irr']:.1f}%")
        
        print(f"\n{'='*80}")
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")

def main():
    project = ReservoirSimulationProject()
    project.run()

if __name__ == "__main__":
    main()
