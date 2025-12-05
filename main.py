"""
Reservoir Simulation Project - Main Execution File
PhD-level reservoir simulation with Google Drive integration
"""

import logging
from pathlib import Path
import sys
from typing import List

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_loader import GoogleDriveReservoirLoader
from src.simulator import AdvancedReservoirSimulator
from src.visualization import ReservoirDashboard
from src.report_generator import SimulationReport


def setup_logging():
    """Setup professional logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('reservoir_simulation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_google_drive_data(links: List[str]):
    """Load data from Google Drive"""
    logger = logging.getLogger(__name__)
    logger.info("Initializing Google Drive data loader...")
    
    # Initialize loader
    loader = GoogleDriveReservoirLoader(
        credentials_path='credentials.json',  # You need to provide this
        download_dir='./data/raw'
    )
    
    # Load data
    data = loader.load_from_drive(links)
    
    logger.info(f"Data loaded: {len(data.get('time', []))} time points, "
                f"{data.get('n_wells', 0)} wells")
    return data


def run_simulation(data, forecast_years: int = 3):
    """Run comprehensive reservoir simulation"""
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing simulator with {forecast_years}-year forecast...")
    
    # Initialize simulator
    simulator = AdvancedReservoirSimulator(data)
    
    # Run simulation
    results = simulator.run_comprehensive_analysis(
        forecast_years=forecast_years,
        include_economics=True,
        include_sensitivity=True
    )
    
    logger.info("Simulation completed successfully")
    return results


def generate_outputs(data, results):
    """Generate all outputs and visualizations"""
    logger = logging.getLogger(__name__)
    
    # Create outputs directory
    output_dir = Path('./outputs')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Generate dashboard
    logger.info("Generating visualization dashboard...")
    dashboard = ReservoirDashboard(data, results)
    dashboard.create_comprehensive_dashboard(
        save_path=output_dir / 'reservoir_dashboard.png'
    )
    
    # 2. Generate interactive dashboard
    dashboard.create_interactive_dashboard(
        save_path=output_dir / 'interactive_dashboard.html'
    )
    
    # 3. Generate report
    logger.info("Generating comprehensive report...")
    report = SimulationReport(data, results)
    report.generate_pdf_report(
        output_path=output_dir / 'simulation_report.pdf'
    )
    
    # 4. Export results
    logger.info("Exporting results...")
    from src.utils import export_results
    export_results(results, output_dir / 'simulation_results')
    
    logger.info(f"All outputs saved to {output_dir}")


def main():
    """Main execution function"""
    print("=" * 70)
    print("ADVANCED RESERVOIR SIMULATION - PhD LEVEL")
    print("=" * 70)
    
    # Setup logging
    logger = setup_logging()
    
    # Google Drive links
    DRIVE_LINKS = [
        "https://drive.google.com/file/d/1ZwEswptUcexDn_kqm_q8qRcHYTl1WHq2/view?usp=sharing",
        "https://drive.google.com/file/d/1sxq7sd4GSL-chE362k8wTLA_arehaD5U/view?usp=sharing",
        "https://drive.google.com/file/d/1f0aJFS99ZBVkT8IXbKdZdVihbIZIpBwZ/view?usp=sharing",
        "https://drive.google.com/file/d/1bdyUFKx-FKPy7YOlq-E9Y4nupcrhOoXi/view?usp=sharing",
        "https://drive.google.com/file/d/1n_auKzsDz5aHglQ4YvskjfHPK8ZuLBqC/view?usp=sharing",
        "https://drive.google.com/file/d/13twFcFA35CKbI8neIzIt-D54dzDd1B-N/view?usp=sharing"
    ]
    
    try:
        # Step 1: Load data
        logger.info("Step 1/4: Loading data from Google Drive...")
        data = load_google_drive_data(DRIVE_LINKS)
        
        # Step 2: Run simulation
        logger.info("Step 2/4: Running reservoir simulation...")
        results = run_simulation(data, forecast_years=3)
        
        # Step 3: Generate outputs
        logger.info("Step 3/4: Generating outputs...")
        generate_outputs(data, results)
        
        # Step 4: Display summary
        logger.info("Step 4/4: Displaying results summary...")
        display_summary(results)
        
        print("\n" + "=" * 70)
        print("✅ PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


def display_summary(results):
    """Display key results summary"""
    print("\n📊 SIMULATION RESULTS SUMMARY")
    print("-" * 40)
    
    # Economic results
    if 'economics' in results:
        econ = results['economics']
        print(f"💰 Economic Analysis:")
        print(f"   NPV: ${econ.get('npv', 0)/1e6:.2f}M")
        print(f"   IRR: {econ.get('irr', 0)*100:.1f}%")
        if econ.get('payback_period'):
            print(f"   Payback: {econ['payback_period']:.1f} years")
    
    # Production results
    if 'production' in results:
        prod = results['production']
        print(f"🛢️ Production Forecast:")
        print(f"   Peak: {prod.get('peak_rate', 0):,.0f} bbl/day")
        print(f"   Cumulative: {prod.get('cumulative', 0)/1e6:.1f}M bbl")
    
    # Reservoir results
    if 'reservoir' in results:
        res = results['reservoir']
        print(f"🏭 Reservoir Analysis:")
        print(f"   OOIP: {res.get('ooip', 0)/1e6:.1f}M STB")
        print(f"   Recovery Factor: {res.get('recovery_factor', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
