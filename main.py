"""
Main entry point for Reservoir Simulation Project
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import GoogleDriveLoader, create_sample_data
from simulator import ReservoirSimulator, SimulationParameters
from visualizer import ReservoirVisualizer
from analyzer import ReservoirAnalyzer
from utils import setup_logging


def run_with_google_drive():
    """Run simulation with Google Drive data"""
    print("\n🔗 Google Drive Mode")
    print("-" * 40)
    
    # Your Google Drive links
    DRIVE_LINKS = [
        "https://drive.google.com/file/d/1ZwEswptUcexDn_kqm_q8qRcHYTl1WHq2/view?usp=sharing",
        "https://drive.google.com/file/d/1sxq7sd4GSL-chE362k8wTLA_arehaD5U/view?usp=sharing",
        "https://drive.google.com/file/d/1f0aJFS99ZBVkT8IXbKdZdVihbIZIpBwZ/view?usp=sharing",
        "https://drive.google.com/file/d/1bdyUFKx-FKPy7YOlq-E9Y4nupcrhOoXi/view?usp=sharing",
        "https://drive.google.com/file/d/1n_auKzsDz5aHglQ4YvskjfHPK8ZuLBqC/view?usp=sharing",
        "https://drive.google.com/file/d/13twFcFA35CKbI8neIzIt-D54dzDd1B-N/view?usp=sharing"
    ]
    
    try:
        # Try to load from Google Drive
        print("Loading data from Google Drive...")
        loader = GoogleDriveLoader(credentials_path='credentials.json')
        data = loader.load_from_drive(DRIVE_LINKS)
        
    except Exception as e:
        print(f"Google Drive loading failed: {e}")
        print("Using sample data instead...")
        data = create_sample_data()
    
    return data


def run_with_sample_data():
    """Run simulation with sample data"""
    print("\n📊 Sample Data Mode")
    print("-" * 40)
    
    print("Creating sample reservoir data...")
    data = create_sample_data()
    
    return data


def main():
    """Main execution function"""
    print("=" * 60)
    print("RESERVOIR SIMULATION PROJECT - PhD LEVEL")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging('reservoir_simulation.log', 'INFO')
    
    # Ask for mode
    print("\nSelect mode:")
    print("1. Google Drive (requires credentials.json)")
    print("2. Sample Data")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        return
    
    if choice == '1':
        data = run_with_google_drive()
    else:
        data = run_with_sample_data()
    
    # Display data summary
    print(f"\n📦 Data Summary:")
    print(f"   • Wells: {data.production.shape[1]}")
    print(f"   • Time Points: {len(data.time)}")
    print(f"   • Layers: {len(data.petrophysical) if not data.petrophysical.empty else 0}")
    
    # Ask for simulation parameters
    print("\n⚙️ Simulation Parameters:")
    
    try:
        forecast_years = int(input("Forecast years (default: 3): ") or "3")
        oil_price = float(input("Oil price USD/bbl (default: 75.0): ") or "75.0")
        operating_cost = float(input("Operating cost USD/bbl (default: 18.0): ") or "18.0")
        
    except ValueError:
        print("Using default values...")
        forecast_years = 3
        oil_price = 75.0
        operating_cost = 18.0
    
    # Set parameters
    params = SimulationParameters(
        forecast_years=forecast_years,
        oil_price=oil_price,
        operating_cost=operating_cost
    )
    
    # Run data analysis
    print("\n🔬 Running data analysis...")
    analyzer = ReservoirAnalyzer(data)
    analysis_results = analyzer.perform_comprehensive_analysis()
    
    # Run simulation
    print("\n⚡ Running reservoir simulation...")
    simulator = ReservoirSimulator(data, params)
    results = simulator.run_comprehensive_simulation()
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    visualizer = ReservoirVisualizer(data, results)
    
    # Create dashboard
    visualizer.create_dashboard(save_path='reservoir_dashboard.png')
    
    # Create interactive dashboard
    visualizer.create_interactive_dashboard(
        save_path='reservoir_interactive.html'
    )
    
    # Export results
    print("\n💾 Exporting results...")
    export_files = simulator.export_results('./outputs')
    
    # Display results summary
    print("\n📈 RESULTS SUMMARY")
    print("-" * 40)
    
    # Material balance
    mb = results.get('material_balance', {})
    if mb:
        print(f"Material Balance:")
        print(f"  • OOIP: {mb.get('ooip_stb', 0):,.0f} STB")
    
    # Production forecast
    prod = results.get('production_forecast', {})
    if prod and 'statistics' in prod:
        stats = prod['statistics']
        print(f"\nProduction Forecast:")
        print(f"  • Peak: {stats.get('peak_production', 0):,.0f} bbl/day")
        print(f"  • Cumulative: {stats.get('total_cumulative', 0)/1e6:.1f}M bbl")
    
    # Economic analysis
    econ = results.get('economic_analysis', {})
    if econ:
        print(f"\nEconomic Analysis:")
        print(f"  • NPV: ${econ.get('npv', 0)/1e6:.2f}M")
        print(f"  • IRR: {econ.get('irr', 0)*100:.1f}%")
        if econ.get('payback_period'):
            print(f"  • Payback: {econ['payback_period']:.1f} years")
    
    # Sensitivity analysis
    sensitivity = results.get('sensitivity_analysis', {})
    if sensitivity:
        key_params = sensitivity.get('key_parameters', [])
        if key_params:
            print(f"\nSensitivity Analysis:")
            print(f"  • Key Parameters: {', '.join(key_params[:3])}")
    
    print("\n" + "=" * 60)
    print("✅ SIMULATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n📁 Generated Files:")
    print(f"  • reservoir_dashboard.png")
    print(f"  • reservoir_interactive.html")
    if export_files.get('json'):
        print(f"  • {Path(export_files['json']).name}")
    if export_files.get('csv'):
        print(f"  • {Path(export_files['csv']).name}")


if __name__ == "__main__":
    main()
