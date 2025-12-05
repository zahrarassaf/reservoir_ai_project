"""
Basic example of reservoir simulation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_loader import create_sample_data
from simulator import ReservoirSimulator, SimulationParameters
from visualizer import ReservoirVisualizer


def main():
    """Run basic reservoir simulation example"""
    print("=" * 60)
    print("Basic Reservoir Simulation Example")
    print("=" * 60)
    
    # 1. Create sample data
    print("\n1. Creating sample reservoir data...")
    data = create_sample_data()
    print(f"   Created data with {data.production.shape[1]} wells and "
          f"{len(data.time)} time points")
    
    # 2. Set simulation parameters
    print("\n2. Setting simulation parameters...")
    params = SimulationParameters(
        forecast_years=3,
        oil_price=72.0,
        operating_cost=16.0,
        discount_rate=0.10
    )
    
    # 3. Run simulation
    print("\n3. Running reservoir simulation...")
    simulator = ReservoirSimulator(data, params)
    results = simulator.run_comprehensive_simulation()
    
    # 4. Display results
    print("\n4. Simulation Results:")
    print("-" * 40)
    
    # Material balance
    mb = results.get('material_balance', {})
    if 'ooip_stb' in mb:
        print(f"   Material Balance:")
        print(f"   • OOIP: {mb['ooip_stb']:,.0f} STB")
        print(f"   • R²: {mb['regression']['r_squared']:.3f}")
    
    # Economic analysis
    econ = results.get('economic_analysis', {})
    if 'npv' in econ:
        print(f"\n   Economic Analysis:")
        print(f"   • NPV: ${econ['npv']/1e6:.2f}M")
        print(f"   • IRR: {econ['irr']*100:.1f}%")
        if econ.get('payback_period'):
            print(f"   • Payback: {econ['payback_period']:.1f} years")
    
    # Production forecast
    prod = results.get('production_forecast', {})
    if 'statistics' in prod:
        stats = prod['statistics']
        print(f"\n   Production Forecast:")
        print(f"   • Peak: {stats['peak_production']:,.0f} bbl/day")
        print(f"   • Cumulative: {stats['total_cumulative']/1e6:.1f}M bbl")
    
    # 5. Create visualizations
    print("\n5. Creating visualizations...")
    visualizer = ReservoirVisualizer(data, results)
    visualizer.create_dashboard(save_path='basic_simulation_dashboard.png')
    
    # 6. Export results
    print("\n6. Exporting results...")
    export_files = simulator.export_results('./outputs/basic_example')
    
    print("\n✅ Example completed successfully!")
    print(f"\nOutput files:")
    for file_type, file_path in export_files.items():
        if file_path:
            print(f"   • {file_type}: {file_path}")


if __name__ == "__main__":
    main()
