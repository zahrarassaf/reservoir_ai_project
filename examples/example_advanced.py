"""
Advanced reservoir simulation example
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_loader import create_sample_data
from simulator import ReservoirSimulator, SimulationParameters
from analyzer import ReservoirAnalyzer
from economics import EconomicAnalyzer
from visualizer import ReservoirVisualizer


def main():
    """Run advanced reservoir simulation example"""
    print("=" * 60)
    print("Advanced Reservoir Simulation Example")
    print("=" * 60)
    
    # 1. Create comprehensive data
    print("\n1. Creating reservoir data...")
    data = create_sample_data()
    
    # 2. Perform data analysis
    print("\n2. Performing data analysis...")
    analyzer = ReservoirAnalyzer(data)
    analysis_results = analyzer.perform_comprehensive_analysis()
    
    print("   Data Analysis Summary:")
    print(f"   • Peak Production: {analysis_results['summary'].get('peak_production', 0):,.0f} bbl/day")
    print(f"   • OOIP: {analysis_results['summary'].get('ooip', 0):,.0f} STB")
    
    # 3. Set advanced simulation parameters
    print("\n3. Setting advanced parameters...")
    params = SimulationParameters(
        forecast_years=5,
        decline_model="hyperbolic",
        oil_price=78.0,
        operating_cost=15.0,
        discount_rate=0.11,
        compressibility=8e-6
    )
    
    # 4. Run comprehensive simulation
    print("\n4. Running comprehensive simulation...")
    simulator = ReservoirSimulator(data, params)
    simulation_results = simulator.run_comprehensive_simulation()
    
    # 5. Perform economic analysis
    print("\n5. Performing economic analysis...")
    economic_analyzer = EconomicAnalyzer()
    
    # Get production forecast
    forecast = simulation_results['production_forecast']
    production_profile = forecast['total_production']
    time_years = np.array(forecast['time']) / 365
    
    # Analyze economic scenario
    economic_results = economic_analyzer.analyze_production_scenario(
        production_profile, time_years
    )
    
    # Perform sensitivity analysis
    sensitivity_results = economic_analyzer.perform_sensitivity_analysis(
        production_profile, time_years
    )
    
    print("   Economic Analysis:")
    print(f"   • NPV: ${economic_results['npv_usd']/1e6:.2f}M")
    print(f"   • IRR: {economic_results['irr']*100:.1f}%")
    print(f"   • Key Sensitivities: {', '.join(sensitivity_results['key_parameters'][:3])}")
    
    # 6. Create comprehensive visualizations
    print("\n6. Creating comprehensive visualizations...")
    visualizer = ReservoirVisualizer(data, simulation_results)
    
    # Static dashboard
    visualizer.create_dashboard(save_path='advanced_dashboard.png')
    
    # Interactive dashboard
    visualizer.create_interactive_dashboard(
        save_path='advanced_interactive.html'
    )
    
    # 7. Export all results
    print("\n7. Exporting all results...")
    
    # Export simulation results
    sim_export = simulator.export_results('./outputs/advanced_example/simulation')
    
    # Export analysis results
    from utils import export_to_json
    export_to_json(
        analysis_results,
        './outputs/advanced_example/analysis_results.json'
    )
    
    # Export economic results
    export_to_json(
        {**economic_results, 'sensitivity': sensitivity_results},
        './outputs/advanced_example/economic_results.json'
    )
    
    print("\n✅ Advanced example completed successfully!")
    print("\nGenerated outputs:")
    print("   • advanced_dashboard.png")
    print("   • advanced_interactive.html")
    print("   • analysis_results.json")
    print("   • economic_results.json")
    print("   • simulation_results.json")


if __name__ == "__main__":
    main()
