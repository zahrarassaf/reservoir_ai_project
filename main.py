#!/usr/bin/env python3
"""
PhD Reservoir Simulator with REAL SPE9 Data Integration
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.real_parser import RealSPE9Parser
from src.core.physics_engine import BlackOilSimulator
from src.economics.cash_flow import EconomicAnalyzer
from src.visualization.dashboard import ReservoirDashboard

def main():
    print("=" * 70)
    print("🎯 PhD RESERVOIR SIMULATOR - REAL SPE9 ANALYSIS")
    print("=" * 70)
    
    # 1. Load REAL SPE9 data
    print("\n📥 Loading REAL SPE9 dataset...")
    parser = RealSPE9Parser(Path("data"))
    real_data = parser.get_simulation_ready_data()
    
    # Show validation results
    validation = real_data['validation']
    print(f"✅ Grid: {real_data['grid_dimensions']} = {real_data['grid_dimensions'][0]*real_data['grid_dimensions'][1]*real_data['grid_dimensions'][2]:,} cells")
    print(f"✅ Wells: {validation['well_count']} wells")
    
    if 'porosity_stats' in validation:
        stats = validation['porosity_stats']
        print(f"✅ Porosity: {stats['mean']:.3f} (min: {stats['min']:.3f}, max: {stats['max']:.3f})")
    
    # 2. Run REAL simulation
    print("\n⚡ Running REAL reservoir simulation...")
    
    # Create simulator with REAL data
    simulator = BlackOilSimulator(
        grid_dimensions=real_data['grid_dimensions'],
        porosity=real_data['porosity'],
        permeability=real_data['permeability'],
        initial_conditions=real_data['initial_conditions'],
        pvt_tables=real_data['pvt_tables']
    )
    
    # Add REAL wells
    for well in real_data['wells']:
        simulator.add_well(
            name=well['name'],
            location=(well['i'], well['j'], well['completions'][0]['k_top']),
            well_type=well['type'],
            control=well.get('control_value', 1000)
        )
    
    # Run simulation
    results = simulator.run(total_time=3650)  # 10 years
    
    print(f"✅ Simulation completed: {len(results['time'])} time steps")
    
    # 3. Economic Analysis with REAL data
    print("\n💰 Running REAL economic analysis...")
    
    analyzer = EconomicAnalyzer(
        discount_rate=0.095,
        oil_price=82.5,
        operating_cost=16.5
    )
    
    # Calculate production profile from simulation results
    production_profile = results['production']
    
    economic_results = analyzer.analyze(
        production_profile=production_profile,
        capex=len(real_data['wells']) * 3.5e6,  # $3.5M per well
        opex=2.5e6  # $2.5M annual OPEX
    )
    
    print(f"✅ NPV: ${economic_results['npv']/1e6:.2f}M")
    print(f"✅ IRR: {economic_results['irr']*100:.2f}%")
    
    # 4. Generate REAL visualizations
    print("\n📊 Generating REAL visualizations...")
    
    dashboard = ReservoirDashboard(Path("results"))
    dashboard_path = dashboard.create_dataset_dashboard(
        dataset_name="SPE9_REAL_ANALYSIS",
        physics_results=results,
        economic_results=economic_results
    )
    
    print(f"✅ Dashboard saved: {dashboard_path}")
    
    # 5. Generate comprehensive report
    print("\n📄 Generating comprehensive report...")
    
    report = {
        'simulation': {
            'grid_dimensions': real_data['grid_dimensions'],
            'total_cells': real_data['grid_dimensions'][0] * real_data['grid_dimensions'][1] * real_data['grid_dimensions'][2],
            'well_count': len(real_data['wells']),
            'simulation_time': results['time'][-1],
            'time_steps': len(results['time'])
        },
        'physics': {
            'final_pressure': float(np.mean(results['pressure'][-1])),
            'final_oil_saturation': float(np.mean(results['saturation_oil'][-1])),
            'final_water_saturation': float(np.mean(results['saturation_water'][-1])),
            'total_oil_produced': float(np.sum(results['production']['oil']))
        },
        'economics': economic_results,
        'validation': validation
    }
    
    # Save report
    import json
    with open("results/spe9_real_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 70)
    print("🎉 REAL SPE9 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"  • Grid: {report['simulation']['grid_dimensions']}")
    print(f"  • Cells: {report['simulation']['total_cells']:,}")
    print(f"  • Wells: {report['simulation']['well_count']}")
    print(f"  • Simulation: {report['simulation']['simulation_time']:.0f} days")
    print(f"  • Oil Produced: {report['physics']['total_oil_produced']/1000:.1f} Mbbl")
    print(f"  • NPV: ${report['economics']['npv']/1e6:.2f}M")
    print(f"  • IRR: {report['economics']['irr']*100:.2f}%")
    
    if not validation['is_valid']:
        print(f"\n⚠️  Validation issues: {validation['errors']}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
