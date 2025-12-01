#!/usr/bin/env python3
"""
SPE9 Reservoir Simulation - Main Execution Script
Complete professional implementation
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from simulation_runner import SimulationRunner
from results_processor import ResultsProcessor
from data_validator import DataValidator
from analysis.performance_calculator import PerformanceCalculator
from analysis.plot_generator import PlotGenerator


def setup_logging():
    """Configure logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"spe9_simulation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def validate_input_data(logger):
    """Validate all input data before simulation"""
    logger.info("🔍 Validating input data...")
    
    validator = DataValidator()
    is_valid, messages = validator.validate_all("data")
    
    if not is_valid:
        logger.error("❌ Data validation failed!")
        for message in messages:
            if message.startswith("ERROR"):
                logger.error(message)
            else:
                logger.warning(message)
        return False
    
    logger.info("✅ All input data validated successfully")
    return True


def run_reservoir_simulation(logger):
    """Run the reservoir simulation"""
    logger.info("🚀 Starting reservoir simulation...")
    
    runner = SimulationRunner("config/simulation_config.yaml")
    
    # Get simulation info
    info = runner.get_simulation_info()
    logger.info(f"Simulator: {info['simulator']}")
    logger.info(f"Grid: {info['config']['grid']['dimensions']}")
    logger.info(f"Wells: {info['config']['wells']['total']}")
    
    # Run simulation
    success = runner.run_simulation()
    
    if not success:
        logger.error("❌ Simulation failed!")
        return False
    
    logger.info("✅ Simulation completed successfully")
    return True


def process_results(logger):
    """Process and analyze simulation results"""
    logger.info("📊 Processing simulation results...")
    
    processor = ResultsProcessor()
    
    # Load summary results
    summary_data = processor.load_summary_results()
    if summary_data.empty:
        logger.error("❌ No summary data found!")
        return False
    
    logger.info(f"Loaded {len(summary_data)} summary records")
    
    # Generate performance report
    report = processor.generate_performance_report("performance_report.txt")
    logger.info("📋 Performance report generated")
    
    # Calculate detailed metrics
    calculator = PerformanceCalculator(summary_data)
    metrics = calculator.calculate_all_metrics()
    
    metrics_df = calculator.generate_detailed_report()
    metrics_path = Path("results/analysis_results") / "detailed_metrics.csv"
    metrics_path.parent.mkdir(exist_ok=True)
    metrics_df.to_csv(metrics_path)
    logger.info(f"📈 Detailed metrics saved to: {metrics_path}")
    
    # Generate plots
    plot_generator = PlotGenerator()
    plots = plot_generator.create_all_plots(summary_data)
    
    logger.info(f"🎨 Generated {len(plots)} plots:")
    for plot_name, plot_path in plots.items():
        logger.info(f"  - {plot_name}: {plot_path}")
    
    return True


def generate_final_report(logger):
    """Generate final simulation report"""
    logger.info("📄 Generating final report...")
    
    report_content = f"""
SPE9 RESERVOIR SIMULATION - FINAL REPORT
========================================

Simulation Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROJECT OVERVIEW:
-----------------
- Project: 9th SPE Comparative Solution Project
- Grid: 24 × 25 × 15 (9,000 cells)
- Phases: Oil, Water, Gas with solution gas
- Wells: 1 injector, 25 producers
- Simulation Period: 900 days

EXECUTION SUMMARY:
------------------
- Status: COMPLETED SUCCESSFULLY
- Results Directory: results/simulation_output/
- Analysis Directory: results/analysis_results/
- Plots Directory: results/plots/

OUTPUT FILES:
-------------
1. Simulation Results:
   - SPE9.UNRST: Restart files
   - SPE9.SMSPEC: Summary data
   - SPE9.INIT: Initialization data
   - SPE9.EGRID: Grid geometry

2. Analysis Results:
   - performance_report.txt: Summary report
   - detailed_metrics.csv: All performance metrics
   - Multiple PNG plots in results/plots/

3. Logs:
   - simulation.log: Simulation execution log
   - spe9_simulation_*.log: This execution log

NEXT STEPS:
-----------
1. Review performance_report.txt for key metrics
2. Examine plots in results/plots/ directory
3. Validate results against SPE9 benchmark
4. Run sensitivity studies if needed

REFERENCES:
-----------
- Killough, J.E. (1995): "Ninth SPE Comparative Solution Project"
- SPE Comparative Solution Project Documentation

---
Report generated automatically by SPE9 Simulation System
"""
    
    report_path = Path("results") / "final_report.txt"
    with open(report_path, "w") as f:
        f.write(report_content)
    
    logger.info(f"📄 Final report saved to: {report_path}")
    return report_path


def main():
    """Main execution function"""
    # Setup
    logger = setup_logging()
    
    print("\n" + "="*60)
    print("SPE9 PROFESSIONAL RESERVOIR SIMULATION")
    print("="*60)
    
    try:
        # Step 1: Data Validation
        logger.info("🔄 STEP 1: Data Validation")
        if not validate_input_data(logger):
            sys.exit(1)
        
        # Step 2: Run Simulation
        logger.info("🔄 STEP 2: Simulation Execution")
        if not run_reservoir_simulation(logger):
            sys.exit(1)
        
        # Step 3: Results Processing
        logger.info("🔄 STEP 3: Results Analysis")
        if not process_results(logger):
            sys.exit(1)
        
        # Step 4: Final Report
        logger.info("🔄 STEP 4: Report Generation")
        report_path = generate_final_report(logger)
        
        # Success
        print("\n" + "="*60)
        print("🎉 SIMULATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📁 Results available in: results/")
        print(f"📄 Final report: {report_path}")
        print(f"📈 Plots: results/plots/")
        print(f"📋 Performance metrics: results/analysis_results/")
        print("\n" + "="*60)
        
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user")
        print("\n⚠️  Simulation interrupted")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
