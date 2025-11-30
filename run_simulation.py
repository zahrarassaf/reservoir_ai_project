#!/usr/bin/env python3
"""
SPE9 Simulation Runner
Professional reservoir simulation management script
"""

import os
import subprocess
import sys
from datetime import datetime

class SPE9Simulation:
    def __init__(self, data_file="SPE9.DATA"):
        self.data_file = data_file
        self.simulator = "flow"  # Change to "eclipse" if using Schlumberger Eclipse
        self.results_dir = "RESULTS"
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directory structure"""
        os.makedirs(self.results_dir, exist_ok=True)
        print("📁 Project structure verified")
        
    def check_prerequisites(self):
        """Verify all required files exist"""
        required_files = [
            self.data_file,
            "SPE9_GRID.INC",
            "SPE9_PORO.INC", 
            "SPE9_SATURATION_TABLES.INC",
            "TOPSVALUES.DATA",
            "PERMVALUES.DATA"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
                
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            return False
            
        print("✅ All required files present")
        return True
        
    def run_simulation(self):
        """Execute the reservoir simulation"""
        if not self.check_prerequisites():
            sys.exit(1)
            
        print(f"🚀 Starting SPE9 simulation at {datetime.now()}")
        print(f"📊 Using data file: {self.data_file}")
        print(f"⚙️  Simulator: {self.simulator}")
        
        try:
            # Run simulation
            cmd = [self.simulator, self.data_file]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print("✅ Simulation completed successfully")
            print(f"📈 Results saved in: {self.results_dir}")
            
            # Save simulation output
            with open(f"{self.results_dir}/simulation_log.txt", "w") as f:
                f.write(result.stdout)
                
            return True
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Simulation failed with error: {e}")
            print(f"Error output: {e.stderr}")
            return False
        except FileNotFoundError:
            print(f"❌ Simulator '{self.simulator}' not found in PATH")
            return False
            
    def generate_report(self):
        """Generate basic simulation report"""
        report = f"""
SPE9 Simulation Report
======================
Date: {datetime.now()}
Data File: {self.data_file}
Simulator: {self.simulator}

Reservoir Characteristics:
- Grid: 24 x 25 x 15 (9,000 cells)
- Phases: Oil, Water, Gas (with solution gas)
- Wells: 1 injector, 25 producers
- Simulation Period: 900 days

Production Strategy:
- Phase 1 (0-300 days): All producers at 1500 STB/day
- Phase 2 (300-360 days): Rate reduction to 100 STB/day  
- Phase 3 (360-900 days): Return to 1500 STB/day

Injection:
- Water injection at 5000 STB/day
- Maximum BHP: 4000 psi

Expected Outputs:
- Field production rates (oil, gas, water)
- Well performance data
- Pressure and saturation distributions
- Recovery factors and production profiles
"""
        
        with open(f"{self.results_dir}/simulation_report.txt", "w") as f:
            f.write(report)
            
        print("📋 Simulation report generated")

def main():
    """Main execution function"""
    print("🎯 SPE9 Professional Reservoir Simulation")
    print("=" * 50)
    
    simulation = SPE9Simulation()
    
    # Generate report
    simulation.generate_report()
    
    # Run simulation
    success = simulation.run_simulation()
    
    if success:
        print("\n🎉 Simulation completed!")
        print("📊 Check RESULTS directory for outputs")
    else:
        print("\n💥 Simulation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
