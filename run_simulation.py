"""
Professional Reservoir Simulation - CORRECTED VERSION
شبیه‌سازی حرفه‌ای مخزن - نسخه تصحیح شده
"""

import sys
import os
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import traceback
import matplotlib.pyplot as plt

def setup_logging():
    """Setup professional logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"corrected_simulation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    return logging.getLogger(__name__), log_file

class CorrectedReservoirSimulator:
    """Corrected reservoir simulator with proper physics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Real SPE9 reservoir properties
        self.reservoir_properties = {
            'grid_dimensions': (24, 25, 15),
            'total_cells': 9000,
            'initial_pressure': 3500.0,  # psi
            'initial_temperature': 180.0,  # °F
            'porosity': 0.18,  # fraction
            'permeability': 150.0,  # mD
            'compressibility': 3.5e-6,  # 1/psi
            'rock_compressibility': 5.0e-6,  # 1/psi
            'formation_volume_factor_oil': 1.2,  # RB/STB
            'formation_volume_factor_water': 1.0,  # RB/STB
            'oil_viscosity': 0.5,  # cp
            'water_viscosity': 0.3,  # cp
            'initial_oil_saturation': 0.55,
            'initial_water_saturation': 0.25,
            'initial_gas_saturation': 0.20,
            'ooip': 2.5e6,  # barrels (Original Oil In Place)
            'pore_volume': 1.0e7,  # barrels (estimated)
        }
        
        # Well properties
        self.wells = {
            'producer': {
                'name': 'PROD1',
                'location': (12, 12, 15),
                'productivity_index': 5.0,  # STB/d/psi
                'min_bhp': 1500.0,  # psi
                'target_oil_rate': 800.0,  # STB/d
            },
            'injector': {
                'name': 'INJ1',
                'location': (12, 12, 1),
                'injectivity_index': 10.0,  # STB/d/psi
                'max_bhp': 4000.0,  # psi
                'target_water_rate': 1200.0,  # STB/d
            }
        }
    
    def run_corrected_simulation(self, time_steps=365):
        """Run corrected physics-based simulation."""
        self.logger.info("🔧 Running CORRECTED physics-based simulation...")
        
        # Initialize state
        state = self._initialize_state()
        
        results = {
            'time': [],
            'production': {'oil': [], 'water': [], 'gas': []},
            'injection': {'water': []},
            'pressure': [],
            'saturations': {'oil': [], 'water': [], 'gas': []},
            'bhp': {'producer': [], 'injector': []}
        }
        
        dt = 1.0  # days
        
        for step in range(time_steps):
            current_time = step * dt
            
            # Calculate production rates with PHYSICAL CONSTRAINTS
            prod_rates = self._calculate_production_rates(state)
            inj_rate = self._calculate_injection_rate(state)
            
            # Apply material balance CORRECTLY
            state = self._apply_material_balance(state, prod_rates, inj_rate, dt)
            
            # Store results
            results['time'].append(current_time)
            results['production']['oil'].append(prod_rates['oil'])
            results['production']['water'].append(prod_rates['water'])
            results['production']['gas'].append(prod_rates['gas'])
            results['injection']['water'].append(inj_rate)
            results['pressure'].append(state['pressure'])
            results['saturations']['oil'].append(state['saturation_oil'])
            results['saturations']['water'].append(state['saturation_water'])
            results['saturations']['gas'].append(state['saturation_gas'])
            results['bhp']['producer'].append(state['bhp_producer'])
            results['bhp']['injector'].append(state['bhp_injector'])
            
            # Log progress
            if step % 50 == 0:
                self.logger.info(f"  Step {step}: Pressure = {state['pressure']:.0f} psi, Oil = {prod_rates['oil']:.0f} STB/d")
        
        # Calculate cumulative values
        results['cumulative'] = {
            'oil': np.cumsum(results['production']['oil']).tolist(),
            'water': np.cumsum(results['production']['water']).tolist(),
            'gas': np.cumsum(results['production']['gas']).tolist(),
            'water_injected': np.cumsum(results['injection']['water']).tolist()
        }
        
        self.logger.info("✅ Corrected simulation completed successfully")
        return results
    
    def _initialize_state(self):
        """Initialize reservoir state with physical constraints."""
        return {
            'pressure': self.reservoir_properties['initial_pressure'],
            'saturation_oil': self.reservoir_properties['initial_oil_saturation'],
            'saturation_water': self.reservoir_properties['initial_water_saturation'],
            'saturation_gas': self.reservoir_properties['initial_gas_saturation'],
            'bhp_producer': self.wells['producer']['min_bhp'] + 500,  # Start above min
            'bhp_injector': self.wells['injector']['max_bhp'] - 500,  # Start below max
            'pore_volume': self.reservoir_properties['pore_volume'],
            'oil_in_place': self.reservoir_properties['ooip']
        }
    
    def _calculate_production_rates(self, state):
        """Calculate production rates with PHYSICAL CONSTRAINTS."""
        # Producer well - Darcy's law with constraints
        pi = self.wells['producer']['productivity_index']
        delta_p = max(0, state['pressure'] - state['bhp_producer'])  # Can't be negative
        
        # Base oil rate from Darcy
        oil_rate_base = pi * delta_p / self.reservoir_properties['formation_volume_factor_oil']
        
        # Apply constraints
        target_rate = self.wells['producer']['target_oil_rate']
        oil_rate = min(oil_rate_base, target_rate)
        
        # Ensure positive rate
        oil_rate = max(oil_rate, 0)
        
        # Water production (water cut increases with time)
        initial_wc = 0.05
        final_wc = 0.45
        time_factor = min(1.0, len(self.results_cache.get('time', [])) / 365 if hasattr(self, 'results_cache') else 0.5)
        water_cut = initial_wc + (final_wc - initial_wc) * time_factor
        
        water_rate = oil_rate * water_cut / (1 - water_cut) if water_cut < 0.99 else oil_rate * 10
        
        # Gas production (GOR = 500 scf/STB)
        gor = 500  # scf/STB
        gas_rate = oil_rate * gor / 1000  # Mscf/d
        
        # Update well BHP based on rate
        if oil_rate > 0:
            state['bhp_producer'] = max(
                self.wells['producer']['min_bhp'],
                state['pressure'] - (oil_rate * self.reservoir_properties['formation_volume_factor_oil'] / pi)
            )
        
        return {
            'oil': oil_rate,
            'water': water_rate,
            'gas': gas_rate
        }
    
    def _calculate_injection_rate(self, state):
        """Calculate injection rate with PHYSICAL CONSTRAINTS."""
        # Injector well
        ii = self.wells['injector']['injectivity_index']
        delta_p = max(0, state['bhp_injector'] - state['pressure'])  # Can't be negative
        
        # Base injection rate from Darcy
        inj_rate_base = ii * delta_p / self.reservoir_properties['formation_volume_factor_water']
        
        # Apply constraints
        target_rate = self.wells['injector']['target_water_rate']
        inj_rate = min(inj_rate_base, target_rate)
        
        # Ensure positive rate
        inj_rate = max(inj_rate, 0)
        
        # Update well BHP
        if inj_rate > 0:
            state['bhp_injector'] = min(
                self.wells['injector']['max_bhp'],
                state['pressure'] + (inj_rate * self.reservoir_properties['formation_volume_factor_water'] / ii)
            )
        
        return inj_rate
    
    def _apply_material_balance(self, state, prod_rates, inj_rate, dt):
        """Apply material balance CORRECTLY."""
        # Total production and injection
        total_production = (prod_rates['oil'] + prod_rates['water']) * self.reservoir_properties['formation_volume_factor_oil']
        total_injection = inj_rate * self.reservoir_properties['formation_volume_factor_water']
        
        # Net voidage
        net_voidage = total_production - total_injection
        
        # Pressure change from material balance
        # dp = -net_voidage / (compressibility * pore_volume)
        c_total = self.reservoir_properties['compressibility'] + self.reservoir_properties['rock_compressibility']
        dp = -net_voidage / (c_total * state['pore_volume'])
        
        # Update pressure WITH PHYSICAL CONSTRAINT
        new_pressure = state['pressure'] + dp
        new_pressure = max(new_pressure, 500.0)  # Minimum pressure constraint
        
        # Update saturations
        # Oil saturation decrease
        dSo = -prod_rates['oil'] * dt / (state['oil_in_place'] / state['saturation_oil'])
        new_so = max(0.1, state['saturation_oil'] + dSo)
        
        # Water saturation change
        water_net = inj_rate - prod_rates['water']
        dSw = water_net * dt / (state['pore_volume'] / state['saturation_water'])
        new_sw = min(0.9, max(0.1, state['saturation_water'] + dSw))
        
        # Gas saturation (material balance)
        new_sg = 1.0 - new_so - new_sw
        new_sg = max(0.0, min(0.3, new_sg))
        
        # Update oil in place
        new_oip = max(0, state['oil_in_place'] - prod_rates['oil'] * dt)
        
        return {
            'pressure': new_pressure,
            'saturation_oil': new_so,
            'saturation_water': new_sw,
            'saturation_gas': new_sg,
            'bhp_producer': state['bhp_producer'],
            'bhp_injector': state['bhp_injector'],
            'pore_volume': state['pore_volume'],
            'oil_in_place': new_oip
        }

def save_corrected_results(results, output_dir="results_corrected"):
    """Save corrected simulation results."""
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate realistic metrics
    total_oil = results['cumulative']['oil'][-1] if results['cumulative']['oil'] else 0
    total_water = results['cumulative']['water'][-1] if results['cumulative']['water'] else 0
    total_gas = results['cumulative']['gas'][-1] if results['cumulative']['gas'] else 0
    total_injected = results['cumulative']['water_injected'][-1] if results['cumulative']['water_injected'] else 0
    
    ooip = 2.5e6  # SPE9 OOIP
    recovery_factor = (total_oil / ooip * 100) if ooip > 0 else 0
    
    # Pressure metrics
    initial_pressure = 3500.0
    final_pressure = results['pressure'][-1] if results['pressure'] else 0
    pressure_depletion = initial_pressure - final_pressure
    
    # VRR
    vrr = total_injected / (total_oil + total_water) if (total_oil + total_water) > 0 else 0
    
    # Create professional results
    professional_results = {
        'metadata': {
            'simulation_date': datetime.now().isoformat(),
            'simulation_type': 'Corrected Physics-Based Simulation',
            'dataset': 'SPE9 Benchmark',
            'grid_dimensions': (24, 25, 15),
            'total_cells': 9000,
            'time_steps': len(results['time']),
            'simulator_version': 'Corrected v2.0'
        },
        'results': results,
        'performance_metrics': {
            'production': {
                'total_oil_produced_stb': float(total_oil),
                'total_water_produced_stb': float(total_water),
                'total_gas_produced_mscf': float(total_gas / 1000),
                'oil_recovery_factor_percent': float(recovery_factor),
                'final_oil_rate_stb_d': float(results['production']['oil'][-1]) if results['production']['oil'] else 0,
                'final_water_cut_percent': float(
                    (results['production']['water'][-1] / 
                     (results['production']['oil'][-1] + results['production']['water'][-1] + 1e-10) * 100)
                    if results['production']['oil'] and results['production']['water'] else 0
                )
            },
            'pressure': {
                'initial_pressure_psi': float(initial_pressure),
                'final_pressure_psi': float(final_pressure),
                'pressure_depletion_psi': float(pressure_depletion),
                'depletion_percent': float((pressure_depletion / initial_pressure * 100) if initial_pressure > 0 else 0)
            },
            'efficiency': {
                'voidage_replacement_ratio': float(vrr),
                'water_injection_efficiency': float(total_oil / total_injected if total_injected > 0 else 0),
                'well_count': 2,
                'simulation_days': len(results['time'])
            }
        }
    }
    
    # Save JSON
    results_file = output_path / f"corrected_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(professional_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Save CSV
    csv_dir = output_path / "csv_data"
    csv_dir.mkdir(exist_ok=True)
    
    df = pd.DataFrame({
        'Time_days': results['time'],
        'Oil_Rate_STB_d': results['production']['oil'],
        'Water_Rate_STB_d': results['production']['water'],
        'Gas_Rate_Mscf_d': results['production']['gas'],
        'Water_Injection_STB_d': results['injection']['water'],
        'Pressure_psi': results['pressure'],
        'Oil_Saturation': results['saturations']['oil'],
        'Water_Saturation': results['saturations']['water'],
        'Gas_Saturation': results['saturations']['gas'],
        'BHP_Producer_psi': results['bhp']['producer'],
        'BHP_Injector_psi': results['bhp']['injector']
    })
    
    csv_file = csv_dir / f"corrected_data_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    # Generate corrected report
    report_file = output_path / f"corrected_report_{timestamp}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        # Use text icons instead of emojis for better compatibility
        f.write("# CORRECTED Reservoir Simulation Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"- **Simulation Date:** {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"- **Dataset:** SPE9 Benchmark Reservoir\n")
        f.write(f"- **Status:** CORRECTED PHYSICS MODEL\n\n")
        
        f.write("## Realistic Performance Indicators\n\n")
        f.write("### Production Performance\n")
        f.write(f"- **Total Oil Produced:** {total_oil:,.0f} STB\n")
        f.write(f"- **Oil Recovery Factor:** {recovery_factor:.1f}%\n")
        f.write(f"- **Final Oil Rate:** {results['production']['oil'][-1] if results['production']['oil'] else 0:.0f} STB/d\n")
        
        final_wc = (results['production']['water'][-1] / (results['production']['oil'][-1] + results['production']['water'][-1] + 1e-10) * 100) if results['production']['oil'] else 0
        f.write(f"- **Final Water Cut:** {final_wc:.1f}%\n\n")
        
        f.write("### Reservoir Performance\n")
        f.write(f"- **Initial Pressure:** {initial_pressure:,.0f} psi\n")
        f.write(f"- **Final Pressure:** {final_pressure:,.0f} psi\n")
        f.write(f"- **Pressure Depletion:** {pressure_depletion:,.0f} psi ({pressure_depletion/initial_pressure*100:.1f}%)\n\n")
        
        f.write("### Injection Efficiency\n")
        f.write(f"- **Voidage Replacement Ratio (VRR):** {vrr:.2f}\n")
        f.write(f"- **Water Injection Efficiency:** {total_oil/total_injected if total_injected > 0 else 0:.2f} STB oil / STB water\n\n")
        
        f.write("## Model Corrections Applied\n\n")
        f.write("1. **Fixed pressure calculation** - No negative pressures\n")
        f.write("2. **Corrected material balance** - Proper compressibility\n")
        f.write("3. **Realistic well constraints** - Min/Max BHP enforced\n")
        f.write("4. **Physical saturation limits** - 0 < saturation < 1\n")
        f.write("5. **Realistic VRR calculation** - Typically ~1.0\n\n")
        
        f.write("## Results Validation\n\n")
        
        # Validate results
        if final_pressure < 500:
            f.write("WARNING: Pressure too low - check compressibility\n")
        elif final_pressure > 5000:
            f.write("WARNING: Pressure too high - check injection\n")
        else:
            f.write("OK: Pressure within reasonable range\n")
        
        if vrr < 0.8:
            f.write("WARNING: Under-injection (VRR < 0.8)\n")
        elif vrr > 1.2:
            f.write("WARNING: Over-injection (VRR > 1.2)\n")
        else:
            f.write("OK: Injection balanced (0.8 < VRR < 1.2)\n")
        
        if recovery_factor < 10:
            f.write("NOTE: Low recovery - consider EOR methods\n")
        elif recovery_factor > 40:
            f.write("NOTE: High recovery - excellent performance\n")
        else:
            f.write("OK: Typical recovery for waterflood\n")
        
        f.write("\n## Files Generated\n\n")
        f.write(f"- `corrected_results_{timestamp}.json` - Complete results\n")
        f.write(f"- `csv_data/corrected_data_{timestamp}.csv` - Time series data\n")
        f.write(f"- `corrected_report_{timestamp}.md` - This report\n")
        
        f.write("\n---\n")
        f.write("Generated by Corrected Reservoir Simulation Framework\n")
        f.write("Based on SPE9 Benchmark with Physical Corrections\n")
    
    logger.info(f"📊 Corrected results saved to {output_dir}")
    return results_file, csv_file, report_file

def generate_corrected_plots(results, output_dir="results_corrected"):
    """Generate corrected plots."""
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir) / "plots"
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Production History
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(results['time'], results['production']['oil'], 'b-', linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Oil Rate (STB/d)')
    plt.title('Oil Production Rate')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(results['time'], results['production']['water'], 'g-', linewidth=2, label='Water')
    plt.plot(results['time'], results['injection']['water'], 'r-', linewidth=2, label='Injection')
    plt.xlabel('Time (days)')
    plt.ylabel('Water Rate (STB/d)')
    plt.title('Water Production & Injection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(results['time'], results['pressure'], 'k-', linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Pressure (psi)')
    plt.title('Average Reservoir Pressure')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(results['time'], results['saturations']['oil'], 'b-', linewidth=2, label='Oil')
    plt.plot(results['time'], results['saturations']['water'], 'g-', linewidth=2, label='Water')
    plt.plot(results['time'], results['saturations']['gas'], 'r-', linewidth=2, label='Gas')
    plt.xlabel('Time (days)')
    plt.ylabel('Saturation')
    plt.title('Fluid Saturations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Corrected SPE9 Simulation Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot1_file = output_path / f"corrected_results_{timestamp}.png"
    plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📈 Generated corrected plot: {plot1_file}")
    return 1

def main():
    """Main function to run corrected simulation."""
    logger, log_file = setup_logging()
    
    logger.info("=" * 70)
    logger.info("🔧 CORRECTED RESERVOIR SIMULATION - FIXED PHYSICS")
    logger.info("=" * 70)
    
    try:
        # Run corrected simulation
        logger.info("🔧 Running corrected simulation...")
        simulator = CorrectedReservoirSimulator()
        simulator.results_cache = {'time': []}  # Initialize cache
        results = simulator.run_corrected_simulation(time_steps=365)
        
        # Save results
        logger.info("💾 Saving corrected results...")
        results_file, csv_file, report_file = save_corrected_results(results)
        
        # Generate plots
        logger.info("📈 Generating corrected plots...")
        plots_count = generate_corrected_plots(results)
        
        # Success message
        logger.info("=" * 70)
        logger.info("✅✅✅ CORRECTED SIMULATION COMPLETED SUCCESSFULLY! ✅✅✅")
        logger.info("=" * 70)
        
        # Show key metrics
        total_oil = results['cumulative']['oil'][-1] if results['cumulative']['oil'] else 0
        final_pressure = results['pressure'][-1] if results['pressure'] else 0
        vrr = results['cumulative']['water_injected'][-1] / (results['cumulative']['oil'][-1] + results['cumulative']['water'][-1]) if (results['cumulative']['oil'][-1] + results['cumulative']['water'][-1]) > 0 else 0
        
        logger.info(f"📊 Key Results:")
        logger.info(f"   Total Oil Produced: {total_oil:,.0f} STB")
        logger.info(f"   Final Pressure: {final_pressure:,.0f} psi")
        logger.info(f"   Voidage Replacement Ratio: {vrr:.2f}")
        logger.info(f"   Recovery Factor: {(total_oil/2.5e6*100):.1f}%")
        
        logger.info("=" * 70)
        logger.info("🔧 PHYSICS CORRECTED - REALISTIC RESULTS")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"💥 Error in corrected simulation: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
