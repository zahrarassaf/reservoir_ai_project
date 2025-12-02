"""
Unit tests for Reservoir Simulation
"""

import unittest
import numpy as np
import tempfile
import json
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from run_simulation import ProfessionalReservoirSimulator, ResultsProcessor

class TestReservoirSimulation(unittest.TestCase):
    """Test reservoir simulation components."""
    
    def setUp(self):
        """Setup test environment."""
        self.simulator = ProfessionalReservoirSimulator()
    
    def test_initialization(self):
        """Test simulator initialization."""
        self.assertIsNotNone(self.simulator)
        self.assertEqual(self.simulator.grid_dim, (24, 25, 15))
        self.assertEqual(self.simulator.total_cells, 9000)
    
    def test_properties_exist(self):
        """Test that all required properties exist."""
        required_props = [
            'initial_pressure', 'porosity', 'permeability',
            'compressibility', 'oil_fvf', 'water_fvf'
        ]
        
        for prop in required_props:
            self.assertIn(prop, self.simulator.properties)
    
    def test_wells_configuration(self):
        """Test well configuration."""
        self.assertIn('PROD', self.simulator.wells)
        self.assertIn('INJ', self.simulator.wells)
        
        prod_well = self.simulator.wells['PROD']
        self.assertEqual(prod_well['type'], 'PRODUCER')
        self.assertIn('target_rate', prod_well)
        self.assertIn('pi', prod_well)
    
    def test_pvt_tables(self):
        """Test PVT tables creation."""
        pvt_tables = self.simulator.pvt_tables
        self.assertIn('oil', pvt_tables)
        self.assertIn('gas', pvt_tables)
        
        oil_table = pvt_tables['oil']
        self.assertIn('pressure', oil_table)
        self.assertIn('fvf', oil_table)
        self.assertIn('viscosity', oil_table)
        
        # Check arrays have same length
        self.assertEqual(len(oil_table['pressure']), 
                        len(oil_table['fvf']))
        self.assertEqual(len(oil_table['pressure']), 
                        len(oil_table['viscosity']))
    
    def test_simulation_run(self):
        """Test simulation execution."""
        # Run short simulation
        results = self.simulator.run_simulation(time_steps=10)
        
        # Check results structure
        required_keys = ['time', 'production', 'injection', 
                        'pressure', 'saturations', 'cumulative']
        
        for key in required_keys:
            self.assertIn(key, results)
    
    def test_results_structure(self):
        """Test results data structure."""
        results = self.simulator.run_simulation(time_steps=5)
        
        # Check time array
        self.assertEqual(len(results['time']), 5)
        
        # Check production rates
        self.assertIn('oil', results['production'])
        self.assertIn('water', results['production'])
        self.assertIn('gas', results['production'])
        
        # Check all arrays have same length
        self.assertEqual(len(results['production']['oil']), 5)
        self.assertEqual(len(results['pressure']), 5)
    
    def test_positive_pressure(self):
        """Test that pressure remains positive."""
        results = self.simulator.run_simulation(time_steps=50)
        pressures = results['pressure']
        
        # All pressures should be positive
        self.assertTrue(all(p > 0 for p in pressures), 
                       "Negative pressure detected")
    
    def test_saturation_limits(self):
        """Test that saturations stay within physical limits."""
        results = self.simulator.run_simulation(time_steps=50)
        
        oil_sat = results['saturations']['oil']
        water_sat = results['saturations']['water']
        gas_sat = results['saturations']['gas']
        
        # Check bounds
        self.assertTrue(all(0 <= s <= 1 for s in oil_sat), 
                       "Oil saturation out of bounds")
        self.assertTrue(all(0 <= s <= 1 for s in water_sat),
                       "Water saturation out of bounds")
        self.assertTrue(all(0 <= s <= 1 for s in gas_sat),
                       "Gas saturation out of bounds")
        
        # Check saturation sum (approximately 1)
        for i in range(len(oil_sat)):
            total = oil_sat[i] + water_sat[i] + gas_sat[i]
            self.assertAlmostEqual(total, 1.0, delta=0.01,
                                 msg=f"Saturation sum not 1 at step {i}")
    
    def test_cumulative_calculation(self):
        """Test cumulative production calculation."""
        results = self.simulator.run_simulation(time_steps=10)
        
        self.assertIn('cumulative', results)
        self.assertIn('oil', results['cumulative'])
        self.assertIn('water', results['cumulative'])
        self.assertIn('gas', results['cumulative'])
        self.assertIn('water_injected', results['cumulative'])
        
        # Cumulative should be non-decreasing
        oil_cum = results['cumulative']['oil']
        self.assertTrue(all(oil_cum[i] <= oil_cum[i+1] 
                          for i in range(len(oil_cum)-1)),
                       "Cumulative oil is decreasing")

class TestResultsProcessor(unittest.TestCase):
    """Test results processing and saving."""
    
    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.processor = ResultsProcessor(self.temp_dir.name)
        self.simulator = ProfessionalReservoirSimulator()
        
        # Create test results
        self.results = self.simulator.run_simulation(time_steps=5)
    
    def tearDown(self):
        """Cleanup test environment."""
        self.temp_dir.cleanup()
    
    def test_results_processing(self):
        """Test results processing."""
        output_files = self.processor.save_results(self.results, self.simulator)
        
        # Check all required files were created
        self.assertIn('json_file', output_files)
        self.assertIn('csv_file', output_files)
        self.assertIn('plot_files', output_files)
        self.assertIn('report_file', output_files)
        
        # Check files exist
        self.assertTrue(output_files['json_file'].exists())
        self.assertTrue(output_files['csv_file'].exists())
        self.assertTrue(output_files['report_file'].exists())
    
    def test_json_output(self):
        """Test JSON output structure."""
        output_files = self.processor.save_results(self.results, self.simulator)
        json_file = output_files['json_file']
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Check JSON structure
        required_sections = ['metadata', 'reservoir_properties', 
                           'well_configuration', 'simulation_results',
                           'performance_metrics']
        
        for section in required_sections:
            self.assertIn(section, data)
    
    def test_csv_output(self):
        """Test CSV output."""
        output_files = self.processor.save_results(self.results, self.simulator)
        csv_file = output_files['csv_file']
        
        # Check CSV can be loaded
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        # Check required columns
        required_columns = ['Time_days', 'Oil_Rate_STB_d', 'Water_Rate_STB_d',
                          'Pressure_psi', 'Oil_Saturation']
        
        for col in required_columns:
            self.assertIn(col, df.columns)
    
    def test_metrics_calculation(self):
        """Test performance metrics calculation."""
        metrics = self.processor._calculate_metrics(self.results)
        
        required_metrics = ['production', 'pressure', 'efficiency']
        
        for category in required_metrics:
            self.assertIn(category, metrics)
            self.assertIsInstance(metrics[category], dict)

class TestPhysicsConstraints(unittest.TestCase):
    """Test physical constraints are enforced."""
    
    def setUp(self):
        self.simulator = ProfessionalReservoirSimulator()
    
    def test_well_rates_positive(self):
        """Test that well rates are non-negative."""
        results = self.simulator.run_simulation(time_steps=20)
        
        oil_rates = results['production']['oil']
        water_rates = results['production']['water']
        gas_rates = results['production']['gas']
        inj_rates = results['injection']['water']
        
        # All rates should be non-negative
        self.assertTrue(all(r >= 0 for r in oil_rates),
                       "Negative oil production rate")
        self.assertTrue(all(r >= 0 for r in water_rates),
                       "Negative water production rate")
        self.assertTrue(all(r >= 0 for r in gas_rates),
                       "Negative gas production rate")
        self.assertTrue(all(r >= 0 for r in inj_rates),
                       "Negative injection rate")
    
    def test_bhp_constraints(self):
        """Test BHP constraints are enforced."""
        results = self.simulator.run_simulation(time_steps=50)
        
        prod_bhp = results['bhp']['producer']
        inj_bhp = results['bhp']['injector']
        
        # Producer BHP should be above minimum
        min_bhp = self.simulator.wells['PROD']['min_bhp']
        self.assertTrue(all(bhp >= min_bhp for bhp in prod_bhp),
                       f"Producer BHP below minimum {min_bhp}")
        
        # Injector BHP should be below maximum
        max_bhp = self.simulator.wells['INJ']['max_bhp']
        self.assertTrue(all(bhp <= max_bhp for bhp in inj_bhp),
                       f"Injector BHP above maximum {max_bhp}")
    
    def test_material_balance(self):
        """Test material balance consistency."""
        results = self.simulator.run_simulation(time_steps=30)
        
        # Get production and injection data
        oil_prod = np.array(results['production']['oil'])
        water_prod = np.array(results['production']['water'])
        water_inj = np.array(results['injection']['water'])
        
        # Calculate voidage replacement
        produced_voidage = (oil_prod * self.simulator.properties['oil_fvf'] +
                          water_prod * self.simulator.properties['water_fvf'])
        injected_voidage = water_inj * self.simulator.properties['water_fvf']
        
        # Net voidage should be reasonable
        net_voidage = produced_voidage - injected_voidage
        self.assertTrue(np.all(np.abs(net_voidage) < 1e6),
                       "Unreasonable net voidage")

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestReservoirSimulation)
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestResultsProcessor))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPhysicsConstraints))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
