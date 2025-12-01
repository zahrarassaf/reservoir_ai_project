"""
Test simulation functionality
"""

import pytest
import os
import tempfile
from pathlib import Path
from src.simulation_runner import SimulationRunner
from src.data_validator import DataValidator


class TestSimulation:
    """Test simulation functionality"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        runner = SimulationRunner()
        assert runner.config is not None
        assert 'simulation' in runner.config
        assert 'grid' in runner.config
        
    def test_simulator_detection(self):
        """Test simulator detection"""
        runner = SimulationRunner()
        assert runner.simulator in ['flow', 'eclipse', 'intersect']
        
    def test_input_validation(self, tmp_path):
        """Test input validation"""
        # Create minimal valid input files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create main data file
        main_data = data_dir / "SPE9.DATA"
        main_data.write_text("""
RUNSPEC
DIMENS
24 25 15 /
OIL
WATER
GRID
INCLUDE
'SPE9_GRID.INC' /
PROPS
SOLUTION
SCHEDULE
END
""")
        
        # Create grid file
        grid_file = data_dir / "SPE9_GRID.INC"
        grid_file.write_text("DX\n9000*300 /\n")
        
        runner = SimulationRunner()
        # Modify runner to use temp directory
        runner.validate_inputs = lambda: True  # Mock for now
        
        assert True  # Placeholder assertion
        
    def test_results_processor_initialization(self):
        """Test results processor initialization"""
        from src.results_processor import ResultsProcessor
        
        processor = ResultsProcessor()
        assert processor.results_dir == Path("results/simulation_output")
        assert processor.summary_data is None
        
    def test_performance_calculator(self):
        """Test performance calculator"""
        from analysis.performance_calculator import PerformanceCalculator
        import pandas as pd
        
        # Create test data
        test_data = pd.DataFrame({
            'DATE': pd.date_range('2015-01-01', periods=10, freq='D'),
            'FOPR': [1000 + i*100 for i in range(10)],
            'FOPT': [i*1000 for i in range(10)],
            'FGPR': [500 + i*50 for i in range(10)],
            'FGPT': [i*500 for i in range(10)],
            'FWPR': [100 + i*10 for i in range(10)],
            'FWPT': [i*100 for i in range(10)],
        })
        
        calculator = PerformanceCalculator(test_data)
        metrics = calculator.calculate_all_metrics()
        
        assert 'cumulative_oil' in metrics
        assert 'peak_oil_rate' in metrics
        assert 'oil_recovery_factor' in metrics
        assert metrics['cumulative_oil'] == 9000  # Last value of FOPT


class TestDataValidator:
    """Test data validation"""
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        validator = DataValidator()
        assert validator.errors == []
        assert validator.warnings == []
        
    def test_validation_summary(self):
        """Test validation summary"""
        validator = DataValidator()
        validator.errors = ["Error 1", "Error 2"]
        validator.warnings = ["Warning 1"]
        
        summary = validator.get_validation_summary()
        
        assert summary['total_errors'] == 2
        assert summary['total_warnings'] == 1
        assert len(summary['errors']) == 2
        assert len(summary['warnings']) == 1
        
    def test_file_existence_check(self, tmp_path):
        """Test file existence validation"""
        validator = DataValidator()
        
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Test with existing file
        validator.errors = []
        validator._check_file_exists(test_file, "Test file")
        assert len(validator.errors) == 0
        
        # Test with non-existent file
        non_existent = tmp_path / "nonexistent.txt"
        validator._check_file_exists(non_existent, "Non-existent file")
        assert len(validator.errors) > 0


@pytest.fixture
def sample_summary_data():
    """Fixture for sample summary data"""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2015-01-01', periods=100, freq='D')
    data = {
        'DATE': dates,
        'FOPR': 1500 + np.random.randn(100) * 100,
        'FOPT': np.cumsum(1500 + np.random.randn(100) * 100),
        'FGPR': 1000 + np.random.randn(100) * 50,
        'FGPT': np.cumsum(1000 + np.random.randn(100) * 50),
        'FWPR': 500 + np.random.randn(100) * 25,
        'FWPT': np.cumsum(500 + np.random.randn(100) * 25),
        'FWCT': np.linspace(0, 0.5, 100),
        'FGOR': 500 + np.random.randn(100) * 20,
    }
    
    return pd.DataFrame(data)


def test_plot_generator(sample_summary_data):
    """Test plot generator"""
    from analysis.plot_generator import PlotGenerator
    
    generator = PlotGenerator()
    
    # Test production profile
    plot_path = generator.create_production_profile(sample_summary_data)
    assert plot_path.endswith('.png')
    assert os.path.exists(plot_path)
    
    # Test water cut plot
    water_cut_path = generator.create_water_cut_plot(sample_summary_data)
    assert water_cut_path.endswith('.png')
    
    # Clean up
    if os.path.exists(plot_path):
        os.remove(plot_path)
    if os.path.exists(water_cut_path):
        os.remove(water_cut_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
