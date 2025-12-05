"""
Unit tests for Reservoir Simulator
"""

import pytest
import numpy as np
import pandas as pd
from src.simulator import ReservoirSimulator, SimulationConfig
from src.data_loader import ReservoirDataLoader


class TestReservoirSimulator:
    """Test cases for ReservoirSimulator class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample reservoir data for testing"""
        loader = ReservoirDataLoader()
        return loader.load_synthetic_data(n_wells=3, n_layers=3, n_days=100)
    
    @pytest.fixture
    def simulator(self, sample_data):
        """Create simulator instance for testing"""
        config = SimulationConfig(forecast_years=1)
        return ReservoirSimulator(sample_data, config)
    
    def test_initialization(self, simulator):
        """Test simulator initialization"""
        assert simulator is not None
        assert simulator.config is not None
        assert simulator.data is not None
    
    def test_parameter_estimation(self, simulator):
        """Test reservoir parameter estimation"""
        params = simulator.reservoir_params
        assert params is not None
        assert hasattr(params, 'compressibility_psi_inv')
        assert hasattr(params, 'formation_volume_factor_rb_per_stb')
    
    def test_simulation_run(self, simulator):
        """Test complete simulation run"""
        results = simulator.run_simulation(forecast_years=1)
        
        assert 'time' in results
        assert 'production' in results
        assert 'pressure' in results
        assert 'economics' in results
        
        # Check data types
        assert isinstance(results['time'], np.ndarray)
        assert isinstance(results['production'], np.ndarray)
        assert isinstance(results['pressure'], np.ndarray)
        assert isinstance(results['economics'], dict)
    
    def test_production_simulation_shape(self, simulator):
        """Test production simulation output shape"""
        results = simulator.run_simulation(forecast_years=1)
        n_wells = simulator.data['n_wells']
        
        # Check shape: (time_steps, n_wells)
        assert results['production'].shape[1] == n_wells
        assert len(results['production']) == len(results['time'])
    
    def test_economic_analysis(self, simulator):
        """Test economic analysis calculations"""
        results = simulator.run_simulation(forecast_years=1)
        economics = results['economics']
        
        assert 'npv_usd' in economics
        assert 'irr' in economics
        assert 'payback_period_years' in economics
        
        # NPV should be a float
        assert isinstance(economics['npv_usd'], float)
    
    def test_sensitivity_analysis(self, simulator):
        """Test sensitivity analysis"""
        results = simulator.run_simulation(forecast_years=1)
        sensitivity = results['sensitivity']
        
        # Should have sensitivity results for key parameters
        assert 'oil_price' in sensitivity
        assert 'discount_rate' in sensitivity
        
        # Each parameter should have values and NPV results
        for param_data in sensitivity.values():
            assert 'values' in param_data
            assert 'npv_usd' in param_data
            assert 'sensitivity_index' in param_data
    
    def test_summary_generation(self, simulator):
        """Test summary dataframe generation"""
        simulator.run_simulation(forecast_years=1)
        summary = simulator.get_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert not summary.empty
        assert 'category' in summary.columns
        assert 'metric' in summary.columns
        assert 'value' in summary.columns
    
    def test_forecast_period(self, simulator):
        """Test different forecast periods"""
        for years in [1, 2, 5]:
            results = simulator.run_simulation(forecast_years=years)
            
            # Check forecast period length
            total_days = len(simulator.data['time']) + years * 365
            assert len(results['time']) <= total_days + 1  # Allow for rounding
    
    def test_error_handling(self):
        """Test error handling for invalid data"""
        # Test with empty data
        empty_data = {}
        simulator = ReservoirSimulator(empty_data)
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(Exception):
            simulator.run_simulation()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
