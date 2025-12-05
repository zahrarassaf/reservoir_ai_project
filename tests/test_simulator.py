"""
Tests for reservoir simulator
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_loader import create_sample_data
from simulator import ReservoirSimulator, SimulationParameters


class TestReservoirSimulator:
    """Test reservoir simulator functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return create_sample_data()
    
    @pytest.fixture
    def simulator(self, sample_data):
        """Create simulator instance"""
        params = SimulationParameters(forecast_years=2)
        return ReservoirSimulator(sample_data, params)
    
    def test_initialization(self, simulator):
        """Test simulator initialization"""
        assert simulator is not None
        assert simulator.params is not None
        assert simulator.data is not None
    
    def test_material_balance(self, simulator):
        """Test material balance analysis"""
        results = simulator._perform_material_balance()
        
        # Should return a dictionary
        assert isinstance(results, dict)
        
        # If there's enough data, should have OOIP
        if results:
            assert 'ooip_stb' in results
            assert 'regression' in results
            assert isinstance(results['ooip_stb'], float)
    
    def test_decline_analysis(self, simulator):
        """Test decline curve analysis"""
        results = simulator._perform_decline_analysis()
        
        assert isinstance(results, dict)
        
        # Should have results for each well
        if results:
            for well_data in results.values():
                assert 'exponential' in well_data
                assert 'statistics' in well_data
    
    def test_production_forecast(self, simulator):
        """Test production forecasting"""
        results = simulator._forecast_production()
        
        assert isinstance(results, dict)
        assert 'time' in results
        assert 'production' in results
        assert 'total_production' in results
        assert 'cumulative_production' in results
        
        # Check data types
        assert isinstance(results['time'], list)
        assert isinstance(results['total_production'], list)
        
        # Check lengths match
        assert len(results['time']) == len(results['total_production'])
    
    def test_economic_analysis(self, simulator):
        """Test economic analysis"""
        # First get production forecast
        forecast = simulator._forecast_production()
        
        # Then run economic analysis
        results = simulator._perform_economic_analysis(forecast)
        
        assert isinstance(results, dict)
        assert 'npv' in results
        assert 'irr' in results
        assert 'cash_flows' in results
        
        # Check data types
        assert isinstance(results['npv'], float)
        assert isinstance(results['irr'], float)
        assert isinstance(results['cash_flows'], list)
    
    def test_comprehensive_simulation(self, simulator):
        """Test comprehensive simulation"""
        results = simulator.run_comprehensive_simulation()
        
        assert isinstance(results, dict)
        
        # Check all components are present
        expected_keys = [
            'material_balance',
            'decline_analysis', 
            'production_forecast',
            'pressure_forecast',
            'economic_analysis',
            'sensitivity_analysis',
            'parameters',
            'timestamp'
        ]
        
        for key in expected_keys:
            assert key in results
