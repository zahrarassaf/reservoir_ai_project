"""
Tests for data loader
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_loader import create_sample_data, ReservoirData


class TestDataLoader:
    """Test data loader functionality"""
    
    def test_create_sample_data(self):
        """Test sample data creation"""
        data = create_sample_data()
        
        assert isinstance(data, ReservoirData)
        assert isinstance(data.time, np.ndarray)
        assert isinstance(data.production, pd.DataFrame)
        assert isinstance(data.pressure, np.ndarray)
        assert isinstance(data.petrophysical, pd.DataFrame)
        
        assert len(data.time) > 0
        assert len(data.production.columns) > 0
        assert len(data.pressure) > 0
        assert len(data.petrophysical) > 0
    
    def test_reservoir_data_structure(self):
        """Test ReservoirData structure"""
        data = create_sample_data()
        
        # Check time and production alignment
        assert len(data.time) == len(data.production)
        
        # Check production data
        assert data.production.shape[1] > 0  # At least one well
        assert np.all(data.production.values >= 0)  # No negative production
        
        # Check pressure data
        assert np.all(data.pressure > 0)  # Positive pressure
        assert len(data.pressure) == len(data.time)
    
    def test_petrophysical_data(self):
        """Test petrophysical data"""
        data = create_sample_data()
        
        if not data.petrophysical.empty:
            required_columns = ['porosity', 'permeability', 'netthickness', 'watersaturation']
            
            for col in required_columns:
                if col in data.petrophysical.columns:
                    values = data.petrophysical[col].values
                    assert np.all(values >= 0)  # Non-negative values
                    assert np.all(~np.isnan(values))  # No NaN values
