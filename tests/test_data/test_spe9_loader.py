import pytest
import numpy as np
from pathlib import Path

from config.data_config import DataConfig
from data.spe9_loader import SPE9Loader

class TestSPE9Loader:
    def test_initialization(self, sample_configs):
        data_config = sample_configs['data_config']
        loader = SPE9Loader(data_config)
        assert loader is not None
        assert loader._grid_dims == (24, 25, 15)
        assert loader._total_cells == 9000

    def test_grid_geometry(self, sample_configs):
        data_config = sample_configs['data_config']
        loader = SPE9Loader(data_config)
        
        geometry = loader.load_grid_geometry()
        
        assert geometry['dimensions'] == (24, 25, 15)
        assert geometry['num_cells'] == 9000
        assert geometry['active_cells'] == 9000
        assert 'coordinates' in geometry
        assert 'x' in geometry['coordinates']
        assert 'y' in geometry['coordinates']
        assert 'z' in geometry['coordinates']

    def test_static_properties_fallback(self, sample_configs):
        data_config = sample_configs['data_config']
        # Modify path to force fallback
        data_config.opm_data_dir = Path("/nonexistent/path")
        
        loader = SPE9Loader(data_config)
        
        # Should raise FileNotFoundError due to validation
        with pytest.raises(FileNotFoundError):
            loader.load_static_properties()

    def test_dynamic_properties_generation(self, sample_configs):
        data_config = sample_configs['data_config']
        data_config.opm_data_dir = Path("/nonexistent/path")
        
        # Create loader without validation for testing
        loader = SPE9Loader(data_config)
        loader._validate_paths = lambda: None  # Skip validation
        
        time_steps = [0, 5, 10]
        dynamic_props = loader.load_dynamic_properties(time_steps)
        
        assert len(dynamic_props) == 3
        for time_key in [f"T{ts:03d}" for ts in time_steps]:
            assert time_key in dynamic_props
            assert 'PRESSURE' in dynamic_props[time_key]
            assert 'SWAT' in dynamic_props[time_key]
            assert 'SOIL' in dynamic_props[time_key]

    def test_training_sequences(self, sample_configs):
        data_config = sample_configs['data_config']
        data_config.sequence_length = 5
        data_config.prediction_horizon = 3
        data_config.opm_data_dir = Path("/nonexistent/path")
        
        loader = SPE9Loader(data_config)
        loader._validate_paths = lambda: None  # Skip validation
        
        features, targets = loader.get_training_sequences()
        
        # Check shapes
        assert len(features.shape) == 3  # (num_sequences, seq_len, num_features)
        assert len(targets.shape) == 3   # (num_sequences, pred_horizon, num_targets)
        
        assert features.shape[1] == 5   # sequence_length
        assert targets.shape[1] == 3    # prediction_horizon
        assert features.shape[2] == 8   # 5 static + 3 dynamic features
        assert targets.shape[2] == 3    # 3 dynamic targets

    def test_property_ranges(self, sample_configs):
        data_config = sample_configs['data_config']
        data_config.opm_data_dir = Path("/nonexistent/path")
        
        loader = SPE9Loader(data_config)
        loader._validate_paths = lambda: None
        
        static_props = loader.load_static_properties()
        
        # Test porosity range
        poro = static_props['PORO']
        assert 0.0 <= poro.min() <= poro.max() <= 1.0
        
        # Test permeability positive
        for perm_key in ['PERMX', 'PERMY', 'PERMZ']:
            perm = static_props[perm_key]
            assert perm.min() >= 0.0

    def test_eclipse_array_parsing_error(self, sample_configs):
        data_config = sample_configs['data_config']
        loader = SPE9Loader(data_config)
        
        # Test with non-existent file
        with pytest.raises(Exception):
            loader._parse_eclipse_array(Path("/nonexistent/file"), "TEST")
