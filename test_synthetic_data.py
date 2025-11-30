import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.data_config import DataConfig
from data.spe9_loader import SPE9Loader
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

def test_synthetic_loader():
    print("Testing Synthetic SPE9 Loader...")
    
    config = DataConfig()
    loader = SPE9Loader(config)
    
    # Test grid
    geometry = loader.load_grid_geometry()
    print(f"Grid: {geometry['dimensions']}")
    
    # Test static properties
    static_props = loader.load_static_properties()
    print(f"Static properties: {list(static_props.keys())}")
    print(f"PORO range: {static_props['PORO'].min():.3f} - {static_props['PORO'].max():.3f}")
    print(f"PERMX range: {static_props['PERMX'].min():.1f} - {static_props['PERMX'].max():.1f}")
    
    # Test dynamic properties
    dynamic_props = loader.load_dynamic_properties(time_steps=[0, 5, 10])
    print(f"Time steps: {list(dynamic_props.keys())}")
    
    # Test training sequences
    features, targets = loader.get_training_sequences()
    print(f"Training data: {features.shape} -> {targets.shape}")

if __name__ == "__main__":
    test_synthetic_loader()
