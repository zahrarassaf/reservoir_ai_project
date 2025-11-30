import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.data_config import DataConfig
from data.spe9_loader import SPE9Loader
import logging

logging.basicConfig(level=logging.INFO)

def test_real_loader():
    print("Testing REAL SPE9 Data Loader...")
    
    config = DataConfig()
    
    try:
        loader = SPE9Loader(config)
        
        # Test with real data
        geometry = loader.load_grid_geometry()
        print(f"Grid dimensions: {geometry['dimensions']}")
        
        static_props = loader.load_static_properties()
        print(f"Loaded properties: {list(static_props.keys())}")
        
        # Show real data statistics
        for prop_name, data in static_props.items():
            if hasattr(data, 'shape'):
                print(f"{prop_name}: shape {data.shape}, "
                      f"range [{data.min():.3f}, {data.max():.3f}], "
                      f"mean {data.mean():.3f}")
        
        # Test dynamic data
        dynamic_props = loader.load_dynamic_properties([0, 10, 20])
        print(f"Dynamic time steps: {list(dynamic_props.keys())}")
        
        # Test training data
        features, targets = loader.get_training_sequences()
        print(f"Training sequences: {features.shape} -> {targets.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure OPM data is cloned: git clone https://github.com/OPM/opm-data.git")

if __name__ == "__main__":
    test_real_loader()
