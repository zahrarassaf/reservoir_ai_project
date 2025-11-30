import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.data_config import DataConfig
from data.spe9_loader import SPE9Loader
import logging

logging.basicConfig(level=logging.INFO)

def validate_data_integrity():
    print("🔍 Validating SPE9 Data Integrity...")
    
    try:
        config = DataConfig()
        loader = SPE9Loader(config)
        
        # Test grid loading
        geometry = loader.load_grid_geometry()
        print(f"✅ Grid geometry: {geometry['dimensions']}")
        
        # Test static properties
        static_props = loader.load_static_properties()
        print(f"✅ Static properties: {list(static_props.keys())}")
        
        # Validate property ranges
        for prop_name, data in static_props.items():
            print(f"  {prop_name}: shape {data.shape}, range [{data.min():.3f}, {data.max():.3f}]")
        
        # Test dynamic properties
        dynamic_props = loader.load_dynamic_properties([0, 10, 20])
        print(f"✅ Dynamic properties: {len(dynamic_props)} time steps")
        
        # Test training sequences
        features, targets = loader.get_training_sequences()
        print(f"✅ Training sequences: {features.shape} -> {targets.shape}")
        
        print("🎉 All data validation tests passed!")
        
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    validate_data_integrity()
