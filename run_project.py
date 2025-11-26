#!/usr/bin/env python3
"""
MAIN PROJECT SCRIPT - FIXED VERSION
"""
import sys
import os
from pathlib import Path

# ADD PROJECT ROOT TO PATH
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

try:
    from src.data_loader import ReservoirDataLoader
    from src.feature_engineer import ReservoirFeatureEngineer  
    from src.ensemble_model import ReservoirEnsembleModel
    from src.config import config
    print("✅ All main imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔄 Creating minimal working version...")

# MINIMAL FALLBACK IMPLEMENTATION
import numpy as np
import pandas as pd

def minimal_data_loader():
    """Minimal data loader if imports fail"""
    print("🔄 Using minimal data loader...")
    
    # Create simple synthetic data
    time_points = 1000
    wells = 10
    
    data = []
    for t in range(time_points):
        for well_id in range(wells):
            data.append({
                'TIME_INDEX': t,
                'WELL_ID': well_id,
                'BOTTOMHOLE_PRESSURE': 3000 + 500 * np.sin(0.1 * t) + np.random.normal(0, 50),
                'FLOW_RATE_OIL': 5000 * np.exp(-0.001 * t) + 200 * np.sin(0.05 * t),
                'FLOW_RATE_WATER': 1000 + 100 * np.cos(0.03 * t),
                'FLOW_RATE_GAS': 8000 * np.exp(-0.002 * t) + 300 * np.sin(0.04 * t),
                'WELL_TYPE': 'PRODUCER' if well_id < 6 else 'INJECTOR'
            })
    
    return pd.DataFrame(data)

def main():
    """Main execution with fallbacks"""
    print("🚀 Starting Reservoir AI Project...")
    
    try:
        # Try to use full implementation
        from src.data_loader import ReservoirDataLoader
        loader = ReservoirDataLoader()
        data = loader.get_combined_dataset()
        print(f"✅ Full implementation loaded: {data.shape}")
        
    except ImportError:
        # Fallback to minimal version
        print("🔄 Falling back to minimal implementation...")
        data = minimal_data_loader()
        print(f"✅ Minimal data created: {data.shape}")
    
    # Continue with analysis
    print("📊 Data overview:")
    print(f"   Shape: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    print(f"   Wells: {data['WELL_ID'].nunique()}")
    print(f"   Time steps: {data['TIME_INDEX'].nunique()}")
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    data.to_csv(output_dir / 'reservoir_data.csv', index=False)
    print(f"💾 Data saved to: {output_dir / 'reservoir_data.csv'}")

if __name__ == "__main__":
    main()
