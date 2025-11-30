#!/usr/bin/env python3

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.model_config import SPE9GridConfig, EnsembleModelConfig
from src.ensemble_model import DeepEnsembleModel

def test_setup():
    print("🧪 Testing project setup...")
    
    # Test configurations
    grid_config = SPE9GridConfig()
    model_config = EnsembleModelConfig()
    
    print("✅ Configurations loaded successfully")
    
    # Test model creation
    model = DeepEnsembleModel(model_config)
    
    # Test forward pass
    batch_size = 10
    input_dim = len(model_config.input_features)
    x = torch.randn(batch_size, input_dim)
    
    mean_pred, std_pred = model(x)
    
    print(f"✅ Model test successful!")
    print(f"   Input shape: {x.shape}")
    print(f"   Output mean shape: {mean_pred.shape}")
    print(f"   Output std shape: {std_pred.shape}")
    print(f"   Ensemble size: {len(model.models)}")
    
    # Test data parser
    from src.spe9_data_parser import SPE9DataParser
    parser = SPE9DataParser(grid_config)
    synthetic_data = parser._generate_spe9_production_data({'simulation_days': 100, 'num_wells': 5})
    print(f"✅ Data parser test successful!")
    print(f"   Generated data keys: {list(synthetic_data.keys())}")

if __name__ == "__main__":
    test_setup()
