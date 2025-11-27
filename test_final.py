#!/usr/bin/env python3
"""
FINAL TEST SCRIPT
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("🔍 TESTING ALL IMPORTS...")

try:
    from src.config import config
    print("✅ config: OK")
    
    from src.data_loader import ReservoirDataLoader
    print("✅ data_loader: OK")
    
    from src.ensemble_model import AdvancedReservoirModel
    print("✅ ensemble_model: OK")
    
    from src.feature_engineer import ReservoirFeatureEngineer
    print("✅ feature_engineer: OK")
    
    print("🎯 ALL IMPORTS SUCCESSFUL! READY FOR TRAINING!")
    
    # تست دیتا
    loader = ReservoirDataLoader()
    data = loader.load_data()
    print(f"📊 DATA: {data.shape}")
    
except Exception as e:
    print(f"❌ IMPORT FAILED: {e}")
    import traceback
    traceback.print_exc()
