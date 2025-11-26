#!/usr/bin/env python3
"""
FINAL IMPORT TEST - NO MORE ERRORS!
"""
import sys
import os
from pathlib import Path

print("🔍 FINAL IMPORT DEBUG...")
print(f"Python: {sys.executable}")
print(f"Workdir: {os.getcwd()}")

# ADD PROJECT ROOT
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
print(f"Project: {project_root}")

# LIST ALL FILES
print("\n📁 PROJECT STRUCTURE:")
for item in project_root.rglob('*.py'):
    print(f"  {item.relative_to(project_root)}")

print("\n🔧 TESTING IMPORTS...")

# TEST EACH MODULE
modules_to_test = [
    'src.config',
    'src.data_loader', 
    'src.feature_engineer',
    'src.ensemble_model',
    'src.evaluator',
    'src.cnn_lstm_model'
]

for module in modules_to_test:
    try:
        __import__(module)
        print(f"✅ {module}")
    except Exception as e:
        print(f"❌ {module}: {e}")

print("\n🚀 TESTING MAIN IMPORTS...")
try:
    from src.config import config
    from src.data_loader import ReservoirDataLoader
    from src.feature_engineer import ReservoirFeatureEngineer
    from src.ensemble_model import ReservoirEnsembleModel
    print("✅ ALL MAIN IMPORTS SUCCESSFUL!")
    
    # TEST INSTANTIATION
    loader = ReservoirDataLoader()
    feature_engineer = ReservoirFeatureEngineer() 
    ensemble = ReservoirEnsembleModel()
    print("✅ ALL CLASSES INSTANTIATED!")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
