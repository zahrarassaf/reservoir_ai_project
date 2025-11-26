#!/usr/bin/env python3
"""
MAIN PROJECT EXECUTION SCRIPT - FIXED IMPORTS
"""
import sys
import os
from pathlib import Path

# ADD PROJECT ROOT TO PYTHON PATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    # NOW THESE IMPORTS SHOULD WORK
    from src.data_loader import ReservoirDataLoader
    from src.feature_engineer import ReservoirFeatureEngineer
    from src.ensemble_model import ReservoirEnsembleModel
    from src.config import config
    from src.evaluator import ModelEvaluator
    print("✅ ALL IMPORTS SUCCESSFUL!")
    
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("🔍 Checking available modules...")
    
    # DEBUG: LIST AVAILABLE MODULES
    import src
    print("Available in src:", [x for x in dir(src) if not x.startswith('_')])
    sys.exit(1)

def main():
    """MAIN EXECUTION WITH PROPER IMPORTS"""
    print("🚀 RESERVOIR AI PROJECT - STARTING...")
    print(f"📁 Project root: {project_root}")
    
    # STEP 1: LOAD DATA
    print("\n" + "="*50)
    print("STEP 1: DATA LOADING")
    print("="*50)
    
    loader = ReservoirDataLoader()
    datasets = loader.load_all_datasets()
    combined_data = loader.get_combined_dataset()
    
    print(f"📊 Combined data shape: {combined_data.shape}")
    print(f"🎯 Target variable: FLOW_RATE_OIL")
    
    # STEP 2: FEATURE ENGINEERING
    print("\n" + "="*50)
    print("STEP 2: FEATURE ENGINEERING") 
    print("="*50)
    
    feature_engineer = ReservoirFeatureEngineer()
    X, y, feature_names, engineered_data = feature_engineer.prepare_features(combined_data)
    
    print(f"🛠️ Features created: {len(feature_names)}")
    print(f"📈 Sequence data: {X.shape}")
    
    # STEP 3: MODEL TRAINING
    print("\n" + "="*50)
    print("STEP 3: MODEL TRAINING")
    print("="*50)
    
    ensemble = ReservoirEnsembleModel()
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Train models
    ensemble.train_ensemble(X_train_flat, y_train)
    history = ensemble.train_cnn_lstm(X_train, y_train, X_test, y_test)
    
    # STEP 4: EVALUATION
    print("\n" + "="*50)
    print("STEP 4: MODEL EVALUATION")
    print("="*50)
    
    predictions = ensemble.predict_ensemble(X_test, X_test_flat)
    
    # Evaluate results
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(predictions, y_test)
    
    # STEP 5: SAVE RESULTS
    print("\n" + "="*50)
    print("STEP 5: SAVING RESULTS")
    print("="*50)
    
    ensemble.save_models()
    evaluator.save_evaluation_results(results)
    
    print("✅ PROJECT EXECUTION COMPLETED!")
    print(f"📁 Models saved to: {config.MODELS_DIR}")
    print(f"📊 Results saved to: {config.RESULTS_DIR}")

if __name__ == "__main__":
    main()
