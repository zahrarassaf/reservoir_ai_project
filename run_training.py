#!/usr/bin/env python3
"""
PRODUCTION TRAINING PIPELINE
END-TO-END MODEL TRAINING AND EVALUATION
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ADD PROJECT PATH
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import ReservoirDataLoader
from src.feature_engineer import ReservoirFeatureEngineer
from src.ensemble_model import AdvancedReservoirModel
from src.evaluator import ModelEvaluator
from src.utils import setup_directories, save_predictions
from src.config import config

def main():
    """MAIN TRAINING EXECUTION"""
    print("🚀 RESERVOIR AI - PRODUCTION TRAINING PIPELINE")
    print("=" * 70)
    
    # SETUP DIRECTORIES
    setup_directories()
    
    # STEP 1: DATA LOADING
    print("\n📊 STEP 1: DATA LOADING & PREPARATION")
    print("-" * 40)
    
    loader = ReservoirDataLoader()
    data = loader.load_data()
    
    print(f"✅ DATA LOADED: {data.shape}")
    print(f"🛢️  WELLS: {data['well_id'].nunique()}")
    print(f"⏰ TIME STEPS: {data['time_index'].nunique()}")
    print(f"🎯 TARGET: oil_rate")
    
    # STEP 2: FEATURE ENGINEERING
    print("\n🛠️ STEP 2: FEATURE ENGINEERING")
    print("-" * 40)
    
    feature_engineer = ReservoirFeatureEngineer()
    X, y, feature_names, engineered_data = feature_engineer.prepare_features(data)
    
    print(f"✅ FEATURES: {len(feature_names)}")
    print(f"📈 SEQUENCES: {X.shape}")
    print(f"🎯 TARGETS: {y.shape}")
    
    if len(X) == 0:
        print("❌ NO SEQUENCES GENERATED - CHECK DATA")
        return
    
    # STEP 3: DATA SPLITTING
    print("\n📋 STEP 3: DATA SPLITTING")
    print("-" * 40)
    
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print(f"🏋️  TRAIN: {X_train.shape} ({len(y_train)} samples)")
    print(f"🧪 TEST: {X_test.shape} ({len(y_test)} samples)")
    
    # STEP 4: MODEL TRAINING
    print("\n🤖 STEP 4: ENSEMBLE MODEL TRAINING")
    print("-" * 40)
    
    ensemble_model = AdvancedReservoirModel()
    
    # TRAIN ML ENSEMBLE
    ensemble_model.train_ensemble(X_train_flat, y_train)
    
    # TRAIN CNN-LSTM
    history = ensemble_model.train_cnn_lstm(X_train, y_train, X_test, y_test)
    
    # STEP 5: PREDICTION & EVALUATION
    print("\n📊 STEP 5: MODEL EVALUATION")
    print("-" * 40)
    
    predictions = ensemble_model.predict_ensemble(X_test, X_test_flat)
    
    evaluator = ModelEvaluator()
    results_df = evaluator.evaluate_predictions(predictions, y_test)
    evaluator.print_performance_summary(results_df)
    
    # STEP 6: SAVE RESULTS
    print("\n💾 STEP 6: SAVING RESULTS")
    print("-" * 40)
    
    # SAVE MODELS
    ensemble_model.save_models()
    
    # SAVE PREDICTIONS
    save_predictions(predictions, y_test)
    
    # SAVE PERFORMANCE
    evaluator.save_evaluation_results(results_df)
    
    # STEP 7: FINAL SUMMARY
    print("\n🏆 FINAL TRAINING SUMMARY")
    print("=" * 50)
    
    best_model = results_df.loc[results_df['r2'].idxmax()]
    print(f"🎯 BEST PERFORMANCE: {best_model['model']}")
    print(f"   R² Score: {best_model['r2']:.3f}")
    print(f"   MAE: {best_model['mae']:.1f} bbl/day")
    print(f"   RMSE: {best_model['rmse']:.1f} bbl/day")
    print(f"   MAPE: {best_model['mape']:.1f}%")
    
    print(f"\n📁 RESULTS SAVED TO:")
    print(f"   🤖 Models: {config.MODELS_DIR}")
    print(f"   📊 Results: {config.RESULTS_DIR}")
    print(f"   📈 Data: {config.DATA_PROCESSED}")
    
    print(f"\n✅ RESERVOIR AI TRAINING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
