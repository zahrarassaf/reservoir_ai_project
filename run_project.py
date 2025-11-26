#!/usr/bin/env python3
"""
MAIN PROJECT EXECUTION - PRODUCTION READY
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# ADD PROJECT ROOT
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🚀 RESERVOIR AI - STARTING EXECUTION...")

try:
    from src.config import config
    from src.data_loader import ReservoirDataLoader
    from src.feature_engineer import ReservoirFeatureEngineer
    from src.ensemble_model import ReservoirEnsembleModel
    print("✅ IMPORTS SUCCESSFUL!")
    
except ImportError as e:
    print(f"❌ IMPORT FAILED: {e}")
    sys.exit(1)

def main():
    """MAIN EXECUTION PIPELINE"""
    
    # STEP 1: DATA GENERATION
    print("\n" + "="*60)
    print("STEP 1: GENERATING RESERVOIR DATA")
    print("="*60)
    
    loader = ReservoirDataLoader()
    
    # GENERATE ALL DATASETS
    datasets = {}
    for dataset_name in ['spe9', 'norne', 'spe10']:
        print(f"🔄 Generating {dataset_name.upper()} data...")
        datasets[dataset_name] = loader.generate_physics_based_data(dataset_name)
    
    # COMBINE FOR TRAINING
    combined_data = loader.get_combined_dataset()
    print(f"📊 COMBINED DATA: {combined_data.shape}")
    
    # STEP 2: FEATURE ENGINEERING
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)
    
    feature_engineer = ReservoirFeatureEngineer()
    X, y, feature_names, engineered_data = feature_engineer.prepare_features(combined_data)
    
    print(f"🛠️  Features: {len(feature_names)}")
    print(f"📈 Sequences: {X.shape}")
    print(f"🎯 Targets: {y.shape}")
    
    # STEP 3: MODEL TRAINING
    print("\n" + "="*60)
    print("STEP 3: ENSEMBLE MODEL TRAINING") 
    print("="*60)
    
    ensemble = ReservoirEnsembleModel()
    
    # TRAIN-TEST SPLIT
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # FLATTEN FOR TRADITIONAL ML
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print(f"📊 Train: {X_train.shape} | Test: {X_test.shape}")
    
    # TRAIN MODELS
    print("🔄 Training ML Ensemble...")
    ensemble.train_ensemble(X_train_flat, y_train)
    
    print("🔄 Training CNN-LSTM...")
    history = ensemble.train_cnn_lstm(X_train, y_train, X_test, y_test)
    
    # STEP 4: PREDICTIONS & EVALUATION
    print("\n" + "="*60)
    print("STEP 4: MODEL EVALUATION")
    print("="*60)
    
    predictions = ensemble.predict_ensemble(X_test, X_test_flat)
    
    # CALCULATE METRICS
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    print("\n📊 MODEL PERFORMANCE:")
    print("-" * 50)
    
    results = {}
    for model_name, y_pred in predictions.items():
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"🔸 {model_name:<20} | MAE: {mae:>7.1f} | RMSE: {rmse:>7.1f} | R²: {r2:>6.3f}")
    
    # STEP 5: SAVE RESULTS
    print("\n" + "="*60)
    print("STEP 5: SAVING RESULTS")
    print("="*60)
    
    # SAVE MODELS
    ensemble.save_models()
    
    # SAVE PREDICTIONS
    predictions_df = pd.DataFrame(predictions)
    predictions_df['ACTUAL'] = y_test
    predictions_df.to_csv(config.RESULTS_DIR / 'all_predictions.csv', index=False)
    
    # SAVE PERFORMANCE
    results_df = pd.DataFrame(results).T
    results_df.to_csv(config.RESULTS_DIR / 'model_performance.csv')
    
    # SAVE ENGINEERED DATA
    engineered_data.to_csv(config.DATA_PROCESSED / 'engineered_features.csv', index=False)
    
    print("💾 RESULTS SAVED:")
    print(f"   📁 Models: {config.MODELS_DIR}")
    print(f"   📊 Results: {config.RESULTS_DIR}")
    print(f"   📈 Data: {config.DATA_PROCESSED}")
    
    # FINAL SUMMARY
    best_model = results_df.loc[results_df['R2'].idxmax()]
    print(f"\n🏆 BEST MODEL: {results_df['R2'].idxmax()}")
    print(f"   R² Score: {best_model['R2']:.3f}")
    print(f"   MAE: {best_model['MAE']:.1f}")
    print(f"   RMSE: {best_model['RMSE']:.1f}")
    
    print("\n✅ RESERVOIR AI PROJECT COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
