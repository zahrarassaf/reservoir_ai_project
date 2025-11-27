#!/usr/bin/env python3
"""
PRODUCTION TRAINING SCRIPT FOR RESERVOIR AI
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
from src.config import config

def main():
    """MAIN TRAINING EXECUTION"""
    print("🚀 RESERVOIR AI - PRODUCTION TRAINING PIPELINE")
    print("=" * 70)
    
    # STEP 1: DATA PREPARATION
    print("\n📊 STEP 1: DATA PREPARATION")
    print("-" * 40)
    
    loader = ReservoirDataLoader(dataset_name='spe9')
    reservoir_data = loader.load_dataset()
    
    print(f"📈 DATASET SHAPE: {reservoir_data.shape}")
    print(f"🎯 TARGET VARIABLE: OIL_RATE")
    print(f"⏰ TIME STEPS: {reservoir_data['time_index'].nunique()}")
    print(f"🛢️  WELLS: {reservoir_data['well_id'].nunique()}")
    
    # STEP 2: FEATURE ENGINEERING
    print("\n🛠️ STEP 2: FEATURE ENGINEERING")
    print("-" * 40)
    
    feature_engineer = ReservoirFeatureEngineer()
    X, y, feature_names, engineered_data = feature_engineer.prepare_features(reservoir_data)
    
    print(f"🔧 FEATURES: {len(feature_names)}")
    print(f"📊 SEQUENCES: {X.shape}")
    print(f"🎯 TARGETS: {y.shape}")
    
    # STEP 3: DATA SPLITTING
    print("\n📋 STEP 3: DATA SPLITTING")
    print("-" * 40)
    
    split_idx = int(0.7 * len(X))
    val_idx = int(0.85 * len(X))
    
    X_train, X_val, X_test = X[:split_idx], X[split_idx:val_idx], X[val_idx:]
    y_train, y_val, y_test = y[:split_idx], y[split_idx:val_idx], y[val_idx:]
    
    # FLATTEN FOR ML MODELS
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print(f"🏋️  TRAIN: {X_train.shape} ({len(y_train)} samples)")
    print(f"📏 VALIDATION: {X_val.shape} ({len(y_val)} samples)")
    print(f"🧪 TEST: {X_test.shape} ({len(y_test)} samples)")
    
    # STEP 4: MODEL TRAINING
    print("\n🤖 STEP 4: MODEL TRAINING")
    print("-" * 40)
    
    ensemble_model = AdvancedReservoirModel()
    
    # TRAIN ML ENSEMBLE
    print("🔄 TRAINING ML ENSEMBLE...")
    ensemble_model.train_ensemble(X_train_flat, y_train, X_val_flat, y_val)
    
    # TRAIN HYBRID MODEL
    print("🔄 TRAINING HYBRID CNN-LSTM...")
    training_history = ensemble_model.train_hybrid_model(X_train, y_train, X_val, y_val)
    
    # STEP 5: MODEL EVALUATION
    print("\n📊 STEP 5: MODEL EVALUATION")
    print("-" * 40)
    
    predictions = ensemble_model.predict_ensemble(X_test, X_test_flat)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    print("\n🎯 MODEL PERFORMANCE COMPARISON:")
    print("=" * 65)
    print(f"{'MODEL':<25} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'MAPE':<10}")
    print("-" * 65)
    
    performance_results = {}
    
    for model_name, y_pred in predictions.items():
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        performance_results[model_name] = {
            'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape
        }
        
        print(f"{model_name:<25} {mae:<10.1f} {rmse:<10.1f} {r2:<10.3f} {mape:<10.1f}%")
    
    # STEP 6: RESULTS SAVING
    print("\n💾 STEP 6: SAVING RESULTS")
    print("-" * 40)
    
    # SAVE MODELS
    ensemble_model.save_models()
    
    # SAVE PREDICTIONS
    predictions_df = pd.DataFrame(predictions)
    predictions_df['ACTUAL'] = y_test
    predictions_df.to_csv(config.RESULTS_DIR / 'model_predictions.csv', index=False)
    
    # SAVE PERFORMANCE METRICS
    performance_df = pd.DataFrame(performance_results).T
    performance_df.to_csv(config.RESULTS_DIR / 'model_performance.csv')
    
    # SAVE FEATURE IMPORTANCE
    if ensemble_model.feature_importance:
        feature_importance_df = pd.DataFrame(ensemble_model.feature_importance)
        feature_importance_df.to_csv(config.RESULTS_DIR / 'feature_importance.csv')
    
    # STEP 7: FINAL SUMMARY
    print("\n🏆 FINAL RESULTS SUMMARY")
    print("=" * 50)
    
    best_model_name = performance_df['R2'].idxmax()
    best_model_perf = performance_df.loc[best_model_name]
    
    print(f"🏅 BEST MODEL: {best_model_name}")
    print(f"   R² Score: {best_model_perf['R2']:.3f}")
    print(f"   MAE: {best_model_perf['MAE']:.1f} bbl/day")
    print(f"   RMSE: {best_model_perf['RMSE']:.1f} bbl/day")
    print(f"   MAPE: {best_model_perf['MAPE']:.1f}%")
    
    print(f"\n📁 RESULTS SAVED TO:")
    print(f"   📊 Performance: {config.RESULTS_DIR / 'model_performance.csv'}")
    print(f"   🔮 Predictions: {config.RESULTS_DIR / 'model_predictions.csv'}")
    print(f"   🤖 Models: {config.MODELS_DIR}")
    
    print(f"\n✅ RESERVOIR AI TRAINING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
