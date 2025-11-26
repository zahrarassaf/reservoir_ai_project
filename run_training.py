#!/usr/bin/env python3
"""
MAIN TRAINING SCRIPT FOR RESERVOIR AI PROJECT
PRODUCTION-READY IMPLEMENTATION
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ADD SRC TO PATH
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_loader import ReservoirDataLoader
from src.feature_engineer import ReservoirFeatureEngineer
from src.ensemble_model import ReservoirEnsembleModel
from src.config import config

def main():
    """MAIN TRAINING EXECUTION"""
    print("🚀 STARTING RESERVOIR AI TRAINING PIPELINE...")
    
    # STEP 1: LOAD DATA
    print("\n" + "="*50)
    print("STEP 1: DATA LOADING & PREPARATION")
    print("="*50)
    
    data_loader = ReservoirDataLoader()
    datasets = data_loader.load_all_datasets()
    
    # COMBINE ALL DATASETS FOR ROBUST TRAINING
    combined_data = data_loader.get_combined_dataset()
    print(f"📊 COMBINED DATASET SHAPE: {combined_data.shape}")
    
    # STEP 2: FEATURE ENGINEERING
    print("\n" + "="*50)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*50)
    
    feature_engineer = ReservoirFeatureEngineer()
    X, y, feature_names, engineered_data = feature_engineer.prepare_features(combined_data)
    
    print(f"🎯 FEATURES: {len(feature_names)}")
    print(f"📈 SEQUENCES: {X.shape}")
    print(f"🎯 TARGETS: {y.shape}")
    
    # STEP 3: TRAIN-TEST SPLIT
    print("\n" + "="*50)
    print("STEP 3: DATA SPLITTING")
    print("="*50)
    
    split_idx = int(0.8 * len(X))
    
    X_train_seq, X_test_seq = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # FLATTEN SEQUENCES FOR TRADITIONAL ML
    X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
    X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)
    
    print(f"📊 TRAIN SET: {X_train_seq.shape} | TEST SET: {X_test_seq.shape}")
    
    # STEP 4: MODEL TRAINING
    print("\n" + "="*50)
    print("STEP 4: MODEL TRAINING")
    print("="*50)
    
    ensemble_model = ReservoirEnsembleModel()
    
    # TRAIN ML ENSEMBLE
    ensemble_model.train_ensemble(X_train_flat, y_train, X_test_flat, y_test)
    
    # TRAIN CNN-LSTM
    history = ensemble_model.train_cnn_lstm(X_train_seq, y_train, X_test_seq, y_test)
    
    # STEP 5: PREDICTION & EVALUATION
    print("\n" + "="*50)
    print("STEP 5: MODEL EVALUATION")
    print("="*50)
    
    predictions = ensemble_model.predict_ensemble(X_test_seq, X_test_flat)
    
    # CALCULATE PERFORMANCE METRICS
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    print("\n📊 MODEL PERFORMANCE COMPARISON:")
    print("-" * 40)
    
    performance_results = {}
    
    for model_name, y_pred in predictions.items():
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        performance_results[model_name] = {
            'MAE': mae,
            'RMSE': rmse, 
            'R2': r2
        }
        
        print(f"🔹 {model_name.upper():<20} | MAE: {mae:>8.2f} | RMSE: {rmse:>8.2f} | R²: {r2:>6.3f}")
    
    # STEP 6: SAVE MODELS & RESULTS
    print("\n" + "="*50)
    print("STEP 6: SAVING RESULTS")
    print("="*50)
    
    ensemble_model.save_models()
    
    # SAVE PERFORMANCE RESULTS
    results_df = pd.DataFrame(performance_results).T
    results_df.to_csv(config.RESULTS_DIR / 'model_performance.csv')
    
    # SAVE PREDICTIONS
    predictions_df = pd.DataFrame(predictions)
    predictions_df['ACTUAL'] = y_test
    predictions_df.to_csv(config.RESULTS_DIR / 'predictions.csv')
    
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📁 MODELS SAVED TO: {config.MODELS_DIR}")
    print(f"📊 RESULTS SAVED TO: {config.RESULTS_DIR}")
    
    # DISPLAY BEST MODEL
    best_model = results_df.loc[results_df['R2'].idxmax()]
    print(f"\n🏆 BEST MODEL: {results_df['R2'].idxmax()} (R² = {best_model['R2']:.3f})")

if __name__ == "__main__":
    main()
