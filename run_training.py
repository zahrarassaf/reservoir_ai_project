import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append('src')

from config.model_config import Config, ModelFactoryConfig
from src.data_loader import ReservoirDataLoader
from src.feature_engineer import AdvancedFeatureEngineer
from src.model_factory import ReservoirModelFactory
from src.ensemble_trainer import AdvancedEnsembleTrainer
from src.evaluator import ComprehensiveEvaluator

def main():
    print("🚀 RESERVOIR AI - PROFESSIONAL TRAINING PIPELINE")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize configuration
    config = Config()
    
    # Step 1: Data Loading & Preparation
    print("\n📊 STEP 1: DATA LOADING & PREPARATION")
    print("-" * 40)
    
    data_loader = ReservoirDataLoader(config)
    df = data_loader.load_and_validate_data()
    
    # Step 2: Sequence Creation
    X, y, feature_names = data_loader.create_sequences(df)
    
    # Step 3: Train-Test Split (Time Series Aware)
    split_idx = int(len(X) * (1 - config.TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n📋 DATA SPLITTING")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Step 4: Feature Scaling
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = data_loader.scale_features(
        X_train, X_test, y_train, y_test
    )
    
    # Step 5: Advanced Feature Engineering
    print("\n🔧 STEP 2: ADVANCED FEATURE ENGINEERING")
    print("-" * 40)
    
    feature_engineer = AdvancedFeatureEngineer(config)
    X_train_engineered = feature_engineer.create_advanced_features(X_train_scaled, feature_names)
    X_test_engineered = feature_engineer.create_advanced_features(X_test_scaled, feature_names)
    
    # Feature Selection
    X_train_selected = feature_engineer.select_features(X_train_engineered, y_train_scaled, k=30)
    X_test_selected = feature_engineer.select_features(X_test_engineered, y_test_scaled, k=30)
    
    # Step 6: Model Training
    print("\n🤖 STEP 3: ENSEMBLE MODEL TRAINING")
    print("-" * 40)
    
    model_factory = ReservoirModelFactory(config)
    ensemble_trainer = AdvancedEnsembleTrainer(config, model_factory)
    
    # Train ensemble
    ensemble_predictions, individual_predictions = ensemble_trainer.train_ensemble(
        X_train_selected, X_test_selected, y_train_scaled, y_test_scaled, feature_names
    )
    
    # Add ensemble to predictions
    individual_predictions['weighted_ensemble'] = ensemble_predictions
    
    # Step 7: Comprehensive Evaluation
    print("\n📊 STEP 4: MODEL EVALUATION")
    print("-" * 40)
    
    evaluator = ComprehensiveEvaluator(data_loader.target_scaler)
    evaluation_results = evaluator.evaluate_models(
        y_test_scaled, individual_predictions, list(individual_predictions.keys())
    )
    
    # Performance report
    best_model = evaluator.create_performance_report(evaluation_results)
    
    # Visualization
    print("\n📈 STEP 5: RESULTS VISUALIZATION")
    print("-" * 40)
    
    evaluator.plot_predictions(
        y_test_scaled, individual_predictions, best_model,
        save_path='results/prediction_analysis.png'
    )
    
    # Save results
    print("\n💾 STEP 6: SAVING RESULTS")
    print("-" * 40)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Save models
    for name, model in ensemble_trainer.models.items():
        if name in ['cnn_lstm', 'transformer']:
            model.save(f'models/{name}_model.keras')
        else:
            import joblib
            joblib.dump(model, f'models/{name}_model.joblib')
    
    # Save results
    evaluator.save_results(
        evaluation_results, individual_predictions, 'results/model_predictions.csv'
    )
    
    print("\n✅ RESERVOIR AI TRAINING COMPLETED SUCCESSFULLY!")
    print(f"🏆 Best Model: {best_model}")
    print(f"📊 Best R² Score: {evaluation_results[best_model]['r2']:.4f}")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
