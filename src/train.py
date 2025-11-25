"""
Main training script for Reservoir AI project
"""
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.model_factory import ModelFactory
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.utils import ensure_directories, set_display_options

def main():
    """Main training pipeline"""
    print("🚀 Starting Reservoir AI Training Pipeline")
    print("=" * 60)
    
    # Set up
    ensure_directories()
    set_display_options()
    
    # Initialize components
    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    try:
        # 1. Load and prepare data
        print("\n📊 Step 1: Loading and preparing data...")
        raw_data = data_loader.load_data(use_synthetic=True)
        
        # 2. Feature engineering
        print("\n🔧 Step 2: Feature engineering...")
        temporal_features = feature_engineer.create_temporal_features(raw_data)
        engineered_features = feature_engineer.create_domain_features(temporal_features)
        
        # 3. Prepare data for different model types
        print("\n⚡ Step 3: Preparing data for models...")
        
        # For traditional ML models
        X_tabular, y_tabular, feature_names = feature_engineer.prepare_tabular_data(
            engineered_features
        )
        
        # For sequential models (CNN-LSTM)
        X_sequential, y_sequential = feature_engineer.prepare_sequences(
            engineered_features
        )
        
        # 4. Train-test split
        print("\n📈 Step 4: Creating train-test splits...")
        
        # Traditional ML split
        split_idx = int(0.8 * len(X_tabular))
        X_train_tab, X_test_tab = X_tabular[:split_idx], X_tabular[split_idx:]
        y_train_tab, y_test_tab = y_tabular[:split_idx], y_tabular[split_idx:]
        
        # Sequential data split
        if len(X_sequential) > 0:
            seq_split_idx = int(0.8 * len(X_sequential))
            X_train_seq, X_test_seq = X_sequential[:seq_split_idx], X_sequential[seq_split_idx:]
            y_train_seq, y_test_seq = y_sequential[:seq_split_idx], y_sequential[seq_split_idx:]
        
        # 5. Scale features
        print("\n⚖️ Step 5: Scaling features...")
        X_train_tab_scaled, X_test_tab_scaled = feature_engineer.scale_features(
            X_train_tab, X_test_tab
        )
        
        # 6. Train models
        print("\n🧠 Step 6: Training models...")
        
        # Train traditional ML models
        traditional_models = ModelFactory.get_all_models()
        
        for model_name, model in traditional_models.items():
            print(f"Training {model_name}...")
            model_trainer.train_sklearn_model(
                model, X_train_tab_scaled, y_train_tab,
                X_test_tab_scaled, y_test_tab, model_name
            )
        
        # Train CNN-LSTM if sequential data is available
        if len(X_sequential) > 0:
            print("Training CNN-LSTM...")
            model_trainer.train_cnn_lstm(
                X_train_seq, y_train_seq, X_test_seq, y_test_seq, "CNN_LSTM"
            )
        
        # 7. Evaluate models
        print("\n📊 Step 7: Evaluating models...")
        training_results = model_trainer.get_training_results()
        
        # Prepare evaluation data
        eval_results = {}
        for model_name, results in training_results.items():
            if 'predictions' in results:
                # Store true values and predictions for evaluation
                eval_results[model_name] = {
                    'true_values': y_test_tab if model_name != 'CNN_LSTM' else y_test_seq,
                    'predictions': results['predictions'],
                    'metrics': results['metrics']
                }
        
        # Generate comprehensive evaluation
        comprehensive_report = evaluator.generate_comprehensive_report(eval_results)
        
        # Create visualizations
        evaluator.create_comparison_plots(eval_results)
        
        # Calculate feature importance for tree-based models
        for model_name, results in training_results.items():
            if model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
                importance_df = evaluator.calculate_feature_importance(
                    results['model'], feature_names, model_name, X_test_tab_scaled
                )
                if not importance_df.empty:
                    print(f"Top 5 features for {model_name}:")
                    print(importance_df.head())
        
        # 8. Save results
        print("\n💾 Step 8: Saving results...")
        performance_summary = model_trainer.get_model_performance_summary()
        performance_summary.to_csv(config.RESULT_DIR / 'model_performance_summary.csv', index=False)
        
        # Save feature names
        pd.DataFrame({'feature_names': feature_names}).to_csv(
            config.RESULT_DIR / 'feature_names.csv', index=False
        )
        
        # 9. Final report
        print("\n🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("FINAL MODEL PERFORMANCE:")
        print(performance_summary.to_string(index=False))
        print(f"\nResults saved to: {config.RESULT_DIR}")
        
        return {
            'performance_summary': performance_summary,
            'comprehensive_report': comprehensive_report,
            'feature_importance': evaluator.feature_importance
        }
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    results = main()
