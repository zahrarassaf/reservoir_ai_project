import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from config.model_config import Config, ModelFactoryConfig
from src.opm_data_loader import OPMDataLoader
from src.feature_engineer import AdvancedFeatureEngineer
from src.model_factory import ReservoirModelFactory
from src.ensemble_trainer import AdvancedEnsembleTrainer
from src.evaluator import ComprehensiveEvaluator

def setup_directories():
    """Create necessary directories for the project"""
    directories = ['models', 'results', 'data/processed', 'logs', 'visualizations']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def print_banner():
    """Print professional banner"""
    banner = """
    🚀 RESERVOIR AI - PROFESSIONAL OPM TRAINING PIPELINE
    ============================================================
    🔬 Advanced Machine Learning for Reservoir Engineering
    📊 OPM Data Integration | 🤖 Ensemble AI Models
    🎯 Production-Grade Forecasting System
    ============================================================
    """
    print(banner)

def validate_environment():
    """Validate system environment and dependencies"""
    print("🔍 ENVIRONMENT VALIDATION")
    print("-" * 40)
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"🐍 Python Version: {python_version}")
    
    # Check critical packages
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not installed")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not installed")
        return False
    
    try:
        import xgboost
        print(f"✅ XGBoost: {xgboost.__version__}")
    except ImportError:
        print("⚠️  XGBoost not installed (optional)")
    
    try:
        import lightgbm
        print(f"✅ LightGBM: {lightgbm.__version__}")
    except ImportError:
        print("⚠️  LightGBM not installed (optional)")
    
    return True

def main():
    """Main training pipeline execution"""
    
    # Initialize
    print_banner()
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validate environment
    if not validate_environment():
        print("❌ Environment validation failed. Please install required packages.")
        return
    
    # Setup directories
    setup_directories()
    
    # Initialize configuration
    config = Config()
    
    try:
        # =========================================================================
        # STEP 1: OPM DATA LOADING & VALIDATION
        # =========================================================================
        print("\n📊 STEP 1: OPM DATA LOADING & VALIDATION")
        print("-" * 50)
        
        opm_loader = OPMDataLoader(config)
        
        # Load OPM data - try multiple sources
        data_sources = [
            "opm-data",
            "../opm-data", 
            "../../opm-data",
            "data/raw"
        ]
        
        df = None
        for source in data_sources:
            if os.path.exists(source):
                print(f"🔍 Attempting to load data from: {source}")
                df = opm_loader.load_opm_data(source)
                if df is not None and not df.empty:
                    print(f"✅ Successfully loaded data from: {source}")
                    break
            else:
                print(f"📭 Data source not found: {source}")
        
        if df is None or df.empty:
            print("⚠️  No external data sources found, using OPM-like synthetic data")
            df = opm_loader._generate_opm_like_synthetic_data()
        
        # Data quality report
        print(f"\n📋 DATA QUALITY REPORT")
        print(f"   Total records: {len(df):,}")
        print(f"   Features: {len(df.columns)}")
        print(f"   Wells: {df['well_id'].nunique()}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # =========================================================================
        # STEP 2: SEQUENCE CREATION & DATA SPLITTING
        # =========================================================================
        print("\n🔄 STEP 2: SEQUENCE CREATION & DATA SPLITTING")
        print("-" * 50)
        
        X, y, feature_names = opm_loader.create_sequences(df)
        
        # Time-series aware splitting
        split_idx = int(len(X) * (1 - config.TEST_SIZE))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📊 DATA SPLITTING RESULTS:")
        print(f"   Training sequences: {X_train.shape}")
        print(f"   Testing sequences: {X_test.shape}")
        print(f"   Training targets: {y_train.shape}")
        print(f"   Testing targets: {y_test.shape}")
        print(f"   Feature names: {len(feature_names)} features")
        
        # =========================================================================
        # STEP 3: FEATURE SCALING
        # =========================================================================
        print("\n⚖️ STEP 3: FEATURE SCALING")
        print("-" * 50)
        
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = opm_loader.scale_features(
            X_train, X_test, y_train, y_test
        )
        
        print("✅ Feature scaling completed:")
        print(f"   Training data range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
        print(f"   Testing data range: [{X_test_scaled.min():.3f}, {X_test_scaled.max():.3f}]")
        print(f"   Target range: [{y_train_scaled.min():.3f}, {y_train_scaled.max():.3f}]")
        
        # =========================================================================
        # STEP 4: ADVANCED FEATURE ENGINEERING
        # =========================================================================
        print("\n🔧 STEP 4: ADVANCED FEATURE ENGINEERING")
        print("-" * 50)
        
        feature_engineer = AdvancedFeatureEngineer(config)
        
        # Apply feature engineering
        X_train_engineered = feature_engineer.create_advanced_features(X_train_scaled, feature_names)
        X_test_engineered = feature_engineer.create_advanced_features(X_test_scaled, feature_names)
        
        # Feature selection
        X_train_selected = feature_engineer.select_features(X_train_engineered, y_train_scaled, k=30)
        X_test_selected = feature_engineer.select_features(X_test_engineered, y_test_scaled, k=30)
        
        print("✅ Feature engineering completed:")
        print(f"   Original features: {X_train.shape[-1]}")
        print(f"   Engineered features: {X_train_engineered.shape[-1]}")
        print(f"   Selected features: {X_train_selected.shape[-1]}")
        
        # =========================================================================
        # STEP 5: ENSEMBLE MODEL TRAINING
        # =========================================================================
        print("\n🤖 STEP 5: ENSEMBLE MODEL TRAINING")
        print("-" * 50)
        
        model_factory = ReservoirModelFactory(config)
        ensemble_trainer = AdvancedEnsembleTrainer(config, model_factory)
        
        # Train ensemble models
        ensemble_predictions, individual_predictions = ensemble_trainer.train_ensemble(
            X_train_selected, X_test_selected, y_train_scaled, y_test_scaled, feature_names
        )
        
        # Add ensemble to predictions
        individual_predictions['weighted_ensemble'] = ensemble_predictions
        
        print("✅ Model training completed:")
        print(f"   Trained models: {len(ensemble_trainer.models)}")
        print(f"   Ensemble predictions: {ensemble_predictions.shape}")
        
        # =========================================================================
        # STEP 6: COMPREHENSIVE MODEL EVALUATION
        # =========================================================================
        print("\n📊 STEP 6: COMPREHENSIVE MODEL EVALUATION")
        print("-" * 50)
        
        evaluator = ComprehensiveEvaluator(opm_loader.target_scaler)
        
        # Evaluate all models
        evaluation_results = evaluator.evaluate_models(
            y_test_scaled, individual_predictions, list(individual_predictions.keys())
        )
        
        # Performance report
        best_model = evaluator.create_performance_report(evaluation_results)
        
        # =========================================================================
        # STEP 7: RESULTS VISUALIZATION
        # =========================================================================
        print("\n📈 STEP 7: RESULTS VISUALIZATION")
        print("-" * 50)
        
        # Create comprehensive visualizations
        evaluator.plot_predictions(
            y_test_scaled, individual_predictions, best_model,
            save_path='results/opm_prediction_analysis.png'
        )
        
        # Additional visualizations
        try:
            from src.visualization import create_comprehensive_dashboard
            create_comprehensive_dashboard(df, evaluation_results, individual_predictions, 
                                        opm_loader.target_scaler, y_test_scaled)
        except ImportError:
            print("⚠️  Advanced visualization module not available")
        
        # =========================================================================
        # STEP 8: SAVE RESULTS & MODELS
        # =========================================================================
        print("\n💾 STEP 8: SAVE RESULTS & MODELS")
        print("-" * 50)
        
        # Save trained models
        print("💾 Saving trained models...")
        for name, model in ensemble_trainer.models.items():
            try:
                if name in ['cnn_lstm', 'transformer']:
                    model_path = f'models/{name}_model.keras'
                    model.save(model_path)
                    print(f"   ✅ {name}: {model_path}")
                else:
                    import joblib
                    model_path = f'models/{name}_model.joblib'
                    joblib.dump(model, model_path)
                    print(f"   ✅ {name}: {model_path}")
            except Exception as e:
                print(f"   ❌ Failed to save {name}: {str(e)}")
        
        # Save predictions and metrics
        print("💾 Saving results and predictions...")
        evaluator.save_results(
            evaluation_results, individual_predictions, 'results/opm_model_predictions.csv'
        )
        
        # Save training configuration
        config_df = pd.DataFrame({
            'parameter': list(config.__dict__.keys()),
            'value': [str(v) for v in config.__dict__.values()]
        })
        config_df.to_csv('results/training_configuration.csv', index=False)
        
        # Save feature importance if available
        try:
            feature_importance = {}
            for name, model in ensemble_trainer.models.items():
                if hasattr(model, 'feature_importances_'):
                    feature_importance[name] = model.feature_importances_
            
            if feature_importance:
                fi_df = pd.DataFrame(feature_importance)
                fi_df.to_csv('results/feature_importance.csv')
                print("   ✅ Feature importance saved")
        except Exception as e:
            print(f"   ⚠️  Feature importance saving skipped: {str(e)}")
        
        # =========================================================================
        # STEP 9: FINAL SUMMARY & DEPLOYMENT PREPARATION
        # =========================================================================
        print("\n🏆 STEP 9: FINAL SUMMARY")
        print("-" * 50)
        
        best_r2 = evaluation_results[best_model]['r2']
        best_mae = evaluation_results[best_model]['mae']
        
        summary = f"""
        🎯 TRAINING COMPLETED SUCCESSFULLY!
        
        📊 PERFORMANCE SUMMARY:
           Best Model: {best_model}
           R² Score: {best_r2:.4f}
           MAE: {best_mae:.2f} bbl/day
           Test Samples: {len(y_test):,}
           
        🤖 MODELS TRAINED: {len(ensemble_trainer.models)}
        📁 RESULTS SAVED:
           Models: ./models/
           Results: ./results/
           Visualizations: ./results/opm_prediction_analysis.png
           
        ⏰ Duration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        print(summary)
        
        # Save summary to file
        with open('results/training_summary.txt', 'w') as f:
            f.write(summary)
        
        print("✅ All operations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        print("🔍 Debug information:")
        import traceback
        traceback.print_exc()
        
        # Save error log
        with open('logs/error_log.txt', 'w') as f:
            f.write(f"Error at {datetime.now()}:\\n")
            f.write(str(e))
            f.write("\\n\\nTraceback:\\n")
            f.write(traceback.format_exc())
        
        return 1
    
    return 0

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    import tensorflow as tf
    tf.random.set_seed(42)
    
    # Execute main pipeline
    exit_code = main()
    
    # Final message
    if exit_code == 0:
        print(f"\n🎉 Reservoir AI Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🌟 Check './results/' directory for outputs and visualizations")
    else:
        print(f"\n💥 Pipeline failed with errors. Check './logs/error_log.txt' for details")
    
    sys.exit(exit_code)
