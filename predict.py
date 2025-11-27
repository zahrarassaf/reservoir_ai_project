#!/usr/bin/env python3
"""
PRODUCTION PREDICTION PIPELINE
LOAD TRAINED MODELS AND GENERATE PREDICTIONS
"""
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import ReservoirDataLoader
from src.feature_engineer import ReservoirFeatureEngineer
from src.evaluator import ModelEvaluator
from src.config import config

class ReservoirPredictor:
    """PRODUCTION PREDICTION SYSTEM"""
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """LOAD TRAINED MODELS"""
        print("📂 LOADING TRAINED MODELS...")
        
        try:
            # LOAD CNN-LSTM
            from tensorflow.keras.models import load_model
            model_path = config.MODELS_DIR / 'cnn_lstm_final.h5'
            if model_path.exists():
                self.models['cnn_lstm'] = load_model(model_path)
                print("✅ CNN-LSTM model loaded")
            
            # LOAD ML MODELS
            ml_models = ['random_forest', 'xgboost', 'lightgbm']
            for model_name in ml_models:
                model_path = config.MODELS_DIR / f'{model_name}_model.pkl'
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    print(f"✅ {model_name} model loaded")
            
            print(f"🎯 {len(self.models)} MODELS LOADED SUCCESSFULLY")
            
        except Exception as e:
            print(f"❌ ERROR LOADING MODELS: {e}")
    
    def predict(self, data: pd.DataFrame):
        """GENERATE PREDICTIONS FOR NEW DATA"""
        if not self.models:
            print("❌ NO MODELS AVAILABLE - RUN TRAINING FIRST")
            return None, None
        
        print("🔮 GENERATING PREDICTIONS...")
        
        # FEATURE ENGINEERING
        feature_engineer = ReservoirFeatureEngineer()
        X, y, feature_names, engineered_data = feature_engineer.prepare_features(data)
        
        if len(X) == 0:
            print("❌ NO VALID SEQUENCES GENERATED")
            return None, None
        
        # FLATTEN FOR ML MODELS
        X_flat = X.reshape(X.shape[0], -1)
        
        predictions = {}
        
        # CNN-LSTM PREDICTIONS
        if 'cnn_lstm' in self.models:
            predictions['cnn_lstm'] = self.models['cnn_lstm'].predict(X, verbose=0).flatten()
        
        # ML MODEL PREDICTIONS
        for name, model in self.models.items():
            if name != 'cnn_lstm':
                try:
                    predictions[name] = model.predict(X_flat)
                except Exception as e:
                    print(f"⚠️ Prediction failed for {name}: {e}")
        
        return predictions, y

def main():
    """MAIN PREDICTION EXECUTION"""
    print("🚀 RESERVOIR AI - PRODUCTION PREDICTION PIPELINE")
    print("=" * 60)
    
    # CHECK IF MODELS EXIST
    model_files = list(config.MODELS_DIR.glob("*.h5")) + list(config.MODELS_DIR.glob("*.pkl"))
    if not model_files:
        print("❌ NO TRAINED MODELS FOUND")
        print("💡 Run: python run_training.py first")
        return
    
    try:
        # INITIALIZE PREDICTOR
        predictor = ReservoirPredictor()
        
        if not predictor.models:
            print("❌ FAILED TO LOAD MODELS")
            return
        
        # LOAD DATA
        print("\n📥 LOADING PREDICTION DATA...")
        loader = ReservoirDataLoader()
        data = loader.load_data()
        
        print(f"📊 PREDICTION DATA: {data.shape}")
        
        # GENERATE PREDICTIONS
        print("\n🔮 GENERATING PREDICTIONS...")
        predictions, actual = predictor.predict(data)
        
        if predictions is None:
            return
        
        # EVALUATE PREDICTIONS
        print("\n📊 EVALUATING PREDICTIONS...")
        evaluator = ModelEvaluator()
        results_df = evaluator.evaluate_predictions(predictions, actual)
        evaluator.print_performance_summary(results_df)
        
        # SAVE RESULTS
        from src.utils import save_predictions
        save_predictions(predictions, actual, "production_predictions.csv")
        evaluator.save_evaluation_results(results_df, "production_performance.csv")
        
        print(f"\n✅ PREDICTION PIPELINE COMPLETED!")
        print(f"📁 Results saved to: {config.RESULTS_DIR}")
        
    except Exception as e:
        print(f"❌ PREDICTION FAILED: {e}")

if __name__ == "__main__":
    main()
