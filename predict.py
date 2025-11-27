#!/usr/bin/env python3
"""
PRODUCTION PREDICTION PIPELINE
"""
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import ReservoirDataLoader
from src.feature_engineer import ReservoirFeatureEngineer
from src.config import config

def main():
    print("🚀 RESERVOIR AI PREDICTION PIPELINE")
    
    # Load data
    loader = ReservoirDataLoader()
    data = loader.load_dataset()
    
    # Load trained model (simplified version)
    try:
        model_path = config.MODELS_DIR / 'hybrid_cnn_lstm_final.h5'
        if model_path.exists():
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            print("✅ Model loaded successfully")
            
            # Generate features and predict
            feature_engineer = ReservoirFeatureEngineer()
            X, y, features, _ = feature_engineer.prepare_features(data)
            
            predictions = model.predict(X, verbose=0).flatten()
            print(f"📊 Predictions generated: {len(predictions)}")
            
            # Save results
            results_df = pd.DataFrame({
                'actual': y,
                'predicted': predictions
            })
            results_df.to_csv(config.RESULTS_DIR / 'predictions.csv', index=False)
            print("💾 Predictions saved to results/predictions.csv")
            
        else:
            print("❌ No trained model found. Run training first.")
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")

if __name__ == "__main__":
    main()
