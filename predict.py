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
from src.config import config

class ReservoirPredictor:
    """PRODUCTION-READY PREDICTION PIPELINE"""
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = ReservoirFeatureEngineer()
        self.load_models()
    
    def load_models(self):
        """LOAD ALL TRAINED MODELS"""
        print("📂 LOADING TRAINED MODELS...")
        
        try:
            # LOAD HYBRID CNN-LSTM
            from tensorflow.keras.models import load_model
            hybrid_path = config.MODELS_DIR / 'hybrid_cnn_lstm_final.h5'
            if hybrid_path.exists():
                self.models['hybrid_cnn_lstm'] = load_model(hybrid_path)
                print("✅ Hybrid CNN-LSTM model loaded")
            
            # LOAD ML MODELS
            ml_models = ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']
            for model_name in ml_models:
                model_path = config.MODELS_DIR / f'{model_name}_model.pkl'
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    print(f"✅ {model_name} model loaded")
            
            # LOAD METADATA
            metadata_path = config.MODELS_DIR / 'model_metadata.pkl'
            if metadata_path.exists():
                self.metadata = joblib.load(metadata_path)
                print("✅ Model metadata loaded")
            
            print(f"🎯 {len(self.models)} MODELS LOADED SUCCESSFULLY")
            
        except Exception as e:
            print(f"❌ ERROR LOADING MODELS: {e}")
            raise
    
    def predict(self, new_data: pd.DataFrame) -> dict:
        """GENERATE PREDICTIONS FOR NEW DATA"""
        print("🔮 GENERATING PREDICTIONS...")
        
        # FEATURE ENGINEERING
        X, y, feature_names, engineered_data = self.feature_engineer.prepare_features(new_data)
        
        if len(X) == 0:
            raise ValueError("No valid sequences generated from input data")
        
        # FLATTEN FOR ML MODELS
        X_flat = X.reshape(X.shape[0], -1)
        
        predictions = {}
        
        # HYBRID MODEL PREDICTION
        if 'hybrid_cnn_lstm' in self.models:
            hybrid_pred = self.models['hybrid_cnn_lstm'].predict(X, verbose=0).flatten()
            predictions['hybrid_cnn_lstm'] = hybrid_pred
        
        # ML MODEL PREDICTIONS
        for name, model in self.models.items():
            if name != 'hybrid_cnn_lstm':
                try:
                    predictions[name] = model.predict(X_flat)
                except Exception as e:
                    print(f"⚠️ Prediction failed for {name}: {e}")
        
        # ENSEMBLE PREDICTIONS
        if len(predictions) > 0:
            predictions['weighted_ensemble'] = np.mean(list(predictions.values()), axis=0)
        
        print(f"✅ PREDICTIONS GENERATED FOR {len(predictions)} MODELS")
        return predictions, engineered_data
    
    def predict_single_well(self, well_data: pd.DataFrame) -> dict:
        """PREDICT FOR A SINGLE WELL"""
        print(f"🛢️ PREDICTING FOR WELL {well_data['well_id'].iloc[0]}...")
        return self.predict(well_data)
    
    def evaluate_predictions(self, predictions: dict, actual: np.ndarray) -> pd.DataFrame:
        """EVALUATE PREDICTION ACCURACY"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        results = []
        
        for model_name, pred in predictions.items():
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            r2 = r2_score(actual, pred)
            mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
            
            results.append({
                'model': model_name,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            })
        
        return pd.DataFrame(results)

def main():
    """MAIN PREDICTION EXECUTION"""
    print("🚀 RESERVOIR AI - PRODUCTION PREDICTION PIPELINE")
    print("=" * 60)
    
    try:
        # INITIALIZE PREDICTOR
        predictor = ReservoirPredictor()
        
        # LOAD NEW DATA
        print("\n📥 LOADING PREDICTION DATA...")
        loader = ReservoirDataLoader(dataset_name='spe9')
        new_data = loader.load_dataset()
        
        # SPLIT FOR PREDICTION (LAST 20%)
        split_idx = int(0.8 * len(new_data))
        prediction_data = new_data.iloc[split_idx:]
        
        print(f"📊 PREDICTION DATA: {prediction_data.shape}")
        
        # GENERATE PREDICTIONS
        print("\n🔮 GENERATING PREDICTIONS...")
        predictions, engineered_data = predictor.predict(prediction_data)
        
        # EVALUATE PREDICTIONS
        print("\n📊 EVALUATING PREDICTIONS...")
        actual_values = prediction_data.groupby('well_id').tail(config.SEQUENCE_LENGTH)['oil_rate'].values
        
        if len(actual_values) > 0:
            evaluation_df = predictor.evaluate_predictions(predictions, actual_values[:len(list(predictions.values())[0])])
            
            print("\n🎯 PREDICTION PERFORMANCE:")
            print("=" * 65)
            print(f"{'MODEL':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'MAPE':<10}")
            print("-" * 65)
            
            for _, row in evaluation_df.iterrows():
                print(f"{row['model']:<20} {row['mae']:<10.1f} {row['rmse']:<10.1f} "
                      f"{row['r2']:<10.3f} {row['mape']:<10.1f}%")
        
        # SAVE PREDICTION RESULTS
        print("\n💾 SAVING PREDICTION RESULTS...")
        predictions_df = pd.DataFrame(predictions)
        predictions_df['actual'] = actual_values[:len(predictions_df)]
        predictions_df.to_csv(config.RESULTS_DIR / 'production_predictions.csv', index=False)
        
        # SAVE EVALUATION
        evaluation_df.to_csv(config.RESULTS_DIR / 'prediction_evaluation.csv', index=False)
        
        print(f"\n✅ PREDICTION PIPELINE COMPLETED!")
        print(f"📁 Results saved to: {config.RESULTS_DIR}")
        
    except Exception as e:
        print(f"❌ PREDICTION FAILED: {e}")
        raise

if __name__ == "__main__":
    main()
