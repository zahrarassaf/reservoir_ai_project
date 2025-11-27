"""
PRODUCTION ENSEMBLE MODEL ARCHITECTURE
ROBUST IMPLEMENTATION WITH FALLBACKS
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.ensemble import RandomForestRegressor
import joblib

from .config import config

class AdvancedReservoirModel:
    """PRODUCTION-READY ENSEMBLE MODEL"""
    
    def __init__(self):
        self.cnn_lstm_model = None
        self.ensemble_models = {}
        self.is_trained = False
    
    def build_cnn_lstm(self, input_shape: Tuple) -> Sequential:
        """BUILD CNN-LSTM MODEL"""
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape, padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            
            Conv1D(128, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            
            LSTM(128, return_sequences=True, dropout=0.2),
            LSTM(64, dropout=0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_ml_ensemble(self):
        """BUILD MACHINE LEARNING ENSEMBLE"""
        self.ensemble_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                max_depth=10
            )
        }
        
        # OPTIONAL MODELS WITH FALLBACK
        try:
            from xgboost import XGBRegressor
            self.ensemble_models['xgboost'] = XGBRegressor(
                n_estimators=100,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
        except ImportError:
            print("⚠️  XGBoost not available")
            
        try:
            from lightgbm import LGBMRegressor
            self.ensemble_models['lightgbm'] = LGBMRegressor(
                n_estimators=100,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
        except ImportError:
            print("⚠️  LightGBM not available")
    
    def train_ensemble(self, X_flat: np.ndarray, y: np.ndarray):
        """TRAIN ML ENSEMBLE MODELS"""
        if not self.ensemble_models:
            self.build_ml_ensemble()
        
        print("🔥 TRAINING ML ENSEMBLE...")
        for name, model in self.ensemble_models.items():
            print(f"🔄 Training {name}...")
            model.fit(X_flat, y)
        
        self.is_trained = True
    
    def train_cnn_lstm(self, X_seq: np.ndarray, y: np.ndarray, 
                      X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """TRAIN CNN-LSTM MODEL"""
        if self.cnn_lstm_model is None:
            self.cnn_lstm_model = self.build_cnn_lstm(X_seq.shape[1:])
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
            ModelCheckpoint(
                config.MODELS_DIR / 'cnn_lstm_best.h5',
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        print("🔥 TRAINING CNN-LSTM...")
        history = self.cnn_lstm_model.fit(
            X_seq, y,
            validation_data=validation_data,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )
        
        self.is_trained = True
        return history.history
    
    def predict_ensemble(self, X_seq: np.ndarray, X_flat: np.ndarray) -> Dict[str, np.ndarray]:
        """GENERATE PREDICTIONS FROM ALL MODELS"""
        predictions = {}
        
        # CNN-LSTM PREDICTIONS
        if self.cnn_lstm_model is not None:
            predictions['cnn_lstm'] = self.cnn_lstm_model.predict(X_seq, verbose=0).flatten()
        
        # ML ENSEMBLE PREDICTIONS
        for name, model in self.ensemble_models.items():
            predictions[name] = model.predict(X_flat)
        
        # WEIGHTED ENSEMBLE
        if len(predictions) > 0:
            predictions['weighted_ensemble'] = np.mean(list(predictions.values()), axis=0)
        
        return predictions
    
    def save_models(self):
        """SAVE ALL TRAINED MODELS"""
        if self.cnn_lstm_model is not None:
            self.cnn_lstm_model.save(config.MODELS_DIR / 'cnn_lstm_final.h5')
        
        for name, model in self.ensemble_models.items():
            joblib.dump(model, config.MODELS_DIR / f'{name}_model.pkl')
        
        print("✅ ALL MODELS SAVED")
