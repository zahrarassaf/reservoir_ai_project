"""
ENSEMBLE MODEL IMPLEMENTATION
"""
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

from .config import config

class ReservoirEnsembleModel:
    """Ensemble model for reservoir forecasting"""
    
    def __init__(self):
        self.cnn_lstm_model = None
        self.ensemble_models = {}
        
    def build_cnn_lstm(self, input_shape):
        """Build CNN-LSTM model"""
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
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train_ensemble(self, X_flat, y):
        """Train traditional ML models"""
        self.ensemble_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': LGBMRegressor(n_estimators=100, random_state=42)
        }
        
        for name, model in self.ensemble_models.items():
            print(f"Training {name}...")
            model.fit(X_flat, y)
    
    def train_cnn_lstm(self, X_seq, y, X_val=None, y_val=None):
        """Train CNN-LSTM model"""
        if self.cnn_lstm_model is None:
            self.cnn_lstm_model = self.build_cnn_lstm(X_seq.shape[1:])
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.cnn_lstm_model.fit(
            X_seq, y,
            validation_data=validation_data,
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict_ensemble(self, X_seq, X_flat):
        """Generate predictions from all models"""
        predictions = {}
        
        if self.cnn_lstm_model is not None:
            predictions['cnn_lstm'] = self.cnn_lstm_model.predict(X_seq).flatten()
        
        for name, model in self.ensemble_models.items():
            predictions[name] = model.predict(X_flat)
        
        return predictions
    
    def save_models(self):
        """Save trained models"""
        if self.cnn_lstm_model is not None:
            self.cnn_lstm_model.save(config.MODELS_DIR / 'cnn_lstm_model.h5')
        
        for name, model in self.ensemble_models.items():
            joblib.dump(model, config.MODELS_DIR / f'{name}.pkl')
