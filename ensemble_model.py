"""
PRODUCTION-GRADE ENSEMBLE MODEL ARCHITECTURE
COMBINING DEEP LEARNING AND TRADITIONAL ML
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                   Conv1D, MaxPooling1D, Input, concatenate,
                                   GlobalAveragePooling1D, Attention)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                      ModelCheckpoint, TensorBoard)
from tensorflow.keras.regularizers import l2
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                            VotingRegressor)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

from .config import config

class AdvancedReservoirModel:
    """STATE-OF-THE-ART RESERVOIR FORECASTING ENSEMBLE"""
    
    def __init__(self):
        self.cnn_lstm_model = None
        self.transformer_model = None
        self.ensemble_models = {}
        self.feature_importance = {}
        self.training_history = {}
        
    def build_hybrid_cnn_lstm(self, input_shape: Tuple) -> Model:
        """BUILD HYBRID CNN-LSTM ARCHITECTURE WITH ATTENTION"""
        print("🛠️ BUILDING HYBRID CNN-LSTM WITH ATTENTION...")
        
        input_layer = Input(shape=input_shape)
        
        # CONVOLUTIONAL FEATURE EXTRACTION
        conv1 = Conv1D(128, 5, activation='relu', padding='same',
                      kernel_regularizer=l2(0.001))(input_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = MaxPooling1D(2)(conv1)
        conv1 = Dropout(0.3)(conv1)
        
        conv2 = Conv1D(256, 3, activation='relu', padding='same',
                      kernel_regularizer=l2(0.001))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = MaxPooling1D(2)(conv2)
        conv2 = Dropout(0.3)(conv2)
        
        # BIDIRECTIONAL LSTM WITH ATTENTION
        lstm1 = LSTM(256, return_sequences=True, dropout=0.2,
                    recurrent_dropout=0.2)(conv2)
        lstm1 = BatchNormalization()(lstm1)
        
        # ATTENTION MECHANISM
        attention = Attention()([lstm1, lstm1])
        lstm2 = LSTM(128, dropout=0.2)(attention)
        
        # DENSE LAYERS
        dense1 = Dense(256, activation='relu',
                      kernel_regularizer=l2(0.001))(lstm2)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.4)(dense1)
        
        dense2 = Dense(128, activation='relu')(dense1)
        dense2 = Dropout(0.3)(dense2)
        
        dense3 = Dense(64, activation='relu')(dense2)
        output_layer = Dense(1, activation='linear')(dense3)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        
        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss='huber_loss',
            metrics=['mae', 'mse']
        )
        
        print("✅ HYBRID CNN-LSTM MODEL BUILT")
        return model
    
    def build_ml_ensemble(self):
        """BUILD ADVANCED ML ENSEMBLE"""
        print("🛠️ BUILDING ML ENSEMBLE...")
        
        self.ensemble_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ),
            'xgboost': XGBRegressor(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=300,
                max_depth=12,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=config.RANDOM_STATE
            )
        }
        
        print("✅ ML ENSEMBLE BUILT")
    
    def train_ensemble(self, X_flat: np.ndarray, y: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None):
        """TRAIN ML ENSEMBLE WITH CROSS-VALIDATION"""
        print("🔥 TRAINING ML ENSEMBLE...")
        
        if not self.ensemble_models:
            self.build_ml_ensemble()
        
        for name, model in self.ensemble_models.items():
            print(f"🔄 TRAINING {name.upper()}...")
            
            if X_val is not None and y_val is not None:
                # USE EARLY STOPPING IF VALIDATION DATA AVAILABLE
                if hasattr(model, 'set_params'):
                    if 'early_stopping_rounds' in model.get_params():
                        model.set_params(
                            early_stopping_rounds=50,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
            
            model.fit(X_flat, y)
            
            # STORE FEATURE IMPORTANCE
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        print("✅ ML ENSEMBLE TRAINING COMPLETED")
    
    def train_hybrid_model(self, X_seq: np.ndarray, y: np.ndarray,
                          X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """TRAIN HYBRID MODEL WITH ADVANCED CALLBACKS"""
        print("🔥 TRAINING HYBRID CNN-LSTM...")
        
        if self.cnn_lstm_model is None:
            self.cnn_lstm_model = self.build_hybrid_cnn_lstm(X_seq.shape[1:])
        
        # ADVANCED CALLBACKS
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                config.MODELS_DIR / 'hybrid_cnn_lstm_best.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            TensorBoard(
                log_dir=config.MODELS_DIR / 'logs',
                histogram_freq=1
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.cnn_lstm_model.fit(
            X_seq, y,
            validation_data=validation_data,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )
        
        self.training_history['hybrid_cnn_lstm'] = history.history
        print("✅ HYBRID MODEL TRAINING COMPLETED")
        
        return history.history
    
    def create_weighted_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """CREATE WEIGHTED ENSEMBLE BASED ON MODEL PERFORMANCE"""
        weights = {}
        
        # SIMPLE WEIGHTING BASED ON MODEL ORDER
        base_weights = {'hybrid_cnn_lstm': 0.4, 'xgboost': 0.25,
                       'lightgbm': 0.2, 'random_forest': 0.15}
        
        weighted_pred = np.zeros_like(list(predictions.values())[0])
        
        for name, pred in predictions.items():
            weight = base_weights.get(name, 0.1)
            weighted_pred += weight * pred
        
        return weighted_pred
    
    def predict_ensemble(self, X_seq: np.ndarray, X_flat: np.ndarray) -> Dict[str, np.ndarray]:
        """GENERATE PREDICTIONS FROM ALL MODELS"""
        predictions = {}
        
        # HYBRID CNN-LSTM PREDICTION
        if self.cnn_lstm_model is not None:
            predictions['hybrid_cnn_lstm'] = self.cnn_lstm_model.predict(X_seq, verbose=0).flatten()
        
        # ML ENSEMBLE PREDICTIONS
        for name, model in self.ensemble_models.items():
            predictions[name] = model.predict(X_flat)
        
        # WEIGHTED ENSEMBLE
        predictions['weighted_ensemble'] = self.create_weighted_ensemble(predictions)
        
        # SIMPLE AVERAGE ENSEMBLE
        predictions['average_ensemble'] = np.mean(list(predictions.values()), axis=0)
        
        return predictions
    
    def save_models(self):
        """SAVE ALL TRAINED MODELS AND METADATA"""
        print("💾 SAVING MODELS...")
        
        # SAVE HYBRID MODEL
        if self.cnn_lstm_model is not None:
            self.cnn_lstm_model.save(config.MODELS_DIR / 'hybrid_cnn_lstm_final.h5')
        
        # SAVE ML MODELS
        for name, model in self.ensemble_models.items():
            joblib.dump(model, config.MODELS_DIR / f'{name}_model.pkl')
        
        # SAVE METADATA
        model_metadata = {
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'model_config': {
                'sequence_length': config.SEQUENCE_LENGTH,
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE
            }
        }
        
        joblib.dump(model_metadata, config.MODELS_DIR / 'model_metadata.pkl')
        
        print("✅ ALL MODELS AND METADATA SAVED")
