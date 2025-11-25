"""
Model factory for creating and configuring ML/DL models
"""
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, LSTM, Dense, Dropout, 
                                   BatchNormalization, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf

from .config import config

class ModelFactory:
    """Factory class for creating machine learning models"""
    
    @staticmethod
    def set_random_seeds():
        """Set random seeds for reproducibility"""
        tf.random.set_seed(config.RANDOM_STATE)
        import random
        random.seed(config.RANDOM_STATE)
        import numpy as np
        np.random.seed(config.RANDOM_STATE)
    
    @staticmethod
    def create_cnn_lstm(input_shape: tuple, 
                       conv_filters: int = 64,
                       lstm_units: int = 128,
                       dense_units: int = 64,
                       dropout_rate: float = 0.3,
                       learning_rate: float = 0.001) -> Sequential:
        """Create CNN-LSTM model for sequential data"""
        ModelFactory.set_random_seeds()
        
        model = Sequential([
            Conv1D(conv_filters, kernel_size=3, activation='relu', 
                  input_shape=input_shape, padding='same',
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Conv1D(conv_filters // 2, kernel_size=3, activation='relu', 
                  padding='same', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            LSTM(lstm_units, activation='tanh', return_sequences=False,
                kernel_regularizer=l2(0.001)),
            Dropout(dropout_rate),
            
            Dense(dense_units, activation='relu', 
                 kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(dense_units // 2, activation='relu',
                 kernel_regularizer=l2(0.001)),
            Dense(1)  # Single output for regression
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', 
                     metrics=['mae', 'mse'])
        
        return model
    
    @staticmethod
    def create_svr(C: float = 1.0, epsilon: float = 0.1, 
                  kernel: str = 'rbf', gamma: str = 'scale') -> SVR:
        """Create Support Vector Regressor"""
        return SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma)
    
    @staticmethod
    def create_random_forest(n_estimators: int = 100, 
                           max_depth: int = None,
                           min_samples_split: int = 2,
                           random_state: int = None) -> RandomForestRegressor:
        """Create Random Forest Regressor"""
        if random_state is None:
            random_state = config.RANDOM_STATE
            
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
    
    @staticmethod
    def create_xgboost(n_estimators: int = 100,
                      max_depth: int = 6,
                      learning_rate: float = 0.1,
                      random_state: int = None) -> XGBRegressor:
        """Create XGBoost Regressor"""
        if random_state is None:
            random_state = config.RANDOM_STATE
            
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1
        )
    
    @staticmethod
    def create_lightgbm(n_estimators: int = 100,
                       max_depth: int = -1,
                       learning_rate: float = 0.1,
                       random_state: int = None) -> LGBMRegressor:
        """Create LightGBM Regressor"""
        if random_state is None:
            random_state = config.RANDOM_STATE
            
        return LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
    
    @staticmethod
    def create_gradient_boosting(n_estimators: int = 100,
                               max_depth: int = 3,
                               learning_rate: float = 0.1,
                               random_state: int = None) -> GradientBoostingRegressor:
        """Create Gradient Boosting Regressor"""
        if random_state is None:
            random_state = config.RANDOM_STATE
            
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
    
    @staticmethod
    def create_ridge(alpha: float = 1.0, 
                    random_state: int = None) -> Ridge:
        """Create Ridge Regression model"""
        if random_state is None:
            random_state = config.RANDOM_STATE
            
        return Ridge(alpha=alpha, random_state=random_state)
    
    @staticmethod
    def create_lasso(alpha: float = 1.0,
                    random_state: int = None) -> Lasso:
        """Create Lasso Regression model"""
        if random_state is None:
            random_state = config.RANDOM_STATE
            
        return Lasso(alpha=alpha, random_state=random_state)
    
    @staticmethod
    def get_all_models():
        """Get dictionary of all available models"""
        return {
            'RandomForest': ModelFactory.create_random_forest(),
            'XGBoost': ModelFactory.create_xgboost(),
            'LightGBM': ModelFactory.create_lightgbm(),
            'GradientBoosting': ModelFactory.create_gradient_boosting(),
            'SVR': ModelFactory.create_svr(),
            'Ridge': ModelFactory.create_ridge(),
            'Lasso': ModelFactory.create_lasso()
        }
