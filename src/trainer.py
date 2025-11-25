"""
Model training and hyperparameter optimization
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

from .config import config
from .model_factory import ModelFactory

class ModelTrainer:
    """Handles model training and hyperparameter optimization"""
    
    def __init__(self):
        self.trained_models = {}
        self.training_history = {}
    
    def train_cnn_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      model_name: str = "cnn_lstm") -> dict:
        """Train CNN-LSTM model with proper validation"""
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = ModelFactory.create_cnn_lstm(input_shape)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                str(config.MODEL_DIR / f"{model_name}_best.h5"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config.PATIENCE // 2,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Important for temporal data
        )
        
        # Save model and history
        model.save(config.MODEL_DIR / f"{model_name}_final.h5")
        self.training_history[model_name] = history.history
        
        # Make predictions
        y_pred = model.predict(X_val).flatten()
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_val, y_pred)
        
        self.trained_models[model_name] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred
        }
        
        print(f"CNN-LSTM training completed. Validation RMSE: {metrics['rmse']:.4f}")
        return self.trained_models[model_name]
    
    def train_sklearn_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          model_name: str, perform_cv: bool = True) -> dict:
        """Train scikit-learn model with optional hyperparameter tuning"""
        
        if perform_cv and hasattr(model, 'get_params'):
            # Hyperparameter tuning for supported models
            best_model = self._perform_hyperparameter_tuning(model, X_train, y_train, model_name)
        else:
            best_model = model
        
        # Train final model
        best_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = best_model.predict(X_val)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_val, y_pred)
        
        # Save model
        joblib.dump(best_model, config.MODEL_DIR / f"{model_name}.pkl")
        
        self.trained_models[model_name] = {
            'model': best_model,
            'metrics': metrics,
            'predictions': y_pred
        }
        
        print(f"{model_name} training completed. Validation RMSE: {metrics['rmse']:.4f}")
        return self.trained_models[model_name]
    
    def _perform_hyperparameter_tuning(self, model, X_train: np.ndarray, 
                                     y_train: np.ndarray, model_name: str):
        """Perform hyperparameter tuning for scikit-learn models"""
        
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'SVR': {
                'C': config.SVR_C_VALUES,
                'gamma': config.SVR_GAMMA_VALUES,
                'epsilon': config.SVR_EPSILON_VALUES
            },
            'Ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }
        
        if model_name in param_grids:
            param_grid = param_grids[model_name]
            
            # Create pipeline with scaling for SVR
            if model_name == 'SVR':
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
                # Adjust parameter names for pipeline
                param_grid = {f'model__{key}': value for key, value in param_grid.items()}
            else:
                pipeline = model
            
            print(f"Performing hyperparameter tuning for {model_name}...")
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best CV score: {-grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        else:
            return model
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate comprehensive regression metrics"""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred)
        }
    
    def get_training_results(self) -> dict:
        """Get all training results"""
        return self.trained_models
    
    def get_model_performance_summary(self) -> pd.DataFrame:
        """Get performance summary of all trained models"""
        performance_data = []
        
        for model_name, results in self.trained_models.items():
            metrics = results['metrics']
            performance_data.append({
                'Model': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R2': metrics['r2'],
                'MSE': metrics['mse']
            })
        
        return pd.DataFrame(performance_data)
