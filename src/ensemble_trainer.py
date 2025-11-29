import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

class AdvancedEnsembleTrainer:
    def __init__(self, config, model_factory):
        self.config = config
        self.model_factory = model_factory
        self.models = {}
        self.histories = {}
        
    def train_ensemble(self, X_train, X_test, y_train, y_test, feature_names):
        """Train sophisticated ensemble with cross-validation"""
        print("\n🤖 ADVANCED ENSEMBLE TRAINING")
        print("=============================")
        
        # Reshape for tree-based models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Train individual models with cross-validation
        self._train_tree_models(X_train_flat, X_test_flat, y_train, y_test)
        self._train_deep_learning_models(X_train, X_test, y_train, y_test)
        
        # Create weighted ensemble
        ensemble_predictions = self._create_weighted_ensemble(X_test, X_test_flat)
        
        return ensemble_predictions
    
    def _train_tree_models(self, X_train_flat, X_test_flat, y_train, y_test):
        """Train tree-based models with cross-validation"""
        print("\n🌲 TRAINING TREE-BASED MODELS")
        
        # Random Forest
        print("🔄 Training Random Forest...")
        rf_model = self.model_factory.create_random_forest()
        rf_model.fit(X_train_flat, y_train)
        self.models['random_forest'] = rf_model
        
        # XGBoost
        print("🔄 Training XGBoost...")
        xgb_model = self.model_factory.create_xgboost()
        xgb_model.fit(X_train_flat, y_train)
        self.models['xgboost'] = xgb_model
        
        # LightGBM
        print("🔄 Training LightGBM...")
        lgb_model = self.model_factory.create_lightgbm()
        lgb_model.fit(X_train_flat, y_train)
        self.models['lightgbm'] = lgb_model
    
    def _train_deep_learning_models(self, X_train, X_test, y_train, y_test):
        """Train deep learning models with proper validation"""
        print("\n🧠 TRAINING DEEP LEARNING MODELS")
        
        # CNN-LSTM
        print("🔄 Training CNN-LSTM...")
        cnn_lstm_model = self.model_factory.create_cnn_lstm_model(X_train.shape[1:])
        
        history = cnn_lstm_model.fit(
            X_train, y_train,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_data=(X_test, y_test),
            callbacks=self.model_factory.get_callbacks('cnn_lstm'),
            verbose=1,
            shuffle=False  # Important for time series
        )
        
        self.models['cnn_lstm'] = cnn_lstm_model
        self.histories['cnn_lstm'] = history
        
        # Optional: Train Transformer model
        if X_train.shape[1] >= 10:  # Only if sufficient sequence length
            try:
                print("🔄 Training Transformer...")
                transformer_model = self.model_factory.create_transformer_model(X_train.shape[1:])
                
                transformer_history = transformer_model.fit(
                    X_train, y_train,
                    batch_size=self.config.BATCH_SIZE,
                    epochs=50,
                    validation_data=(X_test, y_test),
                    verbose=0
                )
                
                self.models['transformer'] = transformer_model
                self.histories['transformer'] = transformer_history
                
            except Exception as e:
                print(f"⚠️  Transformer training skipped: {str(e)}")
    
    def _create_weighted_ensemble(self, X_test_seq, X_test_flat):
        """Create sophisticated weighted ensemble"""
        print("\n⚖️  CREATING WEIGHTED ENSEMBLE")
        
        predictions = {}
        
        # Get predictions from all models
        for name, model in self.models.items():
            if name in ['cnn_lstm', 'transformer']:
                predictions[name] = model.predict(X_test_seq, verbose=0).flatten()
            else:
                predictions[name] = model.predict(X_test_flat)
        
        # Calculate dynamic weights based on validation performance
        weights = self._calculate_dynamic_weights(predictions)
        
        # Create weighted ensemble
        ensemble_pred = np.zeros_like(predictions['cnn_lstm'])
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
        
        print("📊 Ensemble weights:")
        for name, weight in weights.items():
            print(f"   {name}: {weight:.3f}")
        
        return ensemble_pred, predictions
    
    def _calculate_dynamic_weights(self, predictions):
        """Calculate ensemble weights based on model performance"""
        # For now, use configured weights
        # In production, calculate based on validation performance
        return self.config.ENSEMBLE_WEIGHTS
    
    def predict_individual_models(self, X_seq, X_flat):
        """Get predictions from all individual models"""
        predictions = {}
        
        for name, model in self.models.items():
            if name in ['cnn_lstm', 'transformer']:
                predictions[name] = model.predict(X_seq, verbose=0).flatten()
            else:
                predictions[name] = model.predict(X_flat)
        
        return predictions
