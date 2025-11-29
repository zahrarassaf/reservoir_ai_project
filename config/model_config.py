# config/model_config.py

class Config:
    # Data Configuration
    SEQUENCE_LENGTH = 30
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    RANDOM_STATE = 42
    
    # Feature Engineering
    FEATURE_SELECTION_THRESHOLD = 0.85
    ROLLING_WINDOWS = [3, 7, 14]
    
    # Model Architecture
    CNN_LSTM_CONFIG = {
        'filters': [64, 32],
        'kernel_size': 3,
        'lstm_units': [100, 50],
        'dense_units': [64, 32],
        'dropout_rate': 0.3,
        'learning_rate': 0.001
    }
    
    # Training Parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    PATIENCE = 15
    MIN_DELTA = 0.001
    
    # Ensemble Weights
    ENSEMBLE_WEIGHTS = {
        'cnn_lstm': 0.4,
        'random_forest': 0.2,
        'xgboost': 0.2,
        'lightgbm': 0.2
    }

class ModelFactoryConfig:
    RANDOM_FOREST_PARAMS = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': Config.RANDOM_STATE,
        'n_jobs': -1
    }
    
    XGBOOST_PARAMS = {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': Config.RANDOM_STATE,
        'n_jobs': -1
    }
    
    LIGHTGBM_PARAMS = {
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': Config.RANDOM_STATE,
        'n_jobs': -1
    }
