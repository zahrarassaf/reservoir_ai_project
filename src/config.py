"""
Configuration Module - Reservoir AI Project
Professional settings for reproducibility and maintainability
"""
import os
import numpy as np
from pathlib import Path

class Config:
    """Central configuration for the entire project"""
    
    # 🎯 Project Structure
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "trained_models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    SRC_DIR = PROJECT_ROOT / "src"
    
    # 📊 Data Configuration
    DATA_FILE = DATA_DIR / "well_logs.csv"
    ORIGINAL_FEATURES = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
    TARGETS = ['Permeability', 'Porosity']
    
    # 🧪 Modeling Parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    CV_FOLDS = 5
    
    # 🔧 Model Hyperparameters
    RF_PARAMS = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    SVM_PARAMS = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    
    XGB_PARAMS = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    # 📈 Evaluation Metrics
    REGRESSION_METRICS = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    
    def __init__(self):
        """Initialize directories"""
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [self.DATA_DIR, self.MODELS_DIR, self.RESULTS_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        print("✅ All directories created successfully")

# Global configuration instance
config = Config()
