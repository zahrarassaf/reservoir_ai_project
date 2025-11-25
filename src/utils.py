"""
Utility functions and helpers
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
import joblib

from .config import config

def ensure_directories():
    """Ensure all necessary directories exist"""
    directories = [
        config.DATA_DIR, config.SPE9_DIR, config.PROCESSED_DIR,
        config.MODEL_DIR, config.RESULT_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def save_model(model, filename: str):
    """Save trained model to file"""
    ensure_directories()
    filepath = config.MODEL_DIR / filename
    
    if hasattr(model, 'save'):
        # Keras model
        model.save(filepath)
    else:
        # Scikit-learn model
        joblib.dump(model, filepath)
    
    print(f"Model saved: {filepath}")

def load_model(filename: str):
    """Load trained model from file"""
    filepath = config.MODEL_DIR / filename
    
    if filepath.suffix == '.h5':
        # Keras model
        from tensorflow.keras.models import load_model as load_keras_model
        return load_keras_model(filepath)
    else:
        # Scikit-learn model
        return joblib.load(filepath)

def save_plot(fig, filename: str, dpi: int = 300):
    """Save matplotlib figure"""
    ensure_directories()
    filepath = config.RESULT_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Plot saved: {filepath}")

def calculate_confidence_intervals(predictions: np.ndarray, 
                                 confidence: float = 0.95) -> tuple:
    """Calculate confidence intervals for predictions"""
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    z_score = 1.96  # For 95% confidence
    lower_bound = mean_pred - z_score * std_pred
    upper_bound = mean_pred + z_score * std_pred
    
    return lower_bound, upper_bound, mean_pred

def create_time_based_split(data: pd.DataFrame, 
                          test_size: float = 0.2,
                          time_column: str = 'Time') -> tuple:
    """Create time-based train-test split"""
    sorted_data = data.sort_values(time_column)
    split_index = int(len(sorted_data) * (1 - test_size))
    
    train_data = sorted_data.iloc[:split_index]
    test_data = sorted_data.iloc[split_index:]
    
    return train_data, test_data

def check_data_quality(df: pd.DataFrame) -> dict:
    """Perform basic data quality checks"""
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinite_count = np.isinf(df[numeric_cols]).sum().sum()
    quality_report['infinite_values'] = infinite_count
    
    # Basic statistics
    quality_report['basic_stats'] = df[numeric_cols].describe().to_dict()
    
    return quality_report

def set_display_options():
    """Set pandas display options for better readability"""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.precision', 4)

class Timer:
    """Simple timer utility"""
    def __init__(self):
        import time
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
        return self
    
    def stop(self):
        self.end_time = time.time()
        return self.end_time - self.start_time
    
    def lap(self):
        if self.start_time is None:
            raise ValueError("Timer not started")
        return time.time() - self.start_time
