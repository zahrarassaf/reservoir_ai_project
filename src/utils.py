"""
PRODUCTION UTILITY FUNCTIONS
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

from .config import config

def save_predictions(predictions: dict, actual: np.ndarray, filename: str = "predictions.csv"):
    """SAVE PREDICTION RESULTS"""
    predictions_df = pd.DataFrame(predictions)
    predictions_df['actual'] = actual
    
    results_path = config.RESULTS_DIR / filename
    predictions_df.to_csv(results_path, index=False)
    print(f"💾 PREDICTIONS SAVED: {results_path}")

def setup_directories():
    """SETUP REQUIRED DIRECTORIES"""
    for directory in [config.DATA_RAW, config.DATA_PROCESSED, 
                     config.MODELS_DIR, config.RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ ALL DIRECTORIES CREATED")

def get_project_info():
    """GET PROJECT INFORMATION"""
    return {
        "project_name": "Reservoir AI",
        "version": "1.0.0",
        "datasets": list(config.DATASET_SPECS.keys()),
        "sequence_length": config.SEQUENCE_LENGTH,
        "models": ["CNN-LSTM", "Random Forest", "XGBoost", "LightGBM"]
    }
