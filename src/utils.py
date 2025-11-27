"""
UTILITY FUNCTIONS
"""
import pandas as pd
import numpy as np
from pathlib import Path

from .config import config

def save_results(predictions, actual, filepath):
    results_df = pd.DataFrame(predictions)
    results_df['actual'] = actual
    results_df.to_csv(filepath, index=False)
    print(f"Results saved: {filepath}")

def load_results(filepath):
    return pd.read_csv(filepath)

def setup_directories():
    for directory in [config.DATA_RAW, config.DATA_PROCESSED, 
                     config.MODELS_DIR, config.RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
