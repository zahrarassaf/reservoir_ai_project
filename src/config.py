"""
PROJECT CONFIGURATION - FIXED VERSION
"""
from pathlib import Path
import numpy as np

class Config:
    # PATHS
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" 
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # CREATE DIRECTORIES
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # DATASET CONFIG
    DATASETS = {
        'spe9': {'wells': 10, 'time_steps': 100, 'years': 30},
        'norne': {'wells': 30, 'time_steps': 150, 'years': 25},
        'spe10': {'wells': 25, 'time_steps': 80, 'years': 20}
    }
    
    # MODEL CONFIG
    SEQUENCE_LENGTH = 30
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    RANDOM_STATE = 42

# SINGLE CONFIG INSTANCE
config = Config()
