"""
CONFIGURATION FOR REAL OPM DATA INTEGRATION
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
    for directory in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # REAL OPM DATASET CONFIG
    DATASETS = {
        'spe9': {
            'description': 'SPE9 Benchmark - Industry Standard',
            'wells': 10,
            'producers': 6,
            'injectors': 4,
            'grid': '24x25x15',
            'initial_pressure': 3600,
            'initial_oil_rate': 4500
        }
    }
    
    # MODEL CONFIG
    SEQUENCE_LENGTH = 45
    BATCH_SIZE = 64
    EPOCHS = 200
    LEARNING_RATE = 0.001
    RANDOM_STATE = 42

config = Config()
