"""
PRODUCTION CONFIGURATION FOR RESERVOIR AI
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
    
    # SPE9 DATASET SPECIFICATIONS
    DATASET_SPECS = {
        'spe9': {
            'wells': 24,
            'producer_ratio': 0.6,
            'time_steps': 1000,
            'simulation_years': 10,
            'initial_pressure': 3600,
            'initial_oil_rate': 5000,
            'opm_data_url': 'https://github.com/OPM/opm-data/blob/master/spe9/SPE9.DATA'
        }
    }
    
    # MODEL HYPERPARAMETERS
    SEQUENCE_LENGTH = 30
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    RANDOM_STATE = 42

config = Config()
