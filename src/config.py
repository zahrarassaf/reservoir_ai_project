"""
CONFIGURATION FOR MULTI-DATASET RESERVOIR AI
RUTHLESSLY OPTIMIZED FOR PERFORMANCE
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
    
    # DATASET SPECS
    DATASETS = {
        'spe9': {
            'wells': 10,
            'grid_size': (24, 25, 15),
            'time_steps': 100,
            'years': 30
        },
        'norne': {
            'wells': 30, 
            'grid_size': (46, 112, 22),
            'time_steps': 150,
            'years': 25
        },
        'spe10': {
            'wells': 25,
            'grid_size': (60, 220, 85),
            'time_steps': 80,
            'years': 20
        }
    }
    
    # MODEL PARAMS
    SEQUENCE_LENGTH = 30
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    RANDOM_STATE = 42
    
    # FEATURE ENGINEERING
    NUMERICAL_FEATURES = [
        'BOTTOMHOLE_PRESSURE', 'FLOW_RATE_OIL', 'FLOW_RATE_WATER', 
        'FLOW_RATE_GAS', 'CUMULATIVE_OIL', 'CUMULATIVE_WATER', 'CUMULATIVE_GAS'
    ]
    
    CATEGORICAL_FEATURES = ['WELL_TYPE', 'WELL_GROUP']
    
config = Config()
