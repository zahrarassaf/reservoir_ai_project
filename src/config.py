"""
PRODUCTION CONFIGURATION FOR RESERVOIR AI
OPTIMIZED FOR PERFORMANCE AND SCALABILITY
"""
from pathlib import Path
import numpy as np

class ReservoirConfig:
    # PATHS
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # CREATE DIRECTORIES
    for directory in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # DATASET CONFIGURATION
    DATASET_SPECS = {
        'spe9': {
            'wells': 24,
            'producer_ratio': 0.6,
            'time_steps': 2000,
            'simulation_years': 15,
            'initial_pressure': 3600,
            'initial_oil_rate': 5000
        },
        'norne': {
            'wells': 32,
            'producer_ratio': 0.65,
            'time_steps': 2500,
            'simulation_years': 20,
            'initial_pressure': 3800,
            'initial_oil_rate': 4500
        }
    }
    
    # MODEL HYPERPARAMETERS
    SEQUENCE_LENGTH = 45
    BATCH_SIZE = 64
    EPOCHS = 200
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.15
    RANDOM_STATE = 42
    
    # FEATURE CONFIGURATION
    NUMERICAL_FEATURES = [
        'bottomhole_pressure', 'oil_rate', 'water_rate', 'gas_rate',
        'cumulative_oil', 'cumulative_water', 'cumulative_gas',
        'permeability', 'porosity', 'well_depth'
    ]
    
    TEMPORAL_FEATURES = [
        'pressure_derivative', 'rate_derivative', 'moving_avg_7',
        'moving_avg_30', 'seasonal_factor'
    ]
    
    # TRAINING CONFIG
    EARLY_STOPPING_PATIENCE = 25
    REDUCE_LR_PATIENCE = 15
    MODEL_CHECKPOINT_MONITOR = 'val_loss'

# GLOBAL CONFIG INSTANCE
config = ReservoirConfig()
