"""
Configuration settings for Reservoir AI project
"""
from pathlib import Path
import os

class Config:
    """Central configuration management"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    SPE9_DIR = DATA_DIR / "spe9"
    PROCESSED_DIR = DATA_DIR / "processed"
    MODEL_DIR = PROJECT_ROOT / "models"
    RESULT_DIR = PROJECT_ROOT / "results"
    
    # Data settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    SEQUENCE_LENGTH = 10
    N_WELLS = 25
    TIME_STEPS = 200
    
    # Model parameters
    BATCH_SIZE = 32
    EPOCHS = 200
    PATIENCE = 15
    
    # Feature engineering
    LAGS = [1, 2, 3]
    ROLLING_WINDOWS = [3, 5, 7]
    
    # SVR hyperparameters
    SVR_C_VALUES = [0.1, 1, 10, 100]
    SVR_GAMMA_VALUES = [0.001, 0.01, 0.1, 1]
    SVR_EPSILON_VALUES = [0.01, 0.1, 0.5]
    
    def __init__(self):
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.DATA_DIR, self.SPE9_DIR, self.PROCESSED_DIR,
            self.MODEL_DIR, self.RESULT_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

config = Config()
