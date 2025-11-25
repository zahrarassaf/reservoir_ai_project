"""
Configuration for Reservoir AI Project
"""
from pathlib import Path
import os

class Config:
    """Central configuration management"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    MODEL_DIR = PROJECT_ROOT / "models"
    RESULT_DIR = PROJECT_ROOT / "results"
    
    # Data settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    SEQUENCE_LENGTH = 10
    N_WELLS = 26
    TIME_STEPS = 200
    
    # Model parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    
    def __init__(self):
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        for directory in [self.DATA_DIR, self.PROCESSED_DIR, self.MODEL_DIR, self.RESULT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

config = Config()
