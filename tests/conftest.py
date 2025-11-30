import pytest
import torch
import numpy as np
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_configs():
    from config.data_config import DataConfig
    from config.model_config import TemporalModelConfig, EnsembleConfig
    from config.training_config import TrainingConfig
    
    data_config = DataConfig()
    model_config = TemporalModelConfig(input_channels=5, output_channels=2, hidden_dim=64)
    ensemble_config = EnsembleConfig(num_models=3)
    training_config = TrainingConfig(batch_size=16, num_epochs=2)
    
    return {
        'data_config': data_config,
        'model_config': model_config,
        'ensemble_config': ensemble_config,
        'training_config': training_config
    }

@pytest.fixture
def sample_data():
    batch_size, seq_len, num_features = 8, 10, 5
    X = torch.randn(batch_size, seq_len, num_features)
    y = torch.randn(batch_size, seq_len, 2)
    return X, y

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
