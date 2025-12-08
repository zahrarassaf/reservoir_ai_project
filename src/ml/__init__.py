"""
Reservoir AI - Machine Learning Module
Advanced ML models for reservoir engineering and economics
"""

from .cnn_reservoir import CNNReservoirPredictor, PropertyPredictor, ReservoirDataset
from .svr_economics import SVREconomicPredictor, EconomicFeatureEngineer

__all__ = [
    'CNNReservoirPredictor', 
    'PropertyPredictor', 
    'ReservoirDataset',
    'SVREconomicPredictor', 
    'EconomicFeatureEngineer'
]

__version__ = '1.0.0'
