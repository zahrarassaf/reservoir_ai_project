from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

@dataclass
class SPE9GridConfig:
    """Configuration for SPE9 reservoir grid"""
    nx: int = 24
    ny: int = 25  
    nz: int = 15
    total_blocks: int = 9000
    dx: float = 300.0  # ft
    dy: float = 300.0  # ft
    dz: List[float] = None
    
    def __post_init__(self):
        if self.dz is None:
            self.dz = [20, 15, 26, 15, 16, 14, 8, 8, 18, 12, 19, 18, 20, 50, 100]

@dataclass  
class ReservoirPhysicsConfig:
    """Physics-based constraints for reservoir simulation"""
    oil_density: float = 44.9856  # lb/ft³
    water_density: float = 63.0210  # lb/ft³
    gas_density: float = 0.07039  # lb/ft³
    water_compressibility: float = 3e-6  # psi⁻¹
    rock_compressibility: float = 4e-6  # psi⁻¹
    initial_pressure: float = 3600.0  # psi
    reference_depth: float = 9035.0  # ft

@dataclass
class TemporalModelConfig:
    """Configuration for temporal modeling"""
    sequence_length: int = 30  # days for input sequence
    prediction_horizon: int = 10  # days to predict ahead
    time_features: List[str] = None
    
    def __post_init__(self):
        if self.time_features is None:
            self.time_features = ['FOPR', 'FGPR', 'FWPR', 'FGOR', 'BHP']

@dataclass
class PhysicsInformedConfig:
    """Physics-informed neural network constraints"""
    darcy_weight: float = 1.0
    mass_balance_weight: float = 1.0
    boundary_weight: float = 0.1
    use_physics_loss: bool = True

@dataclass
class EnsembleConfig:
    """Ensemble model configuration with proper diversity"""
    n_models: int = 5
    architecture_variants: List[Dict] = None
    bootstrap_sampling: bool = True
    feature_subsampling: float = 0.8
    
    def __post_init__(self):
        if self.architecture_variants is None:
            self.architecture_variants = [
                {'hidden_dims': [256, 128, 64], 'dropout': 0.3},
                {'hidden_dims': [512, 256, 128], 'dropout': 0.2},
                {'hidden_dims': [128, 64, 32], 'dropout': 0.4},
            ]
