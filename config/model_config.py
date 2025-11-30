from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch

@dataclass
class ReservoirConfig:
    """Configuration for reservoir physical properties"""
    grid_dimensions: Tuple[int, int, int] = (24, 25, 15)
    porosity_range: Tuple[float, float] = (0.1, 0.3)
    permeability_range: Tuple[float, float] = (10.0, 500.0)  # mD
    depth_range: Tuple[float, float] = (2000.0, 2500.0)  # meters
    
    # Rock properties
    rock_compressibility: float = 1e-5  # 1/psi
    reference_pressure: float = 3000.0  # psi

@dataclass
class TemporalModelConfig:
    """Configuration for temporal neural network architecture"""
    # Input/Output dimensions
    input_channels: int = 8  # pressure, saturations, etc.
    output_channels: int = 3  # pressure, oil_sat, water_sat
    hidden_dim: int = 128
    num_layers: int = 3
    
    # Architecture choices
    model_type: str = "lstm"  # "lstm", "gru", "tcn"
    bidirectional: bool = True
    dropout: float = 0.2
    
    # Temporal processing
    sequence_length: int = 10
    use_attention: bool = True
    
    def validate_architecture(self) -> None:
        """Validate model architecture parameters"""
        valid_models = ["lstm", "gru", "tcn"]
        if self.model_type not in valid_models:
            raise ValueError(f"Model type must be one of {valid_models}")

@dataclass
class PhysicsInformedConfig:
    """Configuration for physics-informed constraints"""
    use_physics_constraints: bool = True
    
    # Loss weights
    continuity_weight: float = 1.0
    darcy_weight: float = 1.0
    boundary_weight: float = 0.5
    data_weight: float = 1.0
    
    # Physical parameters
    fluid_density: float = 1000.0  # kg/m³
    viscosity: float = 1.0  # cP
    gravity: float = 9.81  # m/s²

@dataclass
class EnsembleConfig:
    """Configuration for ensemble modeling"""
    num_models: int = 5
    diversity_method: str = "random_init"  # "random_init", "bootstrap", "architecture"
    
    # Regularization
    diversity_weight: float = 0.1
    bootstrap_ratio: float = 0.8
    
    # Training
    early_stopping_patience: int = 10
    model_selection_metric: str = "val_loss"
