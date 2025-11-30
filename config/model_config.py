from dataclasses import dataclass
from typing import List, Tuple
import torch

@dataclass
class ReservoirConfig:
    grid_dims: Tuple[int, int, int] = (24, 25, 15)
    porosity_range: Tuple[float, float] = (0.1, 0.3)
    permeability_range: Tuple[float, float] = (10.0, 500.0)

@dataclass  
class TemporalModelConfig:
    input_channels: int = 10
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    bidirectional: bool = True

@dataclass
class PhysicsInformedConfig:
    use_physics: bool = True
    continuity_weight: float = 1.0
    darcy_weight: float = 1.0
    boundary_weight: float = 0.5
