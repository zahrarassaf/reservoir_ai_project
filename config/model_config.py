from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class ReservoirConfig:
    grid_dimensions: Tuple[int, int, int] = (24, 25, 15)
    porosity_range: Tuple[float, float] = (0.1, 0.3)
    permeability_range: Tuple[float, float] = (10.0, 500.0)
    depth_range: Tuple[float, float] = (2000.0, 2500.0)
    rock_compressibility: float = 1e-5
    reference_pressure: float = 3000.0

@dataclass
class TemporalModelConfig:
    input_channels: int = 8
    output_channels: int = 3
    hidden_dim: int = 128
    num_layers: int = 3
    model_type: str = "lstm"
    bidirectional: bool = True
    dropout: float = 0.2
    sequence_length: int = 10
    use_attention: bool = True
    
    def validate_architecture(self) -> None:
        valid_models = ["lstm", "gru", "tcn"]
        if self.model_type not in valid_models:
            raise ValueError(f"Model type must be one of {valid_models}")

@dataclass
class PhysicsInformedConfig:
    use_physics_constraints: bool = True
    continuity_weight: float = 1.0
    darcy_weight: float = 1.0
    boundary_weight: float = 0.5
    data_weight: float = 1.0
    fluid_density: float = 1000.0
    viscosity: float = 1.0
    gravity: float = 9.81

@dataclass
class EnsembleConfig:
    num_models: int = 5
    diversity_method: str = "random_init"
    diversity_weight: float = 0.1
    bootstrap_ratio: float = 0.8
    early_stopping_patience: int = 10
    model_selection_metric: str = "val_loss"
