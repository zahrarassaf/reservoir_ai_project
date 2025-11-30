from dataclasses import dataclass
from typing import Dict, List, Optional
import torch

@dataclass
class SPE9GridConfig:
    nx: int = 24
    ny: int = 25  
    nz: int = 15
    total_blocks: int = 9000
    dx: float = 300.0
    dy: float = 300.0
    dz: List[float] = None
    
    def __post_init__(self):
        if self.dz is None:
            self.dz = [20, 15, 26, 15, 16, 14, 8, 8, 18, 12, 19, 18, 20, 50, 100]

@dataclass
class ReservoirProperties:
    initial_pressure: float = 3600.0
    datum_depth: float = 9035.0
    water_oil_contact: float = 9950.0
    gas_oil_contact: float = 8800.0
    porosity_layers: List[float] = None
    rock_compressibility: float = 4e-6
    
    def __post_init__(self):
        if self.porosity_layers is None:
            self.porosity_layers = [
                0.087, 0.097, 0.111, 0.16, 0.13, 0.17, 0.17, 
                0.08, 0.14, 0.13, 0.12, 0.105, 0.12, 0.116, 0.157
            ]

@dataclass
class EnsembleModelConfig:
    input_features: List[str] = None
    output_features: List[str] = None
    hidden_layers: List[int] = None
    dropout_rate: float = 0.2
    activation: str = "gelu"
    n_models: int = 5
    diversity_regularization: float = 0.1
    uncertainty_calibration: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 20
    max_epochs: int = 1000
    
    def __post_init__(self):
        if self.input_features is None:
            self.input_features = [
                'PERMX', 'PORO', 'DEPTH', 'REGION', 'SWAT', 'SGAS', 'PRESSURE'
            ]
        if self.output_features is None:
            self.output_features = [
                'FOPR', 'FGPR', 'FWPR', 'FGOR', 'BHP', 'WOPR', 'WGPR', 'WWPR'
            ]
        if self.hidden_layers is None:
            self.hidden_layers = [512, 256, 128, 64]
