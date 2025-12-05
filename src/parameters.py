import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class SimulationParameters:
    forecast_years: int = 3
    oil_price: float = 75.0
    operating_cost: float = 18.0
    discount_rate: float = 0.1
    initial_investment: float = 100.0
    abandonment_pressure: float = 500.0
    reservoir_temperature: float = 180.0
    rock_compressibility: float = 3e-6
    water_compressibility: float = 3e-6
    oil_compressibility: float = 10e-6
    gas_compressibility: float = 500e-6
    aquifer_size: float = 1000000.0
    aquifer_permeability: float = 100.0
    relative_permeability_type: str = "stone"
    hysteresis_model: str = "killough"
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)
