from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path

class BaseReservoirLoader(ABC):
    def __init__(self, config):
        self.config = config
        self._validate_paths()
    
    @abstractmethod
    def load_grid_geometry(self) -> Dict[str, np.ndarray]:
        pass
    
    @abstractmethod
    def load_static_properties(self) -> Dict[str, np.ndarray]:
        pass
    
    @abstractmethod
    def load_dynamic_properties(self, time_steps: List[int]) -> Dict[str, np.ndarray]:
        pass
    
    @abstractmethod
    def _validate_paths(self) -> None:
        pass
    
    def get_grid_info(self) -> Dict[str, any]:
        geometry = self.load_grid_geometry()
        return {
            'dimensions': geometry.get('dimensions', (0, 0, 0)),
            'num_cells': geometry.get('num_cells', 0),
            'active_cells': geometry.get('active_cells', 0)
        }
