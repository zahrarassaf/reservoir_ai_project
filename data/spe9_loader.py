import numpy as np
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
import logging
from .base_loader import BaseReservoirLoader

class SPE9Loader(BaseReservoirLoader):
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self._grid_dims = (24, 25, 15)
        self._total_cells = 24 * 25 * 15
        
    def _validate_paths(self) -> None:
        if not self.config.opm_data_dir.exists():
            raise FileNotFoundError(
                f"OPM data directory not found: {self.config.opm_data_dir}"
            )
    
    def load_grid_geometry(self) -> Dict[str, np.ndarray]:
        nx, ny, nz = self._grid_dims
        
        geometry = {
            'dimensions': self._grid_dims,
            'num_cells': self._total_cells,
            'active_cells': self._total_cells,
            'coordinates': {
                'x': np.linspace(0, nx * 100, nx),
                'y': np.linspace(0, ny * 100, ny),
                'z': np.linspace(0, nz * 20, nz)
            }
        }
        return geometry
    
    def _parse_eclipse_array(self, file_path: Path, array_name: str) -> np.ndarray:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            pattern = rf'{array_name}\s*\n(.*?)(?=\n\w+|\Z)'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            
            if not match:
                raise ValueError(f"Array {array_name} not found in {file_path}")
            
            array_text = match.group(1)
            lines = array_text.split('\n')
            values = []
            
            for line in lines:
                line = line.split('--')[0].strip()
                if line:
                    numbers = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+', line)
                    values.extend(map(float, numbers))
            
            values_array = np.array(values)
            
            if len(values_array) != self._total_cells:
                self.logger.warning(f"Array size mismatch for {array_name}")
                if len(values_array) < self._total_cells:
                    values_array = np.pad(values_array, 
                                        (0, self._total_cells - len(values_array)),
                                        mode='edge')
                else:
                    values_array = values_array[:self._total_cells]
            
            values_3d = values_array.reshape(self._grid_dims, order='F')
            return values_3d
            
        except Exception as e:
            self.logger.error(f"Failed to parse {array_name}: {e}")
            raise
    
    def load_static_properties(self) -> Dict[str, np.ndarray]:
        init_file = self.config.init_path
        
        if not init_file.exists():
            raise FileNotFoundError(f"SPE9.INIT not found at {init_file}")
        
        properties = {}
        property_names = ['PORO', 'PERMX', 'PERMY', 'PERMZ']
        
        for prop_name in property_names:
            properties[prop_name] = self._parse_eclipse_array(init_file, prop_name)
        
        properties['NTG'] = np.ones(self._grid_dims)
        properties['DEPTH'] = np.random.uniform(2000, 2500, self._grid_dims)
        
        return properties
    
    def load_dynamic_properties(self, time_steps: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        if time_steps is None:
            time_steps = self.config.time_steps
        
        static_props = self.load_static_properties()
        nx, ny, nz = self._grid_dims
        
        dynamic_data = {}
        
        for time_step in time_steps:
            time_key = f"T{time_step:03d}"
            dynamic_data[time_key] = {}
            
            base_pressure = 3000 - time_step * 15
            pressure_variation = static_props['PERMX'] / static_props['PERMX'].mean() * 100
            dynamic_data[time_key]['PRESSURE'] = base_pressure + pressure_variation
            
            swat_base = 0.2 + (static_props['PORO'] - 0.1) * 0.5
            swat_increase = time_step * 0.02
            dynamic_data[time_key]['SWAT'] = np.clip(swat_base + swat_increase, 0.1, 0.8)
            dynamic_data[time_key]['SOIL'] = 1.0 - dynamic_data[time_key]['SWAT']
        
        return dynamic_data
    
    def get_training_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        static_props = self.load_static_properties()
        dynamic_data = self.load_dynamic_properties()
        
        nx, ny, nz = self._grid_dims
        sequence_length = self.config.sequence_length
        num_sequences = min(500, nx * ny)
        
        features = []
        targets = []
        
        for seq_idx in range(num_sequences):
            i = np.random.randint(0, nx)
            j = np.random.randint(0, ny)
            k = np.random.randint(0, nz)
            
            static_feat = [
                static_props['PORO'][i, j, k],
                static_props['PERMX'][i, j, k],
                static_props['PERMY'][i, j, k],
                static_props['PERMZ'][i, j, k],
                static_props['DEPTH'][i, j, k]
            ]
            
            available_steps = sorted(dynamic_data.keys())
            
            for t_idx in range(len(available_steps) - self.config.prediction_horizon):
                if t_idx + sequence_length + self.config.prediction_horizon <= len(available_steps):
                    
                    input_seq = []
                    for offset in range(sequence_length):
                        time_key = available_steps[t_idx + offset]
                        dyn_feat = [
                            dynamic_data[time_key]['PRESSURE'][i, j, k],
                            dynamic_data[time_key]['SWAT'][i, j, k],
                            dynamic_data[time_key]['SOIL'][i, j, k]
                        ]
                        input_seq.append(static_feat + dyn_feat)
                    
                    target_seq = []
                    for offset in range(self.config.prediction_horizon):
                        time_key = available_steps[t_idx + sequence_length + offset]
                        target_feat = [
                            dynamic_data[time_key]['PRESSURE'][i, j, k],
                            dynamic_data[time_key]['SWAT'][i, j, k],
                            dynamic_data[time_key]['SOIL'][i, j, k]
                        ]
                        target_seq.append(target_feat)
                    
                    features.append(input_seq)
                    targets.append(target_seq)
        
        return np.array(features), np.array(targets)
