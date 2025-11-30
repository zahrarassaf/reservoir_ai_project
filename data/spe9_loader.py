import numpy as np
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, Any
import logging
from .base_loader import BaseReservoirLoader

class SPE9Loader(BaseReservoirLoader):
    """SPE9 case loader with synthetic data for rapid development"""
    
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self._grid_dims = (24, 25, 15)  # Standard SPE9 dimensions
        self._total_cells = 24 * 25 * 15
        
    def _validate_paths(self) -> None:
        """Only check directory existence"""
        if not self.config.opm_data_dir.exists():
            self.logger.warning(
                f"OPM data directory not found: {self.config.opm_data_dir}"
                f"Using synthetic data for development."
            )
    
    def load_grid_geometry(self) -> Dict[str, np.ndarray]:
        """Create synthetic grid geometry"""
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
        
        self.logger.info(f"Created synthetic grid: {nx}x{ny}x{nz}")
        return geometry
    
    def load_static_properties(self) -> Dict[str, np.ndarray]:
        """Create synthetic static properties based on SPE9 characteristics"""
        nx, ny, nz = self._grid_dims
        
        # Create realistic distributions based on SPE9
        np.random.seed(42)  # For reproducibility
        
        # Porosity - realistic SPE9 range
        poro = np.random.uniform(0.1, 0.3, (nx, ny, nz))
        
        # Permeability - based on actual SPE9 values (in milliDarcy)
        permx = self._create_geological_permeability(nx, ny, nz)
        permy = permx * np.random.uniform(0.8, 1.2, (nx, ny, nz))
        permz = permx * np.random.uniform(0.01, 0.1, (nx, ny, nz))
        
        properties = {
            'PORO': poro,
            'PERMX': permx,
            'PERMY': permy,
            'PERMZ': permz,
            'NTG': np.ones((nx, ny, nz)),
            'DEPTH': np.random.uniform(2000, 2500, (nx, ny, nz))
        }
        
        self.logger.info("Generated synthetic static properties")
        return properties
    
    def _create_geological_permeability(self, nx: int, ny: int, nz: int) -> np.ndarray:
        """Create permeability field with realistic geological patterns"""
        # Create random base field
        base_perm = np.random.lognormal(mean=4.0, sigma=1.0, size=(nx, ny, nz))
        
        # Add spatial trends
        x_trend = np.linspace(0.5, 1.5, nx).reshape(-1, 1, 1)
        y_trend = np.linspace(0.7, 1.3, ny).reshape(1, -1, 1)
        z_trend = np.linspace(0.8, 1.2, nz).reshape(1, 1, -1)
        
        # Add high-permeability channels (similar to SPE9)
        channels = self._create_channel_features(nx, ny, nz)
        
        # Combine all elements
        perm_field = base_perm * x_trend * y_trend * z_trend * channels
        
        # Normalize to realistic SPE9 range
        perm_field = np.clip(perm_field, 10, 2000)
        
        return perm_field
    
    def _create_channel_features(self, nx: int, ny: int, nz: int) -> np.ndarray:
        """Create channel features similar to real reservoir"""
        channels = np.ones((nx, ny, nz))
        
        # Create sinuous channel patterns
        for layer in range(nz):
            for i in range(3):  # Create 3 main channels
                channel_center = ny / 3 * (i + 0.5) + np.sin(np.linspace(0, 4*np.pi, nx)) * 3
                channel_width = 2 + np.random.random() * 3
                
                for x in range(nx):
                    for y in range(ny):
                        distance = abs(y - channel_center[x])
                        if distance < channel_width:
                            # High permeability in channels
                            channels[x, y, layer] *= np.random.uniform(2.0, 5.0)
        
        return channels
    
    def load_dynamic_properties(self, time_steps: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """Generate synthetic dynamic properties for training"""
        if time_steps is None:
            time_steps = self.config.time_steps
        
        static_props = self.load_static_properties()
        geometry = self.load_grid_geometry()
        nx, ny, nz = geometry['dimensions']
        
        dynamic_data = {}
        
        for time_step in time_steps:
            time_key = f"T{time_step:03d}"
            dynamic_data[time_key] = {}
            
            # Create pressure depletion over time
            base_pressure = 3000 - time_step * 10  # Simple depletion model
            pressure_variation = np.random.normal(0, 50, (nx, ny, nz))
            dynamic_data[time_key]['PRESSURE'] = base_pressure + pressure_variation
            
            # Create saturation changes
            swat_init = static_props['PORO'] * 0.6 + np.random.normal(0, 0.05, (nx, ny, nz))
            swat = swat_init + time_step * 0.01  # Water saturation increases over time
            dynamic_data[time_key]['SWAT'] = np.clip(swat, 0.1, 0.8)
            dynamic_data[time_key]['SOIL'] = 1.0 - dynamic_data[time_key]['SWAT']
            
            # Add some noise to simulate real data
            for key in ['PRESSURE', 'SWAT', 'SOIL']:
                dynamic_data[time_key][key] += np.random.normal(0, 0.02, (nx, ny, nz))
        
        self.logger.info(f"Generated dynamic properties for {len(time_steps)} time steps")
        return dynamic_data
    
    def get_training_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training sequences for neural network training"""
        static_props = self.load_static_properties()
        dynamic_data = self.load_dynamic_properties()
        
        # Combine static and dynamic features
        nx, ny, nz = self._grid_dims
        sequence_length = self.config.sequence_length
        num_sequences = min(1000, nx * ny)  # Limit for memory
        
        # Initialize arrays
        features = []
        targets = []
        
        # Create sequences
        for seq_idx in range(num_sequences):
            # Random cell location
            i, j, k = np.random.randint(0, nx), np.random.randint(0, ny), np.random.randint(0, nz)
            
            # Static features for this cell
            static_feat = [
                static_props['PORO'][i, j, k],
                static_props['PERMX'][i, j, k],
                static_props['PERMY'][i, j, k], 
                static_props['PERMZ'][i, j, k],
                static_props['DEPTH'][i, j, k]
            ]
            
            # Dynamic sequence
            dyn_sequence = []
            target_sequence = []
            
            # Create input sequence and targets
            for t in range(sequence_length + self.config.prediction_horizon):
                time_key = f"T{t:03d}"
                if time_key in dynamic_data:
                    dyn_feat = [
                        dynamic_data[time_key]['PRESSURE'][i, j, k],
                        dynamic_data[time_key]['SWAT'][i, j, k],
                        dynamic_data[time_key]['SOIL'][i, j, k]
                    ]
                    
                    if t < sequence_length:
                        # Input features: static + dynamic
                        dyn_sequence.append(static_feat + dyn_feat)
                    else:
                        # Target: only dynamic properties
                        target_sequence.append(dyn_feat)
            
            if len(dyn_sequence) == sequence_length and len(target_sequence) == self.config.prediction_horizon:
                features.append(dyn_sequence)
                targets.append(target_sequence)
        
        features_array = np.array(features)
        targets_array = np.array(targets)
        
        self.logger.info(f"Created training sequences: {features_array.shape} -> {targets_array.shape}")
        return features_array, targets_array
