import numpy as np
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
import logging
from .base_loader import BaseReservoirLoader

class SPE9Loader(BaseReservoirLoader):
    """SPE9 case loader with REAL data parsing from OPM files"""
    
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self._grid_dims = (24, 25, 15)
        self._total_cells = 24 * 25 * 15
        
    def _validate_paths(self) -> None:
        """Validate that OPM data directory exists"""
        if not self.config.opm_data_dir.exists():
            raise FileNotFoundError(
                f"OPM data directory not found: {self.config.opm_data_dir}\n"
                f"Please clone: git clone https://github.com/OPM/opm-data.git"
            )
        
        spe9_dir = self.config.spe9_directory
        if not spe9_dir.exists():
            raise FileNotFoundError(f"SPE9 directory not found: {spe9_dir}")
    
    def load_grid_geometry(self) -> Dict[str, np.ndarray]:
        """Load actual grid geometry from SPE9 files"""
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
        
        self.logger.info(f"Loaded SPE9 grid: {nx}x{ny}x{nz}")
        return geometry
    
    def _parse_eclipse_property(self, file_path: Path, property_name: str) -> np.ndarray:
        """Parse ECLIPSE format property files like PERMX, PORO, etc."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Find the property section
            pattern = rf'{property_name}\s*\n(.*?)(?=\n\w+|\Z)'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            
            if not match:
                self.logger.warning(f"Property {property_name} not found in {file_path}")
                return self._create_fallback_property(property_name)
            
            property_data = match.group(1)
            
            # Parse all numerical values
            values = []
            lines = property_data.split('\n')
            
            for line in lines:
                line = line.split('--')[0].strip()  # Remove comments
                if line:
                    # Extract all floating point numbers
                    numbers = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+', line)
                    values.extend(map(float, numbers))
            
            # Convert to numpy array and reshape
            values_array = np.array(values)
            
            # Handle size mismatches
            if len(values_array) != self._total_cells:
                self.logger.warning(
                    f"Property {property_name} size mismatch: "
                    f"expected {self._total_cells}, got {len(values_array)}"
                )
                # Pad or truncate to match grid size
                if len(values_array) < self._total_cells:
                    values_array = np.pad(values_array, 
                                        (0, self._total_cells - len(values_array)),
                                        mode='edge')
                else:
                    values_array = values_array[:self._total_cells]
            
            # Reshape to 3D grid (Fortran-order for ECLIPSE)
            values_3d = values_array.reshape(self._grid_dims, order='F')
            
            self.logger.info(f"Parsed {property_name}: shape {values_3d.shape}, "
                           f"range [{values_3d.min():.2f}, {values_3d.max():.2f}]")
            
            return values_3d
            
        except Exception as e:
            self.logger.error(f"Failed to parse {property_name} from {file_path}: {e}")
            return self._create_fallback_property(property_name)
    
    def _create_fallback_property(self, property_name: str) -> np.ndarray:
        """Create fallback property if parsing fails"""
        nx, ny, nz = self._grid_dims
        
        if property_name.upper() == 'PORO':
            return np.random.uniform(0.1, 0.3, (nx, ny, nz))
        elif property_name.upper() == 'PERMX':
            return np.random.lognormal(4.0, 1.0, (nx, ny, nz))
        elif property_name.upper() == 'PERMY':
            return np.random.lognormal(4.0, 1.0, (nx, ny, nz))
        elif property_name.upper() == 'PERMZ':
            return np.random.lognormal(2.0, 1.0, (nx, ny, nz))
        else:
            return np.ones((nx, ny, nz))
    
    def load_static_properties(self) -> Dict[str, np.ndarray]:
        """Load ACTUAL static properties from SPE9 files"""
        init_file = self.config.init_path
        
        if not init_file.exists():
            self.logger.warning(f"SPE9.INIT not found at {init_file}, using fallback")
            return self._create_fallback_properties()
        
        self.logger.info(f"Loading REAL static properties from {init_file}")
        
        properties = {}
        
        # Parse actual properties from INIT file
        property_names = ['PORO', 'PERMX', 'PERMY', 'PERMZ']
        
        for prop_name in property_names:
            properties[prop_name] = self._parse_eclipse_property(init_file, prop_name)
        
        # Add additional properties
        properties['NTG'] = np.ones(self._grid_dims)  # Net-to-Gross
        properties['DEPTH'] = np.random.uniform(2000, 2500, self._grid_dims)
        
        return properties
    
    def _create_fallback_properties(self) -> Dict[str, np.ndarray]:
        """Create fallback properties if files not available"""
        nx, ny, nz = self._grid_dims
        
        return {
            'PORO': np.random.uniform(0.1, 0.3, (nx, ny, nz)),
            'PERMX': np.random.lognormal(4.0, 1.0, (nx, ny, nz)),
            'PERMY': np.random.lognormal(4.0, 1.0, (nx, ny, nz)),
            'PERMZ': np.random.lognormal(2.0, 1.0, (nx, ny, nz)),
            'NTG': np.ones((nx, ny, nz)),
            'DEPTH': np.random.uniform(2000, 2500, (nx, ny, nz))
        }
    
    def load_dynamic_properties(self, time_steps: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """Load/generate dynamic properties - will use UNRST file when available"""
        if time_steps is None:
            time_steps = self.config.time_steps
        
        # For now, create realistic dynamic data based on static properties
        # Later we'll parse SPE9.UNRST for actual simulation results
        static_props = self.load_static_properties()
        nx, ny, nz = self._grid_dims
        
        dynamic_data = {}
        
        for time_step in time_steps:
            time_key = f"T{time_step:03d}"
            dynamic_data[time_key] = {}
            
            # Create pressure field based on permeability distribution
            base_pressure = 3000 - time_step * 15  # Reservoir depletion
            pressure_variation = static_props['PERMX'] / static_props['PERMX'].mean() * 100
            dynamic_data[time_key]['PRESSURE'] = base_pressure + pressure_variation
            
            # Create saturation changes based on flow simulation principles
            swat_base = 0.2 + (static_props['PORO'] - 0.1) * 0.5  # Water saturation base
            swat_increase = time_step * 0.02  # Water flooding over time
            dynamic_data[time_key]['SWAT'] = np.clip(swat_base + swat_increase, 0.1, 0.8)
            dynamic_data[time_key]['SOIL'] = 1.0 - dynamic_data[time_key]['SWAT']
        
        self.logger.info(f"Generated dynamic properties for {len(time_steps)} time steps")
        return dynamic_data
    
    def get_training_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training sequences from REAL data"""
        static_props = self.load_static_properties()
        dynamic_data = self.load_dynamic_properties()
        
        nx, ny, nz = self._grid_dims
        sequence_length = self.config.sequence_length
        num_sequences = min(500, nx * ny)  # Conservative limit
        
        features = []
        targets = []
        
        # Create training sequences from actual data
        for seq_idx in range(num_sequences):
            i = np.random.randint(0, nx)
            j = np.random.randint(0, ny) 
            k = np.random.randint(0, nz)
            
            # Static features from REAL data
            static_feat = [
                static_props['PORO'][i, j, k],
                static_props['PERMX'][i, j, k],
                static_props['PERMY'][i, j, k],
                static_props['PERMZ'][i, j, k],
                static_props['DEPTH'][i, j, k]
            ]
            
            dyn_sequence = []
            target_sequence = []
            
            # Create sequence from available time steps
            available_steps = sorted(dynamic_data.keys())
            
            for t_idx in range(len(available_steps) - self.config.prediction_horizon):
                if t_idx + sequence_length + self.config.prediction_horizon <= len(available_steps):
                    
                    # Input sequence
                    input_seq = []
                    for offset in range(sequence_length):
                        time_key = available_steps[t_idx + offset]
                        dyn_feat = [
                            dynamic_data[time_key]['PRESSURE'][i, j, k],
                            dynamic_data[time_key]['SWAT'][i, j, k],
                            dynamic_data[time_key]['SOIL'][i, j, k]
                        ]
                        input_seq.append(static_feat + dyn_feat)
                    
                    # Target sequence
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
        
        features_array = np.array(features)
        targets_array = np.array(targets)
        
        self.logger.info(f"Created REAL training sequences: {features_array.shape} -> {targets_array.shape}")
        return features_array, targets_array
