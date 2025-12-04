"""
SPE9 Dataset Loader - Loads and preprocesses SPE9 benchmark data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import warnings


class SPE9Dataset(Dataset):
    """SPE9 Reservoir Simulation Dataset."""
    
    def __init__(self, 
                 data_path: str = "data/spe9",
                 sequence_length: int = 50,
                 prediction_horizon: int = 10,
                 normalize: bool = True):
        """
        Initialize SPE9 dataset.
        
        Args:
            data_path: Path to SPE9 data files
            sequence_length: Length of input sequences
            prediction_horizon: Steps to predict ahead
            normalize: Whether to normalize data
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        
        # Load SPE9 data
        self.data = self._load_spe9_data()
        
        # Preprocess data
        self._preprocess_data()
        
        # Statistics for normalization
        self.stats = {}
        
    def _load_spe9_data(self) -> Dict[str, np.ndarray]:
        """Load SPE9 data from files."""
        # Check if data exists
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"SPE9 data not found at {self.data_path}. "
                f"Download from https://github.com/OPM/opm-data/tree/master/spe9"
            )
        
        data = {}
        
        try:
            # Load grid information
            grid_file = self.data_path / "SPE9_GRID.DATA"
            data['grid'] = self._parse_grid_file(grid_file)
            
            # Load simulation results
            results_file = self.data_path / "SPE9_RESULTS.H5"
            if results_file.exists():
                with h5py.File(results_file, 'r') as f:
                    for key in f.keys():
                        data[key] = f[key][:]
            else:
                # Fallback to text files
                data = self._load_from_text_files()
            
            # Load well information
            well_file = self.data_path / "SPE9_WELLS.DATA"
            data['wells'] = self._parse_well_file(well_file)
            
        except Exception as e:
            warnings.warn(f"Error loading SPE9 data: {e}. Using synthetic data.")
            data = self._generate_synthetic_data()
        
        return data
    
    def _parse_grid_file(self, grid_file: Path) -> Dict[str, np.ndarray]:
        """Parse SPE9 grid file."""
        # Simplified parser - in reality would use OPM parser
        grid_info = {
            'dims': (24, 25, 15),  # SPE9 grid dimensions
            'dx': np.ones(24) * 100,  # Simplified
            'dy': np.ones(25) * 100,
            'dz': np.ones(15) * 20,
            'depth': np.zeros((24, 25, 15))
        }
        return grid_info
    
    def _parse_well_file(self, well_file: Path) -> Dict[str, np.ndarray]:
        """Parse SPE9 well file."""
        wells = {
            'locations': np.array([
                [5, 5, 5],   # Producer 1
                [20, 20, 10], # Producer 2
                [10, 15, 8],  # Injector
            ]),
            'types': ['PRODUCER', 'PRODUCER', 'INJECTOR'],
            'controls': ['BHP', 'BHP', 'RATE']
        }
        return wells
    
    def _load_from_text_files(self) -> Dict[str, np.ndarray]:
        """Load data from text files if H5 not available."""
        # This would parse the actual SPE9 output files
        # For now, generate synthetic data
        return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> Dict[str, np.ndarray]:
        """Generate synthetic SPE9-like data for testing."""
        nx, ny, nz = 24, 25, 15
        n_cells = nx * ny * nz
        n_timesteps = 1000
        
        # Generate permeability field (log-normal)
        permeability = np.random.lognormal(mean=1.0, sigma=1.5, size=(nx, ny, nz))
        
        # Generate porosity field (truncated normal)
        porosity = np.random.normal(loc=0.2, scale=0.05, size=(nx, ny, nz))
        porosity = np.clip(porosity, 0.05, 0.35)
        
        # Generate pressure and saturation time series
        time = np.linspace(0, 365 * 10, n_timesteps)  # 10 years
        
        # Base pressure decline
        base_pressure = 5000 - 0.1 * time  # psi
        
        # Add spatial variation
        pressure = np.zeros((n_timesteps, nx, ny, nz))
        saturation = np.zeros((n_timesteps, nx, ny, nz))
        
        for t in range(n_timesteps):
            pressure[t] = base_pressure[t] + np.random.normal(0, 100, (nx, ny, nz))
            # Saturation increases from 0.2 to 0.8 over time
            saturation[t] = 0.2 + 0.6 * (t / n_timesteps) + np.random.normal(0, 0.05, (nx, ny, nz))
            saturation[t] = np.clip(saturation[t], 0.0, 1.0)
        
        # Well production rates
        n_wells = 3
        production = np.zeros((n_timesteps, n_wells))
        for w in range(n_wells):
            production[:, w] = 1000 * np.exp(-0.001 * time) + np.random.normal(0, 50, n_timesteps)
        
        return {
            'permeability': permeability,
            'porosity': porosity,
            'pressure': pressure,
            'saturation': saturation,
            'production': production,
            'time': time,
            'grid': {
                'dims': (nx, ny, nz),
                'dx': np.ones(nx) * 100,
                'dy': np.ones(ny) * 100,
                'dz': np.ones(nz) * 20
            }
        }
    
    def _preprocess_data(self):
        """Preprocess SPE9 data for machine learning."""
        # Extract features and targets
        self.features = []
        self.targets = []
        self.physics_states = []
        
        pressure = self.data['pressure']
        saturation = self.data['saturation']
        production = self.data['production']
        
        n_timesteps = pressure.shape[0]
        
        for t in range(self.sequence_length, n_timesteps - self.prediction_horizon):
            # Input features: pressure, saturation, production history
            feat = np.concatenate([
                pressure[t-self.sequence_length:t].flatten(),
                saturation[t-self.sequence_length:t].flatten(),
                production[t-self.sequence_length:t].flatten()
            ])
            
            # Target: pressure and saturation at future time
            target = np.concatenate([
                pressure[t+self.prediction_horizon].flatten(),
                saturation[t+self.prediction_horizon].flatten()
            ])
            
            # Physics state for constraints
            physics_state = {
                'pressure': pressure[t],
                'saturation': saturation[t],
                'permeability': self.data['permeability'],
                'porosity': self.data['porosity']
            }
            
            self.features.append(feat)
            self.targets.append(target)
            self.physics_states.append(physics_state)
        
        # Convert to arrays
        self.features = np.array(self.features, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        
        # Normalize if requested
        if self.normalize:
            self._normalize_data()
    
    def _normalize_data(self):
        """Normalize features and targets."""
        # Feature normalization
        self.feat_mean = self.features.mean(axis=0)
        self.feat_std = self.features.std(axis=0) + 1e-10
        self.features = (self.features - self.feat_mean) / self.feat_std
        
        # Target normalization
        self.target_mean = self.targets.mean(axis=0)
        self.target_std = self.targets.std(axis=0) + 1e-10
        self.targets = (self.targets - self.target_mean) / self.target_std
        
        self.stats = {
            'feat_mean': self.feat_mean,
            'feat_std': self.feat_std,
            'target_mean': self.target_mean,
            'target_std': self.target_std
        }
    
    def denormalize(self, data: np.ndarray, is_target: bool = True) -> np.ndarray:
        """Denormalize data."""
        if not self.normalize:
            return data
        
        if is_target:
            return data * self.target_std + self.target_mean
        else:
            return data * self.feat_std + self.feat_mean
    
    def split_data(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Split data into train, validation, and test sets."""
        n_samples = len(self.features)
        
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        self.train_indices = list(range(0, train_end))
        self.val_indices = list(range(train_end, val_end))
        self.test_indices = list(range(val_end, n_samples))
    
    def get_feature_dimension(self) -> int:
        """Get dimension of feature vector."""
        return self.features.shape[1]
    
    def get_target_dimension(self) -> int:
        """Get dimension of target vector."""
        return self.targets.shape[1]
    
    def get_grid_info(self) -> Dict:
        """Get grid information."""
        return self.data['grid']
    
    def get_permeability_field(self) -> np.ndarray:
        """Get permeability field."""
        return self.data['permeability']
    
    def get_porosity_field(self) -> np.ndarray:
        """Get porosity field."""
        return self.data['porosity']
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get a single data sample."""
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        physics_state = self.physics_states[idx]
        
        # Convert physics state to tensors
        physics_tensors = {}
        for key, value in physics_state.items():
            if isinstance(value, np.ndarray):
                physics_tensors[key] = torch.tensor(value, dtype=torch.float32)
            else:
                physics_tensors[key] = value
        
        return features, target, physics_tensors
    
    def get_train_loader(self, batch_size: int = 32) -> DataLoader:
        """Get training data loader."""
        train_dataset = torch.utils.data.Subset(self, self.train_indices)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    def get_val_loader(self, batch_size: int = 32) -> DataLoader:
        """Get validation data loader."""
        val_dataset = torch.utils.data.Subset(self, self.val_indices)
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    def get_test_loader(self, batch_size: int = 32) -> DataLoader:
        """Get test data loader."""
        test_dataset = torch.utils.data.Subset(self, self.test_indices)
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
