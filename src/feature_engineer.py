import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class AdvancedFeatureEngineer:
    """Advanced feature engineer for reservoir data"""
    
    def __init__(self, config: SPE9GridConfig):
        self.config = config
        
    def create_geological_features(self, static_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create advanced geological features"""
        features = {}
        
        # Basic features
        features['permeability_log'] = torch.log(static_tensors['permeability_x'] + 1e-8)
        features['porosity'] = static_tensors['porosity']
        features['depth_normalized'] = self._normalize_depth(static_tensors['depth'])
        
        # Derived features
        features['flow_capacity'] = static_tensors['permeability_x'] * static_tensors['porosity']
        features['storage_capacity'] = static_tensors['porosity'] * self.config.dz[0]  # Simplified
        
        # Statistical features
        features.update(self._create_statistical_features(static_tensors))
        
        # Spatial features
        features.update(self._create_spatial_features(static_tensors))
        
        return features
    
    def _create_statistical_features(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create statistical features"""
        stats = {}
        
        for name, tensor in tensors.items():
            # Local statistics
            stats[f'{name}_mean'] = tensor.mean(dim=(1, 2), keepdim=True).expand_as(tensor)
            stats[f'{name}_std'] = tensor.std(dim=(1, 2), keepdim=True).expand_as(tensor)
            stats[f'{name}_max'] = tensor.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
            
        return stats
    
    def _create_spatial_features(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create spatial features"""
        spatial = {}
        
        nx, ny, nz = self.config.nx, self.config.ny, self.config.nz
        
        # Grid coordinates
        x_coords = torch.linspace(0, 1, nx).reshape(1, 1, nx).expand(nz, ny, nx)
        y_coords = torch.linspace(0, 1, ny).reshape(1, ny, 1).expand(nz, ny, nx)
        z_coords = torch.linspace(0, 1, nz).reshape(nz, 1, 1).expand(nz, ny, nx)
        
        spatial['x_coord'] = x_coords
        spatial['y_coord'] = y_coords  
        spatial['z_coord'] = z_coords
        
        # Well proximity (simplified)
        spatial['well_proximity'] = self._calculate_well_proximity()
        
        return spatial
    
    def _normalize_depth(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize depth tensor"""
        depth_min = depth_tensor.min()
        depth_max = depth_tensor.max()
        return (depth_tensor - depth_min) / (depth_max - depth_min)
    
    def _calculate_well_proximity(self) -> torch.Tensor:
        """Calculate proximity to wells (simplified)"""
        nx, ny, nz = self.config.nx, self.config.ny, self.config.nz
        proximity = torch.ones(nz, ny, nx)
        
        # Simplified well locations (from SPE9 well specs)
        well_locations = [
            (24, 25), (5, 1), (8, 2), (11, 3), (10, 4),  # Sample well locations
            (12, 5), (4, 6), (8, 7), (14, 8), (11, 9)    # More wells
        ]
        
        # Calculate minimum distance to any well
        for k in range(nz):
            for i in range(nx):
                for j in range(ny):
                    min_dist = float('inf')
                    for well_i, well_j in well_locations:
                        dist = ((i - well_i) ** 2 + (j - well_j) ** 2) ** 0.5
                        min_dist = min(min_dist, dist)
                    proximity[k, j, i] = 1.0 / (1.0 + min_dist)  # Inverse distance
        
        return proximity
    
    def create_temporal_features(self, production_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create advanced temporal features"""
        temporal = {}
        time_steps = len(production_data['FOPR'])
        
        # Basic time features
        time = torch.arange(time_steps).float()
        temporal['time'] = time
        temporal['time_normalized'] = time / time_steps
        
        # Time derivatives
        for name, data in production_data.items():
            if len(data.shape) == 1:  # Field-level data
                temporal[f'{name}_derivative'] = self._calculate_derivative(data)
                temporal[f'{name}_cumulative'] = torch.cumsum(data, dim=0)
                
        return temporal
    
    def _calculate_derivative(self, data: torch.Tensor) -> torch.Tensor:
        """Calculate numerical derivative"""
        derivative = torch.zeros_like(data)
        derivative[1:] = data[1:] - data[:-1]
        return derivative
    
    def create_advanced_features(self, complete_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create all advanced features combined"""
        geological_features = self.create_geological_features({
            'permeability_x': complete_data['permeability_x'],
            'porosity': complete_data['porosity'],
            'depth': complete_data['depth'],
            'region': complete_data['region']
        })
        
        temporal_features = self.create_temporal_features({
            'FOPR': complete_data['FOPR'],
            'FGPR': complete_data['FGPR'],
            'FWPR': complete_data['FWPR'],
            'FGOR': complete_data['FGOR']
        })
        
        # Combine all features
        advanced_features = {
            **geological_features,
            **temporal_features,
            'production_data': complete_data  # Keep original data
        }
        
        return advanced_features
