import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import ndimage

class PhysicsAwareFeatureEngineer:
    """Professional feature engineering with physics-based features"""
    
    def __init__(self, config, grid_config):
        self.config = config
        self.grid_config = grid_config
        
    def create_advanced_features(self, reservoir_data) -> Dict[str, torch.Tensor]:
        """
        Create comprehensive feature set with physics-based features
        """
        features = {}
        
        # Static geological features
        features.update(self._create_geological_features(reservoir_data))
        
        # Dynamic production features
        features.update(self._create_temporal_features(reservoir_data.production))
        
        # Physics-based features
        features.update(self._create_physics_features(reservoir_data))
        
        # Spatial features
        features.update(self._create_spatial_features(reservoir_data))
        
        return features
    
    def _create_geological_features(self, reservoir_data) -> Dict[str, torch.Tensor]:
        """Create geological features with proper statistical aggregation"""
        features = {}
        
        # Basic properties
        features['permeability_log'] = torch.log(reservoir_data.permeability + 1e-8)
        features['porosity'] = reservoir_data.porosity
        features['depth_normalized'] = self._normalize_tensor(reservoir_data.depth)
        
        # Geological statistics with proper aggregation
        features.update(self._calculate_geological_statistics(reservoir_data))
        
        # Flow capacity and storage capacity (F-C Phi)
        flow_capacity = reservoir_data.permeability * reservoir_data.porosity
        storage_capacity = reservoir_data.porosity * torch.tensor(self.grid_config.dz).unsqueeze(1).unsqueeze(2)
        
        features['flow_capacity'] = flow_capacity
        features['storage_capacity'] = storage_capacity
        features['f_c_ratio'] = flow_capacity / (storage_capacity + 1e-8)
        
        return features
    
    def _calculate_geological_statistics(self, reservoir_data) -> Dict[str, torch.Tensor]:
        """Calculate proper geological statistics"""
        stats = {}
        permeability = reservoir_data.permeability
        
        # Layer-wise statistics (correct aggregation)
        for k in range(self.grid_config.nz):
            layer_perm = permeability[k]
            stats[f'layer_{k}_perm_mean'] = layer_perm.mean().unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(permeability[k])
            stats[f'layer_{k}_perm_std'] = layer_perm.std().unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(permeability[k])
            stats[f'layer_{k}_dykstra_parsons'] = self._dykstra_parsons(layer_perm).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(permeability[k])
        
        # Spatial statistics using proper convolution
        stats['perm_local_mean'] = self._spatial_convolution(permeability, kernel_size=3)
        stats['perm_local_std'] = self._local_std(permeability, kernel_size=3)
        
        return stats
    
    def _dykstra_parsons(self, permeability: torch.Tensor) -> torch.Tensor:
        """Calculate Dykstra-Parsons coefficient for heterogeneity"""
        perm_flat = permeability.flatten()
        sorted_perm, _ = torch.sort(perm_flat)
        
        # Remove zeros and very small values
        sorted_perm = sorted_perm[sorted_perm > 1e-8]
        
        if len(sorted_perm) == 0:
            return torch.tensor(0.0)
            
        # Dykstra-Parsons coefficient: V = 1 - exp(-σ/μ)
        mean_perm = sorted_perm.mean()
        std_perm = sorted_perm.std()
        
        if mean_perm > 0:
            return 1 - torch.exp(-std_perm / mean_perm)
        else:
            return torch.tensor(0.0)
    
    def _spatial_convolution(self, tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply spatial convolution for local statistics"""
        # Use average pooling for local mean
        weights = torch.ones(1, 1, kernel_size, kernel_size, kernel_size) / (kernel_size ** 3)
        local_mean = F.conv3d(tensor.unsqueeze(0).unsqueeze(0), 
                            weights, padding=kernel_size//2)
        return local_mean.squeeze()
    
    def _local_std(self, tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Calculate local standard deviation"""
        local_mean = self._spatial_convolution(tensor, kernel_size)
        local_mean_sq = self._spatial_convolution(tensor ** 2, kernel_size)
        return torch.sqrt(local_mean_sq - local_mean ** 2)
    
    def _create_physics_features(self, reservoir_data) -> Dict[str, torch.Tensor]:
        """Create physics-based features"""
        features = {}
        
        # Transmissibility calculations
        features.update(self._calculate_transmissibilities(reservoir_data))
        
        # Well drainage areas
        features.update(self._calculate_well_drainage(reservoir_data))
        
        # Reservoir energy indicators
        features.update(self._calculate_energy_indicators(reservoir_data))
        
        return features
    
    def _calculate_transmissibilities(self, reservoir_data) -> Dict[str, torch.Tensor]:
        """Calculate inter-block transmissibilities using Peaceman model"""
        transmissibilities = {}
        
        kx = reservoir_data.permeability
        dx = self.grid_config.dx
        dy = self.grid_config.dy
        dz = torch.tensor(self.grid_config.dz).unsqueeze(1).unsqueeze(2)
        
        # Geometric mean transmissibility in x-direction
        tx = 2 * dy * dz * kx / dx
        transmissibilities['transmissibility_x'] = tx
        
        # Add more directional transmissibilities as needed
        
        return transmissibilities
    
    def _create_temporal_features(self, production_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create meaningful temporal features"""
        features = {}
        time_steps = len(production_data['FOPR'])
        
        # Time features
        time = torch.arange(time_steps).float()
        features['time'] = time
        features['time_normalized'] = time / time_steps
        
        # Production derivatives with proper handling
        for name, data in production_data.items():
            if name != 'time':
                features[f'{name}_derivative'] = self._robust_derivative(data)
                features[f'{name}_cumulative'] = torch.cumsum(data, dim=0)
                features[f'{name}_normalized'] = data / (data.max() + 1e-8)
                
                # Moving averages for trend analysis
                features[f'{name}_ma7'] = self._moving_average(data, window=7)
                features[f'{name}_ma30'] = self._moving_average(data, window=30)
        
        # Production ratios and indicators
        if 'FOPR' in production_data and 'FWPR' in production_data:
            oil_rate = production_data['FOPR']
            water_rate = production_data['FWPR']
            features['water_cut'] = water_rate / (oil_rate + water_rate + 1e-8)
            
        if 'FOPR' in production_data and 'FGPR' in production_data:
            features['gor_actual'] = production_data['FGPR'] / (production_data['FOPR'] + 1e-8)
            
        return features
    
    def _robust_derivative(self, data: torch.Tensor, window: int = 3) -> torch.Tensor:
        """Calculate robust numerical derivative"""
        derivative = torch.zeros_like(data)
        
        # Central differences for interior points
        if len(data) > 2:
            derivative[1:-1] = (data[2:] - data[:-2]) / 2.0
            
        # Forward/backward differences for boundaries
        if len(data) > 1:
            derivative[0] = data[1] - data[0]
            derivative[-1] = data[-1] - data[-2]
            
        return derivative
    
    def _moving_average(self, data: torch.Tensor, window: int) -> torch.Tensor:
        """Calculate moving average with proper padding"""
        if len(data) < window:
            return data
            
        # Use convolution for efficient moving average
        weights = torch.ones(window) / window
        padded_data = F.pad(data, (window//2, window//2), mode='replicate')
        ma = F.conv1d(padded_data.unsqueeze(0).unsqueeze(0), 
                     weights.unsqueeze(0).unsqueeze(0))
        return ma.squeeze()
    
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [0, 1] range"""
        t_min = tensor.min()
        t_max = tensor.max()
        return (tensor - t_min) / (t_max - t_min + 1e-8)
