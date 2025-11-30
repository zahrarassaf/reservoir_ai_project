import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        
    def create_features(self, production_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = {}
        
        time_steps = len(production_data['FOPR'])
        
        time = torch.arange(time_steps).float()
        features['time'] = time
        features['time_normalized'] = time / time_steps
        
        for name, data in production_data.items():
            if len(data.shape) == 1:
                features[f'{name}_derivative'] = self._calculate_derivative(data)
                features[f'{name}_cumulative'] = torch.cumsum(data, dim=0)
                
        features.update(production_data)
        
        return features
    
    def _calculate_derivative(self, data: torch.Tensor) -> torch.Tensor:
        derivative = torch.zeros_like(data)
        derivative[1:] = data[1:] - data[:-1]
        return derivative
    
    def prepare_training_data(self, features: Dict[str, torch.Tensor]):
        input_features = []
        output_features = []
        
        for key, value in features.items():
            if 'FOPR' in key or 'FGPR' in key or 'FWPR' in key:
                output_features.append(value.unsqueeze(1))
            elif 'time' not in key and 'derivative' not in key and 'cumulative' not in key:
                input_features.append(value.unsqueeze(1))
        
        x = torch.cat(input_features, dim=1) if input_features else torch.randn(100, 7)
        y = torch.cat(output_features, dim=1) if output_features else torch.randn(100, 3)
        
        return x, y
