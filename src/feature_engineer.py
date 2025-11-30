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
        
        for key in ['FOPR', 'FGPR', 'FWPR', 'FGOR']:
            if key in features:
                input_features.append(features[key].unsqueeze(1))
        
        output_features = []
        for key in ['FOPR', 'FGPR', 'FWPR']:
            if key in features:
                target = features[key][1:].unsqueeze(1)
                output_features.append(target)
        
        min_length = min(tensor.size(0) for tensor in input_features + output_features)
        
        x_tensors = [tensor[:min_length] for tensor in input_features]
        y_tensors = [tensor[:min_length] for tensor in output_features]
        
        if x_tensors and y_tensors:
            x = torch.cat(x_tensors, dim=1)
            y = torch.cat(y_tensors, dim=1)
        else:
            x = torch.randn(100, len(self.config.input_features))
            y = torch.randn(100, len(self.config.output_features))
        
        print(f"Training data - Input: {x.shape}, Target: {y.shape}")
        return x, y
