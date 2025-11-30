from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
import torch.nn as nn
import numpy as np

class BaseEnsemble(ABC):
    def __init__(self, config):
        self.config = config
        self.models = []
        self._initialize_ensemble()
    
    @abstractmethod
    def _initialize_ensemble(self):
        pass
    
    @abstractmethod
    def train_ensemble(self, train_loader, val_loader):
        pass
    
    def predict_ensemble(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions_tensor = torch.stack(predictions)
        mean_prediction = torch.mean(predictions_tensor, dim=0)
        std_prediction = torch.std(predictions_tensor, dim=0)
        
        return {
            'mean': mean_prediction,
            'std': std_prediction,
            'all_predictions': predictions_tensor
        }
    
    def compute_diversity(self) -> float:
        if len(self.models) < 2:
            return 0.0
        
        diversities = []
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                param_diff = 0.0
                for param1, param2 in zip(self.models[i].parameters(), self.models[j].parameters()):
                    param_diff += torch.norm(param1.data - param2.data).item()
                diversities.append(param_diff)
        
        return np.mean(diversities) if diversities else 0.0
    
    def save_ensemble(self, path: str):
        ensemble_state = {
            'config': self.config,
            'model_states': [model.state_dict() for model in self.models],
            'ensemble_type': self.__class__.__name__
        }
        torch.save(ensemble_state, path)
    
    def load_ensemble(self, path: str):
        checkpoint = torch.load(path, map_location='cpu')
        for model, state_dict in zip(self.models, checkpoint['model_states']):
            model.load_state_dict(state_dict)
        return self
