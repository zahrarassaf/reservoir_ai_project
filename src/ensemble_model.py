import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, features: int, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(features, features),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)

class ReservoirNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                ResidualBlock(hidden_dim, dropout)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class DeepEnsembleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.models = nn.ModuleList([
            ReservoirNet(
                input_dim=len(config.input_features),
                output_dim=len(config.output_features),
                hidden_dims=config.hidden_layers,
                dropout=config.dropout_rate
            ) for _ in range(config.n_models)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
            
        stacked_preds = torch.stack(predictions, dim=0)
        mean_pred = stacked_preds.mean(dim=0)
        std_pred = stacked_preds.std(dim=0)
        
        return mean_pred, std_pred
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> Dict[str, torch.Tensor]:
        all_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                model_preds = []
                for model in self.models:
                    pred = model(x)
                    model_preds.append(pred)
                
                stacked = torch.stack(model_preds, dim=0)
                all_samples.append(stacked)
            
        all_samples = torch.stack(all_samples, dim=0)
        
        return {
            'mean': all_samples.mean(dim=(0, 1)),
            'std': all_samples.std(dim=(0, 1)),
            'aleatoric': all_samples.var(dim=1).mean(dim=0),
            'epistemic': all_samples.mean(dim=1).var(dim=0),
            'samples': all_samples
        }
