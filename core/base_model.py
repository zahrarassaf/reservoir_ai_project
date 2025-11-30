from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class BaseReservoirModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, x, physics_constraints=None):
        pass
    
    @abstractmethod
    def compute_physics_loss(self, predictions, targets, static_properties):
        pass
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        return self
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
