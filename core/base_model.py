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
    def compute_physics_loss(self, predictions, targets):
        pass
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
