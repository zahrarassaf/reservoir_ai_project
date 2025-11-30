import torch
import torch.nn as nn
from .base_model import BaseReservoirModel
from .physics_layers import PhysicsConstraintLayer

class ReservoirNN(BaseReservoirModel):
    def __init__(self, temporal_config, physics_config):
        super().__init__(temporal_config)
        self.temporal_model = TemporalModel(temporal_config)
        self.physics_layer = PhysicsConstraintLayer(physics_config)
    
    def forward(self, x, physics_constraints=None):
        temporal_output = self.temporal_model(x)
        
        if physics_constraints is not None:
            physics_loss = self.physics_layer(
                temporal_output, 
                physics_constraints['static_properties'],
                physics_constraints.get('grid_mask')
            )
            return temporal_output, physics_loss
        
        return temporal_output
    
    def compute_physics_loss(self, predictions, targets, static_properties):
        return self.temporal_model.compute_physics_loss(predictions, targets, static_properties)
