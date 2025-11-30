import torch
import torch.nn as nn
import numpy as np

class PhysicsConstraintLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def darcy_law_constraint(self, pressure, permeability, saturation):
        # Simplified Darcy law implementation
        grad_p = torch.gradient(pressure, dim=[-3, -2, -1])
        darcy_flux = [-k * g for k, g in zip(permeability, grad_p)]
        return sum([torch.mean(f**2) for f in darcy_flux])
    
    def mass_balance_constraint(self, pressure, saturation, dt=1.0):
        # Simplified mass balance
        pressure_change = torch.diff(pressure, dim=0)
        sat_change = torch.diff(saturation, dim=0)
        mass_balance = pressure_change + sat_change
        return torch.mean(mass_balance**2)
    
    def forward(self, predictions, static_properties):
        physics_loss = 0.0
        
        if self.config.use_physics_constraints:
            pressure = predictions['pressure']
            saturation = predictions['saturation']
            permeability = static_properties['permeability']
            
            physics_loss += self.config.darcy_weight * self.darcy_law_constraint(
                pressure, permeability, saturation
            )
            
            physics_loss += self.config.continuity_weight * self.mass_balance_constraint(
                pressure, saturation
            )
        
        return physics_loss
