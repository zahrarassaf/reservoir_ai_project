import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsConstraintLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def darcy_law_constraint(self, pressure, permeability, saturation):
        pressure_grad_x = torch.gradient(pressure, dim=2)[0]
        pressure_grad_y = torch.gradient(pressure, dim=1)[0]
        pressure_grad_z = torch.gradient(pressure, dim=0)[0]
        
        darcy_x = -permeability[0] * pressure_grad_x
        darcy_y = -permeability[1] * pressure_grad_y  
        darcy_z = -permeability[2] * pressure_grad_z
        
        darcy_loss = torch.mean(darcy_x**2) + torch.mean(darcy_y**2) + torch.mean(darcy_z**2)
        return darcy_loss
    
    def mass_balance_constraint(self, pressure, saturation, porosity, dt=1.0):
        pressure_change = torch.diff(pressure, dim=0)
        saturation_change = torch.diff(saturation, dim=0)
        
        storage_term = porosity * saturation_change
        flux_term = pressure_change
        
        mass_balance = storage_term + flux_term
        return torch.mean(mass_balance**2)
    
    def boundary_conditions(self, pressure, saturation, grid_mask):
        boundary_loss = torch.mean((pressure[grid_mask == 0] - 3000)**2)
        return boundary_loss
    
    def forward(self, predictions, static_properties, grid_mask=None):
        physics_loss = 0.0
        
        if not self.config.use_physics_constraints:
            return physics_loss
        
        pressure = predictions['pressure']
        saturation = predictions['saturation']
        porosity = static_properties['porosity']
        permeability = [
            static_properties['permx'],
            static_properties['permy'], 
            static_properties['permz']
        ]
        
        physics_loss += self.config.darcy_weight * self.darcy_law_constraint(
            pressure, permeability, saturation
        )
        
        physics_loss += self.config.continuity_weight * self.mass_balance_constraint(
            pressure, saturation, porosity
        )
        
        if grid_mask is not None:
            physics_loss += self.config.boundary_weight * self.boundary_conditions(
                pressure, saturation, grid_mask
            )
        
        return physics_loss
