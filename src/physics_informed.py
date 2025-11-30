import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

class PhysicsConstraints:
    """Physics-based constraints for reservoir modeling"""
    
    def __init__(self, config, grid_config):
        self.config = config
        self.grid_config = grid_config
        
    def darcy_flow_constraint(self, pressure: torch.Tensor, 
                            permeability: torch.Tensor,
                            fluid_viscosity: float = 1.0) -> torch.Tensor:
        """
        Enforce Darcy's law: v = - (k/μ) ∇P
        
        Args:
            pressure: Pressure field [batch, nz, ny, nx]
            permeability: Permeability field [nz, ny, nx]
            fluid_viscosity: Fluid viscosity [cp]
            
        Returns:
            Darcy flow constraint loss
        """
        batch_size = pressure.shape[0]
        
        # Expand permeability to match batch size
        k_expanded = permeability.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Calculate pressure gradient
        grad_p_x = self._spatial_gradient(pressure, dim=3)  # ∂P/∂x
        grad_p_y = self._spatial_gradient(pressure, dim=2)  # ∂P/∂y  
        grad_p_z = self._spatial_gradient(pressure, dim=1)  # ∂P/∂z
        
        # Darcy velocity components
        v_x = - (k_expanded / fluid_viscosity) * grad_p_x
        v_y = - (k_expanded / fluid_viscosity) * grad_p_y
        v_z = - (k_expanded / fluid_viscosity) * grad_p_z
        
        # Continuity equation: ∇·v = 0 (for incompressible flow)
        div_v = self._spatial_gradient(v_x, dim=3) + \
                self._spatial_gradient(v_y, dim=2) + \
                self._spatial_gradient(v_z, dim=1)
                
        return torch.mean(div_v ** 2)
    
    def mass_balance_constraint(self, production_rates: torch.Tensor,
                              saturation_changes: torch.Tensor,
                              pore_volume: torch.Tensor,
                              time_step: float) -> torch.Tensor:
        """
        Enforce mass balance: Accumulation = In - Out
        
        Args:
            production_rates: Production rates [batch, time, wells]
            saturation_changes: Saturation changes [batch, time, grid_blocks]
            pore_volume: Pore volume [grid_blocks]
            time_step: Time step size
            
        Returns:
            Mass balance constraint loss
        """
        # Convert production to reservoir volumes
        reservoir_production = production_rates  # Simplified
        
        # Calculate accumulation term
        accumulation = pore_volume * saturation_changes / time_step
        
        # Mass balance: Accumulation + Production = 0
        mass_balance = accumulation + reservoir_production.sum(dim=-1, keepdim=True)
        
        return torch.mean(mass_balance ** 2)
    
    def _spatial_gradient(self, field: torch.Tensor, dim: int) -> torch.Tensor:
        """Calculate spatial gradient using central differences"""
        if dim == 1:  # z-direction
            grad = torch.zeros_like(field)
            grad[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / 2.0
        elif dim == 2:  # y-direction  
            grad = torch.zeros_like(field)
            grad[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / 2.0
        else:  # x-direction
            grad = torch.zeros_like(field)
            grad[:, :, :, 1:-1] = (field[:, :, :, 2:] - field[:, :, :, :-2]) / 2.0
            
        return grad
    
    def boundary_conditions(self, pressure: torch.Tensor, 
                          boundary_pressure: float = 3600.0) -> torch.Tensor:
        """
        Enforce boundary conditions
        """
        # Simple no-flow boundary conditions
        boundary_loss = 0.0
        
        # Left boundary (x=0)
        boundary_loss += torch.mean((pressure[:, :, :, 0] - pressure[:, :, :, 1]) ** 2)
        # Right boundary (x=nx-1)
        boundary_loss += torch.mean((pressure[:, :, :, -1] - pressure[:, :, :, -2]) ** 2)
        # Front boundary (y=0)
        boundary_loss += torch.mean((pressure[:, :, 0, :] - pressure[:, :, 1, :]) ** 2)
        # Back boundary (y=ny-1)
        boundary_loss += torch.mean((pressure[:, :, -1, :] - pressure[:, :, -2, :]) ** 2)
        
        return boundary_loss

class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss function combining data and physics constraints"""
    
    def __init__(self, physics_config, grid_config):
        super().__init__()
        self.physics_constraints = PhysicsConstraints(physics_config, grid_config)
        self.darcy_weight = physics_config.darcy_weight
        self.mass_balance_weight = physics_config.mass_balance_weight
        self.boundary_weight = physics_config.boundary_weight
        
    def forward(self, predictions: Dict, targets: Dict, 
                static_data: Dict, physics_enabled: bool = True) -> torch.Tensor:
        """
        Calculate physics-informed loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            static_data: Static reservoir data
            physics_enabled: Whether to include physics constraints
            
        Returns:
            Combined loss value
        """
        # Data loss (MSE)
        data_loss = F.mse_loss(predictions['production'], targets['production'])
        
        if not physics_enabled:
            return data_loss
            
        # Physics constraints
        physics_loss = 0.0
        
        # Darcy flow constraint
        if 'pressure' in predictions:
            darcy_loss = self.physics_constraints.darcy_flow_constraint(
                predictions['pressure'], static_data['permeability']
            )
            physics_loss += self.darcy_weight * darcy_loss
            
        # Mass balance constraint
        if 'saturation' in predictions and 'production_rates' in predictions:
            mass_balance_loss = self.physics_constraints.mass_balance_constraint(
                predictions['production_rates'],
                predictions['saturation_changes'],
                static_data['pore_volume'],
                time_step=1.0
            )
            physics_loss += self.mass_balance_weight * mass_balance_loss
            
        # Boundary conditions
        if 'pressure' in predictions:
            boundary_loss = self.physics_constraints.boundary_conditions(
                predictions['pressure']
            )
            physics_loss += self.boundary_weight * boundary_loss
            
        return data_loss + physics_loss
