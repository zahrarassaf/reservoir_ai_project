"""
Darcy flow physics implementation for reservoir simulation.
Implements physics constraints for neural networks.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from scipy.sparse import csr_matrix, lil_matrix


class DarcyFlowConstraints:
    """Darcy flow physics constraints for reservoir simulation."""
    
    def __init__(self, grid_dims: Tuple[int, int, int], 
                 permeability: np.ndarray,
                 porosity: np.ndarray,
                 fluid_viscosity: float = 1.0,
                 fluid_density: float = 1000.0):
        """
        Initialize Darcy flow constraints.
        
        Args:
            grid_dims: Grid dimensions (nx, ny, nz)
            permeability: Permeability field [nx, ny, nz]
            porosity: Porosity field [nx, ny, nz]
            fluid_viscosity: Fluid viscosity [cp]
            fluid_density: Fluid density [kg/m3]
        """
        self.grid_dims = grid_dims
        self.nx, self.ny, self.nz = grid_dims
        self.n_cells = self.nx * self.ny * self.nz
        
        self.permeability = torch.tensor(permeability, dtype=torch.float32)
        self.porosity = torch.tensor(porosity, dtype=torch.float32)
        self.viscosity = fluid_viscosity
        self.density = fluid_density
        
        # Pre-compute transmissibility matrix
        self.T_matrix = self._compute_transmissibility_matrix()
        
    def _compute_transmissibility_matrix(self) -> torch.Tensor:
        """Compute transmissibility matrix using harmonic averaging."""
        T = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32)
        
        # Create mapping from 3D to 1D
        idx_map = np.arange(self.n_cells).reshape(self.grid_dims)
        
        # Compute transmissibilities in x-direction
        for i in range(self.nx - 1):
            for j in range(self.ny):
                for k in range(self.nz):
                    cell1 = idx_map[i, j, k]
                    cell2 = idx_map[i + 1, j, k]
                    
                    k1 = self.permeability[i, j, k]
                    k2 = self.permeability[i + 1, j, k]
                    
                    # Harmonic average for interface permeability
                    k_interface = 2 * k1 * k2 / (k1 + k2 + 1e-10)
                    
                    # Simple transmissibility (would need geometry in real case)
                    T[cell1, cell2] = k_interface
                    T[cell2, cell1] = k_interface
        
        # Similar for y and z directions
        return T
    
    def darcy_loss(self, pressure: torch.Tensor, 
                   saturation: torch.Tensor) -> torch.Tensor:
        """
        Compute Darcy flow constraint loss.
        
        Args:
            pressure: Pressure field [batch, nx, ny, nz] or [nx, ny, nz]
            saturation: Water saturation field [same shape as pressure]
            
        Returns:
            loss: Scalar loss value
        """
        if pressure.dim() == 4:
            # Batch mode
            batch_size = pressure.shape[0]
            losses = []
            for b in range(batch_size):
                losses.append(self._single_darcy_loss(pressure[b], saturation[b]))
            return torch.stack(losses).mean()
        else:
            return self._single_darcy_loss(pressure, saturation)
    
    def _single_darcy_loss(self, pressure: torch.Tensor, 
                          saturation: torch.Tensor) -> torch.Tensor:
        """Compute Darcy loss for single sample."""
        # Flatten fields
        p_flat = pressure.reshape(-1)
        s_flat = saturation.reshape(-1)
        
        # Compute flow using transmissibility matrix
        # Q = T * ΔP (simplified)
        flow = torch.matmul(self.T_matrix, p_flat)
        
        # Mass conservation: Accumulation = In - Out + Source
        # For incompressible flow: div(Q) = 0
        divergence = torch.sum(torch.abs(flow))
        
        # Also check capillary pressure relationship
        # Pc = f(S) - usually from Brooks-Corey or van Genuchten
        pc_loss = self._capillary_pressure_loss(pressure, saturation)
        
        return divergence + pc_loss
    
    def _capillary_pressure_loss(self, pressure: torch.Tensor,
                                saturation: torch.Tensor) -> torch.Tensor:
        """Capillary pressure constraint (Brooks-Corey)."""
        # Simplified Brooks-Corey model
        lambda_bc = 2.0  # Pore size distribution index
        entry_pressure = 1.0  # Entry pressure
        
        # Normalized saturation
        s_norm = (saturation - saturation.min()) / (saturation.max() - saturation.min() + 1e-10)
        
        # Brooks-Corey: Pc = Pd * Se^{-1/lambda}
        pc_pred = entry_pressure * torch.pow(s_norm + 1e-10, -1/lambda_bc)
        
        # In reality, would compare with actual Pc curve
        return torch.mean(torch.abs(pc_pred))
    
    def material_balance_loss(self, pressure: torch.Tensor,
                            saturation: torch.Tensor,
                            dt: float = 1.0) -> torch.Tensor:
        """Material balance constraint (conservation of mass)."""
        # For water phase
        porosity = self.porosity.reshape(-1)
        s_flat = saturation.reshape(-1)
        
        # Accumulation term: d/dt (φ * S * ρ)
        # Simplified: check if accumulation matches net flow
        accumulation = porosity * s_flat * self.density
        
        # This should equal net flow (simplified)
        mb_error = torch.std(accumulation)  # Should be conserved
        
        return mb_error
