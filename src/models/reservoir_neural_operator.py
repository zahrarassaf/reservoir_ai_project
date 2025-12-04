# src/models/reservoir_neural_operator.py
"""
Reservoir Neural Operator (RNO) - Combines neural operators with reservoir physics.
Specifically designed for multiphase flow in porous media.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
import math

from src.physics.darcy_flow import DarcyFlowConstraints


class ReservoirNeuralOperator(nn.Module):
    """
    Neural operator for reservoir simulation with built-in physics.
    
    Architecture:
    1. Input encoder (MLP)
    2. Fourier neural operator layers
    3. Physics projection layer
    4. Output decoder (MLP)
    5. Physics constraints (loss terms)
    """
    
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 grid_dims: Tuple[int, int, int],
                 hidden_channels: int = 64,
                 num_fno_layers: int = 4,
                 fno_modes: int = 12,
                 physics_weight: float = 0.1,
                 use_attention: bool = True):
        """
        Initialize Reservoir Neural Operator.
        
        Args:
            input_channels: Number of input channels (pressure, saturation, etc.)
            output_channels: Number of output channels
            grid_dims: Grid dimensions (nx, ny, nz)
            hidden_channels: Hidden channel dimension
            num_fno_layers: Number of FNO layers
            fno_modes: Number of Fourier modes
            physics_weight: Weight for physics loss
            use_attention: Whether to use attention mechanisms
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.grid_dims = grid_dims
        self.nx, self.ny, self.nz = grid_dims
        self.n_cells = self.nx * self.ny * self.nz
        self.physics_weight = physics_weight
        self.use_attention = use_attention
        
        # 1. Input encoder (lift to higher dimension)
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 2. Fourier Neural Operator layers
        self.fno_layers = nn.ModuleList([
            FNOBlock(hidden_channels, hidden_channels, fno_modes)
            for _ in range(num_fno_layers)
        ])
        
        # 3. Attention mechanism for spatial features
        if use_attention:
            self.attention = SpatialAttention(hidden_channels)
        
        # 4. Physics-aware projection
        self.physics_projection = PhysicsProjection(
            hidden_channels, hidden_channels, grid_dims
        )
        
        # 5. Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, output_channels)
        )
        
        # 6. Physics constraints module
        self.physics_constraints = PhysicsConstraints(grid_dims)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, 
                x: torch.Tensor,
                permeability: torch.Tensor,
                porosity: torch.Tensor,
                return_physics: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass of Reservoir Neural Operator.
        
        Args:
            x: Input tensor [batch, channels, nx, ny, nz] or [batch, channels, n_cells]
            permeability: Permeability field [batch, nx, ny, nz]
            porosity: Porosity field [batch, nx, ny, nz]
            return_physics: Whether to return physics diagnostics
            
        Returns:
            predictions: Output predictions
            diagnostics: Physics diagnostics and losses
        """
        batch_size = x.shape[0]
        
        # Ensure correct shape
        if x.dim() == 5:  # [batch, channels, nx, ny, nz]
            x = x.permute(0, 2, 3, 4, 1)  # Move channels last
            x = x.reshape(batch_size, -1, self.input_channels)
        elif x.dim() == 3:  # [batch, n_cells, channels]
            x = x.reshape(batch_size, -1, self.input_channels)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Encode input
        h = self.encoder(x)  # [batch, n_cells, hidden_channels]
        
        # Reshape to spatial for FNO
        h_spatial = h.reshape(batch_size, self.nx, self.ny, self.nz, -1)
        h_spatial = h_spatial.permute(0, 4, 1, 2, 3)  # [batch, channels, nx, ny, nz]
        
        # Apply FNO layers
        for fno_layer in self.fno_layers:
            h_spatial = fno_layer(h_spatial)
        
        # Apply attention if enabled
        if self.use_attention:
            h_spatial = self.attention(h_spatial)
        
        # Apply physics projection
        h_spatial = self.physics_projection(h_spatial, permeability, porosity)
        
        # Reshape back to flat
        h_spatial = h_spatial.permute(0, 2, 3, 4, 1)
        h = h_spatial.reshape(batch_size, -1, h_spatial.shape[1])
        
        # Decode to output
        predictions = self.decoder(h)  # [batch, n_cells, output_channels]
        
        # Reshape predictions to spatial if needed
        if predictions.shape[1] == self.n_cells:
            predictions = predictions.reshape(batch_size, self.nx, self.ny, self.nz, -1)
            predictions = predictions.permute(0, 4, 1, 2, 3)  # [batch, channels, nx, ny, nz]
        
        # Compute physics diagnostics if requested
        diagnostics = {}
        if return_physics:
            diagnostics = self._compute_physics_diagnostics(
                predictions, permeability, porosity
            )
        
        return predictions, diagnostics
    
    def _compute_physics_diagnostics(self,
                                    predictions: torch.Tensor,
                                    permeability: torch.Tensor,
                                    porosity: torch.Tensor) -> Dict:
        """
        Compute physics-based diagnostics and losses.
        
        Args:
            predictions: Model predictions
            permeability: Permeability field
            porosity: Porosity field
            
        Returns:
            Dictionary with physics diagnostics
        """
        diagnostics = {}
        
        # Assume predictions contain pressure and saturation
        if predictions.shape[1] >= 2:
            pressure = predictions[:, 0:1]  # First channel: pressure
            saturation = predictions[:, 1:2]  # Second channel: water saturation
            
            # Compute physics constraints
            physics_losses = self.physics_constraints(
                pressure, saturation, permeability, porosity
            )
            
            diagnostics.update(physics_losses)
            
            # Compute additional metrics
            with torch.no_grad():
                # Mass balance error
                mb_error = self._compute_mass_balance(pressure, saturation, porosity)
                diagnostics['mass_balance_error'] = mb_error
                
                # Darcy law violation
                darcy_violation = self._compute_darcy_violation(
                    pressure, permeability
                )
                diagnostics['darcy_violation'] = darcy_violation
        
        return diagnostics
    
    def _compute_mass_balance(self,
                             pressure: torch.Tensor,
                             saturation: torch.Tensor,
                             porosity: torch.Tensor) -> torch.Tensor:
        """
        Compute mass balance error.
        
        Simplified: For incompressible flow, ∇·v = 0
        """
        # Compute divergence of velocity (simplified)
        # v = -k/μ * ∇P
        
        # Compute pressure gradient
        grad_p = torch.gradient(pressure, dim=(2, 3, 4))
        grad_p_mag = torch.sqrt(sum(g**2 for g in grad_p))
        
        # Simplified mass balance: variance of flow
        mb_error = torch.var(grad_p_mag)
        
        return mb_error
    
    def _compute_darcy_violation(self,
                                pressure: torch.Tensor,
                                permeability: torch.Tensor) -> torch.Tensor:
        """
        Compute Darcy's law violation.
        
        Simplified check for consistency.
        """
        # Darcy: v = -k/μ * ∇P
        # Check if ∇P direction is consistent with k distribution
        
        # Compute gradient magnitude
        grad_p = torch.gradient(pressure, dim=(2, 3, 4))
        grad_p_mag = torch.sqrt(sum(g**2 for g in grad_p))
        
        # Expected velocity magnitude proportional to k * |∇P|
        expected_v_mag = permeability * grad_p_mag
        
        # Simplified violation metric
        violation = torch.std(expected_v_mag) / (torch.mean(expected_v_mag) + 1e-10)
        
        return violation


class FNOBlock(nn.Module):
    """Fourier Neural Operator block for 3D spatial processing."""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Fourier transform parameters
        self.scale = 1 / (in_channels * out_channels)
        
        # Fourier weights
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, modes, modes, 2)
        )
        
        # Convolution for local features
        self.conv = nn.Conv3d(in_channels, out_channels, 1)
        
        # Activation
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, channels, nx, ny, nz]
        batch_size, channels, nx, ny, nz = x.shape
        
        # Fourier transform
        x_ft = torch.fft.rfftn(x, dim=(2, 3, 4))
        
        # Multiply with Fourier weights
        out_ft = torch.zeros(
            batch_size, self.out_channels, nx, ny, nz // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        
        # Only keep low frequencies (modes)
        modes = self.modes
        out_ft[:, :, :modes, :modes, :modes] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, :modes, :modes, :modes],
            torch.view_as_complex(self.weights1)
        )
        
        # Inverse Fourier transform
        x_out = torch.fft.irfftn(out_ft, s=(nx, ny, nz))
        
        # Add local features
        x_local = self.conv(x)
        
        # Combine and activate
        output = self.activation(x_out + x_local)
        
        return output


class SpatialAttention(nn.Module):
    """Spatial attention for reservoir features."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        
        self.num_heads = num_heads
        self.channels = channels
        
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.projection = nn.Conv3d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, nx, ny, nz = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(
            batch_size, 3, self.num_heads, channels // self.num_heads, nx, ny, nz
        )
        q, k, v = qkv.unbind(1)
        
        # Reshape for attention
        q = q.reshape(batch_size * self.num_heads, -1, nx * ny * nz)
        k = k.reshape(batch_size * self.num_heads, -1, nx * ny * nz)
        v = v.reshape(batch_size * self.num_heads, -1, nx * ny * nz)
        
        # Scaled dot-product attention
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.reshape(batch_size, channels, nx, ny, nz)
        
        # Final projection
        out = self.projection(out)
        
        return x + out  # Residual connection


class PhysicsProjection(nn.Module):
    """Physics-aware feature projection."""
    
    def __init__(self, in_channels: int, out_channels: int, grid_dims: Tuple[int, int, int]):
        super().__init__()
        
        self.grid_dims = grid_dims
        
        # Learnable physics parameters
        self.physics_encoder = nn.Sequential(
            nn.Conv3d(in_channels + 2, out_channels, 3, padding=1),  # +2 for k, phi
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )
        
        # Residual connection
        self.residual = nn.Conv3d(in_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor, 
                permeability: torch.Tensor,
                porosity: torch.Tensor) -> torch.Tensor:
        """
        Project features using physics information.
        
        Args:
            x: Input features [batch, channels, nx, ny, nz]
            permeability: Permeability field [batch, 1, nx, ny, nz]
            porosity: Porosity field [batch, 1, nx, ny, nz]
        """
        # Normalize physics fields
        k_norm = torch.log1p(permeability) / 10.0  # Log-normalize permeability
        phi_norm = porosity  # Porosity already in [0, 1]
        
        # Concatenate physics with features
        x_physics = torch.cat([x, k_norm, phi_norm], dim=1)
        
        # Apply physics encoder
        physics_features = self.physics_encoder(x_physics)
        
        # Residual connection
        residual = self.residual(x)
        
        return physics_features + residual


class PhysicsConstraints(nn.Module):
    """Physics constraints for reservoir simulation."""
    
    def __init__(self, grid_dims: Tuple[int, int, int]):
        super().__init__()
        
        self.grid_dims = grid_dims
        self.nx, self.ny, self.n
