"""
Physics-Informed Echo State Network for reservoir simulation.
Combines ESN dynamics with physics constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy.sparse import random


@dataclass
class PIESNConfig:
    """Configuration for Physics-Informed ESN."""
    reservoir_size: int = 1000
    spectral_radius: float = 0.95
    input_scaling: float = 1.0
    leak_rate: float = 0.3
    connectivity: float = 0.1
    regularization: float = 1e-6
    physics_weight: float = 0.1
    use_attention: bool = True
    attention_heads: int = 4
    dropout_rate: float = 0.1


class PhysicsInformedESN(nn.Module):
    """Physics-Informed Echo State Network."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 config: PIESNConfig, physics_module=None):
        """
        Initialize Physics-Informed ESN.
        
        Args:
            input_dim: Dimension of input data
            output_dim: Dimension of output predictions
            config: Model configuration
            physics_module: Physics constraints module
        """
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_module = physics_module
        
        # Initialize reservoir weights
        self.W_in = self._initialize_input_weights()
        self.W_res = self._initialize_reservoir_weights()
        self.W_feedback = None  # For output feedback
        
        # Reservoir state
        self.register_buffer('reservoir_state', 
                           torch.zeros(config.reservoir_size))
        
        # Output layer (trained)
        self.output_layer = nn.Linear(config.reservoir_size, output_dim)
        
        # Attention mechanism for physics guidance
        if config.use_attention:
            self.attention = MultiHeadAttention(
                config.reservoir_size, 
                config.attention_heads,
                dropout=config.dropout_rate
            )
        
        # Physics conditioning network
        self.physics_conditioner = nn.Sequential(
            nn.Linear(config.reservoir_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, config.reservoir_size)
        )
        
    def _initialize_input_weights(self) -> torch.Tensor:
        """Initialize input weights with scaling."""
        W_in = torch.randn(self.config.reservoir_size, self.input_dim)
        return W_in * self.config.input_scaling
    
    def _initialize_reservoir_weights(self) -> torch.Tensor:
        """Initialize sparse reservoir weights with spectral radius control."""
        # Create sparse random matrix
        connectivity = self.config.connectivity
        n = self.config.reservoir_size
        
        # Generate sparse matrix
        indices = torch.randperm(n * n)[:int(connectivity * n * n)]
        rows = indices // n
        cols = indices % n
        
        values = torch.randn(len(rows)) * 0.1
        W_res = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), 
            values, 
            (n, n)
        ).to_dense()
        
        # Ensure sparsity pattern
        mask = (torch.rand(n, n) < connectivity).float()
        W_res = W_res * mask
        
        # Scale to desired spectral radius
        if self.config.spectral_radius > 0:
            eigenvalues = torch.linalg.eigvals(W_res)
            spectral_radius = torch.max(torch.abs(eigenvalues))
            W_res = W_res * (self.config.spectral_radius / (spectral_radius + 1e-10))
        
        return W_res
    
    def forward(self, x: torch.Tensor, 
                physics_state: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with physics constraints.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            physics_state: Optional physics state information
            
        Returns:
            predictions: Output predictions
            diagnostics: Dictionary with internal states and losses
        """
        batch_size = x.shape[0]
        
        # Initialize batch reservoir states
        reservoir_states = torch.zeros(batch_size, self.config.reservoir_size, 
                                      device=x.device)
        
        # Store states for analysis
        states_history = []
        physics_losses = []
        
        # Time steps (assuming temporal dimension in features)
        for t in range(x.shape[1] if x.dim() > 2 else 1):
            if x.dim() > 2:
                x_t = x[:, t, :]
            else:
                x_t = x
            
            # Update reservoir state
            input_term = torch.matmul(x_t, self.W_in.T)
            reservoir_term = torch.matmul(reservoir_states, self.W_res.T)
            
            # Apply attention if enabled
            if self.config.use_attention:
                reservoir_states = self.attention(reservoir_states)
            
            # Physics conditioning
            if physics_state is not None and self.physics_module is not None:
                physics_signal = self.physics_conditioner(reservoir_states)
                physics_loss = self._compute_physics_loss(
                    reservoir_states, physics_state
                )
                physics_losses.append(physics_loss)
                
                # Add physics guidance
                reservoir_states = reservoir_states + self.config.physics_weight * physics_signal
            
            # Leaky integration
            new_state = (1 - self.config.leak_rate) * reservoir_states + \
                       self.config.leak_rate * torch.tanh(
                           input_term + reservoir_term
                       )
            
            reservoir_states = new_state
            states_history.append(reservoir_states.detach())
        
        # Generate output
        predictions = self.output_layer(reservoir_states)
        
        # Prepare diagnostics
        diagnostics = {
            'reservoir_states': torch.stack(states_history),
            'physics_losses': torch.stack(physics_losses) if physics_losses else None,
            'spectral_radius': self._compute_effective_spectral_radius(),
            'memory_capacity': self._estimate_memory_capacity(states_history)
        }
        
        return predictions, diagnostics
    
    def _compute_physics_loss(self, reservoir_states: torch.Tensor,
                            physics_state: Dict) -> torch.Tensor:
        """Compute physics constraint loss."""
        if self.physics_module is None:
            return torch.tensor(0.0, device=reservoir_states.device)
        
        # Decode physics-relevant information from reservoir
        # This is problem-specific
        if 'pressure' in physics_state and 'saturation' in physics_state:
            # For reservoir simulation
            physics_loss = self.physics_module.darcy_loss(
                physics_state['pressure'],
                physics_state['saturation']
            )
        else:
            # Generic regularization
            physics_loss = torch.norm(reservoir_states, p=2)
        
        return physics_loss
    
    def _compute_effective_spectral_radius(self) -> float:
        """Compute effective spectral radius of reservoir."""
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(self.W_res)
            return torch.max(torch.abs(eigenvalues)).item()
    
    def _estimate_memory_capacity(self, states_history: List[torch.Tensor]) -> float:
        """Estimate memory capacity using linear memory task."""
        # Simplified memory capacity estimation
        if len(states_history) < 2:
            return 0.0
        
        states = torch.stack(states_history)
        autocorrelation = torch.corrcoef(states.flatten().unsqueeze(0))[0, 1]
        
        return autocorrelation.item()


class MultiHeadAttention(nn.Module):
    """Multi-head attention for reservoir state refinement."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, \
            "Embedding dimension must be divisible by number of heads"
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention."""
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(2)
        
        # Scaled dot-product attention
        scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.einsum('bhqk,bkhd->bqhd', attn, v)
        out = out.reshape(batch_size, seq_len, embed_dim)
        
        # Final projection
        out = self.output_proj(out)
        
        return out
