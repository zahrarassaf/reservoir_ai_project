# src/models/physics_informed_esn.py - FIXED
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class PIESNConfig:
    """PhD-level PI-ESN configuration."""
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
    use_bayesian: bool = True

class PhysicsInformedESN(nn.Module):
    """PhD-level Physics-Informed Echo State Network."""
    
    def __init__(self, input_dim: int, output_dim: int, config: PIESNConfig):
        super().__init__()
        
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Echo State Reservoir
        self.W_in = self._initialize_input_weights()
        self.W_res = self._initialize_reservoir_weights()
        self.W_fb = None  # Optional feedback
        
        # Physics-aware components
        self.physics_encoder = PhysicsEncoder(reservoir_size=config.reservoir_size)
        self.physics_constraints = PhysicsConstraints()
        
        # Attention mechanism
        if config.use_attention:
            self.attention = MultiHeadAttention(
                config.reservoir_size, 
                config.attention_heads,
                dropout=config.dropout_rate
            )
        
        # Bayesian output layer for uncertainty
        if config.use_bayesian:
            self.output_layer = BayesianLinear(
                config.reservoir_size, 
                output_dim,
                prior_sigma=1.0
            )
        else:
            self.output_layer = nn.Linear(config.reservoir_size, output_dim)
        
        # Dropout for MC sampling
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Reservoir state
        self.register_buffer('initial_state', torch.zeros(config.reservoir_size))
        
    def _initialize_input_weights(self):
        """Initialize with spectral radius control."""
        weights = torch.randn(self.config.reservoir_size, self.input_dim)
        return nn.Parameter(weights * self.config.input_scaling, requires_grad=False)
    
    def _initialize_reservoir_weights(self):
        """Initialize sparse reservoir with spectral radius."""
        n = self.config.reservoir_size
        
        # Create sparse matrix
        mask = (torch.rand(n, n) < self.config.connectivity).float()
        weights = torch.randn(n, n) * 0.1
        weights = weights * mask
        
        # Ensure spectral radius
        if self.config.spectral_radius > 0:
            eigenvalues = torch.linalg.eigvals(weights)
            current_radius = torch.max(torch.abs(eigenvalues))
            weights = weights * (self.config.spectral_radius / (current_radius + 1e-10))
        
        return nn.Parameter(weights, requires_grad=False)
    
    def forward(self, x: torch.Tensor, return_physics: bool = False) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with physics constraints."""
        batch_size, seq_len, _ = x.shape
        
        # Initialize reservoir states
        states = self.initial_state.unsqueeze(0).repeat(batch_size, 1)
        state_history = []
        physics_losses = []
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Reservoir update
            input_term = torch.matmul(x_t, self.W_in.T)
            reservoir_term = torch.matmul(states, self.W_res.T)
            
            # Attention on reservoir states
            if self.config.use_attention:
                states_reshaped = states.unsqueeze(1)  # [batch, 1, features]
                states = self.attention(states_reshaped).squeeze(1)
            
            # Physics encoding
            physics_features = self.physics_encoder(states)
            
            # Apply physics constraints
            if return_physics:
                physics_loss = self.physics_constraints(physics_features, x_t)
                physics_losses.append(physics_loss)
            
            # Leaky integration
            new_state = (1 - self.config.leak_rate) * states + \
                       self.config.leak_rate * torch.tanh(input_term + reservoir_term + physics_features)
            
            # Apply dropout (enabled for MC uncertainty)
            if self.training or return_physics:
                new_state = self.dropout(new_state)
            
            states = new_state
            state_history.append(states)
        
        # Generate output
        if self.config.use_bayesian:
            predictions, kl_div = self.output_layer(states)
        else:
            predictions = self.output_layer(states)
            kl_div = torch.tensor(0.0)
        
        # Prepare diagnostics
        diagnostics = {}
        if return_physics:
            diagnostics = {
                'reservoir_states': torch.stack(state_history, dim=1),
                'physics_loss': torch.stack(physics_losses).mean() if physics_losses else torch.tensor(0.0),
                'kl_divergence': kl_div,
                'spectral_properties': self._analyze_spectral_properties()
            }
        
        return predictions, diagnostics
    
    def _analyze_spectral_properties(self):
        """Analyze reservoir spectral properties."""
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(self.W_res)
            spectral_radius = torch.max(torch.abs(eigenvalues))
            
            # Lyapunov exponent approximation
            jacobian_norm = torch.norm(self.W_res)
            lyapunov = torch.log(jacobian_norm)
        
        return {
            'spectral_radius': spectral_radius.item(),
            'lyapunov_exponent': lyapunov.item(),
            'connectivity': (self.W_res != 0).float().mean().item()
        }

class PhysicsEncoder(nn.Module):
    """Encode physics constraints into reservoir."""
    
    def __init__(self, reservoir_size: int, hidden_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(reservoir_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, reservoir_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class PhysicsConstraints(nn.Module):
    """Physics constraints loss computation."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, reservoir_states: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Compute physics constraint losses."""
        # Mass conservation constraint
        mass_loss = self._mass_conservation(reservoir_states, inputs)
        
        # Energy/entropy constraint
        energy_loss = self._energy_constraint(reservoir_states)
        
        # Smoothness constraint
        smoothness_loss = self._smoothness_constraint(reservoir_states)
        
        total_loss = mass_loss + energy_loss + smoothness_loss
        
        return total_loss
    
    def _mass_conservation(self, states, inputs):
        """Mass conservation loss."""
        # Simplified: states should preserve "mass" in some sense
        mass = torch.sum(states, dim=-1)
        mass_variance = torch.var(mass)
        
        return mass_variance * 0.1
    
    def _energy_constraint(self, states):
        """Energy/entropy constraint."""
        # States should have bounded energy
        energy = torch.norm(states, dim=-1)
        energy_mean = torch.mean(energy)
        
        # Penalize extreme energies
        return torch.abs(energy_mean - 1.0)
    
    def _smoothness_constraint(self, states):
        """Spatial/temporal smoothness."""
        # States should vary smoothly
        if len(states.shape) > 1:
            diff = torch.diff(states, dim=0)
            smoothness = torch.mean(torch.abs(diff))
            return smoothness * 0.01
        return torch.tensor(0.0)

class BayesianLinear(nn.Module):
    """Bayesian linear layer for uncertainty."""
    
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # Prior
        self.prior_sigma = prior_sigma
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -3.0)
        
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, -3.0)
    
    def forward(self, x):
        """Forward with reparameterization."""
        # Sample weights
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_epsilon = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_sigma * weight_epsilon
        
        # Sample bias
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias_epsilon = torch.randn_like(self.bias_mu)
        bias = self.bias_mu + bias_sigma * bias_epsilon
        
        # KL divergence
        kl_div = self._kl_divergence(
            self.weight_mu, weight_sigma,
            self.bias_mu, bias_sigma
        )
        
        # Linear operation
        output = F.linear(x, weight, bias)
        
        return output, kl_div
    
    def _kl_divergence(self, w_mu, w_sigma, b_mu, b_sigma):
        """KL divergence with Gaussian prior."""
        # Prior: N(0, prior_sigma^2)
        # Posterior: N(mu, sigma^2)
        
        w_kl = torch.log(self.prior_sigma / w_sigma) + \
               (w_sigma**2 + w_mu**2) / (2 * self.prior_sigma**2) - 0.5
        
        b_kl = torch.log(self.prior_sigma / b_sigma) + \
               (b_sigma**2 + b_mu**2) / (2 * self.prior_sigma**2) - 0.5
        
        return w_kl.sum() + b_kl.sum()

class MultiHeadAttention(nn.Module):
    """Multi-head attention for reservoir."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        return self.proj(out)
