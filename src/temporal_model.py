import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

class TemporalAttention(nn.Module):
    """Multi-head attention for temporal dependencies"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.size()
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        return self.output(context)

class TemporalBlock(nn.Module):
    """Temporal processing block with attention and convolution"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = TemporalAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Temporal convolution for local patterns
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        
        # Temporal convolution
        conv_input = x.transpose(1, 2)  # [batch, hidden, seq]
        conv_output = self.conv(conv_input).transpose(1, 2)  # [batch, seq, hidden]
        x = x + conv_output
        
        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x

class ReservoirTemporalModel(nn.Module):
    """Physics-aware temporal model for reservoir production forecasting"""
    
    def __init__(self, config, temporal_config):
        super().__init__()
        self.config = config
        self.temporal_config = temporal_config
        
        # Input projections
        self.static_projection = nn.Linear(
            len(config.input_features) - len(temporal_config.time_features), 
            config.hidden_layers[0] // 2
        )
        self.temporal_projection = nn.Linear(
            len(temporal_config.time_features), 
            config.hidden_layers[0] // 2
        )
        
        # Temporal processing
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(config.hidden_layers[0], num_heads=8)
            for _ in range(3)
        ])
        
        # Output heads
        self.production_head = nn.Linear(config.hidden_layers[0], len(config.output_features))
        self.uncertainty_head = nn.Linear(config.hidden_layers[0], len(config.output_features))
        
        # Physics-aware layers
        self.physics_projection = nn.Linear(5, config.hidden_layers[0])  # Physics features
        
    def forward(self, static_features: torch.Tensor, 
                temporal_sequence: torch.Tensor,
                physics_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with physics-aware temporal modeling
        
        Args:
            static_features: Static reservoir features [batch, static_dim]
            temporal_sequence: Temporal sequence [batch, seq_len, temporal_dim]
            physics_features: Physics-based features [batch, physics_dim]
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        batch_size, seq_len, _ = temporal_sequence.shape
        
        # Project inputs
        static_proj = self.static_projection(static_features).unsqueeze(1).expand(-1, seq_len, -1)
        temporal_proj = self.temporal_projection(temporal_sequence)
        
        # Combine static and temporal features
        combined = torch.cat([static_proj, temporal_proj], dim=-1)
        
        # Add physics features if available
        if physics_features is not None:
            physics_proj = self.physics_projection(physics_features).unsqueeze(1).expand(-1, seq_len, -1)
            combined = combined + physics_proj
        
        # Temporal processing
        temporal_output = combined
        for block in self.temporal_blocks:
            temporal_output = block(temporal_output)
        
        # Last time step for prediction
        last_output = temporal_output[:, -1, :]
        
        # Predictions
        production_pred = self.production_head(last_output)
        uncertainty_pred = F.softplus(self.uncertainty_head(last_output))  # Positive uncertainty
        
        return {
            'production': production_pred,
            'uncertainty': uncertainty_pred,
            'temporal_features': temporal_output
        }
