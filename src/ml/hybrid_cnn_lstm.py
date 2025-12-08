"""
Hybrid CNN-LSTM Model for Spatio-temporal Reservoir Analysis
PhD Innovation: Combining spatial and temporal analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for spatio-temporal reservoir forecasting.
    CNN extracts spatial features, LSTM captures temporal dynamics.
    """
    
    def __init__(self,
                 spatial_dims: Tuple[int, int, int] = (24, 25, 15),
                 temporal_features: int = 5,
                 num_classes: int = 1,
                 lstm_hidden: int = 64,
                 lstm_layers: int = 2):
        super().__init__()
        
        # CNN for spatial feature extraction
        self.cnn_encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)  # Global average pooling
        )
        
        # Calculate CNN output size
        self.cnn_output_size = 64
        
        # LSTM for temporal sequence processing
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size + temporal_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Output layers
        lstm_output_size = lstm_hidden * 2
        self.fc1 = nn.Linear(lstm_output_size, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, num_classes)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
        # Residual connection
        self.residual = nn.Linear(self.cnn_output_size + temporal_features, lstm_output_size)
        
    def forward(self,
               spatial_input: torch.Tensor,      # 3D spatial data
               temporal_input: torch.Tensor,     # Time series data
               return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through hybrid model.
        
        Args:
            spatial_input: (batch, 1, depth, height, width)
            temporal_input: (batch, seq_len, temporal_features)
        
        Returns:
            predictions, attention_weights
        """
        batch_size, seq_len, _ = temporal_input.shape
        
        # CNN feature extraction for each time step
        spatial_features = []
        for t in range(seq_len):
            # Extract spatial features at time t
            spatial_t = spatial_input[:, t].unsqueeze(1)  # Add channel dim
            features_t = self.cnn_encoder(spatial_t)
            features_t = features_t.view(batch_size, -1)  # Flatten
            spatial_features.append(features_t)
        
        spatial_features = torch.stack(spatial_features, dim=1)  # (batch, seq_len, cnn_features)
        
        # Combine spatial and temporal features
        combined = torch.cat([spatial_features, temporal_input], dim=2)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(combined)
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        residual = self.residual(combined)
        attn_out = attn_out + residual  # Skip connection
        
        # Layer normalization
        attn_out = self.layer_norm(attn_out)
        
        # Take last time step for prediction
        last_output = attn_out[:, -1, :]
        
        # Fully connected layers
        out = F.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        if return_attention:
            return out, attn_weights
        return out, None
    
    def forecast_production(self,
                           spatial_sequence: np.ndarray,
                           temporal_sequence: np.ndarray,
                           forecast_steps: int = 30) -> Dict[str, np.ndarray]:
        """
        Multi-step production forecasting with uncertainty.
        
        Args:
            spatial_sequence: (seq_len, depth, height, width)
            temporal_sequence: (seq_len, temporal_features)
            forecast_steps: Number of steps to forecast
        
        Returns:
            Dictionary with predictions and uncertainty
        """
        self.eval()
        
        predictions = []
        attention_weights = []
        
        current_spatial = torch.FloatTensor(spatial_sequence).unsqueeze(0)
        current_temporal = torch.FloatTensor(temporal_sequence).unsqueeze(0)
        
        with torch.no_grad():
            for step in range(forecast_steps):
                # Predict next value
                pred, attn = self.forward(
                    current_spatial,
                    current_temporal,
                    return_attention=True
                )
                
                predictions.append(pred.item())
                if attn is not None:
                    attention_weights.append(attn.cpu().numpy())
                
                # Update sequences for next step
                # For spatial: assume small change (could be improved)
                new_spatial = torch.cat([
                    current_spatial[:, 1:, :, :, :],
                    current_spatial[:, -1:, :, :, :]  # Repeat last
                ], dim=1)
                
                # Update temporal: add prediction as new feature
                new_temporal_feat = torch.cat([
                    current_temporal[:, 1:, :],
                    pred.unsqueeze(1).unsqueeze(2)
                ], dim=1)
                
                current_spatial = new_spatial
                current_temporal = new_temporal_feat
        
        predictions = np.array(predictions)
        
        # Calculate confidence intervals (simple version)
        # In practice, use Monte Carlo dropout or ensemble
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        
        return {
            'predictions': predictions,
            'mean': pred_mean,
            'std': pred_std,
            'confidence_interval': [
                pred_mean - 1.96 * pred_std,
                pred_mean + 1.96 * pred_std
            ],
            'attention_weights': attention_weights if attention_weights else None
        }
