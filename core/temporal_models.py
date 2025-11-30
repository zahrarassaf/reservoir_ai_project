import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if config.model_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=config.input_channels,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                bidirectional=config.bidirectional,
                dropout=config.dropout if config.num_layers > 1 else 0,
                batch_first=True
            )
        elif config.model_type == "gru":
            self.rnn = nn.GRU(
                input_size=config.input_channels,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                bidirectional=config.bidirectional,
                dropout=config.dropout if config.num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        rnn_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=rnn_output_dim,
                num_heads=8,
                dropout=config.dropout,
                batch_first=True
            )
        else:
            self.attention = None
        
        self.output_layer = nn.Sequential(
            nn.Linear(rnn_output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_channels)
        )
    
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        
        if self.attention is not None:
            attended_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
            output = self.output_layer(attended_out)
        else:
            output = self.output_layer(rnn_out)
        
        return output
    
    def compute_physics_loss(self, predictions, targets, static_properties):
        data_loss = F.mse_loss(predictions, targets)
        return data_loss
