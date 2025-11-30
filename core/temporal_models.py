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
        
        rnn_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        self.output_layer = nn.Sequential(
            nn.Linear(rnn_output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_channels)
        )
    
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        output = self.output_layer(rnn_out)
        return output
