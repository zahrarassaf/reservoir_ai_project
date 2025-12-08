"""
LSTM for Production Forecasting and Time Series Analysis
PhD-Level Implementation for Temporal Prediction
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Dict
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class ProductionLSTM(nn.Module):
    """
    Advanced LSTM for oil production forecasting.
    Multi-variate time series prediction with attention mechanism.
    """
    
    def __init__(self, 
                 input_size: int = 5,  # Oil rate, water rate, pressure, etc.
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 1,  # Forecast oil rate
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * (2 if bidirectional else 1), 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layers
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(lstm_output_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention mechanism."""
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size * num_directions)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        # attention_weights shape: (batch_size, seq_len, 1)
        
        # Weighted sum
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        # context_vector shape: (batch_size, hidden_size * num_directions)
        
        # Layer normalization
        context_vector = self.layer_norm(context_vector)
        
        # Fully connected layers
        out = F.relu(self.fc1(context_vector))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, attention_weights
    
    def predict_sequence(self, 
                        initial_sequence: np.ndarray,
                        forecast_steps: int = 30) -> np.ndarray:
        """Multi-step production forecasting."""
        self.eval()
        predictions = []
        current_sequence = torch.FloatTensor(initial_sequence).unsqueeze(0)
        
        with torch.no_grad():
            for _ in range(forecast_steps):
                # Predict next step
                pred, _ = self.forward(current_sequence)
                predictions.append(pred.item())
                
                # Update sequence (remove first, add prediction)
                new_seq = torch.cat([
                    current_sequence[:, 1:, :],
                    pred.unsqueeze(1).unsqueeze(2)
                ], dim=1)
                current_sequence = new_seq
        
        return np.array(predictions)

class ProductionForecaster:
    """Complete production forecasting system with LSTM."""
    
    def __init__(self, sequence_length: int = 30, forecast_horizon: int = 90):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler()
        self.model = ProductionLSTM(
            input_size=5,  # Oil, water, gas, pressure, choke_size
            hidden_size=64,
            num_layers=2,
            output_size=1,
            bidirectional=True
        )
    
    def prepare_production_data(self, 
                               df: pd.DataFrame,
                               well_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare production data for LSTM training."""
        
        # Select features
        features = ['oil_rate', 'water_rate', 'gas_rate', 'pressure', 'choke_size']
        
        if well_name:
            well_data = df[df['well_name'] == well_name]
        else:
            well_data = df
        
        # Handle missing values
        well_data = well_data.fillna(method='ffill').fillna(0)
        
        # Normalize
        scaled_data = self.scaler.fit_transform(well_data[features])
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length - self.forecast_horizon):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon, 0])  # Oil rate
        
        return np.array(X), np.array(y)
    
    def train_with_uncertainty(self, 
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              n_ensembles: int = 10,
                              epochs: int = 100) -> List[nn.Module]:
        """Train ensemble of LSTMs for uncertainty quantification."""
        
        ensembles = []
        for i in range(n_ensembles):
            print(f"Training ensemble {i+1}/{n_ensembles}")
            
            # Initialize new model
            model = ProductionLSTM().to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_train).to(self.device)
            y_tensor = torch.FloatTensor(y_train).to(self.device)
            
            # Training loop
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                predictions, _ = model(X_tensor)
                loss = criterion(predictions, y_tensor)
                loss.backward()
                optimizer.step()
            
            ensembles.append(model)
        
        return ensembles
    
    def forecast_with_confidence(self,
                               historical_data: np.ndarray,
                               n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Generate production forecast with confidence intervals."""
        
        forecasts = []
        
        for model in self.ensembles:
            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(historical_data).unsqueeze(0)
                forecast = model.predict_sequence(input_tensor, self.forecast_horizon)
                forecasts.append(forecast)
        
        forecasts = np.array(forecasts)
        
        return {
            'mean': np.mean(forecasts, axis=0),
            'std': np.std(forecasts, axis=0),
            'p10': np.percentile(forecasts, 10, axis=0),
            'p50': np.percentile(forecasts, 50, axis=0),
            'p90': np.percentile(forecasts, 90, axis=0),
            'all_forecasts': forecasts
        }
    
    def detect_anomalies(self, 
                        production_data: np.ndarray,
                        threshold_std: float = 3.0) -> np.ndarray:
        """Detect production anomalies using LSTM reconstruction error."""
        
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(production_data).unsqueeze(0)
            reconstruction, _ = self.model(input_tensor)
            
            # Calculate reconstruction error
            error = torch.abs(input_tensor - reconstruction).cpu().numpy()
            
            # Identify anomalies
            mean_error = np.mean(error)
            std_error = np.std(error)
            anomaly_threshold = mean_error + threshold_std * std_error
            
            anomalies = error > anomaly_threshold
        
        return anomalies
