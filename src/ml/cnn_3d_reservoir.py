"""
3D CNN for Reservoir Property Prediction and Segmentation
PhD-Level Implementation for Spatial Analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional

class Reservoir3DCNN(nn.Module):
    """
    3D Convolutional Neural Network for reservoir property prediction.
    Input: 3D grid of properties (24×25×15)
    Output: Pressure/Saturation predictions or property classification
    """
    
    def __init__(self, input_channels: int = 4, output_classes: int = 1):
        super().__init__()
        
        # Encoder pathway
        self.encoder1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.3)
        )
        
        # Decoder pathway
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.output = nn.Conv3d(32, output_classes, kernel_size=1)
        
        # Attention mechanism for PhD innovation
        self.attention = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for 3D reservoir data."""
        # Encoder
        enc1 = self.encoder1(x)  # 24×25×15 → 24×25×15
        pool1 = self.pool1(enc1)  # 24×25×15 → 12×12×7
        
        enc2 = self.encoder2(pool1)  # 12×12×7 → 12×12×7
        pool2 = self.pool2(enc2)  # 12×12×7 → 6×6×3
        
        # Bottleneck with attention
        bottleneck = self.bottleneck(pool2)  # 6×6×3 → 6×6×3
        attention_weights = self.attention(bottleneck)
        bottleneck = bottleneck * attention_weights
        
        # Decoder with skip connections
        up2 = self.upconv2(bottleneck)  # 6×6×3 → 12×12×7
        up2 = torch.cat([up2, enc2], dim=1)  # Skip connection
        dec2 = self.decoder2(up2)  # 12×12×7 → 12×12×7
        
        up1 = self.upconv1(dec2)  # 12×12×7 → 24×25×15
        up1 = torch.cat([up1, enc1], dim=1)  # Skip connection
        dec1 = self.decoder1(up1)  # 24×25×15 → 24×25×15
        
        # Output
        output = self.output(dec1)
        
        return output
    
    def predict_pressure_field(self, 
                              porosity: np.ndarray,
                              permeability: np.ndarray,
                              saturation: np.ndarray,
                              well_locations: np.ndarray) -> np.ndarray:
        """Predict pressure field using CNN."""
        # Prepare input tensor
        input_tensor = self._prepare_3d_input(
            porosity, permeability, saturation, well_locations
        )
        
        with torch.no_grad():
            prediction = self.forward(input_tensor)
        
        return prediction.cpu().numpy()
    
    def _prepare_3d_input(self, *features: np.ndarray) -> torch.Tensor:
        """Prepare 3D input tensor from reservoir features."""
        # Stack features along channel dimension
        stacked = np.stack(features, axis=0)
        
        # Add batch dimension
        stacked = np.expand_dims(stacked, axis=0)
        
        return torch.FloatTensor(stacked)

class ReservoirPropertyCNN:
    """CNN-based reservoir property prediction system."""
    
    def __init__(self, use_gpu: bool = True):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = Reservoir3DCNN().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
    def train(self, 
              train_data: Dict[str, np.ndarray],
              val_data: Optional[Dict[str, np.ndarray]] = None,
              epochs: int = 100,
              batch_size: int = 4):
        """Train CNN on reservoir data."""
        
        # Prepare datasets
        train_loader = self._create_data_loader(train_data, batch_size)
        
        if val_data:
            val_loader = self._create_data_loader(val_data, batch_size)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            if val_data and val_loader:
                val_loss = self._validate(val_loader)
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f}")
    
    def predict_sweet_spots(self, 
                           reservoir_data: np.ndarray,
                           threshold: float = 0.7) -> np.ndarray:
        """Identify sweet spots (high potential areas) using CNN."""
        self.model.eval()
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(reservoir_data).unsqueeze(0).to(self.device)
            predictions = self.model(input_tensor)
            
            # Apply threshold to identify sweet spots
            sweet_spots = (predictions > threshold).cpu().numpy()
        
        return sweet_spots
    
    def generate_uncertainty_map(self, 
                                reservoir_data: np.ndarray,
                                n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Generate uncertainty map using Monte Carlo dropout."""
        self.model.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                input_tensor = torch.FloatTensor(reservoir_data).unsqueeze(0).to(self.device)
                pred = self.model(input_tensor)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
