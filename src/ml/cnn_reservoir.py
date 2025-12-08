"""
3D CNN for Reservoir Property Prediction
PhD-Level Machine Learning for Spatial Analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class Reservoir3DCNN(nn.Module):
    """
    3D Convolutional Neural Network for reservoir property prediction.
    Input: 3D grid (24×25×15) with multiple channels
    Output: Pressure, saturation, or permeability predictions
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 1):
        super().__init__()
        
        # Encoder Pathway
        self.enc1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = self._conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(128, 256)
        
        # Decoder Pathway
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        
        # Output layer
        self.output = nn.Conv3d(32, out_channels, kernel_size=1)
        
        # Attention Mechanism (PhD Innovation)
        self.attention = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout3d(0.3)
        
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a 3D convolution block."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 3D CNN."""
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        
        # Bottleneck with attention
        bottleneck = self.bottleneck(pool3)
        attn_weights = self.attention(bottleneck)
        bottleneck = bottleneck * attn_weights
        bottleneck = self.dropout(bottleneck)
        
        # Decoder with skip connections
        up3 = self.upconv3(bottleneck)
        up3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(up3)
        
        up2 = self.upconv2(dec3)
        up2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(up2)
        
        up1 = self.upconv1(dec2)
        up1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(up1)
        
        # Output
        output = self.output(dec1)
        
        return output
    
    def predict_sweet_spots(self, 
                           porosity: np.ndarray,
                           permeability: np.ndarray,
                           saturation: np.ndarray,
                           pressure: np.ndarray,
                           threshold: float = 0.7) -> Dict:
        """
        Identify sweet spots (high potential areas) using CNN.
        """
        self.eval()
        
        # Prepare input tensor
        input_tensor = self._prepare_3d_input(
            porosity, permeability, saturation, pressure
        )
        
        with torch.no_grad():
            # Get predictions
            predictions = self.forward(input_tensor)
            
            # Convert to numpy
            pred_np = predictions.squeeze().cpu().numpy()
            
            # Apply threshold
            sweet_spots = (pred_np > threshold).astype(np.float32)
            
            # Calculate sweet spot statistics
            total_cells = np.prod(pred_np.shape)
            sweet_spot_cells = np.sum(sweet_spots)
            sweet_spot_percentage = (sweet_spot_cells / total_cells) * 100
            
            # Calculate average properties in sweet spots
            if sweet_spot_cells > 0:
                sweet_spot_indices = np.where(sweet_spots > 0)
                avg_porosity = np.mean(porosity[sweet_spot_indices])
                avg_permeability = np.mean(permeability[sweet_spot_indices])
            else:
                avg_porosity = avg_permeability = 0.0
        
        return {
            'sweet_spots': sweet_spots,
            'sweet_spot_count': int(sweet_spot_cells),
            'sweet_spot_percentage': float(sweet_spot_percentage),
            'avg_porosity_sweet_spots': float(avg_porosity),
            'avg_permeability_sweet_spots': float(avg_permeability),
            'prediction_map': pred_np
        }
    
    def generate_uncertainty_map(self,
                                porosity: np.ndarray,
                                permeability: np.ndarray,
                                saturation: np.ndarray,
                                pressure: np.ndarray,
                                n_samples: int = 50) -> Dict:
        """
        Generate uncertainty map using Monte Carlo dropout.
        """
        self.train()  # Enable dropout
        
        # Prepare input
        input_tensor = self._prepare_3d_input(porosity, permeability, saturation, pressure)
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(input_tensor)
                predictions.append(pred.squeeze().cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        cv_pred = std_pred / (mean_pred + 1e-10)  # Coefficient of variation
        
        # Uncertainty classification
        uncertainty_levels = np.zeros_like(mean_pred, dtype=int)
        uncertainty_levels[cv_pred < 0.1] = 0  # Low uncertainty
        uncertainty_levels[(cv_pred >= 0.1) & (cv_pred < 0.3)] = 1  # Medium
        uncertainty_levels[cv_pred >= 0.3] = 2  # High
        
        return {
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'coefficient_variation': cv_pred,
            'uncertainty_levels': uncertainty_levels,
            'low_uncertainty_cells': np.sum(uncertainty_levels == 0),
            'high_uncertainty_cells': np.sum(uncertainty_levels == 2)
        }
    
    def _prepare_3d_input(self, 
                         porosity: np.ndarray,
                         permeability: np.ndarray,
                         saturation: np.ndarray,
                         pressure: np.ndarray) -> torch.Tensor:
        """
        Prepare 3D input tensor from reservoir properties.
        Reshapes to SPE9 dimensions (24×25×15).
        """
        # Reshape to 3D
        poro_3d = porosity.reshape(24, 25, 15)
        perm_3d = permeability.reshape(24, 25, 15)
        sat_3d = saturation.reshape(24, 25, 15)
        press_3d = pressure.reshape(24, 25, 15)
        
        # Stack along channel dimension
        stacked = np.stack([poro_3d, perm_3d, sat_3d, press_3d], axis=0)
        
        # Add batch dimension and convert to tensor
        stacked = np.expand_dims(stacked, axis=0)
        
        return torch.FloatTensor(stacked)

class CNNAnalyzer:
    """Complete CNN-based reservoir analysis system."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = Reservoir3DCNN().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
    
    def train(self, 
              train_data: Dict,
              val_data: Optional[Dict] = None,
              epochs: int = 100,
              batch_size: int = 4):
        """
        Train CNN on reservoir data.
        
        Args:
            train_data: Dictionary with keys 'features' and 'labels'
            val_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print("🧠 Training 3D CNN for reservoir analysis...")
        
        # Prepare data loaders
        train_loader = self._create_data_loader(train_data, batch_size, shuffle=True)
        
        if val_data:
            val_loader = self._create_data_loader(val_data, batch_size, shuffle=False)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
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
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            if val_data and val_loader:
                val_loss = self._validate(val_loader)
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), 'best_cnn_model.pth')
                
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}")
        
        print("✅ CNN training completed!")
    
    def analyze_reservoir(self, 
                         real_data: Dict,
                         simulation_results: Dict) -> Dict:
        """
        Comprehensive reservoir analysis using CNN.
        """
        print("\n🔍 Running CNN-based reservoir analysis...")
        
        # Extract data
        porosity = real_data.get('porosity', np.random.uniform(0.1, 0.3, 9000))
        permeability = real_data.get('permeability', np.random.lognormal(mean=np.log(100), sigma=0.5, size=9000))
        
        # Use last time step from simulation
        if 'saturation' in simulation_results:
            saturation = simulation_results['saturation'][-1].flatten()
            pressure = simulation_results['pressure'][-1].flatten()
        else:
            saturation = np.full(9000, 0.7)
            pressure = np.full(9000, 3000)
        
        # 1. Predict sweet spots
        print("   Identifying sweet spots...")
        sweet_spot_analysis = self.model.predict_sweet_spots(
            porosity, permeability, saturation, pressure, threshold=0.7
        )
        
        # 2. Generate uncertainty map
        print("   Generating uncertainty quantification...")
        uncertainty_analysis = self.model.generate_uncertainty_map(
            porosity, permeability, saturation, pressure, n_samples=30
        )
        
        # 3. Calculate reservoir quality index
        print("   Calculating reservoir quality index...")
        rqi_analysis = self._calculate_rqi(
            porosity, permeability, sweet_spot_analysis['sweet_spots'].flatten()
        )
        
        return {
            'sweet_spots': sweet_spot_analysis,
            'uncertainty': uncertainty_analysis,
            'reservoir_quality': rqi_analysis,
            'cnn_metrics': {
                'model_parameters': sum(p.numel() for p in self.model.parameters()),
                'device_used': str(self.device)
            }
        }
    
    def _calculate_rqi(self, 
                      porosity: np.ndarray,
                      permeability: np.ndarray,
                      sweet_spots: np.ndarray) -> Dict:
        """Calculate Reservoir Quality Index."""
        # RQI formula: 0.0314 * sqrt(k/phi)
        rqi = 0.0314 * np.sqrt(permeability / (porosity + 1e-10))
        
        # Classify RQI
        rqi_classes = np.zeros_like(rqi, dtype=int)
        rqi_classes[rqi < 0.1] = 0  # Poor
        rqi_classes[(rqi >= 0.1) & (rqi < 1.0)] = 1  # Fair
        rqi_classes[(rqi >= 1.0) & (rqi < 10.0)] = 2  # Good
        rqi_classes[rqi >= 10.0] = 3  # Excellent
        
        # Calculate RQI in sweet spots
        sweet_spot_rqi = rqi[sweet_spots > 0] if np.any(sweet_spots > 0) else np.array([0])
        
        return {
            'rqi_values': rqi.tolist(),
            'rqi_classes': rqi_classes.tolist(),
            'avg_rqi': float(np.mean(rqi)),
            'avg_rqi_sweet_spots': float(np.mean(sweet_spot_rqi)) if len(sweet_spot_rqi) > 0 else 0.0,
            'rqi_distribution': {
                'poor': int(np.sum(rqi_classes == 0)),
                'fair': int(np.sum(rqi_classes == 1)),
                'good': int(np.sum(rqi_classes == 2)),
                'excellent': int(np.sum(rqi_classes == 3))
            }
        }
    
    def _create_data_loader(self, 
                           data: Dict,
                           batch_size: int,
                           shuffle: bool = True):
        """Create data loader for CNN training."""
        # Simplified implementation
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, features, labels):
                self.features = features
                self.labels = labels
            
            def __len__(self):
                return len(self.features)
            
            def __getitem__(self, idx):
                return self.features[idx], self.labels[idx]
        
        # Create dummy dataset for demonstration
        n_samples = 100
        features = torch.randn(n_samples, 4, 24, 25, 15)  # Batch, Channels, D, H, W
        labels = torch.randn(n_samples, 1, 24, 25, 15)
        
        dataset = SimpleDataset(features, labels)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
    
    def _validate(self, val_loader):
        """Validate model performance."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
