"""
CNN for Reservoir Property Prediction
Predicts permeability, porosity, and saturation from spatial data
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class ReservoirDataset(Dataset):
    """Dataset for reservoir properties"""
    
    def __init__(self, grid_data: np.ndarray, 
                 properties: Dict[str, np.ndarray],
                 transform=None):
        """
        Args:
            grid_data: 3D grid data (Nx, Ny, Nz)
            properties: Dictionary of properties (permeability, porosity, etc.)
            transform: Optional transforms
        """
        self.grid_data = torch.FloatTensor(grid_data).unsqueeze(0)  # Add channel dim
        self.properties = {k: torch.FloatTensor(v) for k, v in properties.items()}
        self.transform = transform
        
    def __len__(self):
        return self.grid_data.shape[1] * self.grid_data.shape[2]  # Ny * Nz
    
    def __getitem__(self, idx):
        # Convert flat index to 3D coordinates
        ny, nz = self.grid_data.shape[2], self.grid_data.shape[3]
        y_idx = idx // nz
        z_idx = idx % nz
        
        # Extract local 3D patch
        patch_size = 5
        half_size = patch_size // 2
        
        # Pad if needed
        padded_grid = F.pad(self.grid_data, 
                           (half_size, half_size, half_size, half_size, half_size, half_size),
                           mode='constant', value=0)
        
        # Extract patch
        patch = padded_grid[:, 
                           :,  # All x
                           y_idx:y_idx + patch_size,
                           z_idx:z_idx + patch_size]
        
        # Get properties for center cell
        properties = {k: v[y_idx, z_idx] for k, v in self.properties.items()}
        
        return patch, properties


class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        
        # Shortcut connection
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class CNNReservoirPredictor(nn.Module):
    """
    3D CNN for predicting reservoir properties
    Architecture inspired by U-Net for spatial feature extraction
    """
    
    def __init__(self, input_channels=1, num_properties=3):
        super().__init__()
        
        # Encoder
        self.encoder1 = ResidualBlock(input_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = ResidualBlock(64, 128)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(128, 256)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(256, 128)
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(128, 64)
        
        # Property prediction heads
        self.permeability_head = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, 1)
        )
        
        self.porosity_head = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, 1)
        )
        
        self.saturation_head = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, 1)
        )
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)  # 32 channels
        enc2 = self.encoder2(self.pool1(enc1))  # 64 channels
        enc3 = self.encoder3(self.pool2(enc2))  # 128 channels
        
        # Bottleneck
        bottleneck = self.bottleneck(enc3)  # 256 channels
        
        # Decoder with skip connections
        dec1 = self.upconv1(bottleneck)
        dec1 = torch.cat([dec1, enc3], dim=1)
        dec1 = self.decoder1(dec1)
        
        dec2 = self.upconv2(dec1)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        # Predict properties
        permeability = self.permeability_head(dec2)
        porosity = self.porosity_head(dec2)
        saturation = self.saturation_head(dec2)
        
        return {
            'permeability': permeability.squeeze(1),
            'porosity': porosity.squeeze(1),
            'saturation': saturation.squeeze(1)
        }


class PropertyPredictor:
    """High-level wrapper for property prediction"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model = CNNReservoirPredictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.scaler = {'permeability': None, 'porosity': None, 'saturation': None}
        
    def prepare_data(self, grid_data: np.ndarray, 
                    properties: Dict[str, np.ndarray],
                    train_ratio: float = 0.8):
        """Prepare and split data"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        # Reshape for CNN
        Nx, Ny, Nz = grid_data.shape
        grid_reshaped = grid_data.reshape(1, Nx, Ny, Nz)
        
        # Scale properties
        scaled_properties = {}
        for name, prop in properties.items():
            scaler = StandardScaler()
            prop_flat = prop.reshape(-1, 1)
            prop_scaled = scaler.fit_transform(prop_flat).reshape(prop.shape)
            scaled_properties[name] = prop_scaled
            self.scaler[name] = scaler
        
        # Create dataset
        dataset = ReservoirDataset(grid_reshaped, scaled_properties)
        
        # Split indices
        indices = np.arange(len(dataset))
        train_idx, val_idx = train_test_split(indices, train_size=train_ratio, random_state=42)
        
        # Create samplers
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        # Create dataloaders
        batch_size = 32
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, epochs=50):
        """Train the model"""
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for patches, properties in train_loader:
                patches = patches.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(patches)
                loss = 0
                for prop_name in properties:
                    prop_tensor = properties[prop_name].to(self.device).unsqueeze(1)
                    loss += self.criterion(outputs[prop_name], prop_tensor)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for patches, properties in val_loader:
                    patches = patches.to(self.device)
                    outputs = self.model(patches)
                    
                    batch_loss = 0
                    for prop_name in properties:
                        prop_tensor = properties[prop_name].to(self.device).unsqueeze(1)
                        batch_loss += self.criterion(outputs[prop_name], prop_tensor)
                    
                    val_loss += batch_loss.item()
            
            # Store losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}: '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}')
        
        return train_losses, val_losses
    
    def predict(self, grid_data: np.ndarray):
        """Predict properties for entire grid"""
        self.model.eval()
        
        with torch.no_grad():
            # Prepare input
            Nx, Ny, Nz = grid_data.shape
            grid_tensor = torch.FloatTensor(grid_data).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Predict
            outputs = self.model(grid_tensor)
            
            # Inverse transform
            predictions = {}
            for prop_name, pred in outputs.items():
                pred_np = pred.cpu().numpy().reshape(-1, 1)
                if self.scaler[prop_name]:
                    pred_np = self.scaler[prop_name].inverse_transform(pred_np)
                predictions[prop_name] = pred_np.reshape(Ny, Nz)
        
        return predictions
    
    def evaluate(self, test_data, true_properties):
        """Evaluate model performance"""
        predictions = self.predict(test_data)
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics = {}
        for prop_name in true_properties:
            pred = predictions[prop_name].flatten()
            true = true_properties[prop_name].flatten()
            
            metrics[prop_name] = {
                'MAE': mean_absolute_error(true, pred),
                'RMSE': np.sqrt(mean_squared_error(true, pred)),
                'R2': r2_score(true, pred),
                'MAPE': np.mean(np.abs((true - pred) / true)) * 100
            }
        
        return metrics
    
    def visualize_predictions(self, grid_data, true_properties, save_path=None):
        """Visualize predictions vs true values"""
        predictions = self.predict(grid_data)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        properties = ['permeability', 'porosity', 'saturation']
        
        for i, prop_name in enumerate(properties):
            # True values
            im1 = axes[i, 0].imshow(true_properties[prop_name], cmap='viridis', aspect='auto')
            axes[i, 0].set_title(f'True {prop_name.capitalize()}')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Predicted values
            im2 = axes[i, 1].imshow(predictions[prop_name], cmap='viridis', aspect='auto')
            axes[i, 1].set_title(f'Predicted {prop_name.capitalize()}')
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Difference
            diff = predictions[prop_name] - true_properties[prop_name]
            im3 = axes[i, 2].imshow(diff, cmap='RdBu', aspect='auto', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
            axes[i, 2].set_title(f'Difference ({prop_name})')
            plt.colorbar(im3, ax=axes[i, 2])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, path='cnn_reservoir_model.pth'):
        """Save model and scalers"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='cnn_reservoir_model.pth'):
        """Load model and scalers"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler = checkpoint['scaler']
        print(f"Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    # Example data
    Nx, Ny, Nz = 24, 25, 15
    grid_data = np.random.rand(Nx, Ny, Nz) * 100
    
    # Simulate properties
    properties = {
        'permeability': np.random.lognormal(mean=3.0, sigma=0.5, size=(Ny, Nz)),
        'porosity': np.random.normal(loc=0.2, scale=0.05, size=(Ny, Nz)),
        'saturation': np.random.uniform(0.6, 0.9, size=(Ny, Nz))
    }
    
    # Create and train predictor
    predictor = PropertyPredictor()
    train_loader, val_loader = predictor.prepare_data(grid_data, properties)
    
    print("Training CNN model...")
    train_losses, val_losses = predictor.train(train_loader, val_loader, epochs=20)
    
    # Evaluate
    metrics = predictor.evaluate(grid_data, properties)
    print("\nModel Performance:")
    for prop_name, prop_metrics in metrics.items():
        print(f"\n{prop_name.upper()}:")
        for metric_name, value in prop_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Visualize
    predictor.visualize_predictions(grid_data, properties, save_path='cnn_predictions.png')
