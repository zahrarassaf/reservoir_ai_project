"""
CNN for Reservoir Property Prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class ReservoirDataset(Dataset):
    
    def __init__(self, grid_data: np.ndarray, 
                 properties: Dict[str, np.ndarray],
                 transform=None):
        """
        Args:
            grid_data: 3D grid data (Nx, Ny, Nz)
            properties: Dictionary of properties (permeability, porosity, etc.)
            transform: Optional transforms
        """
        self.grid_data = torch.FloatTensor(grid_data).unsqueeze(0)
        self.properties = {k: torch.FloatTensor(v) for k, v in properties.items()}
        self.transform = transform
        
        # Store dimensions
        self.ny, self.nz = self.grid_data.shape[2], self.grid_data.shape[3]
        
    def __len__(self):
        return self.ny * self.nz
    
    def __getitem__(self, idx):
        # Convert flat index to 2D coordinates
        y_idx = idx // self.nz
        z_idx = idx % self.nz
        
        # Ensure indices are within bounds
        y_idx = max(0, min(y_idx, self.ny - 1))
        z_idx = max(0, min(z_idx, self.nz - 1))
        
        # Extract local 3D patch
        patch_size = 3
        half_size = patch_size // 2
        
        # Get grid dimensions
        nx, ny, nz = self.grid_data.shape[1], self.grid_data.shape[2], self.grid_data.shape[3]
        
        # Initialize patch with zeros
        patch = torch.zeros((1, nx, patch_size, patch_size))
        
        # Extract valid region
        y_start = max(0, y_idx - half_size)
        y_end = min(ny, y_idx + half_size + 1)
        z_start = max(0, z_idx - half_size)
        z_end = min(nz, z_idx + half_size + 1)
        
        # Calculate patch indices
        patch_y_start = half_size - (y_idx - y_start)
        patch_y_end = patch_y_start + (y_end - y_start)
        patch_z_start = half_size - (z_idx - z_start)
        patch_z_end = patch_z_start + (z_end - z_start)
        
        # Fill patch with valid data
        patch[:, :, patch_y_start:patch_y_end, patch_z_start:patch_z_end] = \
            self.grid_data[:, :, y_start:y_end, z_start:z_end]
        
        # Get properties for center cell
        properties = {}
        for prop_name, prop_tensor in self.properties.items():
            properties[prop_name] = prop_tensor[y_idx, z_idx]
        
        return patch, properties

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class CNNReservoirPredictor(nn.Module):
    
    def __init__(self, input_channels=1, num_properties=3):
        super().__init__()
        
        # Simplified architecture for faster training
        self.encoder1 = nn.Sequential(
            nn.Conv3d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        
        self.encoder2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv3d(32, 16, 3, padding=1),
            nn.ReLU()
        )
        
        # Property prediction heads
        self.permeability_head = nn.Conv3d(16, 1, 1)
        self.porosity_head = nn.Conv3d(16, 1, 1)
        self.saturation_head = nn.Conv3d(16, 1, 1)
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        dec = self.decoder(enc2)
        
        permeability = self.permeability_head(dec)
        porosity = self.porosity_head(dec)
        saturation = self.saturation_head(dec)
        
        return {
            'permeability': permeability.squeeze(1),
            'porosity': porosity.squeeze(1),
            'saturation': saturation.squeeze(1)
        }

class PropertyPredictor:
    
    def __init__(self, device=None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNNReservoirPredictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.scaler = {'permeability': None, 'porosity': None, 'saturation': None}
        
    def prepare_data(self, grid_data: np.ndarray, 
                    properties: Dict[str, np.ndarray],
                    train_ratio: float = 0.8):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        # Get dimensions
        nx, ny, nz = grid_data.shape
        
        # Reshape for CNN
        grid_reshaped = grid_data.reshape(1, nx, ny, nz)
        
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
        batch_size = 16
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, epochs=10):
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
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
            
            if (epoch + 1) % 5 == 0:
                print(f'   Epoch {epoch+1}/{epochs}: '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}')
        
        return train_losses, val_losses
    
    def predict(self, grid_data: np.ndarray):
        self.model.eval()
        
        with torch.no_grad():
            nx, ny, nz = grid_data.shape
            grid_tensor = torch.FloatTensor(grid_data).unsqueeze(0).unsqueeze(0).to(self.device)
            
            outputs = self.model(grid_tensor)
            
            predictions = {}
            for prop_name, pred in outputs.items():
                pred_np = pred.cpu().numpy().reshape(-1, 1)
                if self.scaler[prop_name]:
                    pred_np = self.scaler[prop_name].inverse_transform(pred_np)
                predictions[prop_name] = pred_np.reshape(ny, nz)
        
        return predictions
    
    def evaluate(self, test_data, true_properties):
        predictions = self.predict(test_data)
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics = {}
        for prop_name in true_properties:
            pred = predictions[prop_name].flatten()
            true = true_properties[prop_name].flatten()
            
            # Remove NaN values
            mask = ~np.isnan(pred) & ~np.isnan(true)
            pred_clean = pred[mask]
            true_clean = true[mask]
            
            if len(pred_clean) > 0 and len(true_clean) > 0:
                metrics[prop_name] = {
                    'MAE': mean_absolute_error(true_clean, pred_clean),
                    'RMSE': np.sqrt(mean_squared_error(true_clean, pred_clean)),
                    'R2': r2_score(true_clean, pred_clean),
                    'MAPE': np.mean(np.abs((true_clean - pred_clean) / np.maximum(true_clean, 1e-10))) * 100
                }
            else:
                metrics[prop_name] = {
                    'MAE': 0,
                    'RMSE': 0,
                    'R2': 0,
                    'MAPE': 0
                }
        
        return metrics
    
    def visualize_predictions(self, grid_data, true_properties, save_path=None):
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
            im3 = axes[i, 2].imshow(diff, cmap='RdBu', aspect='auto', 
                                   vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
            axes[i, 2].set_title(f'Difference ({prop_name})')
            plt.colorbar(im3, ax=axes[i, 2])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, path='cnn_reservoir_model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler
        }, path)
        print(f"   Model saved to {path}")
    
    def load_model(self, path='cnn_reservoir_model.pth'):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler = checkpoint['scaler']
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    # Test with sample data
    Nx, Ny, Nz = 24, 25, 15
    grid_data = np.random.rand(Nx, Ny, Nz) * 100
    
    properties = {
        'permeability': np.random.lognormal(mean=3.0, sigma=0.5, size=(Ny, Nz)),
        'porosity': np.random.normal(loc=0.2, scale=0.05, size=(Ny, Nz)),
        'saturation': np.random.uniform(0.6, 0.9, size=(Ny, Nz))
    }
    
    predictor = PropertyPredictor()
    train_loader, val_loader = predictor.prepare_data(grid_data, properties)
    
    print("Training CNN model...")
    train_losses, val_losses = predictor.train(train_loader, val_loader, epochs=5)
    
    metrics = predictor.evaluate(grid_data, properties)
    print("\nModel Performance:")
    for prop_name, prop_metrics in metrics.items():
        print(f"\n{prop_name.upper()}:")
        for metric_name, value in prop_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
