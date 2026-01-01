"""
CNN for Reservoir Property Prediction - FIXED VERSION
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class ReservoirDataset(Dataset):
    
    def __init__(self, grid_data: np.ndarray, 
                 properties: Dict[str, np.ndarray],
                 patch_size: int = 3,
                 transform=None):
        """
        Args:
            grid_data: 3D grid data (Nx, Ny, Nz) - shape (24, 25, 15)
            properties: Dictionary of 2D properties arrays (Ny, Nz)
            patch_size: Size of 3D patch to extract
            transform: Optional transforms
        """
        # grid_data shape: (Nx, Ny, Nz) = (24, 25, 15)
        # Add batch and channel dimensions: (1, 1, Nx, Ny, Nz)
        self.grid_data = torch.FloatTensor(grid_data).unsqueeze(0).unsqueeze(0)
        self.properties = {k: torch.FloatTensor(v) for k, v in properties.items()}
        self.patch_size = patch_size
        self.transform = transform
        
        # Store dimensions
        self.nx, self.ny, self.nz = grid_data.shape
        
        # Create list of valid patch centers
        self.valid_indices = []
        half_size = patch_size // 2
        
        for y in range(self.ny):
            for z in range(self.nz):
                # Check if patch can be extracted without going out of bounds
                y_start = max(0, y - half_size)
                y_end = min(self.ny, y + half_size + 1)
                z_start = max(0, z - half_size)
                z_end = min(self.nz, z + half_size + 1)
                
                # Only include if patch is full size (optional, can remove)
                if (y_end - y_start == patch_size and 
                    z_end - z_start == patch_size):
                    self.valid_indices.append((y, z))
        
        # If no valid indices (edge cells), include all
        if not self.valid_indices:
            for y in range(self.ny):
                for z in range(self.nz):
                    self.valid_indices.append((y, z))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        y_idx, z_idx = self.valid_indices[idx]
        half_size = self.patch_size // 2
        
        # Calculate patch boundaries
        y_start = max(0, y_idx - half_size)
        y_end = min(self.ny, y_idx + half_size + 1)
        z_start = max(0, z_idx - half_size)
        z_end = min(self.nz, z_idx + half_size + 1)
        
        # Calculate padding if needed
        pad_top = half_size - (y_idx - y_start)
        pad_bottom = (y_idx + half_size + 1) - y_end
        pad_left = half_size - (z_idx - z_start)
        pad_right = (z_idx + half_size + 1) - z_end
        
        # Extract patch with padding if needed
        patch = self.grid_data[0, 0, :, y_start:y_end, z_start:z_end].clone()
        
        # Apply padding if patch is smaller than patch_size
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            # Note: PyTorch padding order is (left, right, top, bottom, front, back)
            # But we're working with 3D: (depth, height, width) -> (Nx, patch_size, patch_size)
            pad_dims = (pad_left, pad_right, pad_top, pad_bottom)
            patch = F.pad(patch, pad_dims, mode='constant', value=0)
        
        # Add channel dimension: (1, Nx, patch_size, patch_size)
        patch = patch.unsqueeze(0)
        
        # Get properties for center cell
        property_values = {}
        for prop_name, prop_tensor in self.properties.items():
            property_values[prop_name] = prop_tensor[y_idx, z_idx]
        
        return patch, property_values

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
    
    def __init__(self, input_channels=1, num_properties=3, nx=24):
        super().__init__()
        
        # Input shape: (batch, channels, depth, height, width) = (B, 1, 24, 3, 3)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((2, 1, 1)),  # Reduce depth dimension
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        
        # Calculate size after encoder
        self.encoded_depth = nx // 2  # After maxpool with stride 2 in depth
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        
        # Property prediction heads
        self.permeability_head = nn.Sequential(
            nn.Conv3d(16, 8, 1),
            nn.ReLU(),
            nn.Conv3d(8, 1, 1)
        )
        self.porosity_head = nn.Sequential(
            nn.Conv3d(16, 8, 1),
            nn.ReLU(),
            nn.Conv3d(8, 1, 1)
        )
        self.saturation_head = nn.Sequential(
            nn.Conv3d(16, 8, 1),
            nn.ReLU(),
            nn.Conv3d(8, 1, 1)
        )
        
    def forward(self, x):
        # x shape: (B, 1, 24, 3, 3)
        
        # Encoder
        enc = self.encoder(x)  # (B, 64, 12, 3, 3)
        
        # Decoder
        dec = self.decoder(enc)  # (B, 16, 24, 3, 3)
        
        # Predict properties
        permeability = self.permeability_head(dec)  # (B, 1, 24, 3, 3)
        porosity = self.porosity_head(dec)  # (B, 1, 24, 3, 3)
        saturation = self.saturation_head(dec)  # (B, 1, 24, 3, 3)
        
        # Take mean over spatial dimensions for final prediction
        permeability_pred = permeability.mean(dim=[2, 3, 4])  # (B, 1)
        porosity_pred = porosity.mean(dim=[2, 3, 4])  # (B, 1)
        saturation_pred = saturation.mean(dim=[2, 3, 4])  # (B, 1)
        
        return {
            'permeability': permeability_pred.squeeze(-1),
            'porosity': porosity_pred.squeeze(-1),
            'saturation': saturation_pred.squeeze(-1)
        }

class PropertyPredictor:
    
    def __init__(self, device=None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model with correct input dimensions
        self.model = CNNReservoirPredictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.scaler = {'permeability': None, 'porosity': None, 'saturation': None}
        
    def prepare_data(self, grid_data: np.ndarray, 
                    properties: Dict[str, np.ndarray],
                    train_ratio: float = 0.8,
                    patch_size: int = 3):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        # Validate shapes
        nx, ny, nz = grid_data.shape
        print(f"Grid data shape: {grid_data.shape}")
        print(f"Properties shapes: { {k: v.shape for k, v in properties.items()} }")
        
        # Scale properties
        scaled_properties = {}
        for name, prop in properties.items():
            if prop.shape != (ny, nz):
                raise ValueError(f"Property {name} has shape {prop.shape}, expected ({ny}, {nz})")
            
            scaler = StandardScaler()
            prop_flat = prop.reshape(-1, 1)
            prop_scaled = scaler.fit_transform(prop_flat).reshape(prop.shape)
            scaled_properties[name] = prop_scaled
            self.scaler[name] = scaler
        
        # Create dataset
        dataset = ReservoirDataset(grid_data, scaled_properties, patch_size=patch_size)
        print(f"Dataset created with {len(dataset)} samples")
        
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
        
        # Test one batch
        test_batch = next(iter(train_loader))
        patches, props = test_batch
        print(f"Batch patches shape: {patches.shape}")
        print(f"Batch properties shapes: { {k: v.shape for k, v in props.items()} }")
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, epochs=10):
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for patches, properties in train_loader:
                patches = patches.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(patches)
                
                # Calculate loss
                loss = 0
                for prop_name in properties:
                    # properties[prop_name] shape: (batch_size,)
                    prop_tensor = properties[prop_name].to(self.device)
                    pred_tensor = outputs[prop_name]
                    loss += self.criterion(pred_tensor, prop_tensor)
                
                # Backward pass
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
                        prop_tensor = properties[prop_name].to(self.device)
                        pred_tensor = outputs[prop_name]
                        batch_loss += self.criterion(pred_tensor, prop_tensor)
                    
                    val_loss += batch_loss.item()
            
            # Average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}')
        
        return train_losses, val_losses
    
    def predict(self, grid_data: np.ndarray, patch_size: int = 3):
        self.model.eval()
        
        nx, ny, nz = grid_data.shape
        predictions = {
            'permeability': np.zeros((ny, nz)),
            'porosity': np.zeros((ny, nz)),
            'saturation': np.zeros((ny, nz))
        }
        counts = np.zeros((ny, nz))
        
        with torch.no_grad():
            # Create dataset for prediction
            dummy_props = {
                'permeability': np.zeros((ny, nz)),
                'porosity': np.zeros((ny, nz)),
                'saturation': np.zeros((ny, nz))
            }
            dataset = ReservoirDataset(grid_data, dummy_props, patch_size=patch_size)
            
            for i in range(len(dataset)):
                patch, _ = dataset[i]
                patch = patch.unsqueeze(0).to(self.device)  # Add batch dimension
                
                # Get prediction
                outputs = self.model(patch)
                
                # Get coordinates
                y_idx, z_idx = dataset.valid_indices[i]
                
                # Store predictions
                for prop_name in predictions:
                    pred_value = outputs[prop_name].cpu().numpy()[0]
                    
                    # Inverse transform if scaler exists
                    if self.scaler[prop_name]:
                        pred_value = self.scaler[prop_name].inverse_transform(
                            np.array([[pred_value]])
                        )[0, 0]
                    
                    predictions[prop_name][y_idx, z_idx] += pred_value
                    counts[y_idx, z_idx] += 1
        
        # Average predictions for cells with multiple patches
        for prop_name in predictions:
            mask = counts > 0
            predictions[prop_name][mask] /= counts[mask]
        
        return predictions
    
    def evaluate(self, grid_data, true_properties):
        predictions = self.predict(grid_data)
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics = {}
        for prop_name in true_properties:
            pred = predictions[prop_name].flatten()
            true = true_properties[prop_name].flatten()
            
            # Remove NaN and zero values for MAPE
            mask = ~np.isnan(pred) & ~np.isnan(true) & (true != 0)
            pred_clean = pred[mask]
            true_clean = true[mask]
            
            if len(pred_clean) > 0 and len(true_clean) > 0:
                metrics[prop_name] = {
                    'MAE': mean_absolute_error(true_clean, pred_clean),
                    'RMSE': np.sqrt(mean_squared_error(true_clean, pred_clean)),
                    'R2': r2_score(true_clean, pred_clean),
                    'MAPE': np.mean(np.abs((true_clean - pred_clean) / true_clean)) * 100
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
        print(f"Model saved to {path}")
    
    def load_model(self, path='cnn_reservoir_model.pth'):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler = checkpoint['scaler']
        print(f"Model loaded from {path}")

def test_cnn():
    """Test function to verify CNN works correctly"""
    print("Testing CNN model...")
    
    # Create synthetic data matching SPE9 dimensions
    Nx, Ny, Nz = 24, 25, 15
    
    # Grid data - 3D array
    grid_data = np.random.rand(Nx, Ny, Nz) * 100
    
    # Properties - 2D arrays (Ny, Nz)
    properties = {
        'permeability': np.random.lognormal(mean=3.0, sigma=0.5, size=(Ny, Nz)),
        'porosity': np.random.normal(loc=0.2, scale=0.05, size=(Ny, Nz)),
        'saturation': np.random.uniform(0.6, 0.9, size=(Ny, Nz))
    }
    
    print(f"\nData shapes:")
    print(f"Grid data: {grid_data.shape}")
    print(f"Properties: { {k: v.shape for k, v in properties.items()} }")
    
    # Initialize predictor
    predictor = PropertyPredictor()
    
    # Prepare data
    print("\nPreparing data...")
    train_loader, val_loader = predictor.prepare_data(grid_data, properties)
    
    # Train
    print("\nTraining CNN model...")
    train_losses, val_losses = predictor.train(train_loader, val_loader, epochs=5)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = predictor.evaluate(grid_data, properties)
    
    print("\nModel Performance:")
    for prop_name, prop_metrics in metrics.items():
        print(f"\n{prop_name.upper()}:")
        for metric_name, value in prop_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Save model
    predictor.save_model('test_cnn_model.pth')
    
    print("\n✅ CNN test completed successfully!")
    return predictor

if __name__ == "__main__":
    # Run test
    test_cnn()
