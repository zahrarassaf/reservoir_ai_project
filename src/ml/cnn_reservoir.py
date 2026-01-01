"""
CNN for Reservoir Property Prediction - REAL DATA VERSION
Optimized for SPE9 benchmark data - FIXED VERSION
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# Try to import from parent directory
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from main_refactored import ProfessionalSPE9Loader
    HAS_MAIN_MODULE = True
except ImportError:
    HAS_MAIN_MODULE = False
    print("Note: Could not import main_refactored module. Using synthetic data.")

class SPE9ReservoirDataset(Dataset):
    
    def __init__(self, grid_data: np.ndarray, 
                 properties: Dict[str, np.ndarray],
                 patch_size: int = 5,
                 transform=None,
                 augmentation: bool = False):
        """
        Args:
            grid_data: 3D grid data (Nx, Ny, Nz) - SPE9: (24, 25, 15)
            properties: Dictionary of 2D target properties (Ny, Nz)
            patch_size: Size of 3D patch to extract
            transform: Optional transforms
            augmentation: Enable data augmentation
        """
        # Store original data
        self.grid_data = grid_data  # (Nx, Ny, Nz)
        self.properties = properties  # Dict of (Ny, Nz) arrays
        self.patch_size = patch_size
        self.transform = transform
        self.augmentation = augmentation
        
        # Store dimensions
        self.nx, self.ny, self.nz = grid_data.shape
        
        # Generate all possible patch centers
        self.valid_indices = []
        self.property_values = []
        
        half_size = patch_size // 2
        
        print(f"Generating patches from grid: {grid_data.shape}")
        print(f"Property shapes: { {k: v.shape for k, v in properties.items()} }")
        
        # Create sliding window over all positions
        for y in range(self.ny):
            for z in range(self.nz):
                # Check if we can extract a full patch
                y_start = y - half_size
                y_end = y + half_size + 1
                z_start = z - half_size
                z_end = z + half_size + 1
                
                # Check boundaries
                if (y_start >= 0 and y_end <= self.ny and 
                    z_start >= 0 and z_end <= self.nz):
                    self.valid_indices.append((y, z))
                    
                    # Store property values for this location
                    prop_vals = {}
                    for prop_name in properties:
                        prop_vals[prop_name] = properties[prop_name][y, z]
                    self.property_values.append(prop_vals)
        
        print(f"Generated {len(self.valid_indices)} valid patches")
        
        # If not enough patches, allow partial patches with padding
        if len(self.valid_indices) < 100:
            print("Generating additional patches with boundary handling...")
            for y in range(self.ny):
                for z in range(self.nz):
                    if (y, z) not in self.valid_indices:
                        self.valid_indices.append((y, z))
                        prop_vals = {}
                        for prop_name in properties:
                            prop_vals[prop_name] = properties[prop_name][y, z]
                        self.property_values.append(prop_vals)
            print(f"Total patches: {len(self.valid_indices)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        y_center, z_center = self.valid_indices[idx]
        half_size = self.patch_size // 2
        
        # Calculate patch boundaries
        y_start = y_center - half_size
        y_end = y_center + half_size + 1
        z_start = z_center - half_size
        z_end = z_center + half_size + 1
        
        # Initialize patch with zeros
        patch = np.zeros((self.nx, self.patch_size, self.patch_size), 
                        dtype=np.float32)
        
        # Calculate valid region within boundaries
        valid_y_start = max(0, y_start)
        valid_y_end = min(self.ny, y_end)
        valid_z_start = max(0, z_start)
        valid_z_end = min(self.nz, z_end)
        
        # Calculate corresponding positions in patch
        patch_y_start = max(0, -y_start)
        patch_y_end = patch_y_start + (valid_y_end - valid_y_start)
        patch_z_start = max(0, -z_start)
        patch_z_end = patch_z_start + (valid_z_end - valid_z_start)
        
        # Extract valid data
        if valid_y_end > valid_y_start and valid_z_end > valid_z_start:
            patch[:, patch_y_start:patch_y_end, patch_z_start:patch_z_end] = \
                self.grid_data[:, valid_y_start:valid_y_end, valid_z_start:valid_z_end]
        
        # Convert to tensor and add channel dimension
        patch_tensor = torch.FloatTensor(patch).unsqueeze(0)  # (1, Nx, patch_size, patch_size)
        
        # Get property values for this location
        property_tensors = {}
        for prop_name in self.properties:
            property_tensors[prop_name] = torch.FloatTensor(
                [self.property_values[idx][prop_name]]
            )
        
        # Data augmentation (if enabled)
        if self.augmentation and torch.rand(1).item() > 0.5:
            # Random flip
            if torch.rand(1).item() > 0.5:
                patch_tensor = torch.flip(patch_tensor, dims=[2])  # Flip y-axis
            if torch.rand(1).item() > 0.5:
                patch_tensor = torch.flip(patch_tensor, dims=[3])  # Flip z-axis
            
            # Random rotation (0, 90, 180, 270 degrees in y-z plane)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                patch_tensor = torch.rot90(patch_tensor, k, dims=[2, 3])
            
            # Add small noise
            if torch.rand(1).item() > 0.7:
                noise = torch.randn_like(patch_tensor) * 0.01
                patch_tensor = patch_tensor + noise
        
        return patch_tensor, property_tensors

class EnhancedCNNReservoirPredictor(nn.Module):
    
    def __init__(self, input_channels=1, nx=24):
        super().__init__()
        
        # Input shape: (B, 1, 24, patch_size, patch_size)
        
        # Multi-scale feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1))
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((6, 1, 1))  # Fixed size output
        )
        
        # Calculate flattened size
        self.flattened_size = 128 * 6 * 1 * 1
        
        # Separate heads for each property
        self.permeability_head = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self.porosity_head = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self.saturation_head = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (B, 1, nx, patch_size, patch_size)
        
        # Feature extraction
        x1 = self.conv1(x)  # (B, 32, 12, patch_size, patch_size)
        x2 = self.conv2(x1)  # (B, 64, 6, patch_size, patch_size)
        x3 = self.conv3(x2)  # (B, 128, 6, 1, 1)
        
        # Flatten
        x_flat = x3.view(x3.size(0), -1)  # (B, 128*6*1*1)
        
        # Separate predictions
        perm_pred = self.permeability_head(x_flat).squeeze(-1)
        poro_pred = self.porosity_head(x_flat).squeeze(-1)
        sat_pred = self.saturation_head(x_flat).squeeze(-1)
        
        return {
            'permeability': perm_pred,
            'porosity': poro_pred,
            'saturation': sat_pred
        }

class ProfessionalPropertyPredictor:
    
    def __init__(self, device=None, patch_size=5):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.patch_size = patch_size
        self.model = EnhancedCNNReservoirPredictor().to(self.device)
        
        # Use AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.001,
            weight_decay=1e-4
        )
        
        # Huber loss is more robust than MSE
        self.criterion = nn.HuberLoss(delta=1.0)
        
        # Scaler for each property
        self.property_scalers = {}
        self.grid_scaler = StandardScaler()
        
        print(f"Using device: {self.device}")
        print(f"Model architecture: {self.model.__class__.__name__}")
    
    def load_spe9_data(self):
        """Load real SPE9 data from the main project"""
        try:
            if HAS_MAIN_MODULE:
                print("Loading SPE9 data...")
                loader = ProfessionalSPE9Loader("data")
                reservoir_data = loader.load()
                
                # Extract 3D properties
                permeability_3d = reservoir_data['properties']['permeability_3d']
                porosity_3d = reservoir_data['properties']['porosity_3d']
                saturation_3d = reservoir_data['properties']['saturation_3d']
                
                print(f"Loaded 3D properties:")
                print(f"  Permeability: {permeability_3d.shape}, mean={permeability_3d.mean():.2f}")
                print(f"  Porosity: {porosity_3d.shape}, mean={porosity_3d.mean():.3f}")
                print(f"  Saturation: {saturation_3d.shape}, mean={saturation_3d.mean():.3f}")
                
                # Create enhanced grid data by combining properties
                grid_data = np.stack([
                    self._normalize_permeability(permeability_3d),
                    porosity_3d * 100,  # Scale for better numerical stability
                    saturation_3d * 50   # Scale for better numerical stability
                ], axis=0)  # Shape: (3, 24, 25, 15)
                
                # Take weighted mean for input grid
                grid_data_mean = np.average(grid_data, axis=0, weights=[0.5, 0.3, 0.2])
                
                # Create 2D target properties (average over depth/X dimension)
                properties = {
                    'permeability': np.mean(permeability_3d, axis=0),
                    'porosity': np.mean(porosity_3d, axis=0),
                    'saturation': np.mean(saturation_3d, axis=0)
                }
                
                print(f"\nCreated 2D target properties:")
                for prop_name, prop_data in properties.items():
                    print(f"  {prop_name}: shape={prop_data.shape}, "
                          f"mean={prop_data.mean():.4f}, std={prop_data.std():.4f}")
                
                return grid_data_mean, properties
            else:
                raise ImportError("Could not import main_refactored")
            
        except Exception as e:
            print(f"Note: Could not load real SPE9 data: {e}")
            print("Creating synthetic data with SPE9 dimensions...")
            return self._create_synthetic_spe9_data()
    
    def _normalize_permeability(self, perm_data):
        """Apply log transform to permeability"""
        return np.log10(perm_data + 1)  # Add 1 to avoid log(0)
    
    def _create_synthetic_spe9_data(self):
        """Create synthetic data with SPE9-like characteristics"""
        nx, ny, nz = 24, 25, 15
        
        # Create correlated properties
        x, y, z = np.meshgrid(
            np.linspace(0, 1, nx),
            np.linspace(0, 1, ny),
            np.linspace(0, 1, nz),
            indexing='ij'
        )
        
        # Create realistic spatial patterns
        pattern = (np.sin(2*np.pi*x) * np.cos(3*np.pi*y) * np.sin(4*np.pi*z) + 1) / 2
        
        # Permeability (log-normal distribution with spatial correlation)
        permeability_3d = np.exp(3 + 0.5 * pattern + 0.3 * np.random.randn(nx, ny, nz))
        
        # Porosity (correlated with permeability)
        porosity_3d = 0.15 + 0.1 * pattern + 0.02 * np.random.randn(nx, ny, nz)
        porosity_3d = np.clip(porosity_3d, 0.1, 0.35)
        
        # Saturation (inversely related to porosity in some areas)
        saturation_3d = 0.7 + 0.15 * (1 - pattern) + 0.05 * np.random.randn(nx, ny, nz)
        saturation_3d = np.clip(saturation_3d, 0.6, 0.9)
        
        # Create grid data
        grid_data = np.stack([
            self._normalize_permeability(permeability_3d),
            porosity_3d * 100,
            saturation_3d * 50
        ], axis=0)
        grid_data_mean = np.mean(grid_data, axis=0)
        
        # 2D properties
        properties = {
            'permeability': np.mean(permeability_3d, axis=0),
            'porosity': np.mean(porosity_3d, axis=0),
            'saturation': np.mean(saturation_3d, axis=0)
        }
        
        print("Created synthetic SPE9-like data")
        return grid_data_mean, properties
    
    def prepare_data(self, grid_data: np.ndarray, 
                    properties: Dict[str, np.ndarray],
                    train_ratio: float = 0.8,
                    augmentation: bool = True):
        
        print(f"\nPreparing data:")
        print(f"  Grid data shape: {grid_data.shape}")
        print(f"  Properties: {list(properties.keys())}")
        
        # Scale properties
        scaled_properties = {}
        for name, prop in properties.items():
            scaler = StandardScaler()
            prop_flat = prop.reshape(-1, 1)
            prop_scaled = scaler.fit_transform(prop_flat).reshape(prop.shape)
            scaled_properties[name] = prop_scaled
            self.property_scalers[name] = scaler
            print(f"  Scaled {name}: mean={prop_scaled.mean():.4f}, std={prop_scaled.std():.4f}")
        
        # Create dataset with augmentation for training
        full_dataset = SPE9ReservoirDataset(
            grid_data, 
            scaled_properties, 
            patch_size=self.patch_size,
            augmentation=augmentation
        )
        
        # Split indices
        indices = np.arange(len(full_dataset))
        train_idx, val_idx = train_test_split(
            indices, 
            train_size=train_ratio, 
            random_state=42,
            shuffle=True
        )
        
        print(f"  Total samples: {len(full_dataset)}")
        print(f"  Training samples: {len(train_idx)}")
        print(f"  Validation samples: {len(val_idx)}")
        
        # Create samplers
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        # Create dataloaders
        batch_size = 32
        train_loader = DataLoader(
            full_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            num_workers=0
        )
        val_loader = DataLoader(
            full_dataset, 
            batch_size=batch_size, 
            sampler=val_sampler,
            num_workers=0
        )
        
        # Test one batch
        if len(train_loader) > 0:
            test_batch = next(iter(train_loader))
            patches, props = test_batch
            print(f"  Batch patches shape: {patches.shape}")
            print(f"  Batch properties: { {k: v.shape for k, v in props.items()} }")
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, epochs=50, patience=10):
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Learning rate scheduler - REMOVED verbose parameter
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for patches, properties in train_loader:
                patches = patches.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(patches)
                
                # Calculate weighted loss
                loss = 0
                weights = {'permeability': 0.4, 'porosity': 0.3, 'saturation': 0.3}
                
                for prop_name in properties:
                    prop_tensor = properties[prop_name].to(self.device)
                    pred_tensor = outputs[prop_name]
                    prop_loss = self.criterion(pred_tensor, prop_tensor)
                    loss += weights.get(prop_name, 1.0) * prop_loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
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
                    val_batches += 1
            
            # Average losses
            avg_train_loss = train_loss / max(train_batches, 1)
            avg_val_loss = val_loss / max(val_batches, 1)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.save_model('best_cnn_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1:3d}/{epochs} | '
                      f'Train Loss: {avg_train_loss:.6f} | '
                      f'Val Loss: {avg_val_loss:.6f} | '
                      f'LR: {current_lr:.6f} | '
                      f'Patience: {patience_counter}/{patience}')
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.load_model('best_cnn_model.pth')
        
        print(f"\nTraining completed. Best validation loss: {best_val_loss:.6f}")
        
        # Plot training history
        self._plot_training_history(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def _plot_training_history(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('cnn_training_history.png', dpi=150)
        plt.show()
    
    def predict(self, grid_data: np.ndarray):
        self.model.eval()
        
        nx, ny, nz = grid_data.shape
        
        # Create dataset for prediction
        dummy_props = {
            'permeability': np.zeros((ny, nz)),
            'porosity': np.zeros((ny, nz)),
            'saturation': np.zeros((ny, nz))
        }
        
        dataset = SPE9ReservoirDataset(
            grid_data, 
            dummy_props, 
            patch_size=self.patch_size,
            augmentation=False
        )
        
        # Initialize prediction arrays
        predictions = {
            'permeability': np.zeros((ny, nz)),
            'porosity': np.zeros((ny, nz)),
            'saturation': np.zeros((ny, nz))
        }
        counts = np.zeros((ny, nz))
        
        print(f"\nMaking predictions for {len(dataset)} patches...")
        
        with torch.no_grad():
            for i in range(len(dataset)):
                patch, _ = dataset[i]
                patch = patch.unsqueeze(0).to(self.device)
                
                # Get prediction
                outputs = self.model(patch)
                
                # Get coordinates
                y_idx, z_idx = dataset.valid_indices[i]
                
                # Store predictions (inverse transform)
                for prop_name in predictions:
                    pred_scaled = outputs[prop_name].cpu().numpy()[0]
                    
                    # Inverse scaling
                    if prop_name in self.property_scalers:
                        pred_original = self.property_scalers[prop_name].inverse_transform(
                            np.array([[pred_scaled]])
                        )[0, 0]
                    else:
                        pred_original = pred_scaled
                    
                    predictions[prop_name][y_idx, z_idx] += pred_original
                    counts[y_idx, z_idx] += 1
        
        # Average predictions for cells with multiple patches
        for prop_name in predictions:
            mask = counts > 0
            predictions[prop_name][mask] /= counts[mask]
        
        # Fill missing values with nearest neighbor
        for prop_name in predictions:
            pred_array = predictions[prop_name]
            missing_mask = counts == 0
            
            if np.any(missing_mask):
                from scipy import ndimage
                # Use nearest neighbor interpolation for missing values
                distances, indices = ndimage.distance_transform_edt(
                    missing_mask, return_distances=False, return_indices=True
                )
                predictions[prop_name] = pred_array[tuple(indices)]
        
        return predictions
    
    def evaluate(self, grid_data, true_properties):
        predictions = self.predict(grid_data)
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics = {}
        
        print("\nEvaluation Results:")
        print("-" * 60)
        
        for prop_name in true_properties:
            pred = predictions[prop_name].flatten()
            true = true_properties[prop_name].flatten()
            
            # Remove NaN values
            mask = ~np.isnan(pred) & ~np.isnan(true)
            pred_clean = pred[mask]
            true_clean = true[mask]
            
            if len(pred_clean) > 10:  # Need enough samples
                mae = mean_absolute_error(true_clean, pred_clean)
                rmse = np.sqrt(mean_squared_error(true_clean, pred_clean))
                r2 = r2_score(true_clean, pred_clean)
                
                # Calculate relative errors
                mape = np.mean(np.abs((true_clean - pred_clean) / np.maximum(true_clean, 1e-10))) * 100
                
                metrics[prop_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'MAPE': mape,
                    'Mean_True': true_clean.mean(),
                    'Mean_Pred': pred_clean.mean()
                }
                
                print(f"\n{prop_name.upper():<15}")
                print(f"  MAE:      {mae:.4f}")
                print(f"  RMSE:     {rmse:.4f}")
                print(f"  R²:       {r2:.4f}")
                print(f"  MAPE:     {mape:.2f}%")
                print(f"  Mean True: {true_clean.mean():.4f}")
                print(f"  Mean Pred: {pred_clean.mean():.4f}")
            else:
                metrics[prop_name] = {
                    'MAE': 0, 'RMSE': 0, 'R2': 0, 'MAPE': 0,
                    'Mean_True': 0, 'Mean_Pred': 0
                }
                print(f"\n{prop_name.upper():<15} - Insufficient valid data")
        
        return metrics
    
    def visualize_results(self, grid_data, true_properties, save_prefix='cnn_results'):
        predictions = self.predict(grid_data)
        
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        properties = ['permeability', 'porosity', 'saturation']
        
        for i, prop_name in enumerate(properties):
            true_data = true_properties[prop_name]
            pred_data = predictions[prop_name]
            
            # Row 0: True values
            im1 = axes[i, 0].imshow(true_data, cmap='viridis', aspect='auto')
            axes[i, 0].set_title(f'True {prop_name.capitalize()}')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Row 1: Predicted values
            im2 = axes[i, 1].imshow(pred_data, cmap='viridis', aspect='auto')
            axes[i, 1].set_title(f'Predicted {prop_name.capitalize()}')
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Row 2: Absolute difference
            diff = pred_data - true_data
            im3 = axes[i, 2].imshow(diff, cmap='RdBu', aspect='auto',
                                   vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
            axes[i, 2].set_title(f'Difference ({prop_name})')
            plt.colorbar(im3, ax=axes[i, 2])
            
            # Row 3: Scatter plot
            axes[i, 3].scatter(true_data.flatten(), pred_data.flatten(), 
                              alpha=0.5, s=10)
            axes[i, 3].plot([true_data.min(), true_data.max()], 
                           [true_data.min(), true_data.max()], 
                           'r--', linewidth=1)
            axes[i, 3].set_xlabel('True')
            axes[i, 3].set_ylabel('Predicted')
            axes[i, 3].set_title(f'True vs Predicted ({prop_name})')
            axes[i, 3].grid(True, alpha=0.3)
        
        plt.suptitle('CNN Reservoir Property Predictions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save predictions to file
        np.savez(f'{save_prefix}_predictions.npz', 
                 **{f'{k}_pred': v for k, v in predictions.items()},
                 **{f'{k}_true': v for k, v in true_properties.items()})
        
        print(f"\nResults saved to {save_prefix}_visualization.png and {save_prefix}_predictions.npz")
    
    def save_model(self, path='cnn_reservoir_model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'property_scalers': self.property_scalers,
            'patch_size': self.patch_size,
            'model_config': self.model.__class__.__name__
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='cnn_reservoir_model.pth'):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.property_scalers = checkpoint['property_scalers']
        self.patch_size = checkpoint.get('patch_size', 5)
        print(f"Model loaded from {path}")

def run_professional_cnn_training():
    """Main function to run professional CNN training"""
    print("=" * 70)
    print("PROFESSIONAL CNN RESERVOIR PREDICTION SYSTEM")
    print("=" * 70)
    
    try:
        # Initialize predictor
        predictor = ProfessionalPropertyPredictor(patch_size=5)
        
        # Load real SPE9 data
        grid_data, properties = predictor.load_spe9_data()
        
        # Prepare data
        train_loader, val_loader = predictor.prepare_data(
            grid_data, 
            properties, 
            train_ratio=0.8,
            augmentation=True
        )
        
        # Train model
        train_losses, val_losses = predictor.train(
            train_loader, 
            val_loader, 
            epochs=30,  # Reduced for faster testing
            patience=10
        )
        
        # Evaluate on training data
        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)
        
        metrics = predictor.evaluate(grid_data, properties)
        
        # Visualize results
        print("\n" + "=" * 70)
        print("VISUALIZING RESULTS")
        print("=" * 70)
        
        predictor.visualize_results(grid_data, properties, save_prefix='spe9_cnn')
        
        # Save final model
        predictor.save_model('spe9_cnn_final_model.pth')
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return predictor
        
    except Exception as e:
        print(f"\nError in CNN training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run professional training
    predictor = run_professional_cnn_training()
    
    if predictor:
        print("\n✅ CNN model trained and ready for reservoir property prediction!")
        print(f"\nTo use the trained model:")
        print("1. Load model: predictor.load_model('spe9_cnn_final_model.pth')")
        print("2. Make predictions: predictions = predictor.predict(grid_data)")
        print("3. Evaluate: metrics = predictor.evaluate(grid_data, true_properties)")
