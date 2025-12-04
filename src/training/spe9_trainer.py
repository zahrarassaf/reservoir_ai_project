# src/training/spe9_trainer.py
"""
Complete training pipeline for SPE9 reservoir neural operator.
Includes distributed training, checkpointing, and advanced logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import yaml
import json
import warnings
import time
from tqdm import tqdm

from src.models.reservoir_neural_operator import ReservoirNeuralOperator
from src.data.eclipse_parser import EclipseParser
from src.evaluation.physics_metrics import PhysicsMetrics


class SPE9Dataset(Dataset):
    """Dataset for SPE9 time-series simulation data."""
    
    def __init__(self, 
                 parsed_data: Dict,
                 sequence_length: int = 10,
                 prediction_horizon: int = 1,
                 stride: int = 1,
                 normalize: bool = True):
        """
        Initialize SPE9 dataset from parsed data.
        
        Args:
            parsed_data: Parsed SPE9 data from EclipseParser
            sequence_length: Number of time steps in input sequence
            prediction_horizon: Number of steps to predict ahead
            stride: Stride between sequences
            normalize: Whether to normalize data
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.normalize = normalize
        
        # Extract data
        self._prepare_data(parsed_data)
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        print(f"📊 Dataset created: {len(self.sequences)} sequences, "
              f"input shape: {self.sequences[0][0].shape}")
    
    def _prepare_data(self, parsed_data: Dict):
        """Prepare data from parsed SPE9 results."""
        # Get grid dimensions
        grid_dims = parsed_data['grid'].get('dims', (24, 25, 15))
        self.nx, self.ny, self.nz = grid_dims
        self.n_cells = self.nx * self.ny * self.nz
        
        # Get permeability and porosity
        properties = parsed_data['properties']
        
        # Use synthetic data if real data not available
        if 'results' in parsed_data and 'pressures' in parsed_data['results']:
            pressures = parsed_data['results']['pressures']
            saturations = parsed_data['results']['saturations']
        else:
            # Generate synthetic time series
            n_timesteps = 120
            pressures = self._generate_synthetic_field(n_timesteps, 5000, 100)
            saturations = self._generate_synthetic_field(n_timesteps, 0.2, 0.05)
            saturations = np.clip(saturations, 0.0, 1.0)
        
        # Get permeability field
        if 'PERMX' in properties:
            permeability = properties['PERMX'].reshape(grid_dims)
        else:
            # Generate synthetic permeability
            permeability = np.random.lognormal(mean=1.0, sigma=1.5, size=grid_dims)
        
        # Get porosity field
        if 'PORO' in properties:
            porosity = properties['PORO'].reshape(grid_dims)
        else:
            # Generate synthetic porosity
            porosity = np.random.normal(loc=0.2, scale=0.05, size=grid_dims)
            porosity = np.clip(porosity, 0.05, 0.35)
        
        # Store data
        self.pressures = pressures.astype(np.float32)
        self.saturations = saturations.astype(np.float32)
        self.permeability = permeability.astype(np.float32)
        self.porosity = porosity.astype(np.float32)
        self.n_timesteps = len(pressures)
        
        # Normalize if requested
        if self.normalize:
            self._normalize_data()
    
    def _generate_synthetic_field(self, n_timesteps: int, 
                                 base_value: float, 
                                 variability: float) -> np.ndarray:
        """Generate synthetic field time series."""
        field = np.zeros((n_timesteps, self.n_cells))
        
        for t in range(n_timesteps):
            # Time trend
            time_factor = 1.0 - 0.0005 * t
            
            # Spatial variation
            spatial_var = np.random.normal(0, variability, self.n_cells)
            
            field[t] = base_value * time_factor + spatial_var
        
        return field
    
    def _normalize_data(self):
        """Normalize data."""
        # Pressure normalization
        self.pressure_mean = self.pressures.mean()
        self.pressure_std = self.pressures.std() + 1e-10
        self.pressures = (self.pressures - self.pressure_mean) / self.pressure_std
        
        # Saturation normalization (already in [0,1])
        self.saturation_mean = self.saturations.mean()
        self.saturation_std = self.saturations.std() + 1e-10
        self.saturations = (self.saturations - self.saturation_mean) / self.saturation_std
        
        # Permeability normalization (log-normal)
        self.permeability = np.log1p(self.permeability)
        self.perm_mean = self.permeability.mean()
        self.perm_std = self.permeability.std() + 1e-10
        self.permeability = (self.permeability - self.perm_mean) / self.perm_std
        
        # Porosity normalization
        self.porosity_mean = self.porosity.mean()
        self.porosity_std = self.porosity.std() + 1e-10
        self.porosity = (self.porosity - self.porosity_mean) / self.porosity_std
        
        self.normalization_stats = {
            'pressure': (self.pressure_mean, self.pressure_std),
            'saturation': (self.saturation_mean, self.saturation_std),
            'permeability': (self.perm_mean, self.perm_std),
            'porosity': (self.porosity_mean, self.porosity_std)
        }
    
    def _create_sequences(self) -> List[Tuple]:
        """Create input-output sequences."""
        sequences = []
        
        n_possible = (self.n_timesteps - self.sequence_length - 
                     self.prediction_horizon) // self.stride + 1
        
        for i in range(0, n_possible * self.stride, self.stride):
            # Input: sequence of pressure and saturation
            start_idx = i
            end_idx = i + self.sequence_length
            
            # Get input sequence
            input_pressures = self.pressures[start_idx:end_idx]
            input_saturations = self.saturations[start_idx:end_idx]
            
            # Stack along channel dimension
            # Shape: [sequence_length, 2, n_cells]
            input_seq = np.stack([input_pressures, input_saturations], axis=1)
            
            # Reshape to spatial
            input_seq = input_seq.reshape(
                self.sequence_length, 2, self.nx, self.ny, self.nz
            )
            
            # Output: pressure and saturation at prediction horizon
            target_idx = i + self.sequence_length + self.prediction_horizon - 1
            if target_idx >= self.n_timesteps:
                continue
            
            target_pressure = self.pressures[target_idx]
            target_saturation = self.saturations[target_idx]
            
            # Stack targets
            # Shape: [2, n_cells]
            target = np.stack([target_pressure, target_saturation], axis=0)
            
            # Reshape to spatial
            target = target.reshape(2, self.nx, self.ny, self.nz)
            
            # Permeability and porosity (constant in time)
            permeability = self.permeability
            porosity = self.porosity
            
            sequences.append((input_seq, target, permeability, porosity))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get a sequence sample."""
        input_seq, target, permeability, porosity = self.sequences[idx]
        
        # Convert to torch tensors
        input_tensor = torch.FloatTensor(input_seq)
        target_tensor = torch.FloatTensor(target)
        perm_tensor = torch.FloatTensor(permeability)
        porosity_tensor = torch.FloatTensor(porosity)
        
        return input_tensor, target_tensor, perm_tensor, porosity_tensor


class SPE9Trainer:
    """Trainer for Reservoir Neural Operator on SPE9 data."""
    
    def __init__(self,
                 config: Dict,
                 parsed_data: Dict,
                 output_dir: str = "experiments/spe9_training"):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            parsed_data: Parsed SPE9 data
            output_dir: Output directory for results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"🚀 Using device: {self.device}")
        
        # Setup distributed training if requested
        self.distributed = config.get('distributed', False)
        if self.distributed:
            self._setup_distributed()
        
        # Prepare data
        self._prepare_data(parsed_data)
        
        # Initialize model
        self._init_model()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Setup logging
        self._setup_logging()
        
        # Physics metrics
        self.physics_metrics = PhysicsMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'physics_loss': [],
            'learning_rate': []
        }
    
    def _setup_distributed(self):
        """Setup distributed training."""
        dist.init_process_group(backend='nccl')
        self.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
    
    def _prepare_data(self, parsed_data: Dict):
        """Prepare training and validation data."""
        # Create dataset
        dataset_config = self.config['dataset']
        self.dataset = SPE9Dataset(
            parsed_data=parsed_data,
            sequence_length=dataset_config.get('sequence_length', 10),
            prediction_horizon=dataset_config.get('prediction_horizon', 1),
            stride=dataset_config.get('stride', 1),
            normalize=dataset_config.get('normalize', True)
        )
        
        # Split data
        n_samples = len(self.dataset)
        train_ratio = dataset_config.get('train_ratio', 0.7)
        val_ratio = dataset_config.get('val_ratio', 0.15)
        
        train_size = int(train_ratio * n_samples)
        val_size = int(val_ratio * n_samples)
        test_size = n_samples - train_size - val_size
        
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subsets
        train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
        val_dataset = torch.utils.data.Subset(self.dataset, val_indices)
        test_dataset = torch.utils.data.Subset(self.dataset, test_indices)
        
        # Create data loaders
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training'].get('num_workers', 4)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"📊 Data split: Train={len(train_dataset)}, "
              f"Val={len(val_dataset)}, Test={len(test_dataset)}")
        print(f"📊 Batch size: {batch_size}, "
              f"Total batches: Train={len(self.train_loader)}")
    
    def _init_model(self):
        """Initialize the model."""
        model_config = self.config['model']
        
        # Get grid dimensions
        grid_dims = self.dataset.nx, self.dataset.ny, self.dataset.nz
        
        # Input channels: pressure + saturation over sequence
        input_channels = 2  # pressure and saturation
        
        # Output channels: pressure + saturation at next time
        output_channels = 2
        
        # Create model
        self.model = ReservoirNeuralOperator(
            input_channels=input_channels,
            output_channels=output_channels,
            grid_dims=grid_dims,
            hidden_channels=model_config.get('hidden_channels', 64),
            num_fno_layers=model_config.get('num_fno_layers', 4),
            fno_modes=model_config.get('fno_modes', 12),
            physics_weight=model_config.get('physics_weight', 0.1),
            use_attention=model_config.get('use_attention', True)
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Wrap with DDP if distributed
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"🧠 Model initialized:")
        print(f"   Architecture: Reservoir Neural Operator")
        print(f"   Grid dimensions: {grid_dims}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Input shape: [batch, {input_channels}, {grid_dims[0]}, {grid_dims[1]}, {grid_dims[2]}]")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        training_config = self.config['training']
        
        # Optimizer
        optimizer_name = training_config.get('optimizer', 'adamw')
        learning_rate = training_config.get('learning_rate', 1e-3)
        weight_decay = training_config.get('weight_decay', 1e-4)
        
        if optimizer_name.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Scheduler
        scheduler_config = training_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.get('epochs', 100),
                eta_min=1e-6
            )
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=scheduler_config.get('patience', 10),
                verbose=True
            )
        elif scheduler_type == 'onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=learning_rate,
                epochs=training_config.get('epochs', 100),
                steps_per_epoch=len(self.train_loader)
            )
        else:
            self.scheduler = None
        
        print(f"⚙️ Optimizer: {optimizer_name}, LR: {learning_rate:.1e}")
        print(f"⚙️ Scheduler: {scheduler_type if self.scheduler else 'None'}")
    
    def _setup_logging(self):
        """Setup logging and visualization."""
        # TensorBoard
        log_dir = self.output_dir / 'tensorboard'
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Checkpoint directory
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Results directory
        self.results_dir = self.output_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Log file
        self.log_file = self.output_dir / 'training_log.txt'
        
        print(f"📝 Logging to: {self.output_dir}")
        print(f"📊 TensorBoard: tensorboard --logdir={log_dir}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_physics_loss = 0.0
        epoch_data_loss = 0.0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch:03d} [Train]')
        
        for batch_idx, (inputs, targets, permeability, porosity) in enumerate(pbar):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            permeability = permeability.to(self.device)
            porosity = porosity.to(self.device)
            
            # Add channel dimension if needed
            if permeability.dim() == 4:
                permeability = permeability.unsqueeze(1)
            if porosity.dim() == 4:
                porosity = porosity.unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Last time step as input for prediction
            # Here we predict next time step from sequence
            predictions, physics_diagnostics = self.model(
                inputs[:, -1],  # Use last time step
                permeability,
                porosity,
                return_physics=True
            )
            
            # Compute losses
            data_loss = F.mse_loss(predictions, targets)
            
            # Physics loss
            physics_loss = 0.0
            if physics_diagnostics:
                for loss_name, loss_value in physics_diagnostics.items():
                    if 'loss' in loss_name.lower():
                        physics_loss += loss_value
            
            # Total loss
            total_loss = data_loss + self.config['model'].get('physics_weight', 0.1) * physics_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training'].get('grad_clip', 1.0)
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            epoch_data_loss += data_loss.item()
            epoch_physics_loss += physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'data': f'{data_loss.item():.4f}',
                'physics': f'{physics_loss:.4f}' if isinstance(physics_loss, float) else f'{physics_loss.item():.4f}'
            })
            
            # Log batch metrics to TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/batch_loss', total_loss.item(), global_step)
            self.writer.add_scalar('train/batch_data_loss', data_loss.item(), global_step)
            self.writer.add_scalar('train/batch_physics_loss', physics_loss, global_step)
            
            # Learning rate logging
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/learning_rate', current_lr, global_step)
        
        # Compute epoch averages
        avg_loss = epoch_loss / len(self.train_loader)
        avg_data_loss = epoch_data_loss / len(self.train_loader)
        avg_physics_loss = epoch_physics_loss / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'data_loss': avg_data_loss,
            'physics_loss': avg_physics_loss
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        val_loss = 0.0
        val_physics_loss = 0.0
        
        # Physics metrics
        physics_scores = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch:03d} [Val]')
            
            for inputs, targets, permeability, porosity in pbar:
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                permeability = permeability.to(self.device)
                porosity = porosity.to(self.device)
                
                # Add channel dimension
                if permeability.dim() == 4:
                    permeability = permeability.unsqueeze(1)
                if porosity.dim() == 4:
                    porosity = porosity.unsqueeze(1)
                
                # Forward pass
                predictions, physics_diagnostics = self.model(
                    inputs[:, -1],
                    permeability,
                    porosity,
                    return_physics=True
                )
                
                # Compute losses
                data_loss = F.mse_loss(predictions, targets)
                
                # Physics loss
                physics_loss = 0.0
                if physics_diagnostics:
                    for loss_name, loss_value in physics_diagnostics.items():
                        if 'loss' in loss_name.lower():
                            physics_loss += loss_value
                
                # Total loss
                total_loss = data_loss + self.config['model'].get('physics_weight', 0.1) * physics_loss
                
                # Update metrics
                val_loss += total_loss.item()
                val_physics_loss += physics_loss
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'physics': f'{physics_loss:.4f}'
                })
        
        # Compute averages
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_physics = val_physics_loss / len(self.val_loader)
        
        # Log to TensorBoard
        self.writer.add_scalar('val/loss', avg_val_loss, epoch)
        self.writer.add_scalar('val/physics_loss', avg_val_physics, epoch)
        
        return {
            'loss': avg_val_loss,
            'physics_loss': avg_val_physics
        }
    
    def test(self) -> Dict[str, Any]:
        """
        Test model on test set.
        
        Returns:
            Dictionary with test results
        """
        self.model.eval()
        
        test_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets, permeability, porosity in tqdm(self.test_loader, desc='Testing'):
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                permeability = permeability.to(self.device)
                porosity = porosity.to(self.device)
                
                # Add channel dimension
                if permeability.dim() == 4:
                    permeability = permeability.unsqueeze(1)
                if porosity.dim() == 4:
                    porosity = porosity.unsqueeze(1)
                
                # Forward pass
                predictions, _ = self.model(
                    inputs[:, -1],
                    permeability,
                    porosity,
                    return_physics=False
                )
                
                # Compute loss
                loss = F.mse_loss(predictions, targets)
                test_loss += loss.item()
                
                # Store for analysis
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Combine all predictions and targets
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute additional metrics
        mae = np.mean(np.abs(all_predictions - all_targets))
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        r2 = 1 - np.sum((all_predictions - all_targets) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)
        
        avg_test_loss = test_loss / len(self.test_loader)
        
        results = {
            'test_loss': avg_test_loss,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        print(f"\n🧪 Test Results:")
        print(f"   Loss: {avg_test_loss:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   R²: {r2:.6f}")
        
        return results
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'train_history': self.train_history,
            'config': self.config
        }
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model (loss: {val_loss:.6f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        self.train_history = checkpoint['train_history']
        
        print(f"📂 Loaded checkpoint from epoch {self.current_epoch}, "
              f"val loss: {self.best_val_loss:.6f}")
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        
        training_config = self.config['training']
        epochs = training_config.get('epochs', 100)
        patience = training_config.get('patience', 20)
        
        no_improve = 0
        
        for epoch in range(self.current_epoch, epochs):
            print(f"\n📈 Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['physics_loss'].append(train_metrics['physics_loss'])
            
            # Validate
            val_metrics = self.validate(epoch)
            self.train_history['val_loss'].append(val_metrics['loss'])
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_history['learning_rate'].append(current_lr)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Print epoch summary
            print(f"   Train Loss: {train_metrics['loss']:.6f} "
                  f"(Data: {train_metrics['data_loss']:.6f}, "
                  f"Physics: {train_metrics['physics_loss']:.6f})")
            print(f"   Val Loss: {val_metrics['loss']:.6f}")
            print(f"   Learning Rate: {current_lr:.6f}")
            
            # Log to TensorBoard
            self.writer.add_scalar('train/epoch_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('train/epoch_physics_loss', train_metrics['physics_loss'], epoch)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                no_improve = 0
            else:
                no_improve += 1
            
            if epoch % training_config.get('save_every', 10) == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics['loss'], is_best)
            
            # Early stopping
            if no_improve >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch + 1}")
                break
        
        # Final test
        print("\n" + "="*60)
        print("🧪 FINAL TESTING")
        print("="*60)
        
        test_results = self.test()
        
        # Save final results
        self.save_final_results(test_results)
        
        # Close TensorBoard writer
        self.writer.close()
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print("="*60)
        
        return test_results
    
    def save_final_results(self, test_results: Dict[str, Any]):
        """Save final results and plots."""
        # Save test results
        results_path = self.results_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON
            json_results = test_results.copy()
            if 'predictions' in json_results:
                json_results['predictions'] = json_results['predictions'].tolist()
            if 'targets' in json_results:
                json_results['targets'] = json_results['targets'].tolist()
            
            json.dump(json_results, f, indent=2)
        
        # Save training history
        history_path = self.results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        # Save configuration
        config_path = self.results_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Create plots
        self._create_plots()
        
        print(f"💾 Results saved to: {self.results_dir}")
    
    def _create_plots(self):
        """Create training plots."""
        import matplotlib.pyplot as plt
        
        # Loss curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.train_history['train_loss'], label='Train Loss')
        plt.plot(self.train_history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(self.train_history['physics_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Physics Loss')
        plt.title('Physics Constraint Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(self.train_history['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plot_path = self.results_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved training plots to: {plot_path}")


def run_spe9_training(data_path: str, config_path: Optional[str] = None):
    """
    Main function to run SPE9 training.
    
    Args:
        data_path: Path to SPE9 data directory
        config_path: Path to configuration file
        
    Returns:
        Training results
    """
    print("="*60)
    print("🏭 RESERVOIR NEURAL OPERATOR - SPE9 TRAINING")
    print("="*60)
    
    # Load configuration
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'dataset': {
                'sequence_length': 10,
                'prediction_horizon': 1,
                'stride': 1,
                'normalize': True,
                'train_ratio': 0.7,
                'val_ratio': 0.15
            },
            'model': {
                'hidden_channels': 64,
                'num_fno_layers': 4,
                'fno_modes': 12,
                'physics_weight': 0.1,
                'use_attention': True
            },
            'training': {
                'epochs': 100,
                'batch_size': 4,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'optimizer': 'adamw',
                'scheduler': {'type': 'cosine'},
                'grad_clip': 1.0,
                'patience': 20,
                'save_every': 10,
                'num_workers': 4,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'experiment': {
                'name': f'spe9_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'output_dir': 'experiments'
            }
        }
    
    # Parse SPE9 data
    print("\n📂 Parsing SPE9 data...")
    parser = EclipseParser(data_path)
    parsed_data = parser.get_full_dataset()
    
    # Create trainer
    output_dir = Path(config['experiment']['output_dir']) / config['experiment']['name']
    trainer = SPE9Trainer(config, parsed_data, output_dir)
    
    # Train
    results = trainer.train()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Reservoir Neural Operator on SPE9')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to SPE9 data directory')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Run training
    results = run_spe9_training(args.data_path, args.config)
