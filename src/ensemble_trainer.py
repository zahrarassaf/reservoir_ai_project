import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import json

class AdvancedEnsembleTrainer:
    """Advanced trainer for ensemble models"""
    
    def __init__(self, model: nn.Module, config: EnsembleModelConfig):
        self.model = model
        self.config = config
        self.optimizers = []
        self.schedulers = []
        
        # Create optimizers for each model in ensemble
        for i, model in enumerate(self.model.models):
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            self.optimizers.append(optimizer)
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=config.patience // 2, factor=0.5
            )
            self.schedulers.append(scheduler)
            
    def train_ensemble(self, features: Dict[str, torch.Tensor], 
                      epochs: int, batch_size: int) -> Dict[str, List[float]]:
        """Train the ensemble model"""
        print("🎯 Starting ensemble training...")
        
        # Prepare data
        train_loader = self._prepare_data_loader(features, batch_size)
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'diversity_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            train_loss, diversity_loss = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss = self._validate_epoch(train_loader)  # Using same data for simplicity
            
            # Update learning rates
            for scheduler in self.schedulers:
                scheduler.step(val_loss)
            
            # Record history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['diversity_loss'].append(diversity_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self._save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
            
            if epoch % 100 == 0:
                print(f"📊 Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        return training_history
    
    def _train_epoch(self, data_loader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_diversity = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(data_loader):
            batch_loss = 0.0
            
            # Train each model in ensemble
            for i, (model, optimizer) in enumerate(zip(self.model.models, self.optimizers)):
                optimizer.zero_grad()
                
                # Forward pass
                pred = model(x)
                
                # Calculate loss
                loss = nn.MSELoss()(pred, y)
                
                # Add diversity regularization
                if self.config.diversity_regularization > 0:
                    diversity_loss = self.model.ensemble_diversity_loss(x)
                    loss += self.config.diversity_regularization * diversity_loss
                    total_diversity += diversity_loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
            
            total_loss += batch_loss / len(self.model.models)
            num_batches += 1
        
        return total_loss / num_batches, total_diversity / num_batches
    
    def _validate_epoch(self, data_loader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in data_loader:
                # Get ensemble prediction
                mean_pred, _ = self.model(x)
                loss = nn.MSELoss()(mean_pred, y)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _prepare_data_loader(self, features: Dict[str, torch.Tensor], batch_size: int):
        """Prepare data loader for training"""
        # Extract features and targets
        static_features = features['static_features']
        dynamic_features = features['production_data']['FOPR']  # Using FOPR as target for simplicity
        
        # Create dataset (simplified - in practice you'd have proper targets)
        # For demonstration, using static features to predict FOPR
        x_data = static_features
        y_data = dynamic_features.unsqueeze(1).expand(-1, x_data.size(1))  # Match dimensions
        
        dataset = torch.utils.data.TensorDataset(x_data, y_data)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        return data_loader
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dicts': [opt.state_dict() for opt in self.optimizers],
            'val_loss': val_loss,
            'config': self.config
        }
        
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(checkpoint, f"checkpoints/best_model_epoch_{epoch}.pth")
    
    def save_results(self, output_dir: str, results: Dict[str, List[float]]):
        """Save training results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save training history
        with open(output_path / "training_history.json", "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in results.items()}, f, indent=2)
        
        # Save model
        torch.save(self.model.state_dict(), output_path / "final_model.pth")
        
        print(f"💾 Results saved to {output_dir}")
