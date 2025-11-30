import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import json

class EnsembleTrainer:
    def __init__(self, model: nn.Module, config):
        self.model = model
        self.config = config
        self.optimizers = []
        self.schedulers = []
        
        for model in self.model.models:
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
            
    def train_ensemble(self, features, epochs: int, batch_size: int):
        print("Starting ensemble training...")
        
        train_loader = self._prepare_data_loader(features, batch_size)
        
        training_history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(train_loader)
            
            for scheduler in self.schedulers:
                scheduler.step(val_loss)
            
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        return training_history
    
    def _train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(data_loader):
            batch_loss = 0.0
            
            for i, (model, optimizer) in enumerate(zip(self.model.models, self.optimizers)):
                optimizer.zero_grad()
                
                pred = model(x)
                loss = nn.MSELoss()(pred, y)
                
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
            
            total_loss += batch_loss / len(self.model.models)
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in data_loader:
                mean_pred, _ = self.model(x)
                loss = nn.MSELoss()(mean_pred, y)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _prepare_data_loader(self, features, batch_size: int):
        x_data = torch.randn(1000, len(self.config.input_features))
        y_data = torch.randn(1000, len(self.config.output_features))
        
        dataset = torch.utils.data.TensorDataset(x_data, y_data)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        return data_loader
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss
        }
        
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(checkpoint, f"checkpoints/best_model_epoch_{epoch}.pth")
    
    def save_results(self, output_dir: str, results):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / "training_history.json", "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in results.items()}, f, indent=2)
        
        torch.save(self.model.state_dict(), output_path / "final_model.pth")
        
        print(f"Results saved to {output_dir}")
