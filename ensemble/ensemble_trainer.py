import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np
from tqdm import tqdm

from .base_ensemble import BaseEnsemble
from core.temporal_models import TemporalModel
from config.model_config import TemporalModelConfig, EnsembleConfig
from config.training_config import TrainingConfig

class EnsembleTrainer(BaseEnsemble):
    def __init__(self, ensemble_config: EnsembleConfig, model_config: TemporalModelConfig, training_config: TrainingConfig):
        self.ensemble_config = ensemble_config
        self.model_config = model_config
        self.training_config = training_config
        self.logger = logging.getLogger(__name__)
        
        super().__init__(ensemble_config)
        self.optimizers = []
        self.schedulers = []
    
    def _initialize_ensemble(self):
        self.models = []
        self.optimizers = []
        self.schedulers = []
        
        for i in range(self.ensemble_config.num_models):
            model = TemporalModel(self.model_config)
            
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
            
            if self.training_config.use_scheduler:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    patience=self.training_config.patience,
                    factor=self.training_config.factor
                )
            else:
                scheduler = None
            
            self.models.append(model)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
        
        self.logger.info(f"Initialized ensemble with {len(self.models)} models")
        self.logger.info(f"Total parameters per model: {self.models[0].count_parameters():,}")
    
    def train_ensemble(self, train_loader, val_loader: Optional = None) -> Dict[str, List[float]]:
        self.logger.info(f"Training ensemble of {len(self.models)} models")
        
        training_history = {
            'train_loss': [[] for _ in range(len(self.models))],
            'val_loss': [[] for _ in range(len(self.models))] if val_loader else None
        }
        
        for epoch in range(self.training_config.num_epochs):
            epoch_train_losses = self._train_epoch(epoch, train_loader)
            
            for i, loss in enumerate(epoch_train_losses):
                training_history['train_loss'][i].append(loss)
            
            if val_loader is not None:
                epoch_val_losses = self._validate_ensemble(val_loader)
                for i, loss in enumerate(epoch_val_losses):
                    training_history['val_loss'][i].append(loss)
                
                if epoch % 10 == 0:
                    self.logger.info(
                        f"Epoch {epoch:3d} | "
                        f"Train Loss: {np.mean(epoch_train_losses):.4f} | "
                        f"Val Loss: {np.mean(epoch_val_losses):.4f} | "
                        f"Diversity: {self.compute_diversity():.4f}"
                    )
            else:
                if epoch % 10 == 0:
                    self.logger.info(
                        f"Epoch {epoch:3d} | "
                        f"Train Loss: {np.mean(epoch_train_losses):.4f} | "
                        f"Diversity: {self.compute_diversity():.4f}"
                    )
        
        self.logger.info("Ensemble training completed")
        return training_history
    
    def _train_epoch(self, epoch: int, train_loader) -> List[float]:
        epoch_losses = []
        
        for model_idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            model.train()
            model_losses = []
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.training_config.device)
                target = target.to(self.training_config.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = nn.MSELoss()(output, target)
                
                if self.ensemble_config.diversity_weight > 0 and len(self.models) > 1:
                    diversity_loss = self._compute_diversity_loss(model_idx)
                    loss += self.ensemble_config.diversity_weight * diversity_loss
                
                loss.backward()
                
                if self.training_config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.training_config.gradient_clip
                    )
                
                optimizer.step()
                model_losses.append(loss.item())
            
            epoch_losses.append(np.mean(model_losses))
        
        return epoch_losses
    
    def _compute_diversity_loss(self, model_idx: int) -> torch.Tensor:
        diversity_loss = 0.0
        current_model = self.models[model_idx]
        
        for other_idx, other_model in enumerate(self.models):
            if other_idx != model_idx:
                for param1, param2 in zip(current_model.parameters(), other_model.parameters()):
                    diversity_loss += torch.norm(param1 - param2)
        
        return diversity_loss
    
    def _validate_ensemble(self, val_loader) -> List[float]:
        val_losses = []
        
        for model in self.models:
            model.eval()
            model_losses = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(self.training_config.device)
                    target = target.to(self.training_config.device)
                    
                    output = model(data)
                    loss = nn.MSELoss()(output, target)
                    model_losses.append(loss.item())
            
            val_losses.append(np.mean(model_losses))
        
        return val_losses
    
    def get_ensemble_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        predictions = self.predict_ensemble(x)
        
        epistemic_uncertainty = predictions['std']
        total_uncertainty = torch.sqrt(predictions['std']**2 + 0.1)  # Add small aleatoric term
        
        return {
            'epistemic': epistemic_uncertainty,
            'total': total_uncertainty,
            'predictive_variance': predictions['std']**2
        }
