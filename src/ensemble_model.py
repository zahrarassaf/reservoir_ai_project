import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import OrderedDict

class DiverseReservoirModel(nn.Module):
    """Diverse ensemble model with different architectures and training"""
    
    def __init__(self, config, ensemble_config):
        super().__init__()
        self.config = config
        self.ensemble_config = ensemble_config
        self.models = nn.ModuleList()
        
        # Create diverse architectures
        for variant in ensemble_config.architecture_variants:
            model = self._create_model_variant(variant)
            self.models.append(model)
            
        # Additional models if needed
        remaining_models = ensemble_config.n_models - len(ensemble_config.architecture_variants)
        for i in range(remaining_models):
            model = self._create_default_model()
            self.models.append(model)
    
    def _create_model_variant(self, variant: Dict) -> nn.Module:
        """Create model with specific architecture variant"""
        from .temporal_model import ReservoirTemporalModel
        
        # Create modified config for this variant
        variant_config = self.config.__class__(**self.config.__dict__)
        variant_config.hidden_layers = variant['hidden_dims']
        variant_config.dropout_rate = variant['dropout']
        
        return ReservoirTemporalModel(variant_config, self.config)
    
    def _create_default_model(self) -> nn.Module:
        """Create default model architecture"""
        from .temporal_model import ReservoirTemporalModel
        return ReservoirTemporalModel(self.config, self.config)
    
    def forward(self, static_features: torch.Tensor, 
                temporal_sequence: torch.Tensor,
                physics_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Ensemble forward pass
        
        Returns:
            Dictionary with ensemble predictions and uncertainties
        """
        ensemble_predictions = []
        ensemble_uncertainties = []
        
        for model in self.models:
            output = model(static_features, temporal_sequence, physics_features)
            ensemble_predictions.append(output['production'])
            ensemble_uncertainties.append(output['uncertainty'])
        
        # Stack predictions [n_models, batch, output_dim]
        pred_stack = torch.stack(ensemble_predictions, dim=0)
        unc_stack = torch.stack(ensemble_uncertainties, dim=0)
        
        # Ensemble statistics
        mean_pred = pred_stack.mean(dim=0)
        std_pred = pred_stack.std(dim=0)
        
        # Combined uncertainty (aleatoric + epistemic)
        aleatoric_unc = unc_stack.mean(dim=0)  # Average model uncertainties
        epistemic_unc = std_pred  # Std of predictions
        total_unc = torch.sqrt(aleatoric_unc**2 + epistemic_unc**2)
        
        return {
            'mean': mean_pred,
            'std': total_unc,
            'aleatoric': aleatoric_unc,
            'epistemic': epistemic_unc,
            'individual_predictions': pred_stack,
            'individual_uncertainties': unc_stack
        }
    
    def predict_with_quantiles(self, static_features: torch.Tensor,
                             temporal_sequence: torch.Tensor,
                             physics_features: Optional[torch.Tensor] = None,
                             n_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """
        Probabilistic prediction with quantile estimation
        """
        all_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                model_samples = []
                for model in self.models:
                    # Sample from each model's predictive distribution
                    output = model(static_features, temporal_sequence, physics_features)
                    pred = output['production']
                    unc = output['uncertainty']
                    
                    # Sample from Gaussian distribution
                    sample = pred + unc * torch.randn_like(pred)
                    model_samples.append(sample)
                
                model_samples = torch.stack(model_samples, dim=0)
                all_samples.append(model_samples)
            
        # Combine all samples [n_samples, n_models, batch, output_dim]
        all_samples = torch.stack(all_samples, dim=0)
        
        # Calculate quantiles
        samples_flat = all_samples.view(-1, all_samples.size(-2), all_samples.size(-1))
        quantiles = torch.quantile(samples_flat, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]), dim=0)
        
        return {
            'samples': all_samples,
            'quantile_10': quantiles[0],
            'quantile_25': quantiles[1],
            'median': quantiles[2],
            'quantile_75': quantiles[3],
            'quantile_90': quantiles[4]
        }
