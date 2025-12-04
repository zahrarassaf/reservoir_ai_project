"""
Physics-based evaluation metrics for reservoir simulation models.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import scipy.stats as stats


class PhysicsMetrics:
    """Physics-based metrics for model evaluation."""
    
    @staticmethod
    def compute_material_balance_error(predictions: torch.Tensor, 
                                      targets: torch.Tensor,
                                      porosity: torch.Tensor,
                                      dt: float = 1.0) -> float:
        """
        Compute material balance error (conservation of mass).
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            porosity: Porosity field
            dt: Time step
            
        Returns:
            MBE: Material balance error percentage
        """
        # For water phase (assuming saturation predictions)
        pred_accumulation = torch.sum(porosity * predictions)
        target_accumulation = torch.sum(porosity * targets)
        
        mbe = torch.abs(pred_accumulation - target_accumulation) / target_accumulation
        return mbe.item() * 100  # Percentage
    
    @staticmethod
    def compute_darcy_consistency(pressure_grad: torch.Tensor,
                                 flow_rate: torch.Tensor,
                                 permeability: torch.Tensor,
                                 viscosity: float = 1.0) -> float:
        """
        Compute consistency with Darcy's law.
        
        Args:
            pressure_grad: Pressure gradient
            flow_rate: Flow rate
            permeability: Permeability field
            viscosity: Fluid viscosity
            
        Returns:
            darcy_error: Error in Darcy's law
        """
        # Darcy's law: q = - (k/μ) * ∇P
        expected_flow = - (permeability / viscosity) * pressure_grad
        darcy_error = torch.mean(torch.abs(flow_rate - expected_flow))
        
        return darcy_error.item()
    
    @staticmethod
    def compute_capillary_pressure_consistency(saturation: torch.Tensor,
                                              pc_curve: torch.Tensor) -> float:
        """
        Compute consistency with capillary pressure curve.
        
        Args:
            saturation: Predicted saturation
            pc_curve: Capillary pressure curve
            
        Returns:
            pc_error: Capillary pressure consistency error
        """
        # Interpolate to compare with curve
        sorted_s, indices = torch.sort(saturation.flatten())
        sorted_pc = pc_curve.flatten()[indices]
        
        # Smoothness metric
        pc_grad = torch.diff(sorted_pc)
        smoothness_error = torch.std(pc_grad)
        
        return smoothness_error.item()
    
    @staticmethod
    def compute_entropy_production(predictions: torch.Tensor) -> float:
        """
        Compute entropy production (thermodynamic consistency).
        
        Args:
            predictions: Model predictions
            
        Returns:
            entropy_rate: Rate of entropy production
        """
        # Simplified entropy production calculation
        # For reservoir flow, should be non-negative
        grad = torch.gradient(predictions, dim=0)[0]
        entropy_prod = torch.sum(torch.square(grad))
        
        return entropy_prod.item()
    
    @staticmethod
    def compute_physics_violation_score(model, 
                                       data_loader,
                                       physics_module) -> Dict[str, float]:
        """
        Compute comprehensive physics violation score.
        
        Args:
            model: Trained model
            data_loader: Test data loader
            physics_module: Physics constraints module
            
        Returns:
            scores: Dictionary of physics violation scores
        """
        model.eval()
        
        physics_scores = {
            'material_balance_error': [],
            'darcy_consistency': [],
            'capillary_consistency': [],
            'entropy_production': []
        }
        
        with torch.no_grad():
            for batch in data_loader:
                inputs, targets, physics_state = batch
                
                predictions, _ = model(inputs)
                
                # Extract physics information
                if 'pressure' in physics_state and 'saturation' in physics_state:
                    pressure = physics_state['pressure']
                    saturation = physics_state['saturation']
                    
                    # Compute metrics
                    mbe = PhysicsMetrics.compute_material_balance_error(
                        predictions, targets, physics_module.porosity
                    )
                    physics_scores['material_balance_error'].append(mbe)
        
        # Average scores
        avg_scores = {k: np.mean(v) for k, v in physics_scores.items() if v}
        
        # Overall physics score (lower is better)
        overall_score = np.mean(list(avg_scores.values()))
        avg_scores['overall_physics_score'] = overall_score
        
        return avg_scores
