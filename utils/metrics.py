import numpy as np
import torch
from typing import Dict, Any

def reservoir_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    # Normalized metrics
    target_range = np.max(targets) - np.min(targets)
    nrmse = rmse / target_range if target_range > 0 else 0.0
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'nrmse': float(nrmse),
        'r_squared': float(r_squared)
    }

def physics_violation_metrics(predictions: Dict[str, torch.Tensor], 
                            static_properties: Dict[str, torch.Tensor]) -> Dict[str, float]:
    violations = {}
    
    # Mass balance violation
    pressure = predictions.get('pressure')
    saturation = predictions.get('saturation')
    
    if pressure is not None and saturation is not None:
        pressure_change = torch.diff(pressure, dim=0)
        saturation_change = torch.diff(saturation, dim=0)
        mass_balance = pressure_change + saturation_change
        violations['mass_balance_violation'] = torch.mean(mass_balance ** 2).item()
    
    # Saturation bounds violation
    if saturation is not None:
        saturation_violation = torch.mean(
            torch.relu(saturation - 1.0) + torch.relu(0.0 - saturation)
        )
        violations['saturation_bounds_violation'] = saturation_violation.item()
    
    return violations

def calculate_forecast_accuracy(predictions: np.ndarray, 
                              targets: np.ndarray, 
                              horizons: list = [1, 5, 10]) -> Dict[str, float]:
    accuracy_metrics = {}
    
    for horizon in horizons:
        if horizon <= predictions.shape[1]:
            horizon_preds = predictions[:, horizon-1, :]
            horizon_targets = targets[:, horizon-1, :]
            
            horizon_rmse = np.sqrt(np.mean((horizon_preds - horizon_targets) ** 2))
            accuracy_metrics[f'horizon_{horizon}_rmse'] = float(horizon_rmse)
    
    return accuracy_metrics
