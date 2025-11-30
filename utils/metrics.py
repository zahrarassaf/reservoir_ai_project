import numpy as np
import torch

def reservoir_metrics(predictions, targets):
    mse = np.mean((predictions - targets)**2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae, 
        'rmse': rmse
    }

def physics_violation_metrics(predictions, constraints):
    violations = {}
    # Physics constraint violation calculations
    return violations
