"""
Comprehensive metrics for reservoir simulation evaluation.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import jensenshannon
import logging

logger = logging.getLogger(__name__)


class PetroleumMetrics:
    """Comprehensive metrics for petroleum reservoir simulation."""
    
    @staticmethod
    def validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Validate input arrays.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Raises:
            ValueError: If inputs are invalid
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}"
            )
        
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            raise ValueError("Input contains NaN values")
        
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            raise ValueError("Input contains infinite values")
    
    @staticmethod
    def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate point-wise prediction metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of point metrics
        """
        PetroleumMetrics.validate_inputs(y_true, y_pred)
        
        # Flatten if multi-dimensional
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Basic statistics
        residuals = y_true_flat - y_pred_flat
        abs_residuals = np.abs(residuals)
        
        metrics = {
            # Error metrics
            'mse': np.mean(residuals ** 2),
            'rmse': np.sqrt(np.mean(residuals ** 2)),
            'mae': np.mean(abs_residuals),
            'mape': np.mean(abs_residuals / (np.abs(y_true_flat) + 1e-10)) * 100,
            'smape': 100 * np.mean(
                2 * abs_residuals / (np.abs(y_true_flat) + np.abs(y_pred_flat) + 1e-10)
            ),
            
            # Correlation metrics
            'r2': 1 - np.sum(residuals ** 2) / np.sum((y_true_flat - np.mean(y_true_flat)) ** 2),
            'pearson_r': np.corrcoef(y_true_flat, y_pred_flat)[0, 1],
            'spearman_rho': stats.spearmanr(y_true_flat, y_pred_flat)[0],
            
            # Distribution metrics
            'bias': np.mean(residuals),
            'relative_bias': np.mean(residuals / (np.abs(y_true_flat) + 1e-10)) * 100,
            'std_ratio': np.std(y_pred_flat) / np.std(y_true_flat),
            
            # Advanced metrics
            'nash_sutcliffe': 1 - np.sum(residuals ** 2) / \
                             np.sum((y_true_flat - np.mean(y_true_flat)) ** 2),
            'kling_gupta': 1 - np.sqrt(
                (np.corrcoef(y_true_flat, y_pred_flat)[0, 1] - 1) ** 2 +
                (np.std(y_pred_flat) / np.std(y_true_flat) - 1) ** 2 +
                (np.mean(y_pred_flat) / np.mean(y_true_flat) - 1) ** 2
            ),
        }
        
        # Handle edge cases
        metrics['r2'] = max(metrics['r2'], -1.0)  # R² can be negative
        metrics['nash_sutcliffe'] = max(metrics['nash_sutcliffe'], -1.0)
        
        return metrics
    
    @staticmethod
    def temporal_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dt: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate temporal dynamics metrics.
        
        Args:
            y_true: True time series
            y_pred: Predicted time series
            dt: Time step
            
        Returns:
            Dictionary of temporal metrics
        """
        if len(y_true.shape) != 2:
            raise ValueError("Temporal metrics require 2D arrays (time, features)")
        
        n_timesteps, n_features = y_true.shape
        
        metrics = {}
        
        for i in range(n_features):
            true_ts = y_true[:, i]
            pred_ts = y_pred[:, i]
            
            # Autocorrelation similarity
            true_acf = PetroleumMetrics._autocorrelation(true_ts, max_lag=min(20, n_timesteps//4))
            pred_acf = PetroleumMetrics._autocorrelation(pred_ts, max_lag=min(20, n_timesteps//4))
            
            acf_correlation = np.corrcoef(true_acf, pred_acf)[0, 1]
            
            # Power spectrum similarity
            true_psd = PetroleumMetrics._power_spectrum(true_ts, dt)
            pred_psd = PetroleumMetrics._power_spectrum(pred_ts, dt)
            
            psd_correlation = np.corrcoef(true_psd, pred_psd)[0, 1]
            
            # Derivative metrics
            true_deriv = np.gradient(true_ts, dt)
            pred_deriv = np.gradient(pred_ts, dt)
            
            deriv_correlation = np.corrcoef(true_deriv, pred_deriv)[0, 1]
            
            metrics[f'feature_{i}_acf_corr'] = acf_correlation
            metrics[f'feature_{i}_psd_corr'] = psd_correlation
            metrics[f'feature_{i}_deriv_corr'] = deriv_correlation
        
        # Average across features
        acf_corrs = [v for k, v in metrics.items() if 'acf_corr' in k]
        psd_corrs = [v for k, v in metrics.items() if 'psd_corr' in k]
        deriv_corrs = [v for k, v in metrics.items() if 'deriv_corr'
