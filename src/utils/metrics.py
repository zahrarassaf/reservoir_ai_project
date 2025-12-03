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
        deriv_corrs = [v for k, v in metrics.items() if 'deriv_corr' in k]
        
        metrics['mean_acf_correlation'] = np.mean(acf_corrs)
        metrics['mean_psd_correlation'] = np.mean(psd_corrs)
        metrics['mean_deriv_correlation'] = np.mean(deriv_corrs)
        
        return metrics
    
    @staticmethod
    def _autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate autocorrelation function."""
        n = len(x)
        x_normalized = (x - np.mean(x)) / np.std(x)
        
        acf = np.zeros(max_lag)
        for lag in range(max_lag):
            if lag < n:
                acf[lag] = np.corrcoef(x_normalized[:n-lag], x_normalized[lag:])[0, 1]
        
        return acf
    
    @staticmethod
    def _power_spectrum(x: np.ndarray, dt: float) -> np.ndarray:
        """Calculate power spectral density."""
        n = len(x)
        
        # Remove mean
        x_detrended = x - np.mean(x)
        
        # Apply window
        window = np.hanning(n)
        x_windowed = x_detrended * window
        
        # Compute FFT
        fft_result = np.fft.fft(x_windowed)
        psd = np.abs(fft_result) ** 2
        
        # Keep only positive frequencies
        freqs = np.fft.fftfreq(n, dt)
        positive_freqs = freqs > 0
        
        return psd[positive_freqs]
    
    @staticmethod
    def material_balance_metrics(
        pressure_true: np.ndarray,
        pressure_pred: np.ndarray,
        production_true: np.ndarray,
        production_pred: np.ndarray,
        compressibility: float,
        volume: float,
        dt: float
    ) -> Dict[str, float]:
        """
        Calculate material balance metrics for reservoir simulation.
        
        Args:
            pressure_true: True pressure values
            pressure_pred: Predicted pressure values
            production_true: True production rates
            production_pred: Predicted production rates
            compressibility: Rock/fluid compressibility
            volume: Reservoir volume
            dt: Time step
            
        Returns:
            Dictionary of material balance metrics
        """
        # Calculate pressure change
        dp_true = np.diff(pressure_true)
        dp_pred = np.diff(pressure_pred)
        
        # Calculate cumulative production
        cum_prod_true = np.cumsum(production_true[:-1] * dt)
        cum_prod_pred = np.cumsum(production_pred[:-1] * dt)
        
        # Material balance equation: ΔP = - (B * Q_cum) / (c_t * V)
        # where B is formation volume factor (simplified to 1)
        
        # Expected pressure change from production
        dp_expected_true = -cum_prod_true / (compressibility * volume)
        dp_expected_pred = -cum_prod_pred / (compressibility * volume)
        
        # Material balance error
        mbe_true = dp_true - dp_expected_true
        mbe_pred = dp_pred - dp_expected_pred
        
        metrics = {
            'material_balance_error_true': np.mean(np.abs(mbe_true)),
            'material_balance_error_pred': np.mean(np.abs(mbe_pred)),
            'mbe_relative_error': np.mean(np.abs(mbe_pred - mbe_true)) / np.mean(np.abs(mbe_true)),
            'cumulative_production_error': np.mean(np.abs(cum_prod_pred - cum_prod_true)),
        }
        
        return metrics
    
    @staticmethod
    def forecast_skill_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_baseline: np.ndarray,
        horizon: int = 10
    ) -> Dict[str, float]:
        """
        Calculate forecast skill metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_baseline: Baseline predictions (e.g., persistence)
            horizon: Forecast horizon
            
        Returns:
            Dictionary of forecast skill metrics
        """
        # Calculate errors
        error_pred = y_true - y_pred
        error_base = y_true - y_baseline
        
        # Mean squared errors
        mse_pred = np.mean(error_pred ** 2, axis=0)
        mse_base = np.mean(error_base ** 2, axis=0)
        
        # Forecast skill score
        skill_score = 1 - mse_pred / mse_base
        
        # Skill score by horizon
        skill_by_horizon = []
        if len(y_true.shape) == 3:  # Has horizon dimension
            for h in range(min(horizon, y_true.shape[1])):
                mse_pred_h = np.mean((y_true[:, h] - y_pred[:, h]) ** 2)
                mse_base_h = np.mean((y_true[:, h] - y_baseline[:, h]) ** 2)
                skill_h = 1 - mse_pred_h / mse_base_h
                skill_by_horizon.append(skill_h)
        
        metrics = {
            'forecast_skill_score_mean': np.mean(skill_score),
            'forecast_skill_score_std': np.std(skill_score),
            'forecast_skill_by_horizon': skill_by_horizon if skill_by_horizon else [],
            'mse_improvement': (mse_base - mse_pred) / mse_base * 100,
        }
        
        return metrics
    
    @staticmethod
    def uncertainty_metrics(
        y_true: np.ndarray,
        y_pred_mean: np.ndarray,
        y_pred_std: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate uncertainty quantification metrics.
        
        Args:
            y_true: True values
            y_pred_mean: Predicted mean values
            y_pred_std: Predicted standard deviations
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary of uncertainty metrics
        """
        from scipy.stats import norm
        
        # Calculate z-score for confidence level
        z_score = norm.ppf((1 + confidence_level) / 2)
        
        # Prediction intervals
        lower_bound = y_pred_mean - z_score * y_pred_std
        upper_bound = y_pred_mean + z_score * y_pred_std
        
        # Coverage
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        
        # Interval width
        interval_width = np.mean(upper_bound - lower_bound)
        
        # Normalized interval width (relative to data range)
        data_range = np.max(y_true) - np.min(y_true)
        normalized_width = interval_width / data_range
        
        # Calibration metrics
        expected_coverage = confidence_level
        calibration_error = np.abs(coverage - expected_coverage)
        
        # Sharpness (inverse of interval width)
        sharpness = 1 / interval_width
        
        metrics = {
            'coverage': coverage,
            'expected_coverage': expected_coverage,
            'calibration_error': calibration_error,
            'interval_width': interval_width,
            'normalized_interval_width': normalized_width,
            'sharpness': sharpness,
            'mean_std': np.mean(y_pred_std),
            'std_of_std': np.std(y_pred_std),
        }
        
        return metrics
    
    @staticmethod
    def comprehensive_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_baseline: Optional[np.ndarray] = None,
        additional_data: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive set of metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_baseline: Baseline predictions
            additional_data: Additional data for specialized metrics
            
        Returns:
            Comprehensive metrics dictionary
        """
        logger.info("Calculating comprehensive metrics")
        
        PetroleumMetrics.validate_inputs(y_true, y_pred)
        
        metrics = {}
        
        # 1. Point metrics
        point_metrics = PetroleumMetrics.point_metrics(y_true, y_pred)
        metrics.update(point_metrics)
        
        # 2. Temporal metrics (if time series)
        if len(y_true.shape) == 2 and y_true.shape[0] > 10:
            try:
                temporal_metrics = PetroleumMetrics.temporal_metrics(y_true, y_pred)
                metrics.update(temporal_metrics)
            except Exception as e:
                logger.warning(f"Failed to calculate temporal metrics: {e}")
        
        # 3. Forecast skill metrics (if baseline provided)
        if y_baseline is not None:
            try:
                PetroleumMetrics.validate_inputs(y_true, y_baseline)
                skill_metrics = PetroleumMetrics.forecast_skill_metrics(
                    y_true, y_pred, y_baseline
                )
                metrics.update(skill_metrics)
            except Exception as e:
                logger.warning(f"Failed to calculate forecast skill metrics: {e}")
        
        # 4. Material balance metrics (if additional data provided)
        if additional_data is not None:
            required_keys = ['pressure_true', 'pressure_pred', 'production_true',
                           'production_pred', 'compressibility', 'volume', 'dt']
            
            if all(key in additional_data for key in required_keys):
                try:
                    mb_metrics = PetroleumMetrics.material_balance_metrics(
                        pressure_true=additional_data['pressure_true'],
                        pressure_pred=additional_data['pressure_pred'],
                        production_true=additional_data['production_true'],
                        production_pred=additional_data['production_pred'],
                        compressibility=additional_data['compressibility'],
                        volume=additional_data['volume'],
                        dt=additional_data['dt']
                    )
                    metrics.update(mb_metrics)
                except Exception as e:
                    logger.warning(f"Failed to calculate material balance metrics: {e}")
        
        # 5. Overall assessment
        metrics['overall_score'] = PetroleumMetrics._calculate_overall_score(metrics)
        
        logger.info(f"Calculated {len(metrics)} metrics")
        return metrics
    
    @staticmethod
    def _calculate_overall_score(metrics: Dict[str, float]) -> float:
        """
        Calculate overall score from multiple metrics.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Overall score (0-1)
        """
        # Define metric weights
        weights = {
            'nash_sutcliffe': 0.25,
            'r2': 0.20,
            'kling_gupta': 0.15,
            'mape': -0.15,  # Negative weight (lower is better)
            'forecast_skill_score_mean': 0.10,
            'material_balance_error_pred': -0.10,  # Negative weight
            'mean_acf_correlation': 0.05,
            'mean_psd_correlation': 0.05,
            'calibration_error': -0.05,  # Negative weight
        }
        
        # Calculate weighted sum
        total_weight = 0
        weighted_sum = 0
        
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                metric_value = metrics[metric_name]
                
                # Normalize metric to [0, 1] range
                if metric_name in ['nash_sutcliffe', 'r2', 'kling_gupta',
                                 'forecast_skill_score_mean', 'mean_acf_correlation',
                                 'mean_psd_correlation']:
                    # Higher is better, already in reasonable range
                    normalized = (metric_value + 1) / 2  # Map from [-1, 1] to [0, 1]
                
                elif metric_name in ['mape', 'material_balance_error_pred',
                                   'calibration_error']:
                    # Lower is better, invert
                    normalized = 1 / (1 + metric_value)  # Map to [0, 1]
                
                else:
                    normalized = 0.5  # Default
                
                weighted_sum += weight * normalized
                total_weight += abs(weight)
        
        # Normalize by total weight
        if total_weight > 0:
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 0.5
        
        # Ensure score is in [0, 1]
        overall_score = max(0.0, min(1.0, overall_score))
        
        return overall_score
    
    @staticmethod
    def format_metrics_report(metrics: Dict[str, Any]) -> str:
        """
        Format metrics as a readable report.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Formatted report string
        """
        # Group metrics by category
        categories = {
            'Error Metrics': ['mse', 'rmse', 'mae', 'mape', 'smape'],
            'Correlation Metrics': ['r2', 'pearson_r', 'spearman_rho', 'nash_sutcliffe'],
            'Advanced Metrics': ['kling_gupta', 'forecast_skill_score_mean'],
            'Temporal Metrics': ['mean_acf_correlation', 'mean_psd_correlation'],
            'Material Balance': ['material_balance_error_pred', 'mbe_relative_error'],
            'Uncertainty': ['coverage', 'calibration_error', 'interval_width'],
            'Overall': ['overall_score'],
        }
        
        report_lines = ["=" * 60, "Metrics Report", "=" * 60]
        
        for category, metric_names in categories.items():
            category_metrics = []
            
            for name in metric_names:
                if name in metrics:
                    value = metrics[name]
                    
                    # Format based on metric type
                    if 'error' in name.lower() or 'mse' in name or 'mae' in name:
                        formatted = f"{value:.4e}"
                    elif name in ['mape', 'smape', 'relative_bias']:
                        formatted = f"{value:.2f}%"
                    elif name == 'overall_score':
                        formatted = f"{value:.3f}"
                    else:
                        formatted = f"{value:.4f}"
                    
                    category_metrics.append(f"{name:30} {formatted}")
            
            if category_metrics:
                report_lines.append(f"\n{category}:")
                report_lines.extend(category_metrics)
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
