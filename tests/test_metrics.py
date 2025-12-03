"""
Tests for metrics and evaluation functions.
"""

import pytest
import numpy as np
from typing import Dict, Any

from src.utils.metrics import PetroleumMetrics


class TestPetroleumMetrics:
    """Test suite for PetroleumMetrics."""
    
    def test_validate_inputs(self):
        """Test input validation."""
        # Valid inputs
        y_true = np.random.randn(100, 3)
        y_pred = np.random.randn(100, 3)
        
        # Should not raise errors
        PetroleumMetrics.validate_inputs(y_true, y_pred)
        
        # Shape mismatch
        y_pred_wrong = np.random.randn(100, 2)
        with pytest.raises(ValueError):
            PetroleumMetrics.validate_inputs(y_true, y_pred_wrong)
        
        # NaN values
        y_true_nan = y_true.copy()
        y_true_nan[50, 1] = np.nan
        
        with pytest.raises(ValueError):
            PetroleumMetrics.validate_inputs(y_true_nan, y_pred)
        
        # Infinite values
        y_true_inf = y_true.copy()
        y_true_inf[50, 1] = np.inf
        
        with pytest.raises(ValueError):
            PetroleumMetrics.validate_inputs(y_true_inf, y_pred)
    
    def test_point_metrics_basic(self):
        """Test basic point metrics calculation."""
        # Perfect prediction
        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_pred = y_true.copy()
        
        metrics = PetroleumMetrics.point_metrics(y_true, y_pred)
        
        # Perfect predictions should give ideal metrics
        assert np.isclose(metrics['mse'], 0.0)
        assert np.isclose(metrics['rmse'], 0.0)
        assert np.isclose(metrics['mae'], 0.0)
        assert np.isclose(metrics['r2'], 1.0)
        assert np.isclose(metrics['nash_sutcliffe'], 1.0)
        assert np.isclose(metrics['pearson_r'], 1.0)
        assert np.isclose(metrics['bias'], 0.0)
        
        # Random prediction
        np.random.seed(42)
        y_true = np.random.randn(1000, 1)
        y_pred = np.random.randn(1000, 1)
        
        metrics = PetroleumMetrics.point_metrics(y_true, y_pred)
        
        # Check all metrics are calculated
        expected_metrics = ['mse', 'rmse', 'mae', 'mape', 'smape',
                          'r2', 'pearson_r', 'spearman_rho', 'bias',
                          'relative_bias', 'std_ratio', 'nash_sutcliffe',
                          'kling_gupta']
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (float, np.floating))
        
        # Check metric ranges
        assert -1.0 <= metrics['r2'] <= 1.0
        assert -1.0 <= metrics['nash_sutcliffe'] <= 1.0
        assert -1.0 <= metrics['kling_gupta'] <= 1.0
        assert metrics['mape'] >= 0.0
        assert metrics['smape'] >= 0.0
    
    def test_point_metrics_edge_cases(self):
        """Test point metrics with edge cases."""
        # Constant true values
        y_true = np.ones((100, 1)) * 5.0
        y_pred = np.random.randn(100, 1) + 5.0
        
        metrics = PetroleumMetrics.point_metrics(y_true, y_pred)
        
        # R² and NSE should be 0 when variance is 0
        # (implementation returns -inf or similar, but we cap at -1)
        assert metrics['r2'] == -1.0
        assert metrics['nash_sutcliffe'] == -1.0
        
        # Very small values
        y_true = np.ones((100, 1)) * 1e-10
        y_pred = np.ones((100, 1)) * 2e-10
        
        metrics = PetroleumMetrics.point_metrics(y_true, y_pred)
        
        # Should handle small values without numerical issues
        assert not np.any(np.isnan(list(metrics.values())))
        assert not np.any(np.isinf(list(metrics.values())))
        
        # Large values
        y_true = np.ones((100, 1)) * 1e10
        y_pred = np.ones((100, 1)) * 1.1e10
        
        metrics = PetroleumMetrics.point_metrics(y_true, y_pred)
        
        # Should handle large values
        assert not np.any(np.isnan(list(metrics.values())))
        assert not np.any(np.isinf(list(metrics.values())))
    
    def test_temporal_metrics(self):
        """Test temporal metrics calculation."""
        # Create simple time series
        n_samples = 200
        t = np.linspace(0, 4*np.pi, n_samples)
        
        # True signal with multiple frequencies
        y_true = np.column_stack([
            np.sin(t) + 0.5 * np.sin(2*t),
            np.cos(t) + 0.3 * np.cos(3*t),
        ])
        
        # Predicted signal (slightly different)
        y_pred = np.column_stack([
            np.sin(t + 0.1) + 0.4 * np.sin(2*t + 0.2),
            np.cos(t - 0.1) + 0.2 * np.cos(3*t - 0.1),
        ]) + 0.05 * np.random.randn(n_samples, 2)
        
        metrics = PetroleumMetrics.temporal_metrics(y_true, y_pred, dt=0.1)
        
        # Check metrics structure
        assert 'mean_acf_correlation' in metrics
        assert 'mean_psd_correlation' in metrics
        assert 'mean_deriv_correlation' in metrics
        
        # Check individual feature metrics
        for i in range(y_true.shape[1]):
            assert f'feature_{i}_acf_corr' in metrics
            assert f'feature_{i}_psd_corr' in metrics
            assert f'feature_{i}_deriv_corr' in metrics
            
            # Correlation should be between -1 and 1
            assert -1.0 <= metrics[f'feature_{i}_acf_corr'] <= 1.0
            assert -1.0 <= metrics[f'feature_{i}_psd_corr'] <= 1.0
            assert -1.0 <= metrics[f'feature_{i}_deriv_corr'] <= 1.0
        
        # Mean correlations should be reasonable
        assert metrics['mean_acf_correlation'] > 0.5  # Similar autocorrelation
        assert metrics['mean_psd_correlation'] > 0.5  # Similar frequency content
        assert metrics['mean_deriv_correlation'] > 0.5  # Similar derivatives
    
    def test_temporal_metrics_edge_cases(self):
        """Test temporal metrics with edge cases."""
        # Constant time series
        y_true = np.ones((100, 2)) * 5.0
        y_pred = np.ones((100, 2)) * 5.0
        
        metrics = PetroleumMetrics.temporal_metrics(y_true, y_pred)
        
        # Constant series should have undefined correlations
        # Implementation should handle this gracefully
        for key in metrics:
            if 'corr' in key:
                assert np.isfinite(metrics[key])
        
        # Very short time series
        y_true = np.random.randn(10, 2)
        y_pred = np.random.randn(10, 2)
        
        metrics = PetroleumMetrics.temporal_metrics(y_true, y_pred)
        
        # Should still compute without errors
        assert 'mean_acf_correlation' in metrics
    
    def test_material_balance_metrics(self):
        """Test material balance metrics."""
        # Generate synthetic reservoir data
        n_samples = 100
        dt = 1.0  # days
        
        # Pressure decline due to production
        initial_pressure = 3000  # psi
        compressibility = 1e-5  # 1/psi
        volume = 1e6  # reservoir volume
        
        time = np.arange(n_samples) * dt
        production = 100 * np.ones(n_samples)  # constant production
        
        # True pressure (following material balance)
        cumulative_production = np.cumsum(production[:-1] * dt)
        pressure_true = initial_pressure - cumulative_production / (compressibility * volume)
        
        # Predicted pressure (with some error)
        pressure_pred = pressure_true + 10 * np.random.randn(n_samples - 1)
        
        metrics = PetroleumMetrics.material_balance_metrics(
            pressure_true=pressure_true,
            pressure_pred=pressure_pred,
            production_true=production[:-1],
            production_pred=production[:-1],
            compressibility=compressibility,
            volume=volume,
            dt=dt
        )
        
        # Check metrics
        expected_metrics = [
            'material_balance_error_true',
            'material_balance_error_pred',
            'mbe_relative_error',
            'cumulative_production_error'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert metrics[metric] >= 0.0
        
        # True MBE should be very small (we used exact material balance)
        assert metrics['material_balance_error_true'] < 1e-10
        
        # Predicted MBE should be larger due to noise
        assert metrics['material_balance_error_pred'] > 0.0
    
    def test_forecast_skill_metrics(self):
        """Test forecast skill metrics."""
        n_samples = 100
        horizon = 5
        
        # True values
        y_true = np.random.randn(n_samples, horizon, 1)
        
        # Model predictions (better than baseline)
        y_pred = y_true + 0.1 * np.random.randn(n_samples, horizon, 1)
        
        # Baseline predictions (e.g., persistence)
        y_baseline = y_true + 0.5 * np.random.randn(n_samples, horizon, 1)
        
        metrics = PetroleumMetrics.forecast_skill_metrics(
            y_true, y_pred, y_baseline, horizon
        )
        
        # Check metrics
        assert 'forecast_skill_score_mean' in metrics
        assert 'forecast_skill_score_std' in metrics
        assert 'forecast_skill_by_horizon' in metrics
        assert 'mse_improvement' in metrics
        
        # Skill score should be positive (model better than baseline)
        assert metrics['forecast_skill_score_mean'] > 0.0
        
        # Skill by horizon should have correct length
        assert len(metrics['forecast_skill_by_horizon']) == horizon
        
        # MSE improvement should be positive
        assert metrics['mse_improvement'] > 0.0
        
        # Test with model worse than baseline
        y_pred_worse = y_true + 1.0 * np.random.randn(n_samples, horizon, 1)
        
        metrics_worse = PetroleumMetrics.forecast_skill_metrics(
            y_true, y_pred_worse, y_baseline, horizon
        )
        
        # Skill score should be negative
        assert metrics_worse['forecast_skill_score_mean'] < 0.0
    
    def test_uncertainty_metrics(self):
        """Test uncertainty quantification metrics."""
        n_samples = 100
        
        # True values
        y_true = np.random.randn(n_samples, 1)
        
        # Predicted mean and standard deviation
        y_pred_mean = np.random.randn(n_samples, 1) * 0.5
        y_pred_std = np.abs(np.random.randn(n_samples, 1)) * 0.5 + 0.1
        
        metrics = PetroleumMetrics.uncertainty_metrics(
            y_true, y_pred_mean, y_pred_std, confidence_level=0.95
        )
        
        # Check metrics
        expected_metrics = [
            'coverage', 'expected_coverage', 'calibration_error',
            'interval_width', 'normalized_interval_width', 'sharpness',
            'mean_std', 'std_of_std'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
        
        # Coverage should be between 0 and 1
        assert 0.0 <= metrics['coverage'] <= 1.0
        
        # Expected coverage should match confidence level
        assert np.isclose(metrics['expected_coverage'], 0.95)
        
        # Calibration error should be non-negative
        assert metrics['calibration_error'] >= 0.0
        
        # Interval width should be positive
        assert metrics['interval_width'] > 0.0
        
        # Normalized width should be between 0 and 1 (approximately)
        assert 0.0 <= metrics['normalized_interval_width'] <= 1.0
        
        # Sharpness should be positive
        assert metrics['sharpness'] > 0.0
        
        # Mean and std of std should be positive
        assert metrics['mean_std'] > 0.0
        assert metrics['std_of_std'] >= 0.0
    
    def test_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        n_samples = 200
        
        # Generate data
        np.random.seed(42)
        y_true = np.random.randn(n_samples, 3)
        y_pred = y_true + 0.1 * np.random.randn(n_samples, 3)
        y_baseline = y_true + 0.5 * np.random.randn(n_samples, 3)
        
        # Test without additional data
        metrics = PetroleumMetrics.comprehensive_metrics(y_true, y_pred)
        
        # Check that all metric categories are included
        assert 'mse' in metrics  # Point metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'nash_sutcliffe' in metrics
        
        # Should have overall score
        assert 'overall_score' in metrics
        assert 0.0 <= metrics['overall_score'] <= 1.0
        
        # Test with baseline
        metrics_with_baseline = PetroleumMetrics.comprehensive_metrics(
            y_true, y_pred, y_baseline=y_baseline
        )
        
        # Should include forecast skill metrics
        assert 'forecast_skill_score_mean' in metrics_with_baseline
        
        # Test with additional data for material balance
        additional_data = {
            'pressure_true': np.random.randn(n_samples),
            'pressure_pred': np.random.randn(n_samples),
            'production_true': np.random.randn(n_samples),
            'production_pred': np.random.randn(n_samples),
            'compressibility': 1e-5,
            'volume': 1e6,
            'dt': 1.0
        }
        
        metrics_with_physics = PetroleumMetrics.comprehensive_metrics(
            y_true, y_pred, additional_data=additional_data
        )
        
        # Should include material balance metrics
        assert 'material_balance_error_pred' in metrics_with_physics
    
    def test_format_metrics_report(self):
        """Test metrics report formatting."""
        # Create sample metrics
        metrics = {
            'mse': 0.123456,
            'rmse': 0.351363,
            'mae': 0.284521,
            'mape': 15.123456,
            'r2': 0.856231,
            'nash_sutcliffe': 0.842156,
            'forecast_skill_score_mean': 0.723456,
            'material_balance_error_pred': 0.045621,
            'overall_score': 0.782345
        }
        
        report = PetroleumMetrics.format_metrics_report(metrics)
        
        # Check report structure
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Should contain section headers
        assert "ERROR METRICS" in report.upper()
        assert "CORRELATION METRICS" in report.upper()
        assert "OVERALL" in report.upper()
        
        # Should contain metric values
        assert "0.1235" in report  # MSE formatted
        assert "85.62%" not in report  # MAPE should be 15.12%
        assert "0.8562" in report  # R² formatted
        
        # Test with empty metrics
        empty_report = PetroleumMetrics.format_metrics_report({})
        assert len(empty_report) > 0
    
    def test_calculate_overall_score(self):
        """Test overall score calculation."""
        # Test with perfect metrics
        perfect_metrics = {
            'nash_sutcliffe': 1.0,
            'r2': 1.0,
            'kling_gupta': 1.0,
            'mape': 0.0,
            'forecast_skill_score_mean': 1.0,
            'material_balance_error_pred': 0.0,
            'mean_acf_correlation': 1.0,
            'mean_psd_correlation': 1.0,
            'calibration_error': 0.0,
        }
        
        score = PetroleumMetrics._calculate_overall_score(perfect_metrics)
        assert np.isclose(score, 1.0)
        
        # Test with worst metrics
        worst_metrics = {
            'nash_sutcliffe': -1.0,
            'r2': -1.0,
            'kling_gupta': -1.0,
            'mape': 100.0,  # Very high
            'forecast_skill_score_mean': -1.0,
            'material_balance_error_pred': 100.0,  # Very high
            'mean_acf_correlation': -1.0,
            'mean_psd_correlation': -1.0,
            'calibration_error': 1.0,  # Very high
        }
        
        score = PetroleumMetrics._calculate_overall_score(worst_metrics)
        assert 0.0 <= score <= 0.5  # Should be low
        
        # Test with mixed metrics
        mixed_metrics = {
            'nash_sutcliffe': 0.7,
            'r2': 0.6,
            'mape': 20.0,
            'forecast_skill_score_mean': 0.5,
        }
        
        score = PetroleumMetrics._calculate_overall_score(mixed_metrics)
        assert 0.0 <= score <= 1.0
        
        # Test with missing metrics
        partial_metrics = {
            'nash_sutcliffe': 0.8,
            'r2': 0.7,
        }
        
        score = PetroleumMetrics._calculate_overall_score(partial_metrics)
        assert 0.0 <= score <= 1.0
    
    def test_autocorrelation_function(self):
        """Test autocorrelation function calculation."""
        # Create simple signal
        n_samples = 100
        signal = np.sin(np.linspace(0, 4*np.pi, n_samples))
        
        # Calculate autocorrelation
        max_lag = 20
        acf = PetroleumMetrics._autocorrelation(signal, max_lag)
        
        # Check shape
        assert acf.shape == (max_lag,)
        
        # Check properties
        assert np.isclose(acf[0], 1.0)  # Autocorrelation at lag 0 should be 1
        assert np.all(acf <= 1.0)  # Should be <= 1
        assert np.all(acf >= -1.0)  # Should be >= -1
        
        # Test with constant signal
        constant_signal = np.ones(100)
        acf_constant = PetroleumMetrics._autocorrelation(constant_signal, 10)
        
        # Should handle constant signal (might get NaN, but function should handle)
        assert np.all(np.isfinite(acf_constant) | np.isnan(acf_constant))
    
    def test_power_spectrum(self):
        """Test power spectrum calculation."""
        # Create simple signal
        n_samples = 100
        dt = 0.01
        t = np.arange(n_samples) * dt
        signal = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*25*t)
        
        # Calculate power spectrum
        psd = PetroleumMetrics._power_spectrum(signal, dt)
        
        # Check shape and properties
        assert len(psd.shape) == 1
        assert len(psd) <= n_samples // 2  # Only positive frequencies
        assert np.all(psd >= 0.0)  # Power should be non-negative
        
        # Test with constant signal
        constant_signal = np.ones(100)
        psd_constant = PetroleumMetrics._power_spectrum(constant_signal, dt)
        
        # Should return array
        assert isinstance(psd_constant, np.ndarray)
        assert len(psd_constant) > 0
