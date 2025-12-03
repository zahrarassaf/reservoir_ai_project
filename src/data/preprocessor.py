"""
Preprocessor for SPE9 reservoir simulation data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import logging
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    
    # Data splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Sequence parameters
    sequence_length: int = 30
    forecast_horizon: int = 10
    stride: int = 1
    
    # Scaling
    scaling_method: str = "standard"  # standard, minmax, robust, none
    feature_range: Tuple[float, float] = (-1, 1)
    
    # Feature selection
    target_columns: List[str] = field(default_factory=lambda: [
        "PRESSURE", "SATURATION", "PRODUCTION_RATE"
    ])
    
    input_columns: List[str] = field(default_factory=lambda: [
        "TIME", "INJECTION_RATE", "BHP", "THP"
    ])
    
    # Advanced options
    remove_outliers: bool = True
    outlier_threshold: float = 3.0  # sigma threshold
    fill_missing: bool = True
    interpolate_method: str = "linear"
    normalize_time: bool = True
    add_derivatives: bool = True
    derivative_order: int = 1
    
    def validate(self) -> None:
        """Validate configuration."""
        assert 0 < self.train_ratio < 1, "Train ratio must be between 0 and 1"
        assert 0 <= self.val_ratio < 1, "Val ratio must be between 0 and 1"
        assert 0 <= self.test_ratio < 1, "Test ratio must be between 0 and 1"
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-10, \
            "Ratios must sum to 1"
        
        assert self.sequence_length > 0, "Sequence length must be positive"
        assert self.forecast_horizon > 0, "Forecast horizon must be positive"
        assert self.stride > 0, "Stride must be positive"
        
        assert self.scaling_method in ["standard", "minmax", "robust", "none"], \
            f"Invalid scaling method: {self.scaling_method}"


class SPE9Preprocessor:
    """Preprocessor for SPE9 dataset."""
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.config.validate()
        
        # Scalers
        self.input_scaler = None
        self.target_scaler = None
        
        # Statistics
        self.statistics = {}
        
        logger.info(f"Preprocessor initialized with config: {config}")
    
    def parse_eclipse_data(self, data_file: Path) -> pd.DataFrame:
        """
        Parse Eclipse DATA file.
        
        Args:
            data_file: Path to DATA file
            
        Returns:
            Parsed DataFrame
        """
        logger.info(f"Parsing Eclipse data: {data_file}")
        
        # This is a simplified parser - in practice, you would use
        # a proper Eclipse parser or convert from summary files
        
        # For now, create synthetic data for demonstration
        n_samples = 1000
        n_timesteps = 100
        
        # Create synthetic reservoir simulation data
        time = np.linspace(0, 365 * 10, n_samples)  # 10 years
        
        # Synthetic pressure data
        base_pressure = 3000  # psi
        pressure_variation = 500 * np.sin(2 * np.pi * time / 365)  # yearly cycle
        noise = 50 * np.random.randn(n_samples)
        pressure = base_pressure + pressure_variation + noise
        
        # Synthetic saturation data
        saturation = 0.8 - 0.3 * (time / time.max()) + 0.1 * np.random.randn(n_samples)
        saturation = np.clip(saturation, 0.2, 0.9)
        
        # Synthetic production rates
        production = 1000 + 500 * np.sin(2 * np.pi * time / 180) + 200 * np.random.randn(n_samples)
        production = np.maximum(production, 100)
        
        # Other features
        injection = 800 + 400 * np.random.randn(n_samples)
        bhp = 1500 + 200 * np.sin(2 * np.pi * time / 90)
        thp = 500 + 100 * np.cos(2 * np.pi * time / 180)
        
        # Create DataFrame
        data = pd.DataFrame({
            'TIME': time,
            'PRESSURE': pressure,
            'SATURATION': saturation,
            'PRODUCTION_RATE': production,
            'INJECTION_RATE': injection,
            'BHP': bhp,
            'THP': thp,
        })
        
        logger.info(f"Parsed data shape: {data.shape}")
        return data
    
    def preprocess(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Preprocess raw data.
        
        Args:
            data: Raw DataFrame
            
        Returns:
            Dictionary with processed data splits
        """
        logger.info("Starting data preprocessing")
        
        # Store original data
        self.raw_data = data.copy()
        
        # 1. Handle missing values
        if self.config.fill_missing:
            data = self._handle_missing_values(data)
        
        # 2. Remove outliers
        if self.config.remove_outliers:
            data = self._remove_outliers(data)
        
        # 3. Add derived features
        if self.config.add_derivatives:
            data = self._add_derived_features(data)
        
        # 4. Normalize time
        if self.config.normalize_time and 'TIME' in data.columns:
            data['TIME'] = (data['TIME'] - data['TIME'].min()) / \
                          (data['TIME'].max() - data['TIME'].min())
        
        # 5. Separate inputs and targets
        X_columns = [col for col in self.config.input_columns if col in data.columns]
        y_columns = [col for col in self.config.target_columns if col in data.columns]
        
        if not X_columns:
            raise ValueError("No valid input columns found")
        if not y_columns:
            raise ValueError("No valid target columns found")
        
        X = data[X_columns].values
        y = data[y_columns].values
        
        # 6. Scale data
        X_scaled, y_scaled = self._scale_data(X, y)
        
        # 7. Create sequences
        sequences = self._create_sequences(X_scaled, y_scaled)
        
        # 8. Split data
        splits = self._split_data(sequences)
        
        # Store statistics
        self._compute_statistics(data, X_scaled, y_scaled, splits)
        
        logger.info("Preprocessing completed")
        return splits
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in data."""
        # Check for missing values
        missing = data.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"Missing values found: {missing[missing > 0].to_dict()}")
            
            if self.config.fill_missing:
                # Forward fill, then backward fill
                data = data.ffill().bfill()
                
                # If still missing, interpolate
                if data.isnull().any().any():
                    data = data.interpolate(method=self.config.interpolate_method)
        
        return data
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using statistical methods."""
        original_shape = data.shape
        
        for column in data.select_dtypes(include=[np.number]).columns:
            values = data[column].values
            
            # Compute statistics
            mean = np.nanmean(values)
            std = np.nanstd(values)
            
            # Identify outliers
            threshold = self.config.outlier_threshold * std
            outliers = np.abs(values - mean) > threshold
            
            # Replace outliers with NaN
            if outliers.any():
                data.loc[outliers, column] = np.nan
                logger.debug(f"Found {outliers.sum()} outliers in {column}")
        
        # Fill NaN values
        data = data.ffill().bfill()
        
        logger.info(f"Outlier removal: {original_shape} -> {data.shape}")
        return data
    
    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add derived features like derivatives."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Skip time column for derivatives
            if col == 'TIME':
                continue
            
            values = data[col].values
            
            # First derivative
            if self.config.derivative_order >= 1:
                derivative = np.gradient(values)
                data[f'{col}_DERIVATIVE'] = derivative
            
            # Second derivative
            if self.config.derivative_order >= 2:
                second_derivative = np.gradient(derivative)
                data[f'{col}_SECOND_DERIVATIVE'] = second_derivative
        
        return data
    
    def _scale_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale input and target data."""
        # Initialize scalers
        if self.config.scaling_method == "standard":
            self.input_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif self.config.scaling_method == "minmax":
            self.input_scaler = MinMaxScaler(feature_range=self.config.feature_range)
            self.target_scaler = MinMaxScaler(feature_range=self.config.feature_range)
        elif self.config.scaling_method == "none":
            return X, y
        else:
            raise ValueError(f"Unknown scaling method: {self.config.scaling_method}")
        
        # Fit and transform
        X_scaled = self.input_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)
        
        logger.info(f"Data scaled using {self.config.scaling_method} scaler")
        return X_scaled, y_scaled
    
    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            X: Scaled input data
            y: Scaled target data
            
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        n_samples = len(X)
        seq_len = self.config.sequence_length
        horizon = self.config.forecast_horizon
        stride = self.config.stride
        
        # Calculate number of sequences
        n_sequences = (n_samples - seq_len - horizon) // stride + 1
        
        if n_sequences <= 0:
            raise ValueError(
                f"Not enough samples for sequences. "
                f"Need at least {seq_len + horizon} samples, got {n_samples}"
            )
        
        # Initialize arrays
        input_sequences = np.zeros((n_sequences, seq_len, X.shape[1]))
        target_sequences = np.zeros((n_sequences, horizon, y.shape[1]))
        
        # Create sequences
        for i in range(n_sequences):
            start_idx = i * stride
            end_idx = start_idx + seq_len
            
            input_sequences[i] = X[start_idx:end_idx]
            target_sequences[i] = y[end_idx:end_idx + horizon]
        
        logger.info(
            f"Created {n_sequences} sequences: "
            f"input shape {input_sequences.shape}, "
            f"target shape {target_sequences.shape}"
        )
        
        return input_sequences, target_sequences
    
    def _split_data(
        self,
        sequences: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Split sequences into train/val/test sets.
        
        Args:
            sequences: Tuple of (input_sequences, target_sequences)
            
        Returns:
            Dictionary with data splits
        """
        X_seq, y_seq = sequences
        n_sequences = len(X_seq)
        
        # Calculate split indices
        n_train = int(n_sequences * self.config.train_ratio)
        n_val = int(n_sequences * self.config.val_ratio)
        
        # Shuffle indices
        indices = np.random.permutation(n_sequences)
        
        # Split indices
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # Create splits
        splits = {
            'X_train': X_seq[train_idx],
            'y_train': y_seq[train_idx],
            'X_val': X_seq[val_idx],
            'y_val': y_seq[val_idx],
            'X_test': X_seq[test_idx],
            'y_test': y_seq[test_idx],
        }
        
        logger.info(
            f"Data split: "
            f"train={len(train_idx)}, "
            f"val={len(val_idx)}, "
            f"test={len(test_idx)}"
        )
        
        return splits
    
    def _compute_statistics(
        self,
        raw_data: pd.DataFrame,
        X_scaled: np.ndarray,
        y_scaled: np.ndarray,
        splits: Dict[str, np.ndarray]
    ) -> None:
        """Compute and store preprocessing statistics."""
        self.statistics = {
            'raw_data_shape': raw_data.shape,
            'raw_columns': list(raw_data.columns),
            'n_input_features': X_scaled.shape[1],
            'n_output_features': y_scaled.shape[1],
            'n_sequences': len(splits['X_train']) + len(splits['X_val']) + len(splits['X_test']),
            'train_size': len(splits['X_train']),
            'val_size': len(splits['X_val']),
            'test_size': len(splits['X_test']),
            'sequence_length': self.config.sequence_length,
            'forecast_horizon': self.config.forecast_horizon,
            'scaling_method': self.config.scaling_method,
        }
        
        # Add feature statistics
        if self.input_scaler is not None:
            self.statistics['input_scaler_params'] = {
                'mean': self.input_scaler.mean_.tolist(),
                'scale': self.input_scaler.scale_.tolist(),
            }
        
        if self.target_scaler is not None:
            self.statistics['target_scaler_params'] = {
                'mean': self.target_scaler.mean_.tolist(),
                'scale': self.target_scaler.scale_.tolist(),
            }
    
    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled targets.
        
        Args:
            y_scaled: Scaled target values
            
        Returns:
            Original scale target values
        """
        if self.target_scaler is None:
            return y_scaled
        
        return self.target_scaler.inverse_transform(y_scaled)
    
    def save(self, filepath: Path) -> None:
        """
        Save preprocessor state.
        
        Args:
            filepath: Path to save preprocessor
        """
        save_data = {
            'config': self.config,
            'input_scaler': self.input_scaler,
            'target_scaler': self.target_scaler,
            'statistics': self.statistics,
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'SPE9Preprocessor':
        """
        Load preprocessor state.
        
        Args:
            filepath: Path to saved preprocessor
            
        Returns:
            Loaded preprocessor instance
        """
        save_data = joblib.load(filepath)
        
        # Create new instance
        preprocessor = cls(save_data['config'])
        
        # Restore state
        preprocessor.input_scaler = save_data['input_scaler']
        preprocessor.target_scaler = save_data['target_scaler']
        preprocessor.statistics = save_data['statistics']
        
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        return self.statistics.copy()
    
    def summary(self) -> str:
        """Get preprocessor summary."""
        stats = self.statistics
        
        summary_lines = [
            "=" * 60,
            "Data Preprocessor Summary",
            "=" * 60,
            f"Raw data shape: {stats.get('raw_data_shape', 'N/A')}",
            f"Input features: {stats.get('n_input_features', 'N/A')}",
            f"Output features: {stats.get('n_output_features', 'N/A')}",
            f"Sequence length: {stats.get('sequence_length', 'N/A')}",
            f"Forecast horizon: {stats.get('forecast_horizon', 'N/A')}",
            f"Total sequences: {stats.get('n_sequences', 'N/A')}",
            f"Train/Val/Test: {stats.get('train_size', 'N/A')}/"
            f"{stats.get('val_size', 'N/A')}/"
            f"{stats.get('test_size', 'N/A')}",
            f"Scaling method: {stats.get('scaling_method', 'N/A')}",
        ]
        
        summary_lines.append("=" * 60)
        
        return "\n".join(summary_lines)
