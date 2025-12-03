"""
Deep Echo State Network implementation.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

from .esn import EchoStateNetwork, ESNConfig

logger = logging.getLogger(__name__)


@dataclass
class DeepESNConfig:
    """Configuration for Deep Echo State Network."""
    
    n_inputs: int
    n_outputs: int
    n_layers: int = 3
    layer_sizes: List[int] = field(default_factory=lambda: [100, 200, 100])
    inter_scaling: float = 0.5
    layer_configs: List[Dict[str, Any]] = field(default_factory=list)
    aggregation_method: str = "concatenate"  # concatenate, average, weighted
    use_skip_connections: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.n_layers != len(self.layer_sizes):
            raise ValueError(
                f"Number of layers ({self.n_layers}) must match "
                f"layer_sizes length ({len(self.layer_sizes)})"
            )
        
        # Create default configs if not provided
        if not self.layer_configs:
            self.layer_configs = []
            for i, size in enumerate(self.layer_sizes):
                config = {
                    'n_reservoir': size,
                    'spectral_radius': 0.9 * (self.inter_scaling ** i),
                    'sparsity': 0.1,
                    'leaking_rate': 0.3,
                    'reservoir_connectivity': 'uniform',
                    'activation_function': 'tanh',
                }
                self.layer_configs.append(config)


class DeepEchoStateNetwork:
    """
    Deep Echo State Network with multiple reservoir layers.
    
    Features:
    - Multiple reservoir layers
    - Layer-wise training
    - Skip connections
    - Various aggregation methods
    - Layer-wise monitoring
    """
    
    def __init__(self, config: DeepESNConfig):
        """
        Initialize Deep ESN.
        
        Args:
            config: Deep ESN configuration
        """
        self.config = config
        
        # Create layers
        self.layers: List[EchoStateNetwork] = []
        self._initialize_layers()
        
        # Final readout
        from sklearn.linear_model import Ridge
        self.final_readout = Ridge(alpha=1e-6, random_state=42)
        
        # Aggregation weights (if using weighted aggregation)
        if config.aggregation_method == "weighted":
            self.aggregation_weights = np.ones(config.n_layers) / config.n_layers
        else:
            self.aggregation_weights = None
        
        # Statistics
        self.layer_outputs = []
        self.training_stats = {}
        
        logger.info(f"Deep ESN initialized with {config.n_layers} layers")
    
    def _initialize_layers(self):
        """Initialize all reservoir layers."""
        for i, layer_config in enumerate(self.config.layer_configs):
            # Create ESN config for this layer
            n_inputs = self.config.n_inputs if i == 0 else self.config.layer_sizes[i-1]
            
            esn_config = ESNConfig(
                n_inputs=n_inputs,
                n_outputs=self.config.layer_sizes[i],
                **layer_config
            )
            
            # Create and store layer
            layer = EchoStateNetwork(esn_config)
            self.layers.append(layer)
    
    def _aggregate_layer_outputs(self, layer_outputs: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate outputs from all layers.
        
        Args:
            layer_outputs: List of outputs from each layer
            
        Returns:
            Aggregated features
        """
        if self.config.aggregation_method == "concatenate":
            # Concatenate all outputs
            aggregated = np.hstack(layer_outputs)
        
        elif self.config.aggregation_method == "average":
            # Average outputs (requires same shape)
            shapes = [out.shape for out in layer_outputs]
            if len(set(shapes)) > 1:
                raise ValueError("Cannot average outputs with different shapes")
            aggregated = np.mean(layer_outputs, axis=0)
        
        elif self.config.aggregation_method == "weighted":
            # Weighted average
            shapes = [out.shape for out in layer_outputs]
            if len(set(shapes)) > 1:
                raise ValueError("Cannot average outputs with different shapes")
            weights = self.aggregation_weights.reshape(-1, 1, 1)
            aggregated = np.average(layer_outputs, axis=0, weights=weights)
        
        else:
            raise ValueError(f"Unknown aggregation: {self.config.aggregation_method}")
        
        return aggregated
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        layerwise_training: bool = True
    ) -> Dict[str, Any]:
        """
        Train the Deep ESN.
        
        Args:
            X: Input data
            y: Target data
            validation_data: Optional validation data
            layerwise_training: Whether to train layers sequentially
            
        Returns:
            Training statistics
        """
        logger.info(f"Training Deep ESN on {len(X)} samples")
        
        if layerwise_training:
            return self._fit_layerwise(X, y, validation_data)
        else:
            return self._fit_joint(X, y, validation_data)
    
    def _fit_layerwise(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Train layers sequentially."""
        n_samples = len(X)
        
        # Store layer outputs during training
        self.layer_outputs = []
        current_input = X.copy()
        
        # Train each layer
        for i, layer in enumerate(self.layers):
            logger.info(f"Training layer {i+1}/{len(self.layers)}")
            
            # For intermediate layers, use layer output as target
            if i < len(self.layers) - 1:
                # Self-supervised target (reconstruct input)
                layer_target = current_input
            else:
                # Final layer target is actual output
                layer_target = y
            
            # Train layer
            layer_stats = layer.fit(
                current_input,
                layer_target,
                validation_data
            )
            
            # Get layer outputs for next layer
            layer_pred, _ = layer.predict(current_input, return_states=True)
            self.layer_outputs.append(layer_pred)
            
            # Update input for next layer
            current_input = layer_pred
            
            # Store layer statistics
            self.training_stats[f'layer_{i}'] = layer_stats
        
        # Aggregate all layer outputs
        aggregated_features = self._aggregate_layer_outputs(self.layer_outputs)
        
        # Add skip connections if enabled
        if self.config.use_skip_connections:
            # Include original input
            aggregated_features = np.hstack([aggregated_features, X])
        
        # Train final readout
        logger.info("Training final readout")
        self.final_readout.fit(aggregated_features, y)
        
        # Evaluate
        train_pred = self.final_readout.predict(aggregated_features)
        from ..utils.metrics import PetroleumMetrics
        metrics = PetroleumMetrics.comprehensive_metrics(y, train_pred)
        
        self.training_stats['final'] = {
            'training_metrics': metrics,
            'readout_norm': np.linalg.norm(self.final_readout.coef_, 'fro'),
        }
        
        # Validation if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_pred = self.predict(X_val)
            val_metrics = PetroleumMetrics.comprehensive_metrics(y_val, val_pred)
            self.training_stats['final']['validation_metrics'] = val_metrics
        
        logger.info("Deep ESN training completed")
        return self.training_stats
    
    def _fit_joint(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Train all layers jointly."""
        # This is a simplified version - in practice, joint training
        # requires more sophisticated optimization
        
        n_samples = len(X)
        
        # Forward pass to collect all outputs
        self.layer_outputs = []
        current_input = X.copy()
        
        for i, layer in enumerate(self.layers):
            # Get layer outputs
            layer_pred, _ = layer.predict(current_input, return_states=True)
            self.layer_outputs.append(layer_pred)
            
            # Update input for next layer
            current_input = layer_pred
        
        # Aggregate features
        aggregated_features = self._aggregate_layer_outputs(self.layer_outputs)
        
        if self.config.use_skip_connections:
            aggregated_features = np.hstack([aggregated_features, X])
        
        # Train final readout
        self.final_readout.fit(aggregated_features, y)
        
        # Collect statistics
        train_pred = self.final_readout.predict(aggregated_features)
        from ..utils.metrics import PetroleumMetrics
        metrics = PetroleumMetrics.comprehensive_metrics(y, train_pred)
        
        self.training_stats = {
            'joint_training_metrics': metrics,
            'n_layers': len(self.layers),
            'aggregation_method': self.config.aggregation_method,
        }
        
        return self.training_stats
    
    def predict(
        self,
        X: np.ndarray,
        return_layer_outputs: bool = False
    ) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Input data
            return_layer_outputs: Whether to return individual layer outputs
            
        Returns:
            Predictions
        """
        # Forward pass through all layers
        layer_outputs = []
        current_input = X.copy()
        
        for layer in self.layers:
            layer_pred, _ = layer.predict(current_input, return_states=True)
            layer_outputs.append(layer_pred)
            current_input = layer_pred
        
        # Aggregate
        aggregated_features = self._aggregate_layer_outputs(layer_outputs)
        
        if self.config.use_skip_connections:
            aggregated_features = np.hstack([aggregated_features, X])
        
        # Final prediction
        predictions = self.final_readout.predict(aggregated_features)
        
        if return_layer_outputs:
            return predictions, layer_outputs
        return predictions
    
    def get_layer(self, index: int) -> EchoStateNetwork:
        """
        Get specific layer.
        
        Args:
            index: Layer index (0-based)
            
        Returns:
            Requested layer
        """
        if index < 0 or index >= len(self.layers):
            raise IndexError(f"Layer index {index} out of range")
        
        return self.layers[index]
    
    def summary(self) -> str:
        """Get model summary."""
        summary_lines = [
            "=" * 60,
            "Deep Echo State Network Summary",
            "=" * 60,
            f"Number of layers: {self.config.n_layers}",
            f"Layer sizes: {self.config.layer_sizes}",
            f"Aggregation method: {self.config.aggregation_method}",
            f"Skip connections: {self.config.use_skip_connections}",
            "-" * 60,
        ]
        
        # Add layer details
        for i, layer in enumerate(self.layers):
            layer_config = layer.get_config()
            summary_lines.append(
                f"Layer {i}: {layer_config['n_reservoir']} neurons, "
                f"SR={layer_config['spectral_radius']:.3f}"
            )
        
        if self.training_stats:
            final_stats = self.training_stats.get('final', {})
            if 'training_metrics' in final_stats:
                metrics = final_stats['training_metrics']
                summary_lines.extend([
                    "-" * 60,
                    "Final Model Performance:",
                    f"NSE: {metrics.get('nse', 0):.4f}",
                    f"R²: {metrics.get('r2', 0):.4f}",
                    f"RMSE: {metrics.get('rmse', 0):.4e}",
                ])
        
        summary_lines.append("=" * 60)
        
        return "\n".join(summary_lines)
