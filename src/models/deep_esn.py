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
        
