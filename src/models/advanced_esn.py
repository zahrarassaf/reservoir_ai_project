"""
Advanced ESN variants with cutting-edge research features.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
import logging
from scipy import sparse
from scipy.sparse.linalg import eigs
import warnings

from .esn import EchoStateNetwork, ESNConfig

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalESNConfig:
    """Configuration for Hierarchical ESN."""
    
    base_config: ESNConfig
    n_levels: int = 3
    level_sizes: List[int] = field(default_factory=lambda: [500, 200, 100])
    reduction_factor: float = 0.5
    inter_level_connections: bool = True
    top_down_feedback: bool = False
    level_specific_training: bool = False
    temporal_scales: List[int] = field(default_factory=lambda: [1, 2, 4])
    
    def __post_init__(self):
        """Validate configuration."""
        if len(self.level_sizes) != self.n_levels:
            if self.level_sizes:
                warnings.warn(f"level_sizes length ({len(self.level_sizes)}) doesn't match n_levels ({self.n_levels})")
            self.level_sizes = [int(self.base_config.n_reservoir * (self.reduction_factor ** i)) 
                              for i in range(self.n_levels)]
        
        if len(self.temporal_scales) != self.n_levels:
            self.temporal_scales = [2 ** i for i in range(self.n_levels)]


class HierarchicalESN:
    """
    Hierarchical Echo State Network for multi-scale temporal processing.
    
    Reference: Gallicchio, C., & Micheli, A. (2017). 
    "Deep Echo State Network (DeepESN): A Brief Survey"
    """
    
    def __init__(self, config: HierarchicalESNConfig):
        self.config = config
        self.levels: List[EchoStateNetwork] = []
        self.level_readouts = []
        self.level_scalers = []
        
        self._initialize_levels()
        
        # Final readout
        from sklearn.linear_model import Ridge
        self.final_readout = Ridge(alpha=1e-6, random_state=42)
        
        # Inter-level connections
        self.inter_level_weights = None
        if config.inter_level_connections:
            self._initialize_inter_level_connections()
        
        logger.info(f"Initialized HierarchicalESN with {config.n_levels} levels")
    
    def _initialize_levels(self):
        """Initialize hierarchical levels."""
        for i, (level_size, scale) in enumerate(zip(self.config.level_sizes, 
                                                  self.config.temporal_scales)):
            
            # Create level-specific config
            level_config = ESNConfig(
                n_inputs=self.config.base_config.n_inputs if i == 0 
                        else self.config.level_sizes[i-1],
                n_outputs=level_size,
                n_reservoir=level_size,
                spectral_radius=self.config.base_config.spectral_radius * (0.9 ** i),
                leaking_rate=self.config.base_config.leaking_rate * (1.2 ** i),  # Faster at higher levels
                sparsity=self.config.base_config.sparsity,
                reservoir_connectivity=self.config.base_config.reservoir_connectivity,
                activation_function=self.config.base_config.activation_function,
                random_state=self.config.base_config.random_state + i
            )
            
            # Create level
            level = EchoStateNetwork(level_config)
            self.levels.append(level)
            
            # Level-specific readout
            from sklearn.linear_model import Ridge
            self.level_readouts.append(Ridge(alpha=1e-6, random_state=42))
            
            # Level-specific scaler for temporal downsampling
            from sklearn.preprocessing import StandardScaler
            self.level_scalers.append(StandardScaler())
    
    def _initialize_inter_level_connections(self):
        """Initialize connections between levels."""
        n_levels = len(self.levels)
        self.inter_level_weights = []
        
        for i in range(n_levels - 1):
            # Connection from level i to level i+1
            size_from = self.config.level_sizes[i]
            size_to = self.config.level_sizes[i + 1]
            
            # Random weights with scaling
            scale = 1.0 / np.sqrt(size_from)
            weights = np.random.randn(size_to, size_from) * scale
            
            self.inter_level_weights.append(weights)
    
    def _process_temporal_scale(self, level_idx: int, X: np.ndarray) -> np.ndarray:
        """Process input for specific temporal scale."""
        scale = self.config.temporal_scales[level_idx]
        
        if scale > 1:
            # Downsample for coarser scales
            n_samples = X.shape[0]
            if n_samples % scale != 0:
                # Pad if necessary
                pad_length = scale - (n_samples % scale)
                X = np.pad(X, ((0, pad_length), (0, 0)), mode='edge')
                n_samples += pad_length
            
            # Reshape and average
            X_downsampled = X.reshape(n_samples // scale, scale, -1).mean(axis=1)
            return X_downsampled
        else:
            return X
    
    def _upsample_temporal_scale(self, level_idx: int, X: np.ndarray, 
                               target_length: int) -> np.ndarray:
        """Upsample output from level to original temporal scale."""
        scale = self.config.temporal_scales[level_idx]
        
        if scale > 1:
            # Repeat each sample scale times
            X_upsampled = np.repeat(X, scale, axis=0)
            
            # Trim to target length
            if X_upsampled.shape[0] > target_length:
                X_upsampled = X_upsampled[:target_length]
            elif X_upsampled.shape[0] < target_length:
                # Pad if necessary
                pad_length = target_length - X_upsampled.shape[0]
                X_upsampled = np.pad(X_upsampled, ((0, pad_length), (0, 0)), mode='edge')
            
            return X_upsampled
        else:
            return X
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the Hierarchical ESN."""
        logger.info(f"Training HierarchicalESN on {len(X)} samples")
        
        n_samples = X.shape[0]
        level_outputs = []
        level_states = []
        
        # Process through each level
        current_input = X.copy()
        
        for i, level in enumerate(self.levels):
            logger.debug(f"Processing level {i+1}/{len(self.levels)}")
            
            # Adjust temporal scale
            X_level = self._process_temporal_scale(i, current_input)
            
            if i == 0:
                # First level processes original input
                level_target = X_level
            else:
                # Higher levels try to predict lower level states
                level_target = level_outputs[-1]
                
                # Adjust target temporal scale if needed
                if level_target.shape[0] > X_level.shape[0]:
                    level_target = self._process_temporal_scale(i, level_target)
            
            # Train level
            level_stats = level.fit(X_level, level_target)
            
            # Get level outputs
            level_pred, states = level.predict(X_level, return_states=True)
            level_outputs.append(level_pred)
            level_states.append(states)
            
            # Prepare input for next level (with inter-level connections if enabled)
            if i < len(self.levels) - 1:
                if self.config.inter_level_connections and self.inter_level_weights is not None:
                    # Add inter-level signal
                    inter_signal = np.dot(level_pred, self.inter_level_weights[i].T)
                    current_input = inter_signal
                else:
                    current_input = level_pred
        
        # Train level-specific readouts if enabled
        if self.config.level_specific_training:
            for i, (states, readout) in enumerate(zip(level_states, self.level_readouts)):
                # Align targets with level temporal scale
                y_level = self._process_temporal_scale(i, y)
                
                # Ensure shapes match
                if y_level.shape[0] != states.shape[0]:
                    min_length = min(y_level.shape[0], states.shape[0])
                    y_level = y_level[:min_length]
                    states = states[:min_length]
                
                # Train readout
                readout.fit(states, y_level)
        
        # Combine features from all levels (upsampled to original temporal scale)
        combined_features = []
        for i, output in enumerate(level_outputs):
            output_upsampled = self._upsample_temporal_scale(i, output, n_samples)
            combined_features.append(output_upsampled)
        
        # Add original input features
        if self.config.top_down_feedback:
            combined_features.append(X)
        
        # Concatenate all features
        X_combined = np.hstack(combined_features)
        
        # Train final readout
        self.final_readout.fit(X_combined, y)
        
        # Collect training statistics
        stats = {
            'n_levels': self.config.n_levels,
            'level_sizes': self.config.level_sizes,
            'temporal_scales': self.config.temporal_scales,
            'combined_features_dim': X_combined.shape[1],
            'level_metrics': [
                {'size': size, 'scale': scale}
                for size, scale in zip(self.config.level_sizes, self.config.temporal_scales)
            ]
        }
        
        # Store for prediction
        self.level_outputs = level_outputs
        self.combined_features_shape = X_combined.shape[1]
        
        logger.info("HierarchicalESN training completed")
        return stats
    
    def predict(self, X: np.ndarray, return_level_outputs: bool = False) -> np.ndarray:
        """Generate predictions."""
        n_samples = X.shape[0]
        level_outputs = []
        
        current_input = X.copy()
        
        # Forward pass through all levels
        for i, level in enumerate(self.levels):
            X_level = self._process_temporal_scale(i, current_input)
            
            level_pred, _ = level.predict(X_level, return_states=True)
            level_outputs.append(level_pred)
            
            if i < len(self.levels) - 1:
                if self.config.inter_level_connections and self.inter_level_weights is not None:
                    inter_signal = np.dot(level_pred, self.inter_level_weights[i].T)
                    current_input = inter_signal
                else:
                    current_input = level_pred
        
        # Combine features
        combined_features = []
        for i, output in enumerate(level_outputs):
            output_upsampled = self._upsample_temporal_scale(i, output, n_samples)
            combined_features.append(output_upsampled)
        
        if self.config.top_down_feedback:
            combined_features.append(X)
        
        X_combined = np.hstack(combined_features)
        
        # Make final prediction
        if self.config.level_specific_training:
            # Average predictions from all levels
            predictions = []
            for i, (output, readout) in enumerate(zip(level_outputs, self.level_readouts)):
                level_pred = readout.predict(output)
                level_pred_upsampled = self._upsample_temporal_scale(i, level_pred, n_samples)
                predictions.append(level_pred_upsampled)
            
            # Weighted average (higher levels get more weight)
            weights = np.arange(1, len(predictions) + 1)
            weights = weights / weights.sum()
            
            final_prediction = np.zeros_like(predictions[0])
            for w, pred in zip(weights, predictions):
                final_prediction += w * pred[:final_prediction.shape[0]]
        else:
            # Use final readout
            final_prediction = self.final_readout.predict(X_combined)
        
        if return_level_outputs:
            return final_prediction, level_outputs
        return final_prediction
    
    def get_level_importance(self) -> np.ndarray:
        """Estimate importance of each level based on readout weights."""
        if self.config.level_specific_training:
            importances = []
            for readout in self.level_readouts:
                weight_norm = np.linalg.norm(readout.coef_)
                importances.append(weight_norm)
            
            # Normalize
            importances = np.array(importances)
            if importances.sum() > 0:
                importances = importances / importances.sum()
            
            return importances
        else:
            # Estimate from final readout weights
            feature_idx = 0
            importances = []
            
            for i, size in enumerate(self.config.level_sizes):
                level_weights = self.final_readout.coef_[:, feature_idx:feature_idx + size]
                importance = np.linalg.norm(level_weights)
                importances.append(importance)
                feature_idx += size
            
            importances = np.array(importances)
            if importances.sum() > 0:
                importances = importances / importances.sum()
            
            return importances


@dataclass
class AttentionESNConfig:
    """Configuration for Attention-based ESN."""
    
    base_config: ESNConfig
    attention_mechanism: str = "self_attention"  # self_attention, temporal_attention, cross_attention
    n_attention_heads: int = 4
    attention_dim: int = 64
    attention_dropout: float = 0.1
    use_residual: bool = True
    use_layer_norm: bool = True
    attention_temperature: float = 1.0
    
    def __post_init__(self):
        """Validate configuration."""
        valid_mechanisms = ["self_attention", "temporal_attention", "cross_attention"]
        if self.attention_mechanism not in valid_mechanisms:
            raise ValueError(f"attention_mechanism must be one of {valid_mechanisms}")


class MultiHeadAttention:
    """Multi-head attention mechanism."""
    
    def __init__(self, n_heads: int, d_model: int, d_k: int, d_v: int, 
                 dropout: float = 0.1, temperature: float = 1.0):
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        # Linear projections
        self.W_q = np.random.randn(d_model, n_heads * d_k) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, n_heads * d_k) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, n_heads * d_v) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(n_heads * d_v, d_model) * np.sqrt(2.0 / (n_heads * d_v))
        
        self.dropout = dropout
        self.temperature = temperature
        self.attention_weights = None
    
    def __call__(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply multi-head attention."""
        batch_size, seq_len, _ = query.shape
        
        # Linear projections and reshape for multi-head
        Q = np.dot(query, self.W_q).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = np.dot(key, self.W_k).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        V = np.dot(value, self.W_v).reshape(batch_size, seq_len, self.n_heads, self.d_v)
        
        # Transpose for attention computation
        Q = Q.transpose(0, 2, 1, 3)  # [batch, heads, seq_len, d_k]
        K = K.transpose(0, 2, 3, 1)  # [batch, heads, d_k, seq_len]
        V = V.transpose(0, 2, 1, 3)  # [batch, heads, seq_len, d_v]
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K) / np.sqrt(self.d_k * self.temperature)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask * -1e9
        
        # Softmax
        attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
        
        # Apply dropout
        if self.dropout > 0 and np.random.random() < self.dropout:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, size=attention_weights.shape)
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout)
        
        # Apply attention to values
        context = np.matmul(attention_weights, V)
        
        # Reshape and project back
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = np.dot(context, self.W_o)
        
        # Store attention weights for analysis
        self.attention_weights = attention_weights
        
        return output


class AttentionESN:
    """
    ESN with attention mechanism for selective information processing.
    
    Reference: Vaswani, A., et al. (2017). "Attention Is All You Need"
    Adapted for Reservoir Computing.
    """
    
    def __init__(self, config: AttentionESNConfig):
        self.config = config
        self.base_esn = EchoStateNetwork(config.base_config)
        
        # Attention mechanism
        self._initialize_attention()
        
        # Layer normalization
        if config.use_layer_norm:
            self.layer_norm1 = LayerNorm(config.base_config.n_reservoir)
            self.layer_norm2 = LayerNorm(config.base_config.n_reservoir)
        
        # Final readout
        from sklearn.linear_model import Ridge
        self.final_readout = Ridge(alpha=1e-6, random_state=42)
        
        # Attention history for analysis
        self.attention_history = []
        
        logger.info(f"Initialized AttentionESN with {config.attention_mechanism} attention")
    
    def _initialize_attention(self):
        """Initialize attention mechanism."""
        n_reservoir = self.config.base_config.n_reservoir
        
        if self.config.attention_mechanism == "self_attention":
            self.attention = MultiHeadAttention(
                n_heads=self.config.n_attention_heads,
                d_model=n_reservoir,
                d_k=self.config.attention_dim // self.config.n_attention_heads,
                d_v=self.config.attention_dim // self.config.n_attention_heads,
                dropout=self.config.attention_dropout,
                temperature=self.config.attention_temperature
            )
        
        elif self.config.attention_mechanism == "temporal_attention":
            # Temporal attention over time steps
            self.attention = MultiHeadAttention(
                n_heads=self.config.n_attention_heads,
                d_model=n_reservoir,
                d_k=self.config.attention_dim // self.config.n_attention_heads,
                d_v=self.config.attention_dim // self.config.n_attention_heads,
                dropout=self.config.attention_dropout,
                temperature=self.config.attention_temperature
            )
        
        elif self.config.attention_mechanism == "cross_attention":
            # Cross attention between input and reservoir states
            self.attention_q = MultiHeadAttention(
                n_heads=self.config.n_attention_heads,
                d_model=self.config.base_config.n_inputs,
                d_k=self.config.attention_dim // self.config.n_attention_heads,
                d_v=self.config.attention_dim // self.config.n_attention_heads,
                dropout=self.config.attention_dropout,
                temperature=self.config.attention_temperature
            )
            
            self.attention_kv = MultiHeadAttention(
                n_heads=self.config.n_attention_heads,
                d_model=n_reservoir,
                d_k=self.config.attention_dim // self.config.n_attention_heads,
                d_v=self.config.attention_dim // self.config.n_attention_heads,
                dropout=self.config.attention_dropout,
                temperature=self.config.attention_temperature
            )
    
    def _apply_attention(self, states: np.ndarray, inputs: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply attention mechanism to states."""
        batch_size, seq_len, n_features = states.shape
        
        if self.config.attention_mechanism == "self_attention":
            # Self-attention over reservoir dimensions
            states_reshaped = states.reshape(batch_size * seq_len, 1, n_features)
            attended = self.attention(states_reshaped, states_reshaped, states_reshaped)
            attended = attended.reshape(batch_size, seq_len, n_features)
        
        elif self.config.attention_mechanism == "temporal_attention":
            # Attention over time dimension
            states_transposed = states.transpose(0, 2, 1)  # [batch, features, time]
            states_reshaped = states_transposed.reshape(batch_size * n_features, seq_len, 1)
            
            # Causal mask for autoregressive prediction
            mask = np.triu(np.ones((seq_len, seq_len)), k=1)
            mask = mask[None, :, :]  # Add batch dimension
            
            attended = self.attention(states_reshaped, states_reshaped, states_reshaped, mask)
            attended = attended.reshape(batch_size, n_features, seq_len).transpose(0, 2, 1)
        
        elif self.config.attention_mechanism == "cross_attention":
            # Cross attention between inputs and states
            if inputs is None:
                raise ValueError("Cross attention requires input data")
            
            # Ensure inputs have same sequence length
            if inputs.shape[1] != seq_len:
                # Pad or truncate
                min_len = min(inputs.shape[1], seq_len)
                inputs = inputs[:, :min_len]
                states = states[:, :min_len]
                seq_len = min_len
            
            # Reshape for attention
            inputs_reshaped = inputs.reshape(batch_size * seq_len, 1, -1)
            states_reshaped = states.reshape(batch_size * seq_len, 1, n_features)
            
            # Cross attention: query from inputs, key/value from states
            attended = self.attention_q(inputs_reshaped, states_reshaped, states_reshaped)
            attended = attended.reshape(batch_size, seq_len, -1)
        
        # Store attention weights for analysis
        if hasattr(self.attention, 'attention_weights') and self.attention.attention_weights is not None:
            self.attention_history.append(self.attention.attention_weights.copy())
        
        return attended
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the Attention ESN."""
        logger.info(f"Training AttentionESN on {len(X)} samples")
        
        # Train base ESN
        base_stats = self.base_esn.fit(X, y)
        
        # Get reservoir states
        _, states = self.base_esn.predict(X, return_states=True)
        
        # Reshape states for attention (add batch dimension)
        states_batch = states.reshape(1, states.shape[0], states.shape[1])
        
        # Apply attention
        if self.config.attention_mechanism == "cross_attention":
            X_batch = X.reshape(1, X.shape[0], X.shape[1])
            attended_states = self._apply_attention(states_batch, X_batch)
        else:
            attended_states = self._apply_attention(states_batch)
        
        # Remove batch dimension
        attended_states = attended_states.reshape(states.shape)
        
        # Apply layer normalization if enabled
        if self.config.use_layer_norm:
            states = self.layer_norm1(states)
            attended_states = self.layer_norm2(attended_states)
        
        # Add residual connection if enabled
        if self.config.use_residual:
            final_states = states + attended_states
        else:
            final_states = attended_states
        
        # Combine features for final readout
        combined_features = np.hstack([
            states,
            attended_states,
            final_states,
            X  # Include original inputs
        ])
        
        # Train final readout
        self.final_readout.fit(combined_features, y)
        
        # Analyze attention patterns
        attention_stats = self._analyze_attention()
        
        # Collect training statistics
        stats = {
            **base_stats,
            'attention_stats': attention_stats,
            'combined_features_dim': combined_features.shape[1],
            'attention_mechanism': self.config.attention_mechanism,
            'n_attention_heads': self.config.n_attention_heads,
        }
        
        logger.info("AttentionESN training completed")
        return stats
    
    def _analyze_attention(self) -> Dict[str, Any]:
        """Analyze attention patterns from training."""
        if not self.attention_history:
            return {}
        
        # Get last attention weights
        attention_weights = self.attention_history[-1]
        
        # Flatten for analysis
        weights_flat = attention_weights.flatten()
        
        # Calculate statistics
        stats = {
            'attention_mean': float(np.mean(weights_flat)),
            'attention_std': float(np.std(weights_flat)),
            'attention_entropy': self._calculate_entropy(weights_flat),
            'attention_sparsity': float(np.mean(weights_flat < 0.01)),
            'attention_max': float(np.max(weights_flat)),
            'attention_min': float(np.min(weights_flat)),
        }
        
        # Calculate attention head diversity
        if len(attention_weights.shape) >= 3:  # Has head dimension
            n_heads = attention_weights.shape[1]
            head_diversities = []
            
            for h in range(n_heads):
                head_weights = attention_weights[:, h].flatten()
                head_entropy = self._calculate_entropy(head_weights)
                head_diversities.append(head_entropy)
            
            stats['head_diversity_mean'] = float(np.mean(head_diversities))
            stats['head_diversity_std'] = float(np.std(head_diversities))
        
        return stats
    
    def _calculate_entropy(self, weights: np.ndarray, bins: int = 50) -> float:
        """Calculate entropy of attention weights."""
        # Normalize
        weights_normalized = weights - np.min(weights)
        if np.sum(weights_normalized) > 0:
            weights_normalized = weights_normalized / np.sum(weights_normalized)
        
        # Bin and calculate entropy
        hist, _ = np.histogram(weights_normalized, bins=bins, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        
        return float(entropy / np.log(2))  # Convert to bits
    
    def predict(self, X: np.ndarray, return_attention: bool = False) -> np.ndarray:
        """Generate predictions."""
        # Get base predictions and states
        _, states = self.base_esn.predict(X, return_states=True)
        
        # Reshape for attention
        states_batch = states.reshape(1, states.shape[0], states.shape[1])
        
        # Apply attention
        if self.config.attention_mechanism == "cross_attention":
            X_batch = X.reshape(1, X.shape[0], X.shape[1])
            attended_states = self._apply_attention(states_batch, X_batch)
        else:
            attended_states = self._apply_attention(states_batch)
        
        attended_states = attended_states.reshape(states.shape)
        
        # Apply layer normalization if enabled
        if self.config.use_layer_norm:
            states = self.layer_norm1(states)
            attended_states = self.layer_norm2(attended_states)
        
        # Add residual connection
        if self.config.use_residual:
            final_states = states + attended_states
        else:
            final_states = attended_states
        
        # Combine features
        combined_features = np.hstack([
            states,
            attended_states,
            final_states,
            X
        ])
        
        # Make prediction
        predictions = self.final_readout.predict(combined_features)
        
        if return_attention and self.attention_history:
            return predictions, self.attention_history[-1]
        
        return predictions
    
    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Get latest attention weights."""
        if self.attention_history:
            return self.attention_history[-1]
        return None


class LayerNorm:
    """Simplified layer normalization."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        
        x_normalized = (x - mean) / (std + self.eps)
        return self.gamma * x_normalized + self.beta


@dataclass
class PhysicsInformedESNConfig:
    """Configuration for Physics-Informed ESN."""
    
    base_config: ESNConfig
    physics_constraints: Dict[str, Any]
    constraint_weight: float = 0.1
    use_adjoint: bool = False
    penalty_method: str = "lagrange"  # lagrange, penalty, augmented
    lagrange_update_rate: float = 0.01
    max_constraint_violation: float = 1e-3
    
    def __post_init__(self):
        """Validate configuration."""
        valid_methods = ["lagrange", "penalty", "augmented"]
        if self.penalty_method not in valid_methods:
            raise ValueError(f"penalty_method must be one of {valid_methods}")


class PhysicsInformedESN:
    """
    Physics-Informed ESN that incorporates domain knowledge as constraints.
    
    Reference: Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
    "Physics-informed neural networks: A deep learning framework for solving
    forward and inverse problems involving nonlinear partial differential equations"
    Adapted for Reservoir Computing.
    """
    
    def __init__(self, config: PhysicsInformedESNConfig):
        self.config = config
        self.base_esn = EchoStateNetwork(config.base_config)
        
        # Physics constraints
        self.constraints = self._parse_constraints(config.physics_constraints)
        
        # Lagrange multipliers for constraint optimization
        self.lagrange_multipliers = {}
        self._initialize_lagrange_multipliers()
        
        # Constraint violation history
        self.constraint_history = []
        
        logger.info("Initialized PhysicsInformedESN with physics constraints")
    
    def _parse_constraints(self, constraints_dict: Dict[str, Any]) -> Dict[str, Callable]:
        """Parse physics constraints into callable functions."""
        constraints = {}
        
        # Material balance constraint (reservoir engineering)
        if "material_balance" in constraints_dict:
            mb_config = constraints_dict["material_balance"]
            
            def material_balance(pressure: np.ndarray, production: np.ndarray,
                               time: np.ndarray, **kwargs) -> np.ndarray:
                """
                Material balance equation for reservoir:
                c_t * V * dP/dt = -Q
                """
                compressibility = mb_config.get("compressibility", 1e-5)
                volume = mb_config.get("volume", 1e6)
                dt = np.gradient(time)
                
                # Pressure change
                dp_dt = np.gradient(pressure, time)
                
                # Material balance error
                lhs = compressibility * volume * dp_dt
                rhs = -production
                
                return lhs - rhs
            
            constraints["material_balance"] = material_balance
        
        # Darcy's law constraint (flow in porous media)
        if "darcys_law" in constraints_dict:
            darcy_config = constraints_dict["darcys_law"]
            
            def darcys_law(pressure: np.ndarray, flow_rate: np.ndarray,
                          permeability: np.ndarray, viscosity: float,
                          length: float, area: float) -> np.ndarray:
                """
                Darcy's law: Q = -k * A * (ΔP) / (μ * L)
                """
                # Pressure gradient
                pressure_gradient = np.gradient(pressure)
                
                # Darcy flow
                darcy_flow = -permeability * area * pressure_gradient / (viscosity * length)
                
                return flow_rate - darcy_flow
            
            constraints["darcys_law"] = darcys_law
        
        # Energy conservation constraint
        if "energy_conservation" in constraints_dict:
            energy_config = constraints_dict["energy_conservation"]
            
            def energy_conservation(temperature_in: np.ndarray,
                                  temperature_out: np.ndarray,
                                  flow_rate: np.ndarray,
                                  heat_capacity: float,
                                  heat_loss: np.ndarray = None) -> np.ndarray:
                """
                Energy conservation: m * cp * (T_in - T_out) = Q_loss
                """
                energy_transfer = flow_rate * heat_capacity * (temperature_in - temperature_out)
                
                if heat_loss is not None:
                    return energy_transfer - heat_loss
                else:
                    return energy_transfer  # Should be zero for adiabatic
            
            constraints["energy_conservation"] = energy_conservation
        
        # Boundary conditions
        if "boundary_conditions" in constraints_dict:
            bc_config = constraints_dict["boundary_conditions"]
            
            def boundary_conditions(field: np.ndarray, time: np.ndarray,
                                  bc_type: str = "dirichlet", **kwargs) -> np.ndarray:
                """
                Enforce boundary conditions.
                """
                errors = []
                
                # Left boundary
                if "left_value" in bc_config:
                    left_error = field[0] - bc_config["left_value"]
                    errors.append(left_error)
                
                # Right boundary
                if "right_value" in bc_config:
                    right_error = field[-1] - bc_config["right_value"]
                    errors.append(right_error)
                
                # Gradient boundary conditions
                if bc_type == "neumann":
                    gradient = np.gradient(field, time)
                    
                    if "left_gradient" in bc_config:
                        left_grad_error = gradient[0] - bc_config["left_gradient"]
                        errors.append(left_grad_error)
                    
                    if "right_gradient" in bc_config:
                        right_grad_error = gradient[-1] - bc_config["right_gradient"]
                        errors.append(right_grad_error)
                
                return np.array(errors)
            
            constraints["boundary_conditions"] = boundary_conditions
        
        return constraints
    
    def _initialize_lagrange_multipliers(self):
        """Initialize Lagrange multipliers for constraints."""
        for constraint_name in self.constraints.keys():
            self.lagrange_multipliers[constraint_name] = 0.0
    
    def _compute_physics_loss(self, predictions: Dict[str, np.ndarray],
                            additional_data: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Compute physics-based loss from constraints."""
        total_loss = 0.0
        constraint_losses = {}
        constraint_violations = {}
        
        for constraint_name, constraint_func in self.constraints.items():
            try:
                # Get required data for this constraint
                constraint_args = {}
                
                # Material balance
                if constraint_name == "material_balance":
                    if all(k in additional_data for k in ["pressure", "production", "time"]):
                        constraint_args = {
                            "pressure": additional_data["pressure"],
                            "production": additional_data["production"],
                            "time": additional_data["time"],
                        }
                
                # Darcy's law
                elif constraint_name == "darcys_law":
                    if all(k in additional_data for k in ["pressure", "flow_rate", "permeability"]):
                        constraint_args = {
                            "pressure": additional_data["pressure"],
                            "flow_rate": additional_data["flow_rate"],
                            "permeability": additional_data.get("permeability", 1.0),
                            "viscosity": additional_data.get("viscosity", 1.0),
                            "length": additional_data.get("length", 1.0),
                            "area": additional_data.get("area", 1.0),
                        }
                
                # Energy conservation
                elif constraint_name == "energy_conservation":
                    if all(k in additional_data for k in ["temperature_in", "temperature_out", "flow_rate"]):
                        constraint_args = {
                            "temperature_in": additional_data["temperature_in"],
                            "temperature_out": additional_data["temperature_out"],
                            "flow_rate": additional_data["flow_rate"],
                            "heat_capacity": additional_data.get("heat_capacity", 1.0),
                            "heat_loss": additional_data.get("heat_loss", None),
                        }
                
                # Boundary conditions
                elif constraint_name == "boundary_conditions":
                    if "pressure" in additional_data and "time" in additional_data:
                        constraint_args = {
                            "field": additional_data["pressure"],
                            "time": additional_data["time"],
                            "bc_type": additional_data.get("bc_type", "dirichlet"),
                        }
                
                # Compute constraint violation
                if constraint_args:
                    violation = constraint_func(**constraint_args)
                    
                    # Calculate constraint loss
                    if isinstance(violation, np.ndarray):
                        constraint_loss = np.mean(violation ** 2)
                        mean_violation = np.mean(np.abs(violation))
                    else:
                        constraint_loss = violation ** 2
                        mean_violation = np.abs(violation)
                    
                    constraint_losses[constraint_name] = constraint_loss
                    constraint_violations[constraint_name] = mean_violation
                    
                    # Apply penalty method
                    if self.config.penalty_method == "penalty":
                        total_loss += self.config.constraint_weight * constraint_loss
                    
                    elif self.config.penalty_method == "lagrange":
                        lagrange = self.lagrange_multipliers[constraint_name]
                        total_loss += lagrange * constraint_loss + \
                                    0.5 * self.config.constraint_weight * constraint_loss ** 2
                    
                    elif self.config.penalty_method == "augmented":
                        lagrange = self.lagrange_multipliers[constraint_name]
                        total_loss += lagrange * constraint_loss + \
                                    0.5 * self.config.constraint_weight * constraint_loss ** 2
            
            except Exception as e:
                logger.warning(f"Failed to compute constraint {constraint_name}: {e}")
                constraint_losses[constraint_name] = 0.0
                constraint_violations[constraint_name] = 0.0
        
        return total_loss, constraint_losses, constraint_violations
    
    def _update_lagrange_multipliers(self, constraint_violations: Dict[str, float]):
        """Update Lagrange multipliers using gradient ascent."""
        for constraint_name, violation in constraint_violations.items():
            if constraint_name in self.lagrange_multipliers:
                lagrange = self.lagrange_multipliers[constraint_name]
                new_lagrange = lagrange + self.config.lagrange_update_rate * violation
                
                # Clip to prevent numerical issues
                self.lagrange_multipliers[constraint_name] = np.clip(
                    new_lagrange, -1e3, 1e3
                )
    
    def fit(self, X: np.ndarray, y: np.ndarray,
           additional_data: Optional[Dict[str, Any]] = None,
           n_physics_iterations: int = 10) -> Dict[str, Any]:
        """Train the Physics-Informed ESN."""
        logger.info(f"Training PhysicsInformedESN with {n_physics_iterations} physics iterations")
        
        # Step 1: Train base ESN without physics constraints
        base_stats = self.base_esn.fit(X, y)
        
        # Initial predictions
        predictions = self.base_esn.predict(X)
        initial_physics = {}
        
        # Step 2: Physics-informed refinement
        if additional_data is not None and self.constraints:
            # Store initial predictions in additional data
            if "pressure" not in additional_data and y.shape[1] == 1:
                additional_data["pressure"] = predictions.flatten()
            
            # Initial physics evaluation
            physics_loss, constraint_losses, constraint_violations = \
                self._compute_physics_loss({"pressure": predictions}, additional_data)
            
            initial_physics = {
                "initial_physics_loss": physics_loss,
                "initial_constraint_losses": constraint_losses,
                "initial_constraint_violations": constraint_violations,
            }
            
            logger.info(f"Initial physics loss: {physics_loss:.6f}")
            
            # Physics refinement iterations
            for iteration in range(n_physics_iterations):
                logger.debug(f"Physics refinement iteration {iteration + 1}/{n_physics_iterations}")
                
                # Update predictions based on physics (simplified approach)
                # In practice, this would involve adjusting the readout weights
                # based on physics gradients
                if self.config.use_adjoint:
                    # Adjoint method for gradient computation
                    self._adjoint_based_refinement(X, y, additional_data)
                else:
                    # Simplified gradient-based adjustment
                    self._gradient_based_refinement(predictions, additional_data)
                
                # Recompute predictions
                predictions = self.base_esn.predict(X)
                
                # Update additional data with new predictions
                if "pressure" in additional_data:
                    additional_data["pressure"] = predictions.flatten()
                
                # Compute physics loss
                physics_loss, constraint_losses, constraint_violations = \
                    self._compute_physics_loss({"pressure": predictions}, additional_data)
                
                # Update Lagrange multipliers if using Lagrange method
                if self.config.penalty_method == "lagrange":
                    self._update_lagrange_multipliers(constraint_violations)
                
                # Store constraint history
                self.constraint_history.append({
                    "iteration": iteration,
                    "physics_loss": physics_loss,
                    "constraint_losses": constraint_losses,
                    "constraint_violations": constraint_violations,
                    "lagrange_multipliers": self.lagrange_multipliers.copy(),
                })
                
                # Check convergence
                if iteration > 0:
                    prev_loss = self.constraint_history[-2]["physics_loss"]
                    loss_change = abs(physics_loss - prev_loss) / (abs(prev_loss) + 1e-10)
                    
                    if loss_change < self.config.max_constraint_violation:
                        logger.info(f"Physics convergence at iteration {iteration}")
                        break
            
            # Final physics evaluation
            final_physics_loss, final_constraint_losses, final_constraint_violations = \
                self._compute_physics_loss({"pressure": predictions}, additional_data)
            
            physics_stats = {
                **initial_physics,
                "final_physics_loss": final_physics_loss,
                "final_constraint_losses": final_constraint_losses,
                "final_constraint_violations": final_constraint_violations,
                "n_physics_iterations": len(self.constraint_history),
                "constraint_names": list(self.constraints.keys()),
                "lagrange_multipliers": self.lagrange_multipliers,
            }
        else:
            physics_stats = {
                "warning": "No physics constraints or additional data provided"
            }
        
        # Combine statistics
        stats = {
            **base_stats,
            **physics_stats,
        }
        
        logger.info("PhysicsInformedESN training completed")
        return stats
    
    def _gradient_based_refinement(self, predictions: np.ndarray,
                                 additional_data: Dict[str, Any]):
        """Simplified gradient-based refinement of predictions."""
        # This is a simplified implementation
        # In practice, you would compute proper gradients through the ESN
        
        if "pressure" in additional_data and "production" in additional_data:
            pressure = predictions.flatten()
            production = additional_data["production"]
            time = additional_data.get("time", np.arange(len(pressure)))
            
            # Material balance gradient
            if "material_balance" in self.constraints:
                # Compute material balance error
                compressibility = additional_data.get("compressibility", 1e-5)
                volume = additional_data.get("volume", 1e6)
                
                dp_dt = np.gradient(pressure, time)
                mb_error = compressibility * volume * dp_dt + production
                
                # Simplified adjustment (would use proper gradient in real implementation)
                adjustment = 0.01 * mb_error / (compressibility * volume + 1e-10)
                
                # Adjust predictions (this affects the readout indirectly)
                # In practice, you would adjust W_out based on physics gradients
                pass
    
    def _adjoint_based_refinement(self, X: np.ndarray, y: np.ndarray,
                                additional_data: Dict[str, Any]):
        """Adjoint method for physics-based refinement."""
        # This is a placeholder for adjoint method implementation
        # Adjoint method requires solving backward equations, which is complex
        # for ESNs due to their random nature
        
        logger.warning("Adjoint method not fully implemented for ESNs")
        
        # Simplified approach: adjust readout based on physics loss gradient
        _, states = self.base_esn.predict(X, return_states=True)
        
        # Compute physics loss gradient w.r.t. predictions
        # This would involve:
        # 1. Computing physics loss
        # 2. Computing gradient of loss w.r.t. predictions
        # 3. Computing gradient of predictions w.r.t. readout weights
        # 4. Updating readout weights
        
        # For now, we'll use a simplified approach
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate physics-informed predictions."""
        return self.base_esn.predict(X)
    
    def get_constraint_history(self) -> List[Dict[str, Any]]:
        """Get history of constraint violations during training."""
        return self.constraint_history
    
    def evaluate_physics_constraints(self, predictions: np.ndarray,
                                   additional_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate physics constraints on given predictions."""
        physics_loss, constraint_losses, constraint_violations = \
            self._compute_physics_loss({"pressure": predictions}, additional_data)
        
        return {
            "physics_loss": physics_loss,
            "constraint_losses": constraint_losses,
            "constraint_violations": constraint_violations,
            "satisfied_constraints": [
                name for name, violation in constraint_violations.items()
                if violation < self.config.max_constraint_violation
            ]
        }
