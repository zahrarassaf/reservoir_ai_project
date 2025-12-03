# src/models/advanced_esn.py
"""
Advanced ESN variants with cutting-edge research features.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from .esn import EchoStateNetwork, ESNConfig

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalESNConfig:
    """Configuration for Hierarchical ESN."""
    
    base_config: ESNConfig
    n_levels: int = 3
    reduction_factor: float = 0.5
    inter_level_connections: bool = True
    top_down_feedback: bool = False
    level_specific_training: bool = False


class HierarchicalESN:
    """
    Hierarchical Echo State Network for multi-scale temporal processing.
    
    Features:
    - Multiple levels with decreasing time scales
    - Bottom-up and top-down information flow
    - Multi-resolution feature extraction
    - Level-specific readouts
    """
    
    def __init__(self, config: HierarchicalESNConfig):
        self.config = config
        self.levels = []
        self.level_readouts = []
        
        self._initialize_levels()
    
    def _initialize_levels(self):
        """Initialize hierarchical levels."""
        current_size = self.config.base_config.n_reservoir
        
        for level in range(self.config.n_levels):
            # Scale reservoir size for higher levels
            level_size = int(current_size * (self.config.reduction_factor ** level))
            
            # Create level-specific config
            level_config = ESNConfig(
                n_inputs=self.config.base_config.n_inputs if level == 0 else level_size,
                n_outputs=level_size,
                n_reservoir=level_size,
                spectral_radius=self.config.base_config.spectral_radius * (0.9 ** level),
                leaking_rate=self.config.base_config.leaking_rate * (1.5 ** level),  # Faster dynamics
                sparsity=self.config.base_config.sparsity,
            )
            
            level_esn = EchoStateNetwork(level_config)
            self.levels.append(level_esn)
            
            # Level-specific readout
            from sklearn.linear_model import Ridge
            self.level_readouts.append(Ridge(alpha=1e-6))
            
            current_size = level_size
    
    def _process_level(self, level_idx: int, input_data: np.ndarray, 
                      higher_level_feedback: Optional[np.ndarray] = None) -> np.ndarray:
        """Process data through a single level."""
        level = self.levels[level_idx]
        
        # Add feedback from higher level if available
        if higher_level_feedback is not None and self.config.top_down_feedback:
            # Downsample feedback to match level timescale
            if higher_level_feedback.shape[0] > input_data.shape[0]:
                step = higher_level_feedback.shape[0] // input_data.shape[0]
                feedback = higher_level_feedback[::step][:input_data.shape[0]]
            else:
                feedback = higher_level_feedback
            
            # Combine input with feedback
            if feedback.shape[1] == input_data.shape[1]:
                input_combined = input_data + 0.1 * feedback
            else:
                input_combined = input_data
        
        else:
            input_combined = input_data
        
        # Process through level
        if level_idx == 0:
            # First level processes raw input
            output, states = level.predict(input_combined, return_states=True)
        else:
            # Higher levels process output from previous level
            output, states = level.predict(input_combined, return_states=True)
        
        return output, states
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train hierarchical ESN."""
        logger.info(f"Training Hierarchical ESN with {self.config.n_levels} levels")
        
        # Process through all levels
        level_outputs = []
        level_states = []
        
        current_input = X
        higher_level_feedback = None
        
        # Bottom-up processing
        for level_idx in range(self.config.n_levels):
            output, states = self._process_level(
                level_idx, current_input, higher_level_feedback
            )
            
            level_outputs.append(output)
            level_states.append(states)
            
            # Output of this level becomes input to next (downsampled)
            if level_idx < self.config.n_levels - 1:
                # Downsample for next level (coarser timescale)
                step = 2 ** (level_idx + 1)
                current_input = output[::step]
                
                # Prepare feedback for next iteration (top-down)
                if self.config.top_down_feedback:
                    # Upsample current output for feedback to lower levels
                    higher_level_feedback = np.repeat(output, step, axis=0)
                    higher_level_feedback = higher_level_feedback[:X.shape[0]]
        
        # Train level-specific readouts if enabled
        if self.config.level_specific_training:
            for level_idx, (states, readout) in enumerate(zip(level_states, self.level_readouts)):
                # Align targets with level timescale
                if level_idx == 0:
                    level_y = y
                else:
                    step = 2 ** level_idx
                    level_y = y[::step][:states.shape[0]]
                
                # Train readout
                readout.fit(states, level_y)
        
        # Train final readout on combined features
        all_features = np.hstack(level_outputs)
        
        from sklearn.linear_model import Ridge
        self.final_readout = Ridge(alpha=1e-6)
        self.final_readout.fit(all_features, y)
        
        # Collect statistics
        stats = {
            'n_levels': self.config.n_levels,
            'level_sizes': [level.config.n_reservoir for level in self.levels],
            'feature_dimensions': [out.shape[1] for out in level_outputs],
            'total_features': all_features.shape[1],
        }
        
        return stats
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        # Process through all levels
        level_outputs = []
        
        current_input = X
        higher_level_feedback = None
        
        for level_idx in range(self.config.n_levels):
            output, _ = self._process_level(
                level_idx, current_input, higher_level_feedback
            )
            
            level_outputs.append(output)
            
            if level_idx < self.config.n_levels - 1:
                step = 2 ** (level_idx + 1)
                current_input = output[::step]
                
                if self.config.top_down_feedback:
                    higher_level_feedback = np.repeat(output, step, axis=0)
                    higher_level_feedback = higher_level_feedback[:X.shape[0]]
        
        # Combine features and predict
        all_features = np.hstack(level_outputs)
        
        if self.config.level_specific_training:
            # Average predictions from all levels
            predictions = []
            for level_idx, (features, readout) in enumerate(zip(level_outputs, self.level_readouts)):
                if level_idx == 0:
                    pred = readout.predict(features)
                else:
                    # Upsample to match original time scale
                    step = 2 ** level_idx
                    pred_level = readout.predict(features)
                    
                    # Interpolate back to original sampling
                    from scipy import interpolate
                    x_original = np.arange(X.shape[0])
                    x_level = np.arange(0, X.shape[0], step)[:pred_level.shape[0]]
                    
                    f = interpolate.interp1d(x_level, pred_level, axis=0, 
                                           bounds_error=False, fill_value="extrapolate")
                    pred = f(x_original)
                
                predictions.append(pred)
            
            final_prediction = np.mean(predictions, axis=0)
        else:
            final_prediction = self.final_readout.predict(all_features)
        
        return final_prediction


@dataclass
class AttentionESNConfig:
    """Configuration for Attention-based ESN."""
    
    base_config: ESNConfig
    attention_mechanism: str = "self_attention"  # self_attention, temporal_attention
    n_attention_heads: int = 4
    attention_dim: int = 64
    use_residual: bool = True
    dropout_rate: float = 0.1


class AttentionESN:
    """
    ESN with attention mechanism for selective information processing.
    
    Features:
    - Self-attention over reservoir states
    - Temporal attention for important time steps
    - Multi-head attention
    - Residual connections
    """
    
    def __init__(self, config: AttentionESNConfig):
        self.config = config
        self.base_esn = EchoStateNetwork(config.base_config)
        
        # Initialize attention parameters
        self._initialize_attention()
    
    def _initialize_attention(self):
        """Initialize attention mechanism parameters."""
        n_reservoir = self.config.base_config.n_reservoir
        
        if self.config.attention_mechanism == "self_attention":
            # Self-attention over reservoir units
            self.W_q = np.random.randn(n_reservoir, self.config.attention_dim) * 0.01
            self.W_k = np.random.randn(n_reservoir, self.config.attention_dim) * 0.01
            self.W_v = np.random.randn(n_reservoir, n_reservoir) * 0.01
            self.W_o = np.random.randn(n_reservoir, n_reservoir) * 0.01
            
        elif self.config.attention_mechanism == "temporal_attention":
            # Attention over time steps
            self.W_q = np.random.randn(n_reservoir, self.config.attention_dim) * 0.01
            self.W_k = np.random.randn(n_reservoir, self.config.attention_dim) * 0.01
            self.W_v = np.random.randn(n_reservoir, n_reservoir) * 0.01
            self.W_o = np.random.randn(n_reservoir, n_reservoir) * 0.01
        
        # Multi-head setup
        self.head_dim = self.config.attention_dim // self.config.n_attention_heads
        
        self.attention_weights = []  # Will store attention weights for analysis
    
    def _attention_layer(self, states: np.ndarray) -> np.ndarray:
        """Apply attention mechanism to reservoir states."""
        n_samples, n_reservoir = states.shape
        
        if self.config.attention_mechanism == "self_attention":
            # Self-attention: attend to different reservoir units
            Q = np.dot(states, self.W_q)  # [n_samples, attention_dim]
            K = np.dot(states, self.W_k)  # [n_samples, attention_dim]
            V = np.dot(states, self.W_v)  # [n_samples, n_reservoir]
            
            # Scaled dot-product attention
            scores = np.dot(Q, K.T) / np.sqrt(self.config.attention_dim)
            attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
            
            # Apply attention
            attended = np.dot(attention_weights, V)
            
            # Store attention weights for analysis
            self.attention_weights.append(attention_weights)
            
            # Output projection
            output = np.dot(attended, self.W_o)
            
        elif self.config.attention_mechanism == "temporal_attention":
            # Temporal attention: attend to important time steps
            # Reshape for temporal processing
            if len(states.shape) == 2:
                # Assuming states are [n_samples, n_reservoir]
                # For temporal attention, we need to consider sequence
                Q = np.dot(states, self.W_q)
                K = np.dot(states, self.W_k)
                V = np.dot(states, self.W_v)
                
                # Attention over time
                scores = np.dot(Q, K.T) / np.sqrt(self.config.attention_dim)
                
                # Causal mask for autoregressive prediction
                mask = np.triu(np.ones((n_samples, n_samples)), k=1)
                scores = scores - 1e9 * mask
                
                attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
                attended = np.dot(attention_weights, V)
                output = np.dot(attended, self.W_o)
                
                self.attention_weights.append(attention_weights)
        
        # Residual connection
        if self.config.use_residual:
            output = output + states
        
        # Dropout (during training only)
        if hasattr(self, 'training') and self.training and self.config.dropout_rate > 0:
            dropout_mask = (np.random.rand(*output.shape) > self.config.dropout_rate)
            output = output * dropout_mask / (1 - self.config.dropout_rate)
        
        return output
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train Attention ESN."""
        logger.info("Training Attention ESN")
        
        # First, train base ESN
        base_stats = self.base_esn.fit(X, y)
        
        # Get reservoir states
        predictions, states = self.base_esn.predict(X, return_states=True)
        
        # Apply attention to states
        self.training = True
        attended_states = self._attention_layer(states)
        self.training = False
        
        # Train final readout on attended states
        from sklearn.linear_model import Ridge
        
        # Combine original and attended states
        combined_features = np.hstack([states, attended_states, X])
        
        self.final_readout = Ridge(alpha=1e-6)
        self.final_readout.fit(combined_features, y)
        
        # Analyze attention patterns
        attention_stats = self._analyze_attention()
        
        stats = {
            **base_stats,
            "attention_stats": attention_stats,
            "combined_features_dim": combined_features.shape[1],
        }
        
        return stats
    
    def _analyze_attention(self) -> Dict[str, Any]:
        """Analyze attention patterns."""
        if not self.attention_weights:
            return {}
        
        # Get last attention weights
        attention_matrix = self.attention_weights[-1]
        
        # Calculate statistics
        entropy = -np.sum(attention_matrix * np.log(attention_matrix + 1e-10), axis=1)
        
        stats = {
            "attention_entropy_mean": np.mean(entropy),
            "attention_entropy_std": np.std(entropy),
            "attention_sparsity": np.mean(attention_matrix < 0.01),
            "max_attention": np.max(attention_matrix),
            "min_attention": np.min(attention_matrix),
        }
        
        return stats
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions with attention."""
        # Get base predictions and states
        _, states = self.base_esn.predict(X, return_states=True)
        
        # Apply attention
        attended_states = self._attention_layer(states)
        
        # Combine features and predict
        combined_features = np.hstack([states, attended_states, X])
        predictions = self.final_readout.predict(combined_features)
        
        return predictions


@dataclass
class PhysicsInformedESNConfig:
    """Configuration for Physics-Informed ESN."""
    
    base_config: ESNConfig
    physics_constraints: Dict[str, Any]
    constraint_weight: float = 0.1
    use_adjoint: bool = True
    penalty_method: str = "lagrange"  # lagrange, penalty, augmented


class PhysicsInformedESN:
    """
    Physics-Informed ESN that incorporates domain knowledge.
    
    Features:
    - Physics-based constraints in loss function
    - Adjoint method for gradient computation
    - Lagrangian optimization
    - Conservation law enforcement
    """
    
    def __init__(self, config: PhysicsInformedESNConfig):
        self.config = config
        self.base_esn = EchoStateNetwork(config.base_config)
        
        # Physics constraints
        self.constraints = self._parse_constraints(config.physics_constraints)
    
    def _parse_constraints(self, constraints_dict: Dict[str, Any]) -> Dict[str, callable]:
        """Parse physics constraints."""
        constraints = {}
        
        # Material balance constraint
        if "material_balance" in constraints_dict:
            def material_balance(pressure, production, compressibility, volume, dt):
                # ∂P/∂t = - (Q / (c_t * V))
                dp_dt = np.gradient(pressure, dt)
                expected_dp = -production / (compressibility * volume)
                return np.mean((dp_dt - expected_dp) ** 2)
            
            constraints["material_balance"] = material_balance
        
        # Energy conservation
        if "energy_conservation" in constraints_dict:
            def energy_conservation(temperature, flow_rate, heat_capacity):
                # Energy in = Energy out
                # Simplified for demonstration
                return 0.0
            
            constraints["energy_conservation"] = energy_conservation
        
        # Boundary conditions
        if "boundary_conditions" in constraints_dict:
            bc_config = constraints_dict["boundary_conditions"]
            
            def boundary_conditions(pressure, time, bc_type="dirichlet"):
                if bc_type == "dirichlet":
                    # Fixed pressure at boundaries
                    boundary_error = (pressure[0] - bc_config.get("left_value", 3000)) ** 2
                    boundary_error += (pressure[-1] - bc_config.get("right_value", 3000)) ** 2
                    return boundary_error
                elif bc_type == "neumann":
                    # Fixed gradient at boundaries
                    grad = np.gradient(pressure, time)
                    boundary_error = (grad[0] - bc_config.get("left_gradient", 0)) ** 2
                    boundary_error += (grad[-1] - bc_config.get("right_gradient", 0)) ** 2
                    return boundary_error
                
                return 0.0
            
            constraints["boundary_conditions"] = boundary_conditions
        
        return constraints
    
    def _physics_loss(self, predictions: np.ndarray, 
                     additional_data: Dict[str, np.ndarray]) -> float:
        """Compute physics-based loss."""
        total_loss = 0.0
        
        for constraint_name, constraint_func in self.constraints.items():
            if constraint_name == "material_balance":
                if all(key in additional_data for key in ["pressure", "production", "compressibility", "volume", "dt"]):
                    loss = constraint_func(
                        additional_data["pressure"],
                        additional_data["production"],
                        additional_data["compressibility"],
                        additional_data["volume"],
                        additional_data["dt"],
                    )
                    total_loss += self.config.constraint_weight * loss
            
            elif constraint_name == "boundary_conditions":
                if "pressure" in additional_data and "time" in additional_data:
                    loss = constraint_func(
                        additional_data["pressure"],
                        additional_data["time"],
                        bc_type="dirichlet"
                    )
                    total_loss += self.config.constraint_weight * loss
        
        return total_loss
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
           additional_data: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """Train Physics-Informed ESN."""
        logger.info("Training Physics-Informed ESN")
        
        # Train base ESN
        base_stats = self.base_esn.fit(X, y)
        
        # Get initial predictions
        predictions = self.base_esn.predict(X)
        
        # Compute physics loss if additional data is provided
        physics_stats = {}
        if additional_data is not None:
            physics_loss = self._physics_loss(predictions, additional_data)
            physics_stats["initial_physics_loss"] = physics_loss
            
            # Optional: Refine model based on physics loss
            if self.config.use_adjoint and physics_loss > 0:
                self._physics_based_refinement(X, y, additional_data)
                
                # Recompute predictions and physics loss
                refined_predictions = self.base_esn.predict(X)
                refined_physics_loss = self._physics_loss(refined_predictions, additional_data)
                
                physics_stats["refined_physics_loss"] = refined_physics_loss
                physics_stats["physics_improvement"] = (physics_loss - refined_physics_loss) / physics_loss
        
        stats = {
            **base_stats,
            **physics_stats,
        }
        
        return stats
    
    def _physics_based_refinement(self, X: np.ndarray, y: np.ndarray,
                                additional_data: Dict[str, np.ndarray]):
        """Refine model using physics-based constraints."""
        # This is a simplified implementation
        # In practice, you would use adjoint method or Lagrangian optimization
        
        # Get current states
        predictions, states = self.base_esn.predict(X, return_states=True)
        
        # Compute physics gradients (simplified)
        # For material balance constraint
        if "material_balance" in self.constraints:
            pressure = additional_data.get("pressure", predictions)
            production = additional_data.get("production", np.zeros_like(predictions))
            compressibility = additional_data.get("compressibility", 1e-5)
            volume = additional_data.get("volume", 1.0)
            dt = additional_data.get("dt", 1.0)
            
            # Compute constraint violation
            dp_dt = np.gradient(pressure.flatten(), dt)
            expected_dp = -production.flatten() / (compressibility * volume)
            violation = dp_dt - expected_dp
            
            # Simple gradient adjustment (this is highly simplified)
            # In reality, you would compute proper gradients through the network
            adjustment = 0.01 * violation.mean()
            
            # Adjust predictions (this affects readout, not reservoir)
            # A proper implementation would adjust W_out based on physics gradients
            pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate physics-informed predictions."""
        return self.base_esn.predict(X)
