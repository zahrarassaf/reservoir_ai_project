"""
Industrial-grade Echo State Network implementation for reservoir simulation.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from scipy.sparse import random, diags, csr_matrix
from scipy.sparse.linalg import eigs
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


@dataclass
class ESNConfig:
    """Configuration for Echo State Network."""
    
    n_inputs: int
    n_outputs: int
    n_reservoir: int = 1000
    spectral_radius: float = 0.95
    sparsity: float = 0.1
    leaking_rate: float = 0.3
    regularization: float = 1e-6
    input_scaling: float = 1.0
    bias_scaling: float = 0.1
    reservoir_scaling: float = 1.0
    teacher_forcing: bool = False
    teacher_scaling: float = 1.0
    feedback_scaling: float = 0.0
    noise_level: float = 1e-6
    warmup_steps: int = 100
    random_state: int = 42
    
    # Advanced parameters
    reservoir_connectivity: str = "uniform"  # uniform, small_world, scale_free
    activation_function: str = "tanh"  # tanh, relu, sigmoid
    weight_distribution: str = "uniform"  # uniform, normal
    ridge_solver: str = "cholesky"  # cholesky, svd, lsqr
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.n_reservoir > 0, "Reservoir size must be positive"
        assert 0 < self.spectral_radius < 2, "Spectral radius must be between 0 and 2"
        assert 0 <= self.sparsity <= 1, "Sparsity must be between 0 and 1"
        assert 0 < self.leaking_rate <= 1, "Leaking rate must be between 0 and 1"
        assert self.regularization >= 0, "Regularization must be non-negative"
        
        if self.reservoir_connectivity not in ["uniform", "small_world", "scale_free"]:
            raise ValueError(f"Invalid connectivity: {self.reservoir_connectivity}")
        
        if self.activation_function not in ["tanh", "relu", "sigmoid"]:
            raise ValueError(f"Invalid activation: {self.activation_function}")


class EchoStateNetwork:
    """
    Industrial-grade Echo State Network with advanced features.
    
    Features:
    - Multiple reservoir connectivity patterns
    - Advanced activation functions
    - Teacher forcing
    - Noise injection
    - State normalization
    - Multiple readout strategies
    - Serialization support
    """
    
    def __init__(self, config: ESNConfig):
        """
        Initialize ESN with given configuration.
        
        Args:
            config: ESN configuration object
        """
        self.config = config
        self.config.validate()
        
        self._rng = np.random.RandomState(config.random_state)
        
        # Initialize components
        self.W_in = None  # Input weights
        self.W_res = None  # Reservoir weights
        self.W_fb = None  # Feedback weights
        self.W_out = None  # Output weights
        self.bias = None  # Reservoir bias
        
        # Scalers
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.state_scaler = StandardScaler()
        
        # Readout model
        self.readout = Ridge(
            alpha=config.regularization,
            solver=config.ridge_solver,
            random_state=config.random_state,
            fit_intercept=True
        )
        
        # State tracking
        self.states = None
        self.last_state = None
        
        # Statistics
        self.training_stats = {}
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"ESN initialized with config: {config}")
    
    def _initialize_weights(self) -> None:
        """Initialize all network weights according to configuration."""
        config = self.config
        
        # Input weights
        self.W_in = self._create_input_weights()
        
        # Reservoir weights
        self.W_res = self._create_reservoir_weights()
        
        # Feedback weights (if teacher forcing is enabled)
        if config.teacher_forcing or config.feedback_scaling > 0:
            self.W_fb = self._create_feedback_weights()
        
        # Reservoir bias
        self.bias = config.bias_scaling * (self._rng.rand(config.n_reservoir) - 0.5)
    
    def _create_input_weights(self) -> np.ndarray:
        """Create input weight matrix."""
        config = self.config
        
        if config.weight_distribution == "uniform":
            W = self._rng.uniform(
                low=-config.input_scaling,
                high=config.input_scaling,
                size=(config.n_reservoir, config.n_inputs)
            )
        elif config.weight_distribution == "normal":
            W = self._rng.normal(
                scale=config.input_scaling / np.sqrt(config.n_inputs),
                size=(config.n_reservoir, config.n_inputs)
            )
        else:
            raise ValueError(f"Unknown distribution: {config.weight_distribution}")
        
        return W
    
    def _create_reservoir_weights(self) -> np.ndarray:
        """Create reservoir weight matrix with specified connectivity."""
        config = self.config
        
        if config.reservoir_connectivity == "uniform":
            W = self._create_uniform_weights()
        elif config.reservoir_connectivity == "small_world":
            W = self._create_small_world_weights()
        elif config.reservoir_connectivity == "scale_free":
            W = self._create_scale_free_weights()
        else:
            raise ValueError(f"Unknown connectivity: {config.reservoir_connectivity}")
        
        # Scale by spectral radius
        eigenvalues = eigs(W, k=1, return_eigenvectors=False)
        max_eigenvalue = np.abs(eigenvalues[0])
        if max_eigenvalue > 0:
            W *= config.spectral_radius / max_eigenvalue
        
        return W
    
    def _create_uniform_weights(self) -> np.ndarray:
        """Create uniformly connected reservoir weights."""
        config = self.config
        
        # Create sparse matrix
        W_sparse = random(
            config.n_reservoir,
            config.n_reservoir,
            density=config.sparsity,
            random_state=config.random_state,
            data_rvs=lambda size: self._rng.uniform(
                low=-config.reservoir_scaling,
                high=config.reservoir_scaling,
                size=size
            )
        )
        
        return W_sparse.toarray()
    
    def _create_small_world_weights(self) -> np.ndarray:
        """Create small-world network reservoir weights."""
        config = self.config
        
        # Simplified Watts-Strogatz small-world network
        n = config.n_reservoir
        k = int(config.sparsity * n / 2)  # Average degree
        p = 0.1  # Rewiring probability
        
        W = np.zeros((n, n))
        
        # Create ring lattice
        for i in range(n):
            for j in range(1, k // 2 + 1):
                W[i, (i + j) % n] = self._rng.uniform(-1, 1)
                W[i, (i - j) % n] = self._rng.uniform(-1, 1)
        
        # Rewire edges with probability p
        for i in range(n):
            for j in range(n):
                if W[i, j] != 0 and self._rng.random() < p:
                    # Find new target
                    new_j = self._rng.randint(0, n)
                    while new_j == i or W[i, new_j] != 0:
                        new_j = self._rng.randint(0, n)
                    
                    W[i, j] = 0
                    W[i, new_j] = self._rng.uniform(-1, 1)
        
        return config.reservoir_scaling * W
    
    def _create_scale_free_weights(self) -> np.ndarray:
        """Create scale-free network reservoir weights."""
        config = self.config
        
        # Simplified Barabási-Albert scale-free network
        n = config.n_reservoir
        m = int(config.sparsity * n)  # Edges to attach from new node
        
        W = np.zeros((n, n))
        
        # Start with fully connected m nodes
        for i in range(m):
            for j in range(i + 1, m):
                weight = self._rng.uniform(-1, 1)
                W[i, j] = weight
                W[j, i] = weight
        
        # Add remaining nodes
        for i in range(m, n):
            # Calculate probabilities proportional to degree
            degrees = np.sum(W[:i, :i] != 0, axis=1)
            probabilities = degrees / np.sum(degrees)
            
            # Select m nodes to connect to
            targets = self._rng.choice(
                i, size=m, replace=False, p=probabilities
            )
            
            for target in targets:
                weight = self._rng.uniform(-1, 1)
                W[i, target] = weight
                W[target, i] = weight
        
        return config.reservoir_scaling * W
    
    def _create_feedback_weights(self) -> np.ndarray:
        """Create feedback weight matrix."""
        config = self.config
        
        W = self._rng.uniform(
            low=-config.feedback_scaling,
            high=config.feedback_scaling,
            size=(config.n_reservoir, config.n_outputs)
        )
        
        return W
    
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        config = self.config
        
        if config.activation_function == "tanh":
            return np.tanh(x)
        elif config.activation_function == "relu":
            return np.maximum(0, x)
        elif config.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError(f"Unknown activation: {config.activation_function}")
    
    def _compute_state(
        self,
        input_vec: np.ndarray,
        prev_state: np.ndarray,
        feedback: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute reservoir state update.
        
        Args:
            input_vec: Current input vector
            prev_state: Previous reservoir state
            feedback: Optional feedback from previous output
            
        Returns:
            Updated reservoir state
        """
        config = self.config
        
        # Base computation
        state_update = (
            np.dot(self.W_res, prev_state) +
            np.dot(self.W_in, input_vec) +
            self.bias
        )
        
        # Add feedback if provided
        if feedback is not None and self.W_fb is not None:
            state_update += np.dot(self.W_fb, feedback)
        
        # Add noise
        if config.noise_level > 0:
            noise = config.noise_level * self._rng.randn(config.n_reservoir)
            state_update += noise
        
        # Apply activation
        activated = self._activation(state_update)
        
        # Leaky integration
        new_state = (1 - config.leaking_rate) * prev_state + \
                    config.leaking_rate * activated
        
        return new_state
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Train the ESN.
        
        Args:
            X: Input data of shape (n_samples, n_inputs)
            y: Target data of shape (n_samples, n_outputs)
            validation_data: Optional validation data
            
        Returns:
            Dictionary with training statistics
        """
        logger.info(f"Training ESN with {len(X)} samples")
        
        # Store original shapes
        n_samples, n_inputs = X.shape
        _, n_outputs = y.shape
        
        assert n_inputs == self.config.n_inputs, \
            f"Expected {self.config.n_inputs} inputs, got {n_inputs}"
        assert n_outputs == self.config.n_outputs, \
            f"Expected {self.config.n_outputs} outputs, got {n_outputs}"
        
        # Scale data
        X_scaled = self.input_scaler.fit_transform(X)
        y_scaled = self.output_scaler.fit_transform(y)
        
        # Initialize state collection
        self.states = np.zeros((n_samples, self.config.n_reservoir))
        state = np.zeros(self.config.n_reservoir)
        
        # Warmup phase
        logger.debug(f"Warmup phase: {self.config.warmup_steps} steps")
        for t in range(self.config.warmup_steps):
            if t < n_samples:
                feedback = y_scaled[t] if self.config.teacher_forcing else None
                state = self._compute_state(X_scaled[t], state, feedback)
        
        # Collect states
        logger.debug("Collecting reservoir states")
        for t in range(self.config.warmup_steps, n_samples):
            if self.config.teacher_forcing:
                feedback = y_scaled[t]
            elif self.W_fb is not None and t > self.config.warmup_steps:
                # Use previous prediction as feedback
                features = np.hstack([state, X_scaled[t-1]])
                pred_scaled = self.readout.predict(features.reshape(1, -1))
                feedback = pred_scaled.flatten()
            else:
                feedback = None
            
            state = self._compute_state(X_scaled[t], state, feedback)
            self.states[t] = state
        
        # Prepare features for readout training
        train_start = self.config.warmup_steps
        train_features = np.hstack([
            self.states[train_start:],
            X_scaled[train_start:]
        ])
        train_targets = y_scaled[train_start:]
        
        # Train readout
        logger.debug("Training readout layer")
        self.readout.fit(train_features, train_targets)
        self.W_out = self.readout.coef_.T
        
        # Store last state for future predictions
        self.last_state = state
        
        # Collect training statistics
        self.training_stats = self._collect_statistics(
            X_scaled, y_scaled, train_features, train_targets
        )
        
        # Evaluate on validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_pred = self.predict(X_val)
            val_stats = self._evaluate_predictions(y_val, val_pred, "validation")
            self.training_stats.update(val_stats)
        
        logger.info("Training completed")
        return self.training_stats
    
    def predict(
        self,
        X: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
        return_states: bool = False
    ) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Input data of shape (n_samples, n_inputs)
            initial_state: Initial reservoir state
            return_states: Whether to return reservoir states
            
        Returns:
            Predictions of shape (n_samples, n_outputs)
        """
        X_scaled = self.input_scaler.transform(X)
        n_samples = X.shape[0]
        
        # Initialize states and predictions
        states = np.zeros((n_samples, self.config.n_reservoir))
        predictions = np.zeros((n_samples, self.config.n_outputs))
        
        # Set initial state
        if initial_state is not None:
            state = initial_state.copy()
        elif self.last_state is not None:
            state = self.last_state.copy()
        else:
            state = np.zeros(self.config.n_reservoir)
        
        # Generate predictions
        for t in range(n_samples):
            # Compute state
            if self.config.teacher_forcing and t > 0:
                # Use previous prediction as feedback
                feedback = predictions[t-1]
                feedback_scaled = self.output_scaler.transform(
                    feedback.reshape(1, -1)
                ).flatten()
            else:
                feedback_scaled = None
            
            state = self._compute_state(X_scaled[t], state, feedback_scaled)
            states[t] = state
            
            # Generate prediction
            features = np.hstack([state, X_scaled[t]])
            pred_scaled = self.readout.predict(features.reshape(1, -1))
            predictions[t] = self.output_scaler.inverse_transform(pred_scaled)
        
        # Update last state
        self.last_state = state
        
        if return_states:
            return predictions, states
        return predictions
    
    def _collect_statistics(
        self,
        X_scaled: np.ndarray,
        y_scaled: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, Any]:
        """Collect training statistics."""
        
        # Predict on training data
        train_pred_scaled = self.readout.predict(features)
        train_pred = self.output_scaler.inverse_transform(train_pred_scaled)
        train_targets = self.output_scaler.inverse_transform(targets)
        
        # Compute metrics
        from ..utils.metrics import PetroleumMetrics
        metrics = PetroleumMetrics.comprehensive_metrics(train_targets, train_pred)
        
        # Additional statistics
        stats = {
            "training_metrics": metrics,
            "feature_norm": np.linalg.norm(features, 'fro'),
            "target_norm": np.linalg.norm(targets, 'fro'),
            "readout_norm": np.linalg.norm(self.W_out, 'fro'),
            "state_mean": np.mean(self.states),
            "state_std": np.std(self.states),
            "state_entropy": self._compute_state_entropy(self.states),
            "memory_capacity": self._estimate_memory_capacity(),
        }
        
        return stats
    
    def _compute_state_entropy(self, states: np.ndarray) -> float:
        """Compute entropy of reservoir states."""
        # Discretize states
        n_bins = min(50, len(states) // 10)
        if n_bins < 2:
            return 0.0
        
        hist, _ = np.histogram(states.flatten(), bins=n_bins, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        
        return entropy / np.log(2)  # Convert to bits
    
    def _estimate_memory_capacity(self) -> float:
        """Estimate memory capacity of the reservoir."""
        # Simplified memory capacity estimation
        if self.states is None:
            return 0.0
        
        n_states = len(self.states)
        if n_states < 10:
            return 0.0
        
        # Compute autocorrelation of states
        autocorr = np.correlate(
            self.states[:, 0],
            self.states[:, 0],
            mode='full'
        )[n_states-1:] / n_states
        
        # Memory capacity is related to autocorrelation decay
        tau = np.argmax(autocorr < 0.5)  # Half-life
        capacity = min(1.0, tau / n_states)
        
        return capacity
    
    def _evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str
    ) -> Dict[str, Any]:
        """Evaluate predictions and return metrics."""
        from ..utils.metrics import PetroleumMetrics
        
        metrics = PetroleumMetrics.comprehensive_metrics(y_true, y_pred)
        
        return {
            f"{dataset_name}_metrics": metrics,
            f"{dataset_name}_mse": np.mean((y_true - y_pred) ** 2),
            f"{dataset_name}_mae": np.mean(np.abs(y_true - y_pred)),
        }
    
    def reset_state(self) -> None:
        """Reset reservoir state to zero."""
        self.last_state = np.zeros(self.config.n_reservoir)
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'config': self.config,
            'W_in': self.W_in,
            'W_res': self.W_res,
            'W_fb': self.W_fb,
            'W_out': self.W_out,
            'bias': self.bias,
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler,
            'last_state': self.last_state,
            'training_stats': self.training_stats,
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EchoStateNetwork':
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded ESN instance
        """
        model_data = joblib.load(filepath)
        
        # Create new instance
        esn = cls(model_data['config'])
        
        # Restore weights and state
        esn.W_in = model_data['W_in']
        esn.W_res = model_data['W_res']
        esn.W_fb = model_data['W_fb']
        esn.W_out = model_data['W_out']
        esn.bias = model_data['bias']
        esn.input_scaler = model_data['input_scaler']
        esn.output_scaler = model_data['output_scaler']
        esn.last_state = model_data['last_state']
        esn.training_stats = model_data['training_stats']
        
        logger.info(f"Model loaded from {filepath}")
        return esn
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return self.config.__dict__
    
    def summary(self) -> str:
        """Get model summary."""
        config = self.config
        
        summary_lines = [
            "=" * 60,
            "Echo State Network Summary",
            "=" * 60,
            f"Input dimension: {config.n_inputs}",
            f"Output dimension: {config.n_outputs}",
            f"Reservoir size: {config.n_reservoir}",
            f"Spectral radius: {config.spectral_radius:.3f}",
            f"Sparsity: {config.sparsity:.3f}",
            f"Leaking rate: {config.leaking_rate:.3f}",
            f"Regularization: {config.regularization:.2e}",
            f"Connectivity: {config.reservoir_connectivity}",
            f"Activation: {config.activation_function}",
        ]
        
        if self.training_stats:
            summary_lines.extend([
                "-" * 60,
                "Training Statistics:",
                f"Memory capacity: {self.training_stats.get('memory_capacity', 0):.3f}",
                f"State entropy: {self.training_stats.get('state_entropy', 0):.3f} bits",
            ])
        
        summary_lines.append("=" * 60)
        
        return "\n".join(summary_lines)
