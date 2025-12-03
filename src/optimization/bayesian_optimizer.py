"""
Bayesian optimization for hyperparameter tuning.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver
import logging
import joblib
from pathlib import Path

from ..models.esn import EchoStateNetwork, ESNConfig
from ..utils.metrics import PetroleumMetrics

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for Bayesian optimization."""
    
    # Search space
    n_reservoir_range: Tuple[int, int] = (100, 2000)
    spectral_radius_range: Tuple[float, float] = (0.5, 1.5)
    sparsity_range: Tuple[float, float] = (0.01, 0.5)
    leaking_rate_range: Tuple[float, float] = (0.01, 0.5)
    regularization_range: Tuple[float, float] = (1e-8, 1e-2)
    input_scaling_range: Tuple[float, float] = (0.1, 2.0)
    
    # Categorical parameters
    connectivity_options: List[str] = field(default_factory=lambda: 
                                           ["uniform", "small_world", "scale_free"])
    activation_options: List[str] = field(default_factory=lambda:
                                         ["tanh", "relu", "sigmoid"])
    
    # Optimization parameters
    n_calls: int = 50
    n_initial_points: int = 10
    random_state: int = 42
    acq_func: str = "EI"  # EI, LCB, PI
    acq_optimizer: str = "auto"  # auto, sampling, lbfgs
    kappa: float = 1.96  # For LCB acquisition function
    xi: float = 0.01  # For EI, PI acquisition functions
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Validation
    cv_folds: int = 3
    validation_split: float = 0.2
    
    def validate(self) -> None:
        """Validate optimization configuration."""
        assert self.n_calls > self.n_initial_points, \
            "n_calls must be greater than n_initial_points"
        
        assert self.acq_func in ["EI", "LCB", "PI"], \
            f"Invalid acquisition function: {self.acq_func}"
        
        assert self.acq_optimizer in ["auto", "sampling", "lbfgs", "mixed"], \
            f"Invalid acquisition optimizer: {self.acq_optimizer}"


class ESNBayesianOptimizer:
    """Bayesian optimizer for Echo State Network hyperparameters."""
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: OptimizationConfig,
        model_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            config: Optimization configuration
            model_config: Base model configuration
        """
        self.X_train = X_train
        self.y_train = y_train
        self.config = config
        self.config.validate()
        
        # Base model configuration
        self.model_config = model_config or {}
        
        # Optimization results
        self.optimization_result = None
        self.best_params = None
        self.best_score = None
        self.history = []
        
        # Create search space
        self.search_space = self._create_search_space()
        
        logger.info(f"Bayesian optimizer initialized with {self.config.n_calls} calls")
    
    def _create_search_space(self) -> List:
        """Create search space for Bayesian optimization."""
        space = [
            Integer(*self.config.n_reservoir_range, name='n_reservoir'),
            Real(*self.config.spectral_radius_range, name='spectral_radius'),
            Real(*self.config.sparsity_range, name='sparsity'),
            Real(*self.config.leaking_rate_range, name='leaking_rate'),
            Real(*self.config.regularization_range, prior='log-uniform',
                 name='regularization'),
            Real(*self.config.input_scaling_range, name='input_scaling'),
            Categorical(self.config.connectivity_options, name='reservoir_connectivity'),
            Categorical(self.config.activation_options, name='activation_function'),
        ]
        
        return space
    
    def _objective_function(self, params: List) -> float:
        """
        Objective function for Bayesian optimization.
        
        Args:
            params: List of parameter values
            
        Returns:
            Negative of the validation score (to be minimized)
        """
        # Unpack parameters
        (n_reservoir, spectral_radius, sparsity, leaking_rate,
         regularization, input_scaling, connectivity, activation) = params
        
        try:
            # Create ESN configuration
            esn_config = ESNConfig(
                n_inputs=self.X_train.shape[-1],
                n_outputs=self.y_train.shape[-1],
                n_reservoir=int(n_reservoir),
                spectral_radius=spectral_radius,
                sparsity=sparsity,
                leaking_rate=leaking_rate,
                regularization=regularization,
                input_scaling=input_scaling,
                reservoir_connectivity=connectivity,
                activation_function=activation,
                **self.model_config
            )
            
            # Create and train ESN
            esn = EchoStateNetwork(esn_config)
            
            # Split data for validation
            n_samples = len(self.X_train)
            n_val = int(n_samples * self.config.validation_split)
            
            if n_val > 0:
                # Use last part for validation
                X_train_split = self.X_train[:-n_val]
                y_train_split = self.y_train[:-n_val]
                X_val = self.X_train[-n_val:]
                y_val = self.y_train[-n_val:]
                
                # Train with validation
                esn.fit(X_train_split, y_train_split, validation_data=(X_val, y_val))
                
                # Get validation metrics
                val_pred = esn.predict(X_val)
                metrics = PetroleumMetrics.comprehensive_metrics(y_val, val_pred)
                
                # Use negative NSE as objective (minimize)
                score = -metrics.get('nash_sutcliffe', 0)
                
            else:
                # Train on all data
                esn.fit(self.X_train, self.y_train)
                score = 0.0  # Default score
            
            # Store in history
            self.history.append({
                'params': params,
                'score': -score,  # Store positive score
                'config': esn_config.__dict__,
            })
            
            logger.debug(f"Evaluation: score={-score:.4f}, params={params}")
            
            return score
            
        except Exception as e:
            logger.error(f"Evaluation failed with params {params}: {e}")
            return 1e6  # Large penalty for failed evaluations
    
    @use_named_args(dimensions=self.search_space)
    def _skopt_objective(self, **params):
        """Wrapper for skopt objective function."""
        # Convert parameters to list in correct order
        param_list = [
            params['n_reservoir'],
            params['spectral_radius'],
            params['sparsity'],
            params['leaking_rate'],
            params['regularization'],
            params['input_scaling'],
            params['reservoir_connectivity'],
            params['activation_function'],
        ]
        
        return self._objective_function(param_list)
    
    def optimize(self, checkpoint_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run Bayesian optimization.
        
        Args:
            checkpoint_dir: Directory for saving checkpoints
            
        Returns:
            Optimization results
        """
        logger.info("Starting Bayesian optimization")
        
        # Create checkpoint saver if directory provided
        callbacks = []
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pkl"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_saver = CheckpointSaver(str(checkpoint_path), compress=9)
            callbacks.append(checkpoint_saver)
        
        # Run optimization
        self.optimization_result = gp_minimize(
            func=self._skopt_objective,
            dimensions=self.search_space,
            n_calls=self.config.n_calls,
            n_initial_points=self.config.n_initial_points,
            random_state=self.config.random_state,
            acq_func=self.config.acq_func,
            acq_optimizer=self.config.acq_optimizer,
            kappa=self.config.kappa,
            xi=self.config.xi,
            callback=callbacks,
            verbose=True
        )
        
        # Extract best results
        self.best_params = self.optimization_result.x
        self.best_score = -self.optimization_result.fun  # Convert back to positive
        
        logger.info(f"Optimization completed. Best score: {self.best_score:.4f}")
        
        return self._compile_results()
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile optimization results."""
        if self.optimization_result is None:
            raise RuntimeError("Optimization not yet run")
        
        # Parameter names
        param_names = [
            'n_reservoir', 'spectral_radius', 'sparsity', 'leaking_rate',
            'regularization', 'input_scaling', 'reservoir_connectivity',
            'activation_function'
        ]
        
        # Best parameters dictionary
        best_params_dict = dict(zip(param_names, self.best_params))
        
        # Convert types
        best_params_dict['n_reservoir'] = int(best_params_dict['n_reservoir'])
        
        # Create results dictionary
        results = {
            'best_params': best_params_dict,
            'best_score': self.best_score,
            'n_evaluations': len(self.optimization_result.func_vals),
            'convergence': self._check_convergence(),
            'history': self.history,
            'func_vals': self.optimization_result.func_vals.tolist(),
            'x_iters': self.optimization_result.x_iters,
            'models': self.optimization_result.models,
        }
        
        return results
    
    def _check_convergence(self) -> Dict[str, Any]:
        """Check if optimization has converged."""
        if len(self.optimization_result.func_vals) < self.config.patience:
            return {'converged': False, 'reason': 'Not enough evaluations'}
        
        # Get recent scores (positive values)
        recent_scores = -np.array(self.optimization_result.func_vals[-self.config.patience:])
        
        # Check for improvement
        best_recent = np.max(recent_scores)
        improvement = best_recent - np.max(recent_scores[:-1]) if len(recent_scores) > 1 else 0
        
        if improvement < self.config.min_delta:
            return {
                'converged': True,
                'reason': f'No improvement > {self.config.min_delta} in {self.config.patience} iterations',
                'best_recent_score': best_recent,
                'improvement': improvement,
            }
        
        return {
            'converged': False,
            'reason': f'Still improving (improvement: {improvement:.6f})',
            'best_recent_score': best_recent,
            'improvement': improvement,
        }
    
    def create_best_model(self, X_train: np.ndarray, y_train: np.ndarray) -> EchoStateNetwork:
        """
        Create and train model with best parameters.
        
        Args:
            X_train: Training data
            y_train: Training targets
            
        Returns:
            Trained ESN with best parameters
        """
        if self.best_params is None:
            raise RuntimeError("Optimization not yet run or no best parameters found")
        
        # Create configuration with best parameters
        esn_config = ESNConfig(
            n_inputs=X_train.shape[-1],
            n_outputs=y_train.shape[-1],
            **self.best_params
        )
        
        # Create and train model
        model = EchoStateNetwork(esn_config)
        model.fit(X_train, y_train)
        
        return model
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Estimate parameter importance from optimization results.
        
        Returns:
            Dictionary of parameter importances
        """
        if self.optimization_result is None:
            raise RuntimeError("Optimization not yet run")
        
        # This is a simplified importance estimation
        # In practice, you might use more sophisticated methods
        
        param_names = [
            'n_reservoir', 'spectral_radius', 'sparsity', 'leaking_rate',
            'regularization', 'input_scaling', 'reservoir_connectivity',
            'activation_function'
        ]
        
        importances = {}
        
        # Estimate importance based on parameter range exploration
        for i, name in enumerate(param_names):
            # Get all values tried for this parameter
            if self.optimization_result.x_iters:
                param_values = [x[i] for x in self.optimization_result.x_iters]
                
                # Calculate correlation with objective (simplified)
                if len(param_values) > 1:
                    # Use absolute values for correlation
                    scores = -np.array(self.optimization_result.func_vals[:len(param_values)])
                    
                    # Normalize
                    param_norm = (param_values - np.mean(param_values)) / np.std(param_values)
                    scores_norm = (scores - np.mean(scores)) / np.std(scores)
                    
                    # Correlation
                    correlation = np.abs(np.corrcoef(param_norm, scores_norm)[0, 1])
                    importances[name] = float(correlation)
                else:
                    importances[name] = 0.0
            else:
                importances[name] = 0.0
        
        # Normalize importances to sum to 1
        total = sum(importances.values())
        if total > 0:
            importances = {k: v/total for k, v in importances.items()}
        
        return importances
    
    def save(self, filepath: str) -> None:
        """
        Save optimizer state.
        
        Args:
            filepath: Path to save optimizer
        """
        save_data = {
            'config': self.config,
            'model_config': self.model_config,
            'optimization_result': self.optimization_result,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'history': self.history,
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Optimizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, X_train: np.ndarray, y_train: np.ndarray) -> 'ESNBayesianOptimizer':
        """
        Load optimizer state.
        
        Args:
            filepath: Path to saved optimizer
            X_train: Training data (required for re-initialization)
            y_train: Training targets
            
        Returns:
            Loaded optimizer instance
        """
        save_data = joblib.load(filepath)
        
        # Create new instance
        optimizer = cls(
            X_train=X_train,
            y_train=y_train,
            config=save_data['config'],
            model_config=save_data.get('model_config', {})
        )
        
        # Restore state
        optimizer.optimization_result = save_data['optimization_result']
        optimizer.best_params = save_data['best_params']
        optimizer.best_score = save_data['best_score']
        optimizer.history = save_data['history']
        
        logger.info(f"Optimizer loaded from {filepath}")
        return optimizer
    
    def summary(self) -> str:
        """Get optimization summary."""
        if self.optimization_result is None:
            return "Optimization not yet run"
        
        lines = [
            "=" * 60,
            "Bayesian Optimization Summary",
            "=" * 60,
            f"Total evaluations: {len(self.optimization_result.func_vals)}",
            f"Best score: {self.best_score:.4f}",
        ]
        
        if self.best_params:
            lines.append("\nBest Parameters:")
            for name, value in self.best_params.items():
                if isinstance(value, float):
                    lines.append(f"  {name:25} {value:.4f}")
                else:
                    lines.append(f"  {name:25} {value}")
        
        convergence = self._check_convergence()
        lines.append(f"\nConvergence: {convergence['converged']}")
        lines.append(f"Reason: {convergence.get('reason', 'N/A')}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
