"""
SPE9 Benchmark Experiment - Industry standard evaluation.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import yaml
import json
from dataclasses import dataclass, asdict

from src.data.spe9_loader import SPE9Dataset
from src.models.physics_informed_esn import PhysicsInformedESN, PIESNConfig
from src.physics.darcy_flow import DarcyFlowConstraints
from src.evaluation.physics_metrics import PhysicsMetrics
from src.evaluation.industry_benchmarks import IndustryBenchmarks


@dataclass
class ExperimentConfig:
    """Experiment configuration for SPE9 benchmark."""
    # Data settings
    data_path: str = "data/spe9"
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    sequence_length: int = 50
    prediction_horizon: int = 10
    
    # Model settings
    model_type: str = "physics_informed_esn"
    reservoir_size: int = 2000
    spectral_radius: float = 0.9
    physics_weight: float = 0.1
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    patience: int = 20
    
    # Evaluation settings
    industry_metrics: bool = True
    uncertainty_quantification: bool = True
    compare_with_eclipse: bool = True
    
    # Output settings
    output_dir: str = "results/spe9_experiment"
    save_checkpoints: bool = True
    wandb_logging: bool = True
    wandb_project: str = "reservoir-physics-ml"
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class SPE9Experiment:
    """SPE9 Benchmark Experiment Class."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize SPE9 experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load data
        self.dataset = self._load_data()
        
        # Setup physics constraints
        self.physics_module = self._setup_physics()
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Evaluation modules
        self.physics_metrics = PhysicsMetrics()
        self.industry_benchmarks = IndustryBenchmarks()
        
    def _setup_logging(self):
        """Setup experiment logging."""
        import wandb
        
        if self.config.wandb_logging:
            wandb.init(
                project=self.config.wandb_project,
                config=asdict(self.config),
                name=f"spe9_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.wandb = wandb
        else:
            self.wandb = None
        
        # Create experiment log file
        self.log_file = self.output_dir / "experiment_log.txt"
        
    def _load_data(self) -> SPE9Dataset:
        """Load and preprocess SPE9 data."""
        dataset = SPE9Dataset(
            data_path=self.config.data_path,
            sequence_length=self.config.sequence_length,
            prediction_horizon=self.config.prediction_horizon
        )
        
        # Split data
        dataset.split_data(
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.validation_ratio
        )
        
        return dataset
    
    def _setup_physics(self) -> DarcyFlowConstraints:
        """Setup physics constraints from SPE9 data."""
        # Extract grid and rock properties from SPE9
        grid_info = self.dataset.get_grid_info()
        permeability = self.dataset.get_permeability_field()
        porosity = self.dataset.get_porosity_field()
        
        return DarcyFlowConstraints(
            grid_dims=grid_info['dims'],
            permeability=permeability,
            porosity=porosity,
            fluid_viscosity=1.0,  # SPE9 default
            fluid_density=1000.0  # Water density
        )
    
    def _initialize_model(self) -> PhysicsInformedESN:
        """Initialize physics-informed model."""
        # Get input/output dimensions from data
        input_dim = self.dataset.get_feature_dimension()
        output_dim = self.dataset.get_target_dimension()
        
        # Model configuration
        model_config = PIESNConfig(
            reservoir_size=self.config.reservoir_size,
            spectral_radius=self.config.spectral_radius,
            physics_weight=self.config.physics_weight
        )
        
        # Create model
        model = PhysicsInformedESN(
            input_dim=input_dim,
            output_dim=output_dim,
            config=model_config,
            physics_module=self.physics_module
        ).to(self.device)
        
        return model
    
    def run(self) -> Dict[str, Any]:
        """
        Run complete SPE9 experiment.
        
        Returns:
            results: Dictionary with all experiment results
        """
        self.log("Starting SPE9 experiment")
        
        # 1. Training phase
        train_results = self._train_model()
        
        # 2. Validation phase
        val_results = self._validate_model()
        
        # 3. Industry benchmark comparison
        if self.config.compare_with_eclipse:
            benchmark_results = self._run_industry_benchmarks()
        else:
            benchmark_results = {}
        
        # 4. Uncertainty quantification
        if self.config.uncertainty_quantification:
            uq_results = self._run_uncertainty_analysis()
        else:
            uq_results = {}
        
        # Compile all results
        results = {
            'training': train_results,
            'validation': val_results,
            'benchmarks': benchmark_results,
            'uncertainty': uq_results,
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        self._save_results(results)
        
        # Final logging
        self.log(f"Experiment completed. Final validation loss: {val_results['final_loss']:.6f}")
        
        if self.wandb:
            self.wandb.finish()
        
        return results
    
    def _train_model(self) -> Dict[str, Any]:
        """Train the model."""
        self.log("Starting model training")
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=self.config.patience // 2
        )
        
        # Early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        train_loader = self.dataset.get_train_loader(self.config.batch_size)
        val_loader = self.dataset.get_val_loader(self.config.batch_size)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'physics_loss': [],
            'learning_rate': []
        }
        
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            physics_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                inputs, targets, physics_state = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                predictions, diagnostics = self.model(inputs, physics_state)
                
                # Compute losses
                data_loss = F.mse_loss(predictions, targets)
                phys_loss = diagnostics['physics_losses'].mean() if diagnostics['physics_losses'] is not None else 0
                
                total_loss = data_loss + self.config.physics_weight * phys_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += data_loss.item()
                physics_loss += phys_loss.item() if isinstance(phys_loss, torch.Tensor) else phys_loss
            
            # Validation phase
            val_loss = self._evaluate(val_loader)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Logging
            avg_train_loss = train_loss / len(train_loader)
            avg_physics_loss = physics_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['physics_loss'].append(avg_physics_loss)
            history['learning_rate'].append(current_lr)
            
            self.log(f"Epoch {epoch+1}/{self.config.epochs}: "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"Physics Loss: {avg_physics_loss:.6f}, "
                    f"LR: {current_lr:.6f}")
            
            if self.wandb:
                self.wandb.log({
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'physics_loss': avg_physics_loss,
                    'learning_rate': current_lr
                })
            
            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save best model
                self._save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    self.log(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'history': history,
            'best_val_loss': best_loss,
            'final_epoch': epoch + 1
        }
    
    def _evaluate(self, data_loader) -> float:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in data_loader:
                inputs, targets, _ = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                predictions, _ = self.model(inputs)
                loss = F.mse_loss(predictions, targets)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def _validate_model(self) -> Dict[str, Any]:
        """Comprehensive model validation."""
        self.log("Running comprehensive validation")
        
        test_loader = self.dataset.get_test_loader(self.config.batch_size)
        
        # Basic evaluation
        test_loss = self._evaluate(test_loader)
        
        # Physics-based validation
        physics_metrics = self.physics_metrics.evaluate_model(
            self.model, test_loader, self.physics_module
        )
        
        # Statistical tests
        statistical_tests = self._run_statistical_tests(test_loader)
        
        return {
            'test_loss': test_loss,
            'physics_metrics': physics_metrics,
            'statistical_tests': statistical_tests
        }
    
    def _run_industry_benchmarks(self) -> Dict[str, Any]:
        """Compare with industry-standard simulators."""
        self.log("Running industry benchmarks")
        
        benchmark_results = self.industry_benchmarks.compare_with_eclipse(
            self.model,
            self.dataset,
            output_dir=self.output_dir / "benchmarks"
        )
        
        return benchmark_results
    
    def _run_uncertainty_analysis(self) -> Dict[str, Any]:
        """Run uncertainty quantification analysis."""
        from src.uncertainty.uncertainty_quantification import UncertaintyQuantifier
        
        uq = UncertaintyQuantifier(self.model, self.dataset)
        
        results = {
            'calibration_curve': uq.compute_calibration(),
            'confidence_intervals': uq.compute_confidence_intervals(),
            'sensitivity_analysis': uq.run_sensitivity_analysis(),
            'bayesian_metrics': uq.compute_bayesian_metrics()
        }
        
        return results
    
    def _run_statistical_tests(self, data_loader) -> Dict[str, Any]:
        """Run statistical significance tests."""
        from scipy import stats
        
        predictions = []
        targets = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                inputs, target_batch, _ = batch
                inputs = inputs.to(self.device)
                
                pred_batch, _ = self.model(inputs)
                predictions.append(pred_batch.cpu().numpy())
                targets.append(target_batch.numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Kolmogorov-Smirnov test for distribution similarity
        ks_statistic, ks_pvalue = stats.ks_2samp(predictions.flatten(), targets.flatten())
        
        # Anderson-Darling test
        anderson_result = stats.anderson_ksamp([predictions.flatten(), targets.flatten()])
        
        return {
            'ks_test': {'statistic': ks_statistic, 'p_value': ks_pvalue},
            'anderson_darling': {
                'statistic': anderson_result.statistic,
                'critical_values': anderson_result.critical_values.tolist(),
                'significance_level': anderson_result.significance_level.tolist()
            }
        }
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'config': asdict(self.config)
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as best model
        best_path = self.output_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results."""
        # Save as JSON
        results_path = self.output_dir / "experiment_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_path = self.output_dir / "experiment_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(self._generate_summary(results))
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate experiment summary."""
        summary = []
        summary.append("=" * 80)
        summary.append("SPE9 EXPERIMENT SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Timestamp: {results['timestamp']}")
        summary.append(f"Model: {self.config.model_type}")
        summary.append(f"Reservoir Size: {self.config.reservoir_size}")
        summary.append(f"Physics Weight: {self.config.physics_weight}")
        summary.append("")
        summary.append("PERFORMANCE METRICS:")
        summary.append("-" * 40)
        summary.append(f"Final Validation Loss: {results['validation']['test_loss']:.6f}")
        
        if 'physics_metrics' in results['validation']:
            physics_metrics = results['validation']['physics_metrics']
            for metric, value in physics_metrics.items():
                summary.append(f"{metric}: {value:.6f}")
        
        if 'benchmarks' in results and results['benchmarks']:
            summary.append("")
            summary.append("INDUSTRY BENCHMARKS:")
            summary.append("-" * 40)
            for benchmark, value in results['benchmarks'].items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        summary.append(f"{benchmark}.{k}: {v}")
                else:
                    summary.append(f"{benchmark}: {value}")
        
        summary.append("")
        summary.append("=" * 80)
        
        return "\n".join(summary)
    
    def log(self, message: str):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
        
        if self.wandb:
            self.wandb.log({'log_message': message})


def run_spe9_experiment(config_path: Optional[str] = None, 
                       config_dict: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Main function to run SPE9 experiment.
    
    Args:
        config_path: Path to YAML configuration file
        config_dict: Configuration dictionary
        
    Returns:
        Experiment results
    """
    # Load configuration
    if config_path:
        config = ExperimentConfig.load(config_path)
    elif config_dict:
        config = ExperimentConfig(**config_dict)
    else:
        # Default configuration
        config = ExperimentConfig()
    
    # Create and run experiment
    experiment = SPE9Experiment(config)
    results = experiment.run()
    
    return results


if __name__ == "__main__":
    # Example usage
    results = run_spe9_experiment()
    print("Experiment completed successfully!")
