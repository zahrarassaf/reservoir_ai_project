"""
Experiment runner for comprehensive model evaluation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import json
import yaml
import joblib

from ..models.esn import EchoStateNetwork, ESNConfig
from ..models.deep_esn import DeepEchoStateNetwork, DeepESNConfig
from ..optimization.bayesian_optimizer import ESNBayesianOptimizer, OptimizationConfig
from ..utils.metrics import PetroleumMetrics
from ..data.preprocessor import SPE9Preprocessor, PreprocessingConfig

logger = logging.getLogger(__name__)


class ExperimentConfig:
    """Configuration for experiments."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        
        # Experiment settings
        self.experiment_name = config_dict.get('experiment_name', 'default')
        self.random_seed = config_dict.get('random_seed', 42)
        self.save_results = config_dict.get('save_results', True)
        self.results_dir = Path(config_dict.get('results_dir', 'results'))
        
        # Model settings
        self.model_type = config_dict.get('model_type', 'esn')  # esn, deep_esn
        self.esn_config = config_dict.get('esn_config', {})
        self.deep_esn_config = config_dict.get('deep_esn_config', {})
        
        # Optimization settings
        self.optimize_hyperparameters = config_dict.get('optimize_hyperparameters', True)
        self.optimization_config = config_dict.get('optimization_config', {})
        
        # Data settings
        self.data_config = config_dict.get('data_config', {})
        
        # Evaluation settings
        self.evaluation_metrics = config_dict.get('evaluation_metrics', [
            'nash_sutcliffe', 'r2', 'mape', 'rmse', 'forecast_skill_score_mean'
        ])
    
    def validate(self) -> None:
        """Validate configuration."""
        assert self.model_type in ['esn', 'deep_esn'], \
            f"Invalid model type: {self.model_type}"
        
        if self.optimize_hyperparameters:
            assert 'optimization_config' in self.config, \
                "optimization_config required when optimize_hyperparameters is True"


class ExperimentRunner:
    """Runner for comprehensive experiments."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.config.validate()
        
        # Set random seed
        np.random.seed(self.config.random_seed)
        
        # Create results directory
        if self.config.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = self.config.results_dir / f"{self.config.experiment_name}_{timestamp}"
            self.results_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.results_dir = None
        
        # Experiment state
        self.preprocessor = None
        self.data_splits = None
        self.models = {}
        self.results = {}
        
        logger.info(f"Experiment runner initialized: {self.config.experiment_name}")
    
    def load_and_preprocess_data(self, data_file: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """
        Load and preprocess data.
        
        Args:
            data_file: Optional path to data file
            
        Returns:
            Data splits
        """
        logger.info("Loading and preprocessing data")
        
        # Create preprocessor
        preproc_config = PreprocessingConfig(**self.config.data_config)
        self.preprocessor = SPE9Preprocessor(preproc_config)
        
        # Load data
        if data_file is None:
            # Use synthetic data for demonstration
            from ..data.preprocessor import SPE9Preprocessor
            import pandas as pd
            
            # Create synthetic data
            n_samples = 1000
            data = pd.DataFrame({
                'TIME': np.linspace(0, 3650, n_samples),
                'PRESSURE': 3000 + 500 * np.sin(np.linspace(0, 10*np.pi, n_samples)) + 50 * np.random.randn(n_samples),
                'SATURATION': 0.8 - 0.3 * np.linspace(0, 1, n_samples) + 0.1 * np.random.randn(n_samples),
                'PRODUCTION_RATE': 1000 + 500 * np.sin(np.linspace(0, 20*np.pi, n_samples)) + 200 * np.random.randn(n_samples),
                'INJECTION_RATE': 800 + 400 * np.random.randn(n_samples),
                'BHP': 1500 + 200 * np.sin(np.linspace(0, 20*np.pi, n_samples)),
                'THP': 500 + 100 * np.cos(np.linspace(0, 10*np.pi, n_samples)),
            })
        else:
            # Parse real data
            data = self.preprocessor.parse_eclipse_data(data_file)
        
        # Preprocess data
        self.data_splits = self.preprocessor.preprocess(data)
        
        # Save preprocessor
        if self.results_dir:
            preprocessor_path = self.results_dir / "preprocessor.pkl"
            self.preprocessor.save(preprocessor_path)
        
        logger.info("Data preprocessing completed")
        return self.data_splits
    
    def run_baseline_experiment(self) -> Dict[str, Any]:
        """
        Run baseline experiment with default parameters.
        
        Returns:
            Baseline results
        """
        logger.info("Running baseline experiment")
        
        # Get data
        X_train = self.data_splits['X_train'].reshape(
            self.data_splits['X_train'].shape[0], -1
        )
        y_train = self.data_splits['y_train'].reshape(
            self.data_splits['y_train'].shape[0], -1
        )
        
        X_val = self.data_splits['X_val'].reshape(
            self.data_splits['X_val'].shape[0], -1
        )
        y_val = self.data_splits['y_val'].reshape(
            self.data_splits['y_val'].shape[0], -1
        )
        
        # Create baseline model
        if self.config.model_type == 'esn':
            baseline_config = ESNConfig(
                n_inputs=X_train.shape[1],
                n_outputs=y_train.shape[1],
                **self.config.esn_config
            )
            baseline_model = EchoStateNetwork(baseline_config)
        else:
            baseline_config = DeepESNConfig(
                n_inputs=X_train.shape[1],
                n_outputs=y_train.shape[1],
                **self.config.deep_esn_config
            )
            baseline_model = DeepEchoStateNetwork(baseline_config)
        
        # Train model
        baseline_model.fit(X_train, y_train, validation_data=(X_val, y_val))
        
        # Evaluate
        y_pred = baseline_model.predict(X_val)
        baseline_metrics = PetroleumMetrics.comprehensive_metrics(y_val, y_pred)
        
        # Store results
        self.results['baseline'] = {
            'model_type': self.config.model_type,
            'config': baseline_config.__dict__,
            'metrics': baseline_metrics,
            'model_summary': baseline_model.summary(),
        }
        
        # Save model
        if self.results_dir:
            model_path = self.results_dir / "baseline_model.pkl"
            joblib.dump(baseline_model, model_path)
        
        logger.info(f"Baseline experiment completed. NSE: {baseline_metrics.get('nash_sutcliffe', 0):.4f}")
        return self.results['baseline']
    
    def run_optimization_experiment(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization experiment.
        
        Returns:
            Optimization results
        """
        if not self.config.optimize_hyperparameters:
            logger.info("Hyperparameter optimization disabled")
            return {}
        
        logger.info("Running hyperparameter optimization experiment")
        
        # Get training data
        X_train = self.data_splits['X_train'].reshape(
            self.data_splits['X_train'].shape[0], -1
        )
        y_train = self.data_splits['y_train'].reshape(
            self.data_splits['y_train'].shape[0], -1
        )
        
        # Create optimizer
        opt_config = OptimizationConfig(**self.config.optimization_config)
        
        optimizer = ESNBayesianOptimizer(
            X_train=X_train,
            y_train=y_train,
            config=opt_config,
            model_config=self.config.esn_config
        )
        
        # Run optimization
        if self.results_dir:
            checkpoint_dir = self.results_dir / "optimization_checkpoints"
        else:
            checkpoint_dir = None
        
        optimization_results = optimizer.optimize(checkpoint_dir=checkpoint_dir)
        
        # Create and train best model
        X_val = self.data_splits['X_val'].reshape(
            self.data_splits['X_val'].shape[0], -1
        )
        y_val = self.data_splits['y_val'].reshape(
            self.data_splits['y_val'].shape[0], -1
        )
        
        best_model = optimizer.create_best_model(X_train, y_train)
        
        # Evaluate best model
        y_pred = best_model.predict(X_val)
        best_metrics = PetroleumMetrics.comprehensive_metrics(y_val, y_pred)
        
        # Store results
        self.results['optimized'] = {
            'optimization_results': optimization_results,
            'best_metrics': best_metrics,
            'model_summary': best_model.summary(),
            'parameter_importance': optimizer.get_parameter_importance(),
        }
        
        # Save optimizer and model
        if self.results_dir:
            optimizer_path = self.results_dir / "optimizer.pkl"
            optimizer.save(optimizer_path)
            
            model_path = self.results_dir / "optimized_model.pkl"
            joblib.dump(best_model, model_path)
        
        logger.info(f"Optimization experiment completed. Best NSE: {best_metrics.get('nash_sutcliffe', 0):.4f}")
        return self.results['optimized']
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """
        Run ablation study on key hyperparameters.
        
        Returns:
            Ablation study results
        """
        logger.info("Running ablation study")
        
        # Define parameters to ablate
        ablation_params = {
            'spectral_radius': [0.5, 0.8, 0.95, 1.2, 1.5],
            'leaking_rate': [0.1, 0.3, 0.5, 0.7, 0.9],
            'sparsity': [0.01, 0.05, 0.1, 0.2, 0.5],
            'n_reservoir': [100, 500, 1000, 1500, 2000],
        }
        
        ablation_results = {}
        
        # Get data
        X_train = self.data_splits['X_train'].reshape(
            self.data_splits['X_train'].shape[0], -1
        )
        y_train = self.data_splits['y_train'].reshape(
            self.data_splits['y_train'].shape[0], -1
        )
        
        X_val = self.data_splits['X_val'].reshape(
            self.data_splits['X_val'].shape[0], -1
        )
        y_val = self.data_splits['y_val'].reshape(
            self.data_splits['y_val'].shape[0], -1
        )
        
        # Base configuration
        base_config = {
            'n_inputs': X_train.shape[1],
            'n_outputs': y_train.shape[1],
            'spectral_radius': 0.95,
            'leaking_rate': 0.3,
            'sparsity': 0.1,
            'n_reservoir': 1000,
            'regularization': 1e-6,
        }
        
        # Test each parameter
        for param_name, param_values in ablation_params.items():
            param_results = []
            
            for value in param_values:
                # Create configuration
                config_dict = base_config.copy()
                config_dict[param_name] = value
                
                # Create and train model
                esn_config = ESNConfig(**config_dict)
                model = EchoStateNetwork(esn_config)
                
                try:
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_val)
                    metrics = PetroleumMetrics.comprehensive_metrics(y_val, y_pred)
                    
                    param_results.append({
                        'value': value,
                        'metrics': metrics,
                        'nse': metrics.get('nash_sutcliffe', 0),
                    })
                    
                except Exception as e:
                    logger.warning(f"Ablation failed for {param_name}={value}: {e}")
                    param_results.append({
                        'value': value,
                        'error': str(e),
                    })
            
            ablation_results[param_name] = param_results
        
        self.results['ablation_study'] = ablation_results
        
        # Save results
        if self.results_dir:
            ablation_path = self.results_dir / "ablation_study.json"
            with open(ablation_path, 'w') as f:
                json.dump(ablation_results, f, indent=2, default=str)
        
        logger.info("Ablation study completed")
        return ablation_results
    
    def run_comparison_experiment(self, n_models: int = 5) -> Dict[str, Any]:
        """
        Run comparison experiment with multiple random initializations.
        
        Args:
            n_models: Number of models to compare
            
        Returns:
            Comparison results
        """
        logger.info(f"Running comparison experiment with {n_models} models")
        
        # Get data
        X_train = self.data_splits['X_train'].reshape(
            self.data_splits['X_train'].shape[0], -1
        )
        y_train = self.data_splits['y_train'].reshape(
            self.data_splits['y_train'].shape[0], -1
        )
        
        X_val = self.data_splits['X_val'].reshape(
            self.data_splits['X_val'].shape[0], -1
        )
        y_val = self.data_splits['y_val'].reshape(
            self.data_splits['y_val'].shape[0], -1
        )
        
        comparison_results = []
        
        for i in range(n_models):
            logger.info(f"Training model {i+1}/{n_models}")
            
            # Create model with different random seed
            esn_config = ESNConfig(
                n_inputs=X_train.shape[1],
                n_outputs=y_train.shape[1],
                random_state=self.config.random_seed + i,
                **self.config.esn_config
            )
            
            model = EchoStateNetwork(esn_config)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            metrics = PetroleumMetrics.comprehensive_metrics(y_val, y_pred)
            
            comparison_results.append({
                'model_id': i,
                'random_seed': self.config.random_seed + i,
                'metrics': metrics,
                'nse': metrics.get('nash_sutcliffe', 0),
                'r2': metrics.get('r2', 0),
                'rmse': metrics.get('rmse', 0),
            })
        
        # Calculate statistics
        nse_values = [r['nse'] for r in comparison_results]
        r2_values = [r['r2'] for r in comparison_results]
        rmse_values = [r['rmse'] for r in comparison_results]
        
        statistics = {
            'nse_mean': np.mean(nse_values),
            'nse_std': np.std(nse_values),
            'nse_min': np.min(nse_values),
            'nse_max': np.max(nse_values),
            'r2_mean': np.mean(r2_values),
            'r2_std': np.std(r2_values),
            'rmse_mean': np.mean(rmse_values),
            'rmse_std': np.std(rmse_values),
        }
        
        self.results['comparison'] = {
            'models': comparison_results,
            'statistics': statistics,
        }
        
        # Save results
        if self.results_dir:
            comparison_path = self.results_dir / "comparison_results.json"
            with open(comparison_path, 'w') as f:
                json.dump(self.results['comparison'], f, indent=2, default=str)
        
        logger.info(f"Comparison experiment completed. Mean NSE: {statistics['nse_mean']:.4f} ± {statistics['nse_std']:.4f}")
        return self.results['comparison']
    
    def run_all_experiments(self, data_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run all experiments.
        
        Args:
            data_file: Optional path to data file
            
        Returns:
            Comprehensive results
        """
        logger.info("Running all experiments")
        
        # Start timer
        start_time = datetime.now()
        
        try:
            # 1. Load and preprocess data
            self.load_and_preprocess_data(data_file)
            
            # 2. Run baseline experiment
            baseline_results = self.run_baseline_experiment()
            
            # 3. Run optimization experiment
            if self.config.optimize_hyperparameters:
                optimization_results = self.run_optimization_experiment()
            else:
                optimization_results = {}
            
            # 4. Run ablation study
            ablation_results = self.run_ablation_study()
            
            # 5. Run comparison experiment
            comparison_results = self.run_comparison_experiment()
            
            # Compile final results
            self.results['summary'] = {
                'experiment_name': self.config.experiment_name,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                'data_statistics': self.preprocessor.get_statistics() if self.preprocessor else {},
                'baseline_performance': baseline_results.get('metrics', {}),
                'optimized_performance': optimization_results.get('best_metrics', {}),
                'comparison_statistics': comparison_results.get('statistics', {}),
            }
            
            # Save final results
            if self.results_dir:
                # Save configuration
                config_path = self.results_dir / "experiment_config.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(self.config.config, f)
                
                # Save summary
                summary_path = self.results_dir / "experiment_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(self.results['summary'], f, indent=2, default=str)
                
                # Save all results
                results_path = self.results_dir / "all_results.json"
                with open(results_path, 'w') as f:
                    json.dump(self.results, f, indent=2, default=str)
                
                logger.info(f"All results saved to {self.results_dir}")
            
            logger.info("All experiments completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
    
    def generate_report(self) -> str:
        """
        Generate comprehensive experiment report.
        
        Returns:
            Formatted report
        """
        if not self.results:
            return "No results available. Run experiments first."
        
        report_lines = [
            "=" * 80,
            f"EXPERIMENT REPORT: {self.config.experiment_name}",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            "",
        ]
        
        # Summary section
        if 'summary' in self.results:
            summary = self.results['summary']
            report_lines.extend([
                "SUMMARY",
                "-" * 80,
                f"Duration: {summary.get('duration_seconds', 0):.1f} seconds",
                f"Data samples: {summary.get('data_statistics', {}).get('n_sequences', 'N/A')}",
                "",
            ])
        
        # Performance comparison
        report_lines.extend([
            "PERFORMANCE COMPARISON",
            "-" * 80,
        ])
        
        if 'baseline' in self.results and 'optimized' in self.results:
            baseline_nse = self.results['baseline']['metrics'].get('nash_sutcliffe', 0)
            optimized_nse = self.results['optimized']['best_metrics'].get('nash_sutcliffe', 0)
            improvement = ((optimized_nse - baseline_nse) / abs(baseline_nse)) * 100 if baseline_nse != 0 else 0
            
            report_lines.extend([
                f"Baseline NSE:     {baseline_nse:.4f}",
                f"Optimized NSE:    {optimized_nse:.4f}",
                f"Improvement:      {improvement:+.1f}%",
                "",
            ])
        
        # Ablation study highlights
        if 'ablation_study' in self.results:
            report_lines.extend([
                "ABLATION STUDY HIGHLIGHTS",
                "-" * 80,
            ])
            
            for param, results in self.results['ablation_study'].items():
                if results:
                    nse_values = [r.get('nse', 0) for r in results if 'nse' in r]
                    if nse_values:
                        best_idx = np.argmax(nse_values)
                        best_value = results[best_idx]['value']
                        best_nse = nse_values[best_idx]
                        
                        report_lines.append(
                            f"{param:20} Best: {best_value} (NSE: {best_nse:.4f})"
                        )
            
            report_lines.append("")
        
        # Comparison statistics
        if 'comparison' in self.results:
            stats = self.results['comparison']['statistics']
            report_lines.extend([
                "MODEL STABILITY",
                "-" * 80,
                f"NSE:  {stats['nse_mean']:.4f} ± {stats['nse_std']:.4f} "
                f"(range: {stats['nse_min']:.4f} - {stats['nse_max']:.4f})",
                f"R²:   {stats['r2_mean']:.4f} ± {stats['r2_std']:.4f}",
                f"RMSE: {stats['rmse_mean']:.2e} ± {stats['rmse_std']:.2e}",
                "",
            ])
        
        # Best parameters (if optimized)
        if 'optimized' in self.results and 'optimization_results' in self.results['optimized']:
            best_params = self.results['optimized']['optimization_results'].get('best_params', {})
            
            report_lines.extend([
                "BEST HYPERPARAMETERS",
                "-" * 80,
            ])
            
            for param, value in best_params.items():
                if isinstance(value, float):
                    report_lines.append(f"{param:25} {value:.4f}")
                else:
                    report_lines.append(f"{param:25} {value}")
            
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_report(self, filepath: Optional[Path] = None) -> Path:
        """
        Save report to file.
        
        Args:
            filepath: Optional file path
            
        Returns:
            Path to saved report
        """
        if filepath is None and self.results_dir:
            filepath = self.results_dir / "experiment_report.txt"
        elif filepath is None:
            raise ValueError("No filepath provided and no results directory configured")
        
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {filepath}")
        return filepath
