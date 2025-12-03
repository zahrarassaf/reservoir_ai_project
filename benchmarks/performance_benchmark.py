
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging

from src.models.esn import EchoStateNetwork, ESNConfig
from src.models.deep_esn import DeepEchoStateNetwork, DeepESNConfig

logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Benchmark performance of different configurations."""
    
    def __init__(self, output_dir: Path = Path("benchmarks/results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Benchmark configurations
        self.benchmark_configs = {
            "small": {
                "n_reservoir": 100,
                "n_samples": 1000,
                "n_features": 10,
            },
            "medium": {
                "n_reservoir": 500,
                "n_samples": 5000,
                "n_features": 20,
            },
            "large": {
                "n_reservoir": 2000,
                "n_samples": 20000,
                "n_features": 50,
            },
        }
    
    def generate_benchmark_data(self, config: Dict) -> tuple:
        """Generate synthetic benchmark data."""
        n_samples = config["n_samples"]
        n_features = config["n_features"]
        
        # Generate correlated time series
        np.random.seed(42)
        
        # Base signal
        t = np.linspace(0, 10*np.pi, n_samples)
        base_signal = np.sin(t) + 0.5 * np.sin(3*t)
        
        # Features with varying correlations
        X = np.zeros((n_samples, n_features))
        for i in range(n_features):
            noise = np.random.randn(n_samples) * 0.1
            phase_shift = np.random.uniform(0, 2*np.pi)
            freq_mult = np.random.uniform(0.5, 2.0)
            
            X[:, i] = np.sin(freq_mult * t + phase_shift) + 0.3 * base_signal + noise
        
        # Target with nonlinear combination
        y = np.sum(X[:, :5] ** 2, axis=1) - np.prod(X[:, 5:10], axis=1)
        y = y.reshape(-1, 1)
        
        # Add noise
        y += 0.1 * np.std(y) * np.random.randn(*y.shape)
        
        return X, y
    
    def benchmark_training_time(self) -> pd.DataFrame:
        """Benchmark training time for different configurations."""
        results = []
        
        for config_name, config in self.benchmark_configs.items():
            logger.info(f"Benchmarking {config_name} configuration")
            
            # Generate data
            X, y = self.generate_benchmark_data(config)
            
            # Test different model configurations
            model_configs = [
                {
                    "name": "ESN_basic",
                    "config": ESNConfig(
                        n_inputs=config["n_features"],
                        n_outputs=1,
                        n_reservoir=config["n_reservoir"],
                        sparsity=0.1,
                    ),
                    "model_class": EchoStateNetwork,
                },
                {
                    "name": "ESN_dense",
                    "config": ESNConfig(
                        n_inputs=config["n_features"],
                        n_outputs=1,
                        n_reservoir=config["n_reservoir"],
                        sparsity=0.5,  # Denser
                    ),
                    "model_class": EchoStateNetwork,
                },
                {
                    "name": "DeepESN_2layer",
                    "config": DeepESNConfig(
                        n_inputs=config["n_features"],
                        n_outputs=1,
                        n_layers=2,
                        layer_sizes=[config["n_reservoir"]//2, config["n_reservoir"]//2],
                    ),
                    "model_class": DeepEchoStateNetwork,
                },
            ]
            
            for model_info in model_configs:
                logger.info(f"  Testing {model_info['name']}")
                
                # Warmup run
                model = model_info["model_class"](model_info["config"])
                model.fit(X[:100], y[:100])
                
                # Actual benchmark
                start_time = time.perf_counter()
                model = model_info["model_class"](model_info["config"])
                model.fit(X, y)
                training_time = time.perf_counter() - start_time
                
                # Memory usage (approximate)
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                # Prediction speed
                start_time = time.perf_counter()
                y_pred = model.predict(X[:1000])
                prediction_time = time.perf_counter() - start_time
                
                results.append({
                    "config": config_name,
                    "model": model_info["name"],
                    "n_reservoir": config["n_reservoir"],
                    "n_samples": config["n_samples"],
                    "n_features": config["n_features"],
                    "training_time_sec": training_time,
                    "prediction_time_ms": prediction_time * 1000 / 1000,  # per sample
                    "memory_mb": memory_mb,
                    "samples_per_second": config["n_samples"] / training_time,
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        output_file = self.output_dir / "training_benchmark.csv"
        df.to_csv(output_file, index=False)
        
        return df
    
    def benchmark_scalability(self) -> pd.DataFrame:
        """Benchmark scalability with increasing reservoir size."""
        scalability_results = []
        
        # Fixed data size
        X, y = self.generate_benchmark_data({"n_samples": 5000, "n_features": 20})
        
        # Test different reservoir sizes
        reservoir_sizes = [100, 200, 500, 1000, 2000, 5000]
        
        for n_reservoir in reservoir_sizes:
            logger.info(f"Testing scalability with reservoir size {n_reservoir}")
            
            try:
                # Training time
                config = ESNConfig(
                    n_inputs=20,
                    n_outputs=1,
                    n_reservoir=n_reservoir,
                    sparsity=1000 / n_reservoir if n_reservoir > 1000 else 0.1,
                )
                
                start_time = time.perf_counter()
                model = EchoStateNetwork(config)
                model.fit(X, y)
                training_time = time.perf_counter() - start_time
                
                # Memory usage
                import sys
                model_size = sys.getsizeof(model) / 1024 / 1024  # MB
                
                # Performance metrics
                y_pred = model.predict(X[:1000])
                mse = np.mean((y[:1000] - y_pred) ** 2)
                
                scalability_results.append({
                    "n_reservoir": n_reservoir,
                    "training_time_sec": training_time,
                    "model_size_mb": model_size,
                    "mse": mse,
                    "flops_estimate": self._estimate_flops(n_reservoir, 20, 5000),
                })
                
            except MemoryError:
                logger.warning(f"Memory error at reservoir size {n_reservoir}")
                scalability_results.append({
                    "n_reservoir": n_reservoir,
                    "training_time_sec": None,
                    "model_size_mb": None,
                    "mse": None,
                    "flops_estimate": self._estimate_flops(n_reservoir, 20, 5000),
                    "error": "MemoryError",
                })
        
        # Create DataFrame
        df = pd.DataFrame(scalability_results)
        
        # Save results
        output_file = self.output_dir / "scalability_benchmark.csv"
        df.to_csv(output_file, index=False)
        
        return df
    
    def _estimate_flops(self, n_reservoir: int, n_inputs: int, n_samples: int) -> float:
        """Estimate FLOPs for ESN training."""
        # Matrix multiplication FLOPs: 2 * n * m * k
        # State update: W_res * state (n_reservoir^2)
        # Input injection: W_in * input (n_reservoir * n_inputs)
        
        flops_per_sample = (
            2 * n_reservoir * n_reservoir +  # W_res * state
            2 * n_reservoir * n_inputs +     # W_in * input
            2 * n_reservoir * 1              # Activation (tanh)
        )
        
        # Readout training (Ridge regression): O(n_reservoir^3)
        readout_flops = n_reservoir ** 3
        
        total_flops = n_samples * flops_per_sample + readout_flops
        
        return total_flops / 1e9  # GFLOPs
    
    def run_all_benchmarks(self) -> Dict[str, pd.DataFrame]:
        """Run all benchmarks."""
        logger.info("Starting comprehensive performance benchmarks")
        
        results = {
            "training_time": self.benchmark_training_time(),
            "scalability": self.benchmark_scalability(),
        }
        
        # Generate summary report
        self.generate_benchmark_report(results)
        
        return results
    
    def generate_benchmark_report(self, results: Dict[str, pd.DataFrame]):
        """Generate benchmark report."""
        report_lines = [
            "=" * 80,
            "PERFORMANCE BENCHMARK REPORT",
            "=" * 80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        # Training time summary
        df_train = results["training_time"]
        
        report_lines.append("TRAINING TIME BENCHMARKS")
        report_lines.append("-" * 80)
        
        for config_name in df_train["config"].unique():
            config_data = df_train[df_train["config"] == config_name]
            
            report_lines.append(f"\n{config_name.upper()} Configuration:")
            for _, row in config_data.iterrows():
                report_lines.append(
                    f"  {row['model']:20} "
                    f"{row['training_time_sec']:6.2f}s "
                    f"({row['samples_per_second']:6.0f} samples/s)"
                )
        
        # Scalability summary
        df_scale = results["scalability"]
        
        report_lines.append("\n\nSCALABILITY ANALYSIS")
        report_lines.append("-" * 80)
        
        for _, row in df_scale.iterrows():
            if pd.notna(row["training_time_sec"]):
                report_lines.append(
                    f"Reservoir {row['n_reservoir']:5d}: "
                    f"{row['training_time_sec']:6.2f}s, "
                    f"{row['model_size_mb']:6.1f} MB, "
                    f"MSE: {row['mse']:.4e}"
                )
        
        # Recommendations
        report_lines.append("\n\nRECOMMENDATIONS")
        report_lines.append("-" * 80)
        
        # Find optimal reservoir size
        valid_data = df_scale[df_scale["training_time_sec"].notna()]
        if len(valid_data) > 0:
            # Find point where MSE improvement diminishes
            mse_improvement = -np.diff(valid_data["mse"].values) / valid_data["mse"].values[:-1]
            time_increase = np.diff(valid_data["training_time_sec"].values) / valid_data["training_time_sec"].values[:-1]
            
            # Find optimal trade-off
            benefit_cost_ratio = mse_improvement / time_increase
            
            if len(benefit_cost_ratio) > 0:
                optimal_idx = np.argmax(benefit_cost_ratio)
                optimal_size = valid_data["n_reservoir"].iloc[optimal_idx]
                
                report_lines.append(
                    f"Optimal reservoir size: {optimal_size} neurons"
                )
                report_lines.append(
                    f"  - MSE improvement per second: {benefit_cost_ratio[optimal_idx]:.3f}"
                )
        
        report_lines.append("\n" + "=" * 80)
        
        # Save report
        report_file = self.output_dir / "benchmark_report.txt"
        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"Benchmark report saved to {report_file}")
