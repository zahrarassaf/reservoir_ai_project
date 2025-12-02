# experiments/run_experiments.py
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

from src.reservoir_models.esn import IndustrialESN
from src.reservoir_models.deep_esn import DeepESN
from src.utils.metrics import PetroleumMetrics
from src.optimization.bayesian_opt import ESNOptimizer

class SPE9Experiment:
    """آزمایش کامل روی داده‌های SPE9"""
    
    def __init__(self, config_path="experiments/configs/spe9_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.load_data()
    
    def setup_logging(self):
        """تنظیم لاگ‌گیری"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs") / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "experiment.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self):
        """بارگذاری و پیش‌پردازش داده‌های SPE9"""
        self.logger.info("Loading SPE9 data...")
        
        # اینجا داده‌های واقعی SPE9 را بارگذاری کن
        # برای شروع، از داده‌های شبیه‌سازی شده استفاده می‌کنیم
        n_samples = 1000
        n_features = 10  # فشار، اشباع، نرخ تولید و...
        
        # داده‌های مصنوعی برای نمونه
        np.random.seed(42)
        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.randn(n_samples, 3)  # خروجی‌های اصلی
        
        # تقسیم داده
        split_idx = int(0.7 * n_samples)
        self.X_train, self.X_test = self.X[:split_idx], self.X[split_idx:]
        self.y_train, self.y_test = self.y[:split_idx], self.y[split_idx:]
        
        self.logger.info(f"Data loaded: {n_samples} samples, {n_features} features")
    
    def run_baseline(self):
        """اجرای مدل baseline"""
        self.logger.info("Running baseline ESN...")
        
        esn = IndustrialESN(
            n_inputs=self.X_train.shape[1],
            n_outputs=self.y_train.shape[1],
            **self.config['model_params']
        )
        
        esn.fit(self.X_train, self.y_train)
        y_pred = esn.predict(self.X_test)
        
        # ارزیابی
        metrics = self.evaluate_predictions(y_pred)
        
        self.logger.info(f"Baseline results: {metrics}")
        return esn, metrics
    
    def run_optimized(self):
        """اجرای مدل بهینه‌سازی شده"""
        self.logger.info("Running optimized ESN...")
        
        # بهینه‌سازی hyperparameters
        optimizer = ESNOptimizer(
            self.X_train[:500], self.y_train[:500],  # زیرمجموعه برای سرعت
            self.X_test[:200], self.y_test[:200]
        )
        
        opt_result = optimizer.optimize(n_calls=30)
        
        # آموزش با بهترین پارامترها
        best_esn = IndustrialESN(
            n_inputs=self.X_train.shape[1],
            n_outputs=self.y_train.shape[1],
            **dict(zip(
                ['n_reservoir', 'spectral_radius', 'leaking_rate', 'sparsity', 'regularization'],
                opt_result['best_params']
            ))
        )
        
        best_esn.fit(self.X_train, self.y_train)
        y_pred = best_esn.predict(self.X_test)
        
        metrics = self.evaluate_predictions(y_pred)
        
        self.logger.info(f"Optimized results: {metrics}")
        return best_esn, metrics, opt_result
    
    def run_deep_esn(self):
        """اجرای Deep ESN"""
        self.logger.info("Running Deep ESN...")
        
        deep_esn = DeepESN(
            n_layers=3,
            layer_sizes=[100, 200, 100],
            **self.config['model_params']
        )
        
        deep_esn.fit(self.X_train, self.y_train)
        y_pred = deep_esn.predict(self.X_test)
        
        metrics = self.evaluate_predictions(y_pred)
        
        self.logger.info(f"Deep ESN results: {metrics}")
        return deep_esn, metrics
    
    def evaluate_predictions(self, y_pred):
        """ارزیابی جامع پیش‌بینی‌ها"""
        metrics = {
            'NSE': PetroleumMetrics.nash_sutcliffe(self.y_test, y_pred),
            'R2': PetroleumMetrics.r2_eff(self.y_test, y_pred),
            'MAPE': PetroleumMetrics.mape(self.y_test, y_pred),
            'RMSE': np.sqrt(np.mean((self.y_test - y_pred) ** 2)),
            'MAE': np.mean(np.abs(self.y_test - y_pred))
        }
        
        return metrics
    
    def run_all(self):
        """اجرای همه آزمایش‌ها"""
        self.logger.info("Starting comprehensive experiments...")
        
        results = {}
        
        # 1. Baseline
        baseline_model, baseline_metrics = self.run_baseline()
        results['baseline'] = baseline_metrics
        
        # 2. Optimized
        optimized_model, optimized_metrics, opt_results = self.run_optimized()
        results['optimized'] = optimized_metrics
        
        # 3. Deep ESN
        deep_model, deep_metrics = self.run_deep_esn()
        results['deep_esn'] = deep_metrics
        
        # مقایسه
        comparison_df = pd.DataFrame(results).T
        self.logger.info(f"\nComparison:\n{comparison_df}")
        
        # ذخیره نتایج
        self.save_results(results, comparison_df)
        
        return results
    
    def save_results(self, results, comparison_df):
        """ذخیره نتایج"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results") / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # ذخیره dataframe
        comparison_df.to_csv(results_dir / "comparison.csv")
        
        # ذخیره config
        with open(results_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        # ذخیره metrics
        with open(results_dir / "detailed_metrics.json", 'w') as f:
            import json
            json.dump(results, f, indent=4)
        
        self.logger.info(f"Results saved to {results_dir}")

if __name__ == "__main__":
    experiment = SPE9Experiment()
    results = experiment.run_all()
