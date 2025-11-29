import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf

class ComprehensiveEvaluator:
    def __init__(self, target_scaler):
        self.target_scaler = target_scaler
        self.results = {}
        
    def evaluate_models(self, y_true_scaled, predictions_dict, model_names):
        """Comprehensive model evaluation with proper metrics"""
        print("\n📊 COMPREHENSIVE MODEL EVALUATION")
        print("=================================")
        
        # Inverse transform to original scale
        y_true = self.target_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        
        evaluation_results = {}
        
        for name in model_names:
            if name in predictions_dict:
                y_pred_scaled = predictions_dict[name]
                y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                
                metrics = self._calculate_metrics(y_true, y_pred)
                evaluation_results[name] = metrics
                
                print(f"\n🎯 {name.upper()} PERFORMANCE:")
                print(f"   R² Score: {metrics['r2']:.4f}")
                print(f"   MAE: {metrics['mae']:.2f} bbl/day")
                print(f"   RMSE: {metrics['rmse']:.2f} bbl/day")
                print(f"   MAPE: {metrics['mape']:.2f}%")
                print(f"   RMSLE: {metrics['rmsle']:.4f}")
        
        self.results = evaluation_results
        return evaluation_results
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        # Filter out invalid values
        valid_mask = (y_true > 0) & (y_pred > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        if len(y_true_valid) == 0:
            return {k: 0 for k in ['r2', 'mae', 'rmse', 'mape', 'rmsle']}
        
        metrics = {
            'r2': r2_score(y_true_valid, y_pred_valid),
            'mae': mean_absolute_error(y_true_valid, y_pred_valid),
            'rmse': np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
            'mape': self._mean_absolute_percentage_error(y_true_valid, y_pred_valid),
            'rmsle': self._root_mean_squared_log_error(y_true_valid, y_pred_valid)
        }
        
        return metrics
    
    def _mean_absolute_percentage_error(self, y_true, y_pred):
        """Calculate MAPE safely"""
        return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    
    def _root_mean_squared_log_error(self, y_true, y_pred):
        """Calculate RMSLE safely"""
        return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
    
    def create_performance_report(self, evaluation_results):
        """Create comprehensive performance report"""
        print("\n🏆 MODEL PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"{'MODEL':<20} {'R²':<8} {'MAE':<10} {'RMSE':<10} {'MAPE':<12}")
        print("-" * 60)
        
        best_r2 = -np.inf
        best_model = None
        
        for model, metrics in evaluation_results.items():
            print(f"{model:<20} {metrics['r2']:<8.4f} {metrics['mae']:<10.2f} "
                  f"{metrics['rmse']:<10.2f} {metrics['mape']:<12.2f}")
            
            if metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                best_model = model
        
        print("=" * 60)
        print(f"🏆 BEST MODEL: {best_model} (R²: {best_r2:.4f})")
        
        return best_model
    
    def plot_predictions(self, y_true_scaled, predictions_dict, best_model, save_path=None):
        """Create comprehensive visualization of results"""
        y_true = self.target_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        y_pred_best = self.target_scaler.inverse_transform(
            predictions_dict[best_model].reshape(-1, 1)
        ).flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted scatter plot
        axes[0, 0].scatter(y_true, y_pred_best, alpha=0.6, s=20)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Oil Rate (bbl/day)')
        axes[0, 0].set_ylabel('Predicted Oil Rate (bbl/day)')
        axes[0, 0].set_title(f'Actual vs Predicted - {best_model}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = y_true - y_pred_best
        axes[0, 1].scatter(y_pred_best, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Analysis')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Time series comparison (first 200 samples)
        axes[1, 0].plot(y_true[:200], label='Actual', alpha=0.8)
        axes[1, 0].plot(y_pred_best[:200], label='Predicted', alpha=0.8)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Oil Rate (bbl/day)')
        axes[1, 0].set_title('Time Series Prediction')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Model comparison bar chart
        models = list(evaluation_results.keys())
        r2_scores = [evaluation_results[model]['r2'] for model in models]
        
        axes[1, 1].barh(models, r2_scores, color='skyblue')
        axes[1, 1].set_xlabel('R² Score')
        axes[1, 1].set_title('Model Comparison (R² Scores)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved: {save_path}")
        
        plt.show()
    
    def save_results(self, evaluation_results, predictions_dict, file_path):
        """Save evaluation results and predictions"""
        # Save metrics
        metrics_df = pd.DataFrame(evaluation_results).T
        metrics_df.to_csv(file_path.replace('.csv', '_metrics.csv'))
        
        # Save predictions
        predictions_df = pd.DataFrame(predictions_dict)
        predictions_df['actual'] = self.target_scaler.inverse_transform(
            np.array(list(predictions_dict.values())[0].reshape(-1, 1))
        ).flatten()
        predictions_df.to_csv(file_path)
        
        print(f"💾 Results saved: {file_path}")
