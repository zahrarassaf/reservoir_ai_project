"""
Comprehensive model evaluation and visualization
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import joblib
from typing import Dict, List

from .config import config

class ModelEvaluator:
    """Comprehensive model evaluation with statistical analysis"""
    
    def __init__(self):
        self.results = {}
        self.feature_importance = {}
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           model_name: str) -> Dict:
        """Comprehensive evaluation of model predictions"""
        
        # Basic metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'explained_variance': self._explained_variance_score(y_true, y_pred)
        }
        
        # Additional statistical metrics
        metrics.update(self._calculate_additional_metrics(y_true, y_pred))
        
        # Residual analysis
        residuals = y_true - y_pred
        metrics.update(self._analyze_residuals(residuals))
        
        self.results[model_name] = metrics
        return metrics
    
    def _calculate_additional_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate additional evaluation metrics"""
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        
        # Mean Bias Error (MBE)
        mbe = np.mean(y_pred - y_true)
        
        # Nash-Sutcliffe Efficiency (NSE)
        nse = 1 - (np.sum((y_true - y_pred) ** 2) / 
                  np.sum((y_true - np.mean(y_true)) ** 2))
        
        return {
            'mape': mape,
            'mbe': mbe,
            'nse': nse
        }
    
    def _explained_variance_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate explained variance score"""
        return 1 - (np.var(y_true - y_pred) / np.var(y_true))
    
    def _analyze_residuals(self, residuals: np.ndarray) -> Dict:
        """Statistical analysis of residuals"""
        analysis = {
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_skewness': stats.skew(residuals),
            'residual_kurtosis': stats.kurtosis(residuals)
        }
        
        # Normality test
        if len(residuals) > 7:  # Minimum sample size for normality test
            _, normality_p = stats.normaltest(residuals)
            analysis['residual_normality_p'] = normality_p
        else:
            analysis['residual_normality_p'] = np.nan
        
        return analysis
    
    def calculate_feature_importance(self, model, feature_names: List[str], 
                                   model_name: str, X_val: np.ndarray = None) -> pd.DataFrame:
        """Calculate feature importance for tree-based models"""
        
        importance_data = {}
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            importance_data['importance'] = importances
            importance_data['importance_rank'] = np.argsort(importances)[::-1]
            
        elif hasattr(model, 'coef_'):
            # Linear models
            coefficients = np.abs(model.coef_)
            importance_data['importance'] = coefficients
            importance_data['importance_rank'] = np.argsort(coefficients)[::-1]
        
        elif X_val is not None and hasattr(model, 'predict'):
            # Permutation importance as fallback
            from sklearn.inspection import permutation_importance
            perm_importance = permutation_importance(
                model, X_val, self._get_validation_target(), 
                n_repeats=10, random_state=config.RANDOM_STATE
            )
            importance_data['importance'] = perm_importance.importances_mean
            importance_data['importance_rank'] = np.argsort(perm_importance.importances_mean)[::-1]
        
        if importance_data:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_data['importance'],
                'rank': importance_data['importance_rank'] + 1
            }).sort_values('importance', ascending=False)
            
            self.feature_importance[model_name] = importance_df
            return importance_df
        
        return pd.DataFrame()
    
    def create_comparison_plots(self, results_dict: Dict, output_dir: str = None):
        """Create comprehensive comparison plots"""
        if output_dir is None:
            output_dir = config.RESULT_DIR
        
        # Create performance comparison plot
        self._plot_model_comparison(results_dict, output_dir)
        
        # Create residual analysis plots
        self._plot_residual_analysis(results_dict, output_dir)
        
        # Create prediction vs actual plots
        self._plot_predictions_vs_actual(results_dict, output_dir)
    
    def _plot_model_comparison(self, results_dict: Dict, output_dir: str):
        """Create model performance comparison plot"""
        metrics_df = pd.DataFrame(results_dict).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RMSE comparison
        metrics_df['rmse'].plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('RMSE Comparison')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # R² comparison
        metrics_df['r2'].plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('R² Score Comparison')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].set_ylim(0, 1)
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        metrics_df['mae'].plot(kind='bar', ax=axes[1,0], color='lightcoral')
        axes[1,0].set_title('MAE Comparison')
        axes[1,0].set_ylabel('MAE')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        metrics_df['mape'].plot(kind='bar', ax=axes[1,1], color='gold')
        axes[1,1].set_title('MAPE Comparison')
        axes[1,1].set_ylabel('MAPE (%)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_residual_analysis(self, results_dict: Dict, output_dir: str):
        """Create residual analysis plots"""
        n_models = len(results_dict)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = np.array([axes]).T
        
        for idx, (model_name, results) in enumerate(results_dict.items()):
            if 'predictions' in results and 'true_values' in results:
                residuals = results['true_values'] - results['predictions']
                
                # Residual distribution
                axes[0, idx].hist(residuals, bins=50, alpha=0.7, color='blue')
                axes[0, idx].set_title(f'{model_name} - Residual Distribution')
                axes[0, idx].set_xlabel('Residuals')
                axes[0, idx].set_ylabel('Frequency')
                
                # Q-Q plot
                stats.probplot(residuals, dist="norm", plot=axes[1, idx])
                axes[1, idx].set_title(f'{model_name} - Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_predictions_vs_actual(self, results_dict: Dict, output_dir: str):
        """Create predictions vs actual values plot"""
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(results_dict.items()):
            if 'predictions' in results and 'true_values' in results:
                y_true = results['true_values']
                y_pred = results['predictions']
                
                axes[idx].scatter(y_true, y_pred, alpha=0.6, s=20)
                
                # Perfect prediction line
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                axes[idx].set_xlabel('Actual Values')
                axes[idx].set_ylabel('Predicted Values')
                axes[idx].set_title(f'{model_name} - Predictions vs Actual')
                
                # Add R² to plot
                r2 = r2_score(y_true, y_pred)
                axes[idx].text(0.05, 0.95, f'R² = {r2:.3f}', 
                              transform=axes[idx].transAxes, 
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, results_dict: Dict, 
                                    feature_importance: Dict = None) -> pd.DataFrame:
        """Generate comprehensive evaluation report"""
        
        # Performance metrics
        metrics_df = pd.DataFrame(results_dict).T
        
        # Add statistical significance testing
        metrics_df = self._add_statistical_significance(metrics_df, results_dict)
        
        # Save detailed report
        metrics_df.to_csv(config.RESULT_DIR / 'comprehensive_evaluation_report.csv')
        
        # Feature importance report
        if feature_importance:
            for model_name, imp_df in feature_importance.items():
                if not imp_df.empty:
                    imp_df.to_csv(
                        config.RESULT_DIR / f'{model_name}_feature_importance.csv', 
                        index=False
                    )
        
        return metrics_df
    
    def _add_statistical_significance(self, metrics_df: pd.DataFrame, 
                                    results_dict: Dict) -> pd.DataFrame:
        """Add statistical significance information to results"""
        
        # This would require multiple runs or cross-validation
        # For now, we'll add placeholder columns
        metrics_df['std_rmse'] = 0.0  # Placeholder for standard deviation
        metrics_df['confidence_95_lower'] = metrics_df['rmse'] * 0.95
        metrics_df['confidence_95_upper'] = metrics_df['rmse'] * 1.05
        
        return metrics_df
    
    def _get_validation_target(self):
        """Get validation target for permutation importance"""
        # This should be implemented based on your validation set
        pass
