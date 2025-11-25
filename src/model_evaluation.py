"""
Professional Model Evaluation Module
Comprehensive model analysis with statistical testing and visualization
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, 
                           explained_variance_score, median_absolute_error)
from sklearn.inspection import permutation_importance
import joblib
import warnings
warnings.filterwarnings('ignore')

from .config import config

class ComprehensiveModelEvaluator:
    """
    Comprehensive model evaluation with statistical analysis
    and professional visualization
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.visualizations = {}
        
    def calculate_comprehensive_metrics(self, y_true, y_pred, set_name):
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100,
            'explained_variance': explained_variance_score(y_true, y_pred),
            'median_ae': median_absolute_error(y_true, y_pred),
            'max_error': np.max(np.abs(y_true - y_pred))
        }
        
        # Statistical tests
        residuals = y_true - y_pred
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        metrics['residual_skew'] = stats.skew(residuals)
        metrics['residual_kurtosis'] = stats.kurtosis(residuals)
        
        # Normality test of residuals
        _, metrics['residual_normality_p'] = stats.normaltest(residuals)
        
        print(f"📈 {set_name} Metrics:")
        print(f"   • R²: {metrics['r2']:.4f}")
        print(f"   • RMSE: {metrics['rmse']:.4f}")
        print(f"   • MAE: {metrics['mae']:.4f}")
        print(f"   • MAPE: {metrics['mape']:.2f}%")
        
        return metrics, residuals
    
    def create_prediction_plots(self, y_true, y_pred, set_name, model_name):
        """Create comprehensive prediction plots"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Actual vs Predicted - {set_name}',
                'Residual Analysis',
                'Error Distribution',
                'Q-Q Plot of Residuals'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Scatter plot: Actual vs Predicted
        fig.add_trace(
            go.Scatter(x=y_true, y=y_pred, mode='markers', 
                      name='Predictions', opacity=0.7),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='Perfect Prediction', 
                      line=dict(dash='dash', color='red')),
            row=1, col=1
        )
        
        # Residual plot
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers', 
                      name='Residuals', opacity=0.7),
            row=1, col=2
        )
        
        # Zero residual line
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[0, 0], 
                      mode='lines', name='Zero Residual', 
                      line=dict(dash='dash', color='red')),
            row=1, col=2
        )
        
        # Error distribution
        fig.add_trace(
            go.Histogram(x=residuals, name='Error Distribution', 
                        nbinsx=50, opacity=0.7),
            row=2, col=1
        )
        
        # Q-Q plot
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(sorted_residuals))
        )
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_residuals, 
                      mode='markers', name='Q-Q Plot', opacity=0.7),
            row=2, col=2
        )
        
        # Q-Q reference line
        qq_min, qq_max = theoretical_quantiles.min(), theoretical_quantiles.max()
        fig.add_trace(
            go.Scatter(x=[qq_min, qq_max], y=[qq_min, qq_max], 
                      mode='lines', name='Normal Reference', 
                      line=dict(dash='dash', color='red')),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'{model_name} - {set_name} Analysis',
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text='Actual Values', row=1, col=1)
        fig.update_yaxes(title_text='Predicted Values', row=1, col=1)
        fig.update_xaxes(title_text='Predicted Values', row=1, col=2)
        fig.update_yaxes(title_text='Residuals', row=1, col=2)
        fig.update_xaxes(title_text='Residual Value', row=2, col=1)
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        fig.update_xaxes(title_text='Theoretical Quantiles', row=2, col=2)
        fig.update_yaxes(title_text='Sample Quantiles', row=2, col=2)
        
        return fig
    
    def feature_importance_analysis(self, model, X, y, feature_names, model_name):
        """Comprehensive feature importance analysis"""
        
        print(f"🔍 Analyzing feature importance for {model_name}...")
        
        # Get feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            importance_type = "Feature Importance"
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_)
            importance_type = "Coefficient Magnitude"
        else:
            # Use permutation importance as fallback
            perm_importance = permutation_importance(
                model, X, y, n_repeats=10, random_state=config.RANDOM_STATE
            )
            importances = perm_importance.importances_mean
            importance_type = "Permutation Importance"
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'importance_normalized': importances / np.sum(importances) * 100
        }).sort_values('importance', ascending=False)
        
        # Create visualization
        fig = px.bar(
            importance_df.head(15),
            x='importance_normalized',
            y='feature',
            orientation='h',
            title=f'{model_name} - {importance_type}',
            labels={'importance_normalized': 'Importance (%)', 'feature': 'Features'}
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        print("📊 Top 5 Features:")
        for idx, row in importance_df.head().iterrows():
            print(f"   {row['feature']}: {row['importance_normalized']:.2f}%")
        
        return importance_df, fig
    
    def model_comparison_analysis(self, training_results, target_name):
        """Compare performance across all models"""
        
        comparison_data = []
        
        for model_name, results in training_results.items():
            if 'evaluation' in results:
                test_metrics = results['evaluation']['test']['metrics']
                train_metrics = results['evaluation']['train']['metrics']
                
                comparison_data.append({
                    'Model': model_name,
                    'Test_R2': test_metrics['r2'],
                    'Test_RMSE': test_metrics['rmse'],
                    'Test_MAE': test_metrics['mae'],
                    'Train_R2': train_metrics['r2'],
                    'CV_Score': results.get('best_cv_score', np.nan),
                    'Overfitting_Index': train_metrics['r2'] - test_metrics['r2']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test_R2', ascending=False)
        
        # Create comparison visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Test R²',
            x=comparison_df['Model'],
            y=comparison_df['Test_R2'],
            text=comparison_df['Test_R2'].round(3),
            textposition='auto',
        ))
        
        fig.add_trace(go.Bar(
            name='Train R²',
            x=comparison_df['Model'],
            y=comparison_df['Train_R2'],
            text=comparison_df['Train_R2'].round(3),
            textposition='auto',
        ))
        
        fig.update_layout(
            title=f'Model Comparison - {target_name}',
            xaxis_title='Models',
            yaxis_title='R² Score',
            barmode='group',
            height=500
        )
        
        print(f"\n🏆 Model Ranking for {target_name}:")
        for idx, row in comparison_df.iterrows():
            print(f"   {idx+1}. {row['Model']}: R² = {row['Test_R2']:.4f}")
        
        return comparison_df, fig
    
    def residual_analysis(self, y_true, y_pred, set_name):
        """Comprehensive residual analysis"""
        
        residuals = y_true - y_pred
        
        analysis = {
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_skewness': stats.skew(residuals),
            'residual_kurtosis': stats.kurtosis(residuals),
            'homoscedasticity_pvalue': self.test_homoscedasticity(y_pred, residuals),
            'autocorrelation': self.test_autocorrelation(residuals)
        }
        
        # Normality test
        _, analysis['normality_pvalue'] = stats.normaltest(residuals)
        
        print(f"📊 Residual Analysis - {set_name}:")
        print(f"   • Mean: {analysis['residual_mean']:.4f}")
        print(f"   • Std: {analysis['residual_std']:.4f}")
        print(f"   • Normality p-value: {analysis['normality_pvalue']:.4f}")
        
        return analysis, residuals
    
    def test_homoscedasticity(self, y_pred, residuals):
        """Test for homoscedasticity using Breusch-Pagan test"""
        try:
            # Simple correlation test between predictions and squared residuals
            correlation = np.corrcoef(y_pred, residuals**2)[0, 1]
            return correlation
        except:
            return np.nan
    
    def test_autocorrelation(self, residuals):
        """Test for autocorrelation using Ljung-Box test"""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            result = acorr_ljungbox(residuals, lags=1)
            return result['lb_pvalue'].iloc[0]
        except:
            return np.nan
    
    def generate_comprehensive_report(self, training_results, datasets, target_name):
        """Generate comprehensive evaluation report"""
        
        print(f"\n{'='*60}")
        print(f"📊 COMPREHENSIVE MODEL EVALUATION REPORT")
        print(f"🎯 Target: {target_name}")
        print(f"{'='*60}")
        
        report = {
            'target': target_name,
            'model_comparison': None,
            'best_model': None,
            'feature_analysis': {},
            'residual_analysis': {}
        }
        
        # Model comparison
        comparison_df, comparison_plot = self.model_comparison_analysis(
            training_results, target_name
        )
        report['model_comparison'] = comparison_df
        
        # Identify best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_model_results = training_results[best_model_name]
        report['best_model'] = {
            'name': best_model_name,
            'performance': comparison_df.iloc[0].to_dict()
        }
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   Test R²: {comparison_df.iloc[0]['Test_R2']:.4f}")
        print(f"   Test RMSE: {comparison_df.iloc[0]['Test_RMSE']:.4f}")
        
        # Feature importance for best model
        X_test = datasets[target_name]['X_test']
        y_test = datasets[target_name]['y_test']
        feature_names = datasets[target_name]['feature_names']
        best_model = best_model_results['model']
        
        importance_df, importance_plot = self.feature_importance_analysis(
            best_model, X_test, y_test, feature_names, best_model_name
        )
        report['feature_analysis'] = {
            'importance_df': importance_df,
            'plot': importance_plot
        }
        
        # Residual analysis for best model
        y_pred = best_model_results['evaluation']['test']['predictions']
        residual_analysis, residuals = self.residual_analysis(y_test, y_pred, 'Test Set')
        report['residual_analysis'] = residual_analysis
        
        # Prediction plots
        prediction_plot = self.create_prediction_plots(
            y_test, y_pred, 'Test Set', best_model_name
        )
        report['prediction_plot'] = prediction_plot
        
        # Save plots
        self.save_visualizations(report, target_name)
        
        return report
    
    def save_visualizations(self, report, target_name):
        """Save all visualizations"""
        
        # Save comparison plot
        comparison_file = config.RESULTS_DIR / f"{target_name}_model_comparison.html"
        report['model_comparison_plot'].write_html(str(comparison_file))
        
        # Save feature importance plot
        importance_file = config.RESULTS_DIR / f"{target_name}_feature_importance.html"
        report['feature_analysis']['plot'].write_html(str(importance_file))
        
        # Save prediction plot
        prediction_file = config.RESULTS_DIR / f"{target_name}_prediction_analysis.html"
        report['prediction_plot'].write_html(str(prediction_file))
        
        print(f"💾 Visualizations saved to {config.RESULTS_DIR}")

def evaluate_all_models(training_results, datasets):
    """Comprehensive evaluation for all targets"""
    
    evaluator = ComprehensiveModelEvaluator()
    evaluation_reports = {}
    
    for target_name in training_results.keys():
        print(f"\n{'='*60}")
        print(f"Evaluating models for: {target_name}")
        print(f"{'='*60}")
        
        report = evaluator.generate_comprehensive_report(
            training_results[target_name], datasets, target_name
        )
        evaluation_reports[target_name] = report
    
    return evaluation_reports, evaluator
