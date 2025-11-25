"""
Advanced SHAP Analysis Module
Model interpretability and explanation using SHAP values
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

from .config import config

class SHAPAnalyzer:
    """
    Advanced SHAP analysis for model interpretability
    Provides comprehensive explanations for model predictions
    """
    
    def __init__(self):
        self.shap_explainers = {}
        self.shap_values = {}
        self.analysis_results = {}
        
    def initialize_shap_explainer(self, model, X, model_name):
        """Initialize appropriate SHAP explainer based on model type"""
        
        print(f"🔍 Initializing SHAP explainer for {model_name}...")
        
        try:
            if hasattr(model, 'predict_proba') or 'tree' in str(type(model)).lower():
                # Tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
            elif hasattr(model, 'coef_'):
                # Linear models
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer.shap_values(X)
                
            else:
                # Kernel explainer for other models
                explainer = shap.KernelExplainer(model.predict, X)
                shap_values = explainer.shap_values(X)
                
            self.shap_explainers[model_name] = explainer
            self.shap_values[model_name] = shap_values
            
            print(f"✅ SHAP explainer initialized for {model_name}")
            return explainer, shap_values
            
        except Exception as e:
            print(f"❌ Error initializing SHAP for {model_name}: {e}")
            return None, None
    
    def create_summary_plot(self, shap_values, X, feature_names, model_name):
        """Create SHAP summary plot"""
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = config.RESULTS_DIR / f"{model_name}_shap_summary.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved SHAP summary plot: {filename}")
        return filename
    
    def create_feature_importance_plot(self, shap_values, feature_names, model_name):
        """Create interactive feature importance plot"""
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap,
            'importance_percentage': (mean_abs_shap / mean_abs_shap.sum()) * 100
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Create interactive plot
        fig = px.bar(
            importance_df.head(15),
            x='importance_percentage',
            y='feature',
            orientation='h',
            title=f'SHAP Feature Importance - {model_name}',
            labels={'importance_percentage': 'Importance (%)', 'feature': 'Features'},
            color='importance_percentage',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500
        )
        
        # Save interactive plot
        filename = config.RESULTS_DIR / f"{model_name}_shap_importance.html"
        fig.write_html(str(filename))
        
        print("📊 SHAP Feature Importance:")
        for idx, row in importance_df.head().iterrows():
            print(f"   {row['feature']}: {row['importance_percentage']:.2f}%")
        
        return importance_df, fig
    
    def create_dependence_plots(self, shap_values, X, feature_names, model_name, top_features=4):
        """Create dependence plots for top features"""
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        top_feature_names = importance_df.head(top_features)['feature'].tolist()
        
        dependence_plots = {}
        
        for feature in top_feature_names:
            try:
                feature_idx = feature_names.index(feature)
                
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    feature_idx, shap_values, X, 
                    feature_names=feature_names,
                    show=False
                )
                plt.title(f'SHAP Dependence Plot - {feature}', fontweight='bold')
                plt.tight_layout()
                
                # Save plot
                filename = config.RESULTS_DIR / f"{model_name}_dependence_{feature}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                dependence_plots[feature] = filename
                
            except Exception as e:
                print(f"❌ Error creating dependence plot for {feature}: {e}")
                continue
        
        return dependence_plots
    
    def create_force_plot(self, explainer, shap_values, X, instance_idx, feature_names, model_name):
        """Create force plot for individual prediction explanation"""
        
        try:
            # Create force plot
            plt.figure(figsize=(12, 4))
            shap.force_plot(
                explainer.expected_value, 
                shap_values[instance_idx, :], 
                X[instance_idx, :],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f'SHAP Force Plot - Instance {instance_idx}', fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            filename = config.RESULTS_DIR / f"{model_name}_force_plot_{instance_idx}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"❌ Error creating force plot: {e}")
            return None
    
    def create_waterfall_plot(self, explainer, shap_values, X, instance_idx, feature_names, model_name):
        """Create waterfall plot for individual prediction"""
        
        try:
            # Create waterfall plot
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(
                explainer.expected_value,
                shap_values[instance_idx, :],
                X[instance_idx, :],
                feature_names=feature_names,
                show=False
            )
            plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}', fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            filename = config.RESULTS_DIR / f"{model_name}_waterfall_{instance_idx}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"❌ Error creating waterfall plot: {e}")
            return None
    
    def analyze_feature_interactions(self, shap_values, X, feature_names, model_name):
        """Analyze feature interactions using SHAP"""
        
        try:
            # Calculate interaction values
            interaction_values = shap.TreeExplainer(model).shap_interaction_values(X)
            
            # Find strongest interactions
            interaction_strength = np.abs(interaction_values).sum(axis=0)
            np.fill_diagonal(interaction_strength, 0)  # Remove self-interactions
            
            # Get top interactions
            interactions_list = []
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    interactions_list.append({
                        'feature_i': feature_names[i],
                        'feature_j': feature_names[j],
                        'interaction_strength': interaction_strength[i, j]
                    })
            
            interactions_df = pd.DataFrame(interactions_list)
            interactions_df = interactions_df.sort_values('interaction_strength', ascending=False)
            
            print("🔗 Top Feature Interactions:")
            for idx, row in interactions_df.head(5).iterrows():
                print(f"   {row['feature_i']} & {row['feature_j']}: {row['interaction_strength']:.4f}")
            
            return interactions_df.head(10)
            
        except Exception as e:
            print(f"❌ Error analyzing interactions: {e}")
            return pd.DataFrame()
    
    def generate_comprehensive_shap_report(self, model, X, y, feature_names, model_name, target_name):
        """Generate comprehensive SHAP analysis report"""
        
        print(f"\n{'='*60}")
        print(f"🔍 COMPREHENSIVE SHAP ANALYSIS")
        print(f"🤖 Model: {model_name}")
        print(f"🎯 Target: {target_name}")
        print(f"{'='*60}")
        
        report = {
            'model_name': model_name,
            'target_name': target_name,
            'feature_importance': None,
            'summary_plot': None,
            'dependence_plots': {},
            'force_plots': {},
            'interaction_analysis': None
        }
        
        # Initialize SHAP explainer
        explainer, shap_vals = self.initialize_shap_explainer(model, X, model_name)
        
        if explainer is None:
            print("❌ SHAP analysis failed - skipping...")
            return report
        
        # 1. Summary plot
        summary_plot_file = self.create_summary_plot(shap_vals, X, feature_names, model_name)
        report['summary_plot'] = summary_plot_file
        
        # 2. Feature importance
        importance_df, importance_plot = self.create_feature_importance_plot(
            shap_vals, feature_names, model_name
        )
        report['feature_importance'] = {
            'dataframe': importance_df,
            'plot': importance_plot
        }
        
        # 3. Dependence plots
        dependence_plots = self.create_dependence_plots(
            shap_vals, X, feature_names, model_name
        )
        report['dependence_plots'] = dependence_plots
        
        # 4. Individual prediction explanations (for first 3 instances)
        report['force_plots'] = {}
        for i in range(min(3, len(X))):
            force_plot = self.create_force_plot(
                explainer, shap_vals, X, i, feature_names, model_name
            )
            if force_plot:
                report['force_plots'][f'instance_{i}'] = force_plot
        
        # 5. Interaction analysis (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            interactions_df = self.analyze_feature_interactions(
                shap_vals, X, feature_names, model_name
            )
            report['interaction_analysis'] = interactions_df
        
        # 6. Global model insights
        report['global_insights'] = self.extract_global_insights(importance_df, shap_vals, feature_names)
        
        print(f"✅ SHAP analysis completed for {model_name}")
        return report
    
    def extract_global_insights(self, importance_df, shap_values, feature_names):
        """Extract global insights from SHAP analysis"""
        
        insights = {
            'top_features': importance_df.head(5)['feature'].tolist(),
            'feature_directions': {},
            'model_behavior': ''
        }
        
        # Analyze feature directions
        mean_shap_values = np.mean(shap_values, axis=0)
        
        for feature, mean_shap in zip(feature_names, mean_shap_values):
            direction = "positive" if mean_shap > 0 else "negative"
            insights['feature_directions'][feature] = {
                'direction': direction,
                'impact_strength': abs(mean_shap)
            }
        
        # Overall model behavior
        positive_impact_features = sum(1 for vals in insights['feature_directions'].values() 
                                     if vals['direction'] == 'positive')
        total_features = len(insights['feature_directions'])
        
        if positive_impact_features / total_features > 0.7:
            insights['model_behavior'] = "Mostly positive feature impacts"
        elif positive_impact_features / total_features < 0.3:
            insights['model_behavior'] = "Mostly negative feature impacts"
        else:
            insights['model_behavior'] = "Balanced positive and negative impacts"
        
        print("💡 Model Insights:")
        print(f"   • Top features: {', '.join(insights['top_features'])}")
        print(f"   • Behavior: {insights['model_behavior']}")
        
        return insights
    
    def run_comprehensive_analysis(self, training_results, datasets):
        """Run comprehensive SHAP analysis for all best models"""
        
        shap_reports = {}
        
        for target_name, results in training_results.items():
            # Get best model
            best_model_name = max(results.keys(), 
                                key=lambda x: results[x]['evaluation']['test']['metrics']['r2'])
            best_model = results[best_model_name]['model']
            
            # Get test data for explanation
            X_test = datasets[target_name]['X_test']
            y_test = datasets[target_name]['y_test']
            feature_names = datasets[target_name]['feature_names']
            
            print(f"\n🎯 Analyzing best model for {target_name}: {best_model_name}")
            
            # Generate SHAP report
            shap_report = self.generate_comprehensive_shap_report(
                best_model, X_test, y_test, feature_names, best_model_name, target_name
            )
            
            shap_reports[target_name] = shap_report
        
        return shap_reports

def perform_complete_shap_analysis(training_results, datasets):
    """Complete SHAP analysis pipeline"""
    
    analyzer = SHAPAnalyzer()
    shap_reports = analyzer.run_comprehensive_analysis(training_results, datasets)
    
    return shap_reports, analyzer
