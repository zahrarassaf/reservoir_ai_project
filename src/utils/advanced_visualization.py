# src/utils/advanced_visualization.py
"""
Advanced visualization tools for reservoir computing analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from ..utils.metrics import PetroleumMetrics


class ReservoirVisualizer:
    """Advanced visualizations for reservoir computing analysis."""
    
    def __init__(self, style: str = "seaborn"):
        """Initialize visualizer with style."""
        if style == "seaborn":
            sns.set_style("whitegrid")
            sns.set_palette("husl")
        elif style == "plotly":
            self.use_plotly = True
        else:
            plt.style.use(style)
        
        self.use_plotly = False
    
    def plot_reservoir_dynamics(self, model, X: np.ndarray, 
                              save_path: Optional[Path] = None) -> go.Figure:
        """Visualize reservoir state dynamics."""
        if not hasattr(model, 'predict') or not callable(model.predict):
            raise ValueError("Model must have predict method")
        
        # Get predictions and states
        predictions, states = model.predict(X, return_states=True)
        
        if self.use_plotly:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=("Reservoir States", "State Distribution",
                              "State Correlation", "State PCA",
                              "Predictions vs True", "Error Distribution"),
                specs=[[{"type": "heatmap"}, {"type": "histogram"}],
                       [{"type": "heatmap"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "histogram"}]]
            )
            
            # 1. Reservoir states heatmap
            fig.add_trace(
                go.Heatmap(
                    z=states.T,
                    colorscale="Viridis",
                    colorbar=dict(title="Activation"),
                ),
                row=1, col=1
            )
            
            # 2. State distribution
            state_mean = np.mean(states, axis=0)
            fig.add_trace(
                go.Histogram(x=state_mean, nbinsx=50, name="State Distribution"),
                row=1, col=2
            )
            
            # 3. State correlation matrix
            correlation = np.corrcoef(states.T)
            fig.add_trace(
                go.Heatmap(
                    z=correlation,
                    colorscale="RdBu",
                    zmid=0,
                    colorbar=dict(title="Correlation"),
                ),
                row=2, col=1
            )
            
            # 4. PCA of states
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            states_pca = pca.fit_transform(states)
            
            fig.add_trace(
                go.Scatter(
                    x=states_pca[:, 0],
                    y=states_pca[:, 1],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=np.arange(len(states_pca)),
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Time")
                    ),
                    name="State Trajectory"
                ),
                row=2, col=2
            )
            
            # 5. Predictions (if we have true values)
            if hasattr(model, 'last_y_true'):
                y_true = model.last_y_true
                fig.add_trace(
                    go.Scatter(
                        x=y_true.flatten(),
                        y=predictions.flatten(),
                        mode="markers",
                        marker=dict(size=5, opacity=0.5),
                        name="Predictions"
                    ),
                    row=3, col=1
                )
                
                # Add perfect prediction line
                min_val = min(y_true.min(), predictions.min())
                max_val = max(y_true.max(), predictions.max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        name="Perfect"
                    ),
                    row=3, col=1
                )
            
            # 6. Error distribution
            if hasattr(model, 'last_y_true'):
                errors = y_true - predictions
                fig.add_trace(
                    go.Histogram(x=errors.flatten(), nbinsx=50, name="Errors"),
                    row=3, col=2
                )
            
            fig.update_layout(
                height=1200,
                showlegend=True,
                title_text="Reservoir Dynamics Analysis"
            )
            
            if save_path:
                fig.write_html(str(save_path / "reservoir_dynamics.html"))
            
            return fig
        
        else:
            # Matplotlib version
            fig, axes = plt.subplots(3, 2, figsize=(15, 18))
            
            # 1. Reservoir states heatmap
            im = axes[0, 0].imshow(states.T, aspect='auto', cmap='viridis')
            axes[0, 0].set_title("Reservoir States")
            axes[0, 0].set_xlabel("Time")
            axes[0, 0].set_ylabel("Neuron")
            plt.colorbar(im, ax=axes[0, 0])
            
            # 2. State distribution
            state_mean = np.mean(states, axis=0)
            axes[0, 1].hist(state_mean, bins=50, edgecolor='black')
            axes[0, 1].set_title("State Distribution")
            axes[0, 1].set_xlabel("Activation")
            axes[0, 1].set_ylabel("Frequency")
            
            # 3. State correlation matrix
            correlation = np.corrcoef(states.T)
            im = axes[1, 0].imshow(correlation, cmap='RdBu', vmin=-1, vmax=1)
            axes[1, 0].set_title("State Correlation Matrix")
            axes[1, 0].set_xlabel("Neuron")
            axes[1, 0].set_ylabel("Neuron")
            plt.colorbar(im, ax=axes[1, 0])
            
            # 4. PCA of states
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            states_pca = pca.fit_transform(states)
            
            scatter = axes[1, 1].scatter(
                states_pca[:, 0], states_pca[:, 1],
                c=np.arange(len(states_pca)), cmap='viridis', s=3
            )
            axes[1, 1].set_title(f"State PCA (Explained: {pca.explained_variance_ratio_.sum():.2%})")
            axes[1, 1].set_xlabel("PC1")
            axes[1, 1].set_ylabel("PC2")
            plt.colorbar(scatter, ax=axes[1, 1], label="Time")
            
            # 5. Predictions
            if hasattr(model, 'last_y_true'):
                y_true = model.last_y_true
                axes[2, 0].scatter(y_true, predictions, alpha=0.5, s=10)
                
                # Perfect prediction line
                min_val = min(y_true.min(), predictions.min())
                max_val = max(y_true.max(), predictions.max())
                axes[2, 0].plot([min_val, max_val], [min_val, max_val], 
                               'r--', alpha=0.5, label='Perfect')
                
                axes[2, 0].set_title("Predictions vs True")
                axes[2, 0].set_xlabel("True Values")
                axes[2, 0].set_ylabel("Predictions")
                axes[2, 0].legend()
            
            # 6. Error distribution
            if hasattr(model, 'last_y_true'):
                errors = y_true - predictions
                axes[2, 1].hist(errors.flatten(), bins=50, edgecolor='black')
                axes[2, 1].set_title("Error Distribution")
                axes[2, 1].set_xlabel("Error")
                axes[2, 1].set_ylabel("Frequency")
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path / "reservoir_dynamics.png", dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_hyperparameter_sensitivity(self, optimization_results: Dict[str, Any],
                                      save_path: Optional[Path] = None) -> go.Figure:
        """Visualize hyperparameter sensitivity from optimization."""
        
        if 'history' not in optimization_results:
            raise ValueError("Optimization results must contain history")
        
        # Extract data
        history = optimization_results['history']
        param_names = list(history[0]['params'].keys()) if history else []
        
        # Create DataFrame
        data = []
        for entry in history:
            row = entry['params'].copy()
            row['score'] = entry.get('score', 0)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if self.use_plotly:
            # Parallel coordinates plot
            dimensions = []
            for param in param_names:
                dimensions.append(dict(
                    label=param,
                    values=df[param],
                    range=[df[param].min(), df[param].max()]
                ))
            
            dimensions.append(dict(
                label='Score',
                values=df['score'],
                range=[df['score'].min(), df['score'].max()]
            ))
            
            fig = go.Figure(data=
                go.Parcoords(
                    line=dict(
                        color=df['score'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Score')
                    ),
                    dimensions=dimensions
                )
            )
            
            fig.update_layout(
                title="Hyperparameter Sensitivity Analysis",
                height=600
            )
            
            if save_path:
                fig.write_html(str(save_path / "hyperparameter_sensitivity.html"))
            
            return fig
        
        else:
            # Matplotlib version
            n_params = len(param_names)
            fig, axes = plt.subplots(n_params, n_params, figsize=(4*n_params, 4*n_params))
            
            for i, param_i in enumerate(param_names):
                for j, param_j in enumerate(param_names):
                    if i == j:
                        # Diagonal: histogram
                        axes[i, j].hist(df[param_i], bins=20, edgecolor='black')
                        axes[i, j].set_title(param_i)
                    else:
                        # Off-diagonal: scatter plot
                        scatter = axes[i, j].scatter(
                            df[param_j], df[param_i],
                            c=df['score'], cmap='viridis', s=20, alpha=0.6
                        )
                        
                        if j == n_params - 1:
                            plt.colorbar(scatter, ax=axes[i, j], label='Score')
                    
                    # Set labels
                    if i == n_params - 1:
                        axes[i, j].set_xlabel(param_j)
                    if j == 0:
                        axes[i, j].set_ylabel(param_i)
            
            plt.suptitle("Hyperparameter Sensitivity Analysis", y=1.02, fontsize=16)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path / "hyperparameter_sensitivity.png", 
                           dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_forecast_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                             horizon: int = 10, save_path: Optional[Path] = None) -> go.Figure:
        """Analyze forecast performance across horizons."""
        
        if len(y_true.shape) != 3 or len(y_pred.shape) != 3:
            raise ValueError("Inputs must be 3D: (n_sequences, horizon, n_features)")
        
        n_sequences, horizon, n_features = y_true.shape
        
        if self.use_plotly:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Forecast by Horizon", "Error Distribution by Horizon",
                              "Skill Score by Horizon", "Feature-wise Performance"),
                specs=[[{"type": "scatter"}, {"type": "box"}],
                       [{"type": "bar"}, {"type": "heatmap"}]]
            )
            
            # 1. Forecast by horizon
            for h in range(min(horizon, 5)):  # Plot first 5 horizons
                fig.add_trace(
                    go.Scatter(
                        x=y_true[:, h, 0],
                        y=y_pred[:, h, 0],
                        mode="markers",
                        name=f"Horizon {h+1}",
                        opacity=0.6
                    ),
                    row=1, col=1
                )
            
            # Add perfect line
            all_values = np.concatenate([y_true.flatten(), y_pred.flatten()])
            min_val, max_val = all_values.min(), all_values.max()
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="Perfect",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 2. Error distribution by horizon
            errors = []
            horizon_labels = []
            for h in range(horizon):
                error_h = y_true[:, h, 0] - y_pred[:, h, 0]
                errors.append(error_h)
                horizon_labels.append(f"H{h+1}")
            
            fig.add_trace(
                go.Box(
                    y=errors,
                    x=horizon_labels,
                    name="Forecast Errors",
                    boxpoints="outliers"
                ),
                row=1, col=2
            )
            
            # 3. Skill score by horizon
            skill_scores = []
            for h in range(horizon):
                mse_h = np.mean((y_true[:, h, 0] - y_pred[:, h, 0]) ** 2)
                variance_h = np.var(y_true[:, h, 0])
                skill_h = 1 - mse_h / variance_h if variance_h > 0 else 0
                skill_scores.append(skill_h)
            
            fig.add_trace(
                go.Bar(
                    x=list(range(1, horizon + 1)),
                    y=skill_scores,
                    name="Skill Score"
                ),
                row=2, col=1
            )
            
            # 4. Feature-wise performance heatmap
            feature_metrics = []
            feature_names = [f"Feature {i+1}" for i in range(n_features)]
            
            for f in range(n_features):
                metrics = PetroleumMetrics.comprehensive_metrics(
                    y_true[:, :, f], y_pred[:, :, f]
                )
                feature_metrics.append([
                    metrics.get('nash_sutcliffe', 0),
                    metrics.get('r2', 0),
                    metrics.get('mape', 0),
                    metrics.get('rmse', 0),
                ])
            
            feature_metrics = np.array(feature_metrics)
            
            fig.add_trace(
                go.Heatmap(
                    z=feature_metrics.T,
                    x=feature_names,
                    y=["NSE", "R²", "MAPE", "RMSE"],
                    colorscale="RdYlGn",
                    zmid=0,
                    text=np.round(feature_metrics.T, 3),
                    texttemplate="%{text}",
                    textfont={"size": 10}
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=1000,
                title_text="Forecast Performance Analysis"
            )
            
            if save_path:
                fig.write_html(str(save_path / "forecast_analysis.html"))
            
            return fig
        
        else:
            # Matplotlib version
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Forecast by horizon
            for h in range(min(horizon, 5)):
                axes[0, 0].scatter(
                    y_true[:, h, 0], y_pred[:, h, 0],
                    alpha=0.6, s=10, label=f"Horizon {h+1}"
                )
            
            all_values = np.concatenate([y_true.flatten(), y_pred.flatten()])
            min_val, max_val = all_values.min(), all_values.max()
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 
                           'r--', alpha=0.5, label='Perfect')
            axes[0, 0].set_title("Forecast by Horizon")
            axes[0, 0].set_xlabel("True Values")
            axes[0, 0].set_ylabel("Predictions")
            axes[0, 0].legend()
            
            # 2. Error distribution by horizon
            errors = []
            horizon_labels = []
            for h in range(min(horizon, 10)):
                error_h = y_true[:, h, 0] - y_pred[:, h, 0]
                errors.append(error_h)
                horizon_labels.append(f"H{h+1}")
            
            axes[0, 1].boxplot(errors, labels=horizon_labels)
            axes[0, 1].set_title("Error Distribution by Horizon")
            axes[0, 1].set_xlabel("Horizon")
            axes[0, 1].set_ylabel("Error")
            axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            # 3. Skill score by horizon
            skill_scores = []
            for h in range(horizon):
                mse_h = np.mean((y_true[:, h, 0] - y_pred[:, h, 0]) ** 2)
                variance_h = np.var(y_true[:, h, 0])
                skill_h = 1 - mse_h / variance_h if variance_h > 0 else 0
                skill_scores.append(skill_h)
            
            axes[1, 0].bar(range(1, horizon + 1), skill_scores)
            axes[1, 0].set_title("Skill Score by Horizon")
            axes[1, 0].set_xlabel("Horizon")
            axes[1, 0].set_ylabel("Skill Score")
            axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            # 4. Feature-wise performance heatmap
            feature_metrics = []
            for f in range(n_features):
                metrics = PetroleumMetrics.comprehensive_metrics(
                    y_true[:, :, f], y_pred[:, :, f]
                )
                feature_metrics.append([
                    metrics.get('nash_sutcliffe', 0),
                    metrics.get('r2', 0),
                    metrics.get('mape', 0),
                    metrics.get('rmse', 0),
                ])
            
            feature_metrics = np.array(feature_metrics)
            
            im = axes[1, 1].imshow(feature_metrics.T, cmap='RdYlGn', vmin=-1, vmax=1)
            axes[1, 1].set_title("Feature-wise Performance")
            axes[1, 1].set_xlabel("Feature")
            axes[1, 1].set_ylabel("Metric")
            axes[1, 1].set_xticks(range(n_features))
            axes[1, 1].set_xticklabels([f"F{i+1}" for i in range(n_features)])
            axes[1, 1].set_yticks(range(4))
            axes[1, 1].set_yticklabels(["NSE", "R²", "MAPE", "RMSE"])
            
            # Add text annotations
            for i in range(n_features):
                for j in range(4):
                    text = f"{feature_metrics[i, j]:.2f}"
                    axes[1, 1].text(i, j, text, ha='center', va='center', fontsize=8)
            
            plt.colorbar(im, ax=axes[1, 1])
            
            plt.suptitle("Forecast Performance Analysis", fontsize=16, y=1.02)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path / "forecast_analysis.png", 
                           dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_model_comparison(self, results_dict: Dict[str, Dict[str, Any]],
                            save_path: Optional[Path] = None) -> go.Figure:
        """Compare multiple models using radar chart."""
        
        # Extract metrics for each model
        model_names = list(results_dict.keys())
        metric_names = ["NSE", "R²", "Skill", "MAPE", "MBE"]
        
        metrics_data = []
        for model_name in model_names:
            model_results = results_dict[model_name]
            
            # Extract or compute metrics
            metrics = model_results.get('metrics', {})
            
            model_metrics = [
                metrics.get('nash_sutcliffe', 0),
                metrics.get('r2', 0),
                metrics.get('forecast_skill_score_mean', 0),
                1 / (1 + metrics.get('mape', 1)),  # Invert MAPE (higher is better)
                1 / (1 + metrics.get('material_balance_error_pred', 1)),  # Invert MBE
            ]
            
            # Normalize to [0, 1]
            model_metrics = [(m + 1) / 2 for m in model_metrics]  # Map from [-1,1] to [0,1]
            metrics_data.append(model_metrics)
        
        if self.use_plotly:
            fig = go.Figure()
            
            for i, (model_name, model_metrics) in enumerate(zip(model_names, metrics_data)):
                # Close the radar chart
                radar_metrics = model_metrics + [model_metrics[0]]
                radar_angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()
                radar_angles += [radar_angles[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=radar_metrics,
                    theta=metric_names + [metric_names[0]],
                    name=model_name,
                    fill='toself',
                    opacity=0.6
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                title="Model Comparison Radar Chart",
                height=600
            )
            
            if save_path:
                fig.write_html(str(save_path / "model_comparison.html"))
            
            return fig
        
        else:
            # Matplotlib radar chart
            from math import pi
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            # Set angles
            angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()
            angles += angles[:1]  # Close the circle
            
            # Plot each model
            for i, (model_name, model_metrics) in enumerate(zip(model_names, metrics_data)):
                # Close the radar chart
                radar_metrics = model_metrics + [model_metrics[0]]
                
                ax.plot(angles, radar_metrics, 'o-', linewidth=2, label=model_name)
                ax.fill(angles, radar_metrics, alpha=0.1)
            
            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_names)
            ax.set_ylim(0, 1)
            ax.set_title("Model Comparison Radar Chart", size=16, y=1.1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path / "model_comparison.png", 
                           dpi=300, bbox_inches='tight')
            
            return fig
