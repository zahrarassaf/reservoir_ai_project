import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional
import seaborn as sns

def plot_reservoir_properties(properties: Dict[str, np.ndarray], 
                            layer: int = 0,
                            figsize: tuple = (15, 10)):
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()
    
    property_keys = ['PORO', 'PERMX', 'PERMY', 'PERMZ', 'PRESSURE', 'SWAT']
    
    for idx, prop_key in enumerate(property_keys):
        if prop_key in properties:
            data = properties[prop_key]
            if len(data.shape) == 3:
                im = axes[idx].imshow(data[:, :, layer], cmap='viridis')
                axes[idx].set_title(f'{prop_key} - Layer {layer}')
                plt.colorbar(im, ax=axes[idx])
            else:
                axes[idx].text(0.5, 0.5, f'Invalid shape: {data.shape}', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(prop_key)
        
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    
    plt.tight_layout()
    return fig

def plot_training_history(training_history: Dict[str, List[float]], 
                         figsize: tuple = (12, 8)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot training loss
    if 'train_loss' in training_history:
        for i, model_losses in enumerate(training_history['train_loss']):
            ax1.plot(model_losses, label=f'Model {i+1}', alpha=0.7)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot validation loss
    if 'val_loss' in training_history and training_history['val_loss']:
        for i, model_losses in enumerate(training_history['val_loss']):
            if model_losses:
                ax2.plot(model_losses, label=f'Model {i+1}', alpha=0.7)
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_uncertainty(predictions: Dict[str, torch.Tensor],
                    targets: torch.Tensor,
                    time_steps: Optional[List[int]] = None,
                    figsize: tuple = (15, 10)):
    if time_steps is None:
        time_steps = list(range(predictions['mean'].shape[1]))
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    
    # Plot mean predictions vs targets
    mean_pred = predictions['mean'].cpu().numpy()
    std_pred = predictions['std'].cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Take first sample for visualization
    sample_idx = 0
    feature_idx = 0  # First output feature (pressure)
    
    # Prediction with uncertainty
    axes[0].plot(time_steps, mean_pred[sample_idx, :, feature_idx], 
                'b-', label='Prediction', linewidth=2)
    axes[0].fill_between(time_steps,
                        mean_pred[sample_idx, :, feature_idx] - 2 * std_pred[sample_idx, :, feature_idx],
                        mean_pred[sample_idx, :, feature_idx] + 2 * std_pred[sample_idx, :, feature_idx],
                        alpha=0.3, label='±2σ Uncertainty')
    axes[0].plot(time_steps, targets_np[sample_idx, :, feature_idx], 
                'r--', label='True', linewidth=2)
    axes[0].set_title('Predictions with Uncertainty')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Uncertainty over time
    axes[1].plot(time_steps, std_pred[sample_idx, :, feature_idx], 
                'g-', linewidth=2)
    axes[1].set_title('Uncertainty Over Time')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].grid(True, alpha=0.3)
    
    # Error distribution
    errors = mean_pred[sample_idx, :, feature_idx] - targets_np[sample_idx, :, feature_idx]
    axes[2].hist(errors, bins=20, alpha=0.7, edgecolor='black')
    axes[2].set_title('Prediction Error Distribution')
    axes[2].set_xlabel('Error')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, alpha=0.3)
    
    # Calibration plot
    sorted_errors = np.sort(np.abs(errors))
    confidence_levels = np.linspace(0, 1, len(sorted_errors))
    axes[3].plot(confidence_levels, sorted_errors, 'purple', linewidth=2)
    axes[3].set_title('Calibration Plot')
    axes[3].set_xlabel('Confidence Level')
    axes[3].set_ylabel('Absolute Error')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
