from dataclasses import dataclass
from typing import List
import torch

@dataclass
class TrainingConfig:
    """Configuration for training procedure"""
    # Basic training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    
    # Optimization
    optimizer: str = "adam"  # "adam", "sgd", "rmsprop"
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "reduce_on_plateau"  # "step", "cosine", "plateau"
    patience: int = 5
    factor: float = 0.5
    
    # Device and reproducibility
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_frequency: int = 10
    
    def get_optimizer_config(self):
        """Get optimizer configuration dictionary"""
        return {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay
        }
