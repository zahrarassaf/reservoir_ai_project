from dataclasses import dataclass
from typing import List
import torch

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    use_scheduler: bool = True
    scheduler_type: str = "reduce_on_plateau"
    patience: int = 5
    factor: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_dir: str = "checkpoints"
    save_frequency: int = 10
    
    def get_optimizer_config(self):
        return {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay
        }
