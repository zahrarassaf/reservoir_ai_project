from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing"""
    # Paths
    data_dir: Path = Path("data/spe9")
    grid_file: str = "SPE9.GRID"
    init_file: str = "SPE9.INIT" 
    restart_file: str = "SPE9.X0000"
    
    # Preprocessing
    normalize_porosity: bool = True
    normalize_permeability: bool = True
    clip_extreme_values: bool = True
    
    # Splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Temporal
    sequence_length: int = 10
    prediction_horizon: int = 5
    
    def validate_paths(self) -> None:
        """Validate that all data files exist"""
        required_files = [
            self.data_dir / self.grid_file,
            self.data_dir / self.init_file,
            self.data_dir / self.restart_file
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required data file not found: {file_path}"
                )
