from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

@dataclass
class DataConfig:
    opm_data_dir: Path = Path("../../opm-data")
    spe9_case: str = "spe9"
    
    grid_file: str = "SPE9.GRID"
    init_file: str = "SPE9.INIT"
    data_file: str = "SPE9.DATA"
    unrest_file: str = "SPE9.UNRST"
    
    extract_properties: List[str] = None
    time_steps: List[int] = None
    
    normalize_features: bool = True
    remove_outliers: bool = True
    outlier_std_threshold: float = 3.0
    
    sequence_length: int = 10
    prediction_horizon: int = 5
    stride: int = 1
    
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    temporal_split: bool = True
    
    def __post_init__(self):
        if self.extract_properties is None:
            self.extract_properties = [
                'PRESSURE', 'SWAT', 'SOIL', 
                'PORO', 'PERMX', 'PERMY', 'PERMZ'
            ]
        if self.time_steps is None:
            self.time_steps = list(range(0, 100, 10))
    
    @property
    def spe9_directory(self) -> Path:
        return self.opm_data_dir / self.spe9_case
    
    def validate_paths(self) -> None:
        if not self.opm_data_dir.exists():
            raise FileNotFoundError(
                f"OPM data directory not found: {self.opm_data_dir}"
            )
