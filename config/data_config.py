from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class DataConfig:
    # Use the CORRECT path to OPM data
    opm_data_dir: Path = Path("opm-data")  # Changed to relative path
    spe9_case: str = "spe9"
    
    # REAL file names from OPM repository
    grid_file: str = "SPE9.GRID"
    init_file: str = "SPE9.INIT"
    data_file: str = "SPE9.DATA"
    unrest_file: str = "SPE9.UNRST"
    
    # Use REAL data now
    development_mode: bool = False  # Changed to False for real data
    
    # Data extraction settings
    extract_properties: List[str] = None
    time_steps: List[int] = None
    normalize_features: bool = True
    sequence_length: int = 10
    prediction_horizon: int = 5
    
    def __post_init__(self):
        if self.extract_properties is None:
            self.extract_properties = [
                'PRESSURE', 'SWAT', 'SOIL',  # Dynamic properties
                'PORO', 'PERMX', 'PERMY', 'PERMZ'  # Static properties
            ]
        if self.time_steps is None:
            self.time_steps = list(range(0, 10))  # First 10 time steps

    @property
    def spe9_directory(self) -> Path:
        return self.opm_data_dir / self.spe9_case

    @property
    def grid_path(self) -> Path:
        return self.spe9_directory / self.grid_file

    @property
    def init_path(self) -> Path:
        return self.spe9_directory / self.init_file

    @property
    def unrest_path(self) -> Path:
        return self.spe9_directory / self.unrest_path

    def validate_paths(self) -> None:
        """Validate that REAL SPE9 files exist"""
        if not self.opm_data_dir.exists():
            raise FileNotFoundError(
                f"OPM data directory not found: {self.opm_data_dir}\n"
                f"Clone it with: git clone https://github.com/OPM/opm-data.git"
            )
        
        required_files = [
            self.grid_path,
            self.init_path,
            self.data_path,
            self.unrest_path
        ]
        
        missing_files = [str(f) for f in required_files if not f.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing SPE9 files: {missing_files}")
