@dataclass  
class DataConfig:
    # ... existing code ...
    
    development_mode: bool = True  # Allow synthetic data when files missing
    synthetic_data_seed: int = 42
    
    def validate_paths(self) -> None:
        if not self.opm_data_dir.exists():
            raise FileNotFoundError(
                f"OPM data directory not found: {self.opm_data_dir}"
            )
        
        required_files = [
            self.grid_path,
            self.init_path, 
            self.data_path,
            self.unrest_path
        ]
        
        missing_files = [str(f) for f in required_files if not f.exists()]
        
        if missing_files and not self.development_mode:
            raise FileNotFoundError(f"Missing SPE9 files: {missing_files}")
        elif missing_files:
            print(f"⚠️  Missing files but continuing in development mode: {missing_files}")
