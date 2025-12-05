"""
Reservoir Data Loader Module

This module handles loading, validating, and preprocessing reservoir data
from various sources including Google Drive, local files, and databases.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings

try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False
    warnings.warn("gdown not installed. Google Drive download disabled.")

logger = logging.getLogger(__name__)


@dataclass
class ReservoirDataConfig:
    """Configuration for reservoir data loading"""
    production_columns: List[str] = None
    pressure_columns: List[str] = None
    injection_columns: List[str] = None
    petrophysical_columns: List[str] = None
    time_column: str = "time"
    date_format: str = "%Y-%m-%d"
    default_units: Dict = None
    
    def __post_init__(self):
        if self.default_units is None:
            self.default_units = {
                "production": "bbl/day",
                "pressure": "psi",
                "injection": "bbl/day",
                "porosity": "fraction",
                "permeability": "mD",
                "thickness": "ft"
            }


class ReservoirDataLoader:
    """
    Load and manage reservoir simulation data from multiple sources.
    
    This class handles:
    - Data loading from Google Drive links
    - Data validation and cleaning
    - Unit conversion
    - Time series alignment
    - Missing data handling
    """
    
    def __init__(self, config: Optional[ReservoirDataConfig] = None):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        config : ReservoirDataConfig, optional
            Configuration for data loading
        """
        self.config = config or ReservoirDataConfig()
        self.data = {}
        self.metadata = {}
        logger.info("ReservoirDataLoader initialized")
    
    def load_from_drive(self, drive_links: List[str], 
                       download_dir: str = "./data/raw") -> Dict:
        """
        Load data from Google Drive links.
        
        Parameters
        ----------
        drive_links : List[str]
            List of Google Drive file links
        download_dir : str
            Directory to download files to
            
        Returns
        -------
        Dict
            Dictionary containing all loaded data
        """
        if not HAS_GDOWN:
            raise ImportError("gdown required for Google Drive downloads")
        
        logger.info(f"Loading data from {len(drive_links)} Google Drive links")
        
        downloaded_files = []
        for i, link in enumerate(drive_links):
            try:
                file_id = self._extract_file_id(link)
                output_path = Path(download_dir) / f"reservoir_data_{i}.csv"
                
                logger.info(f"Downloading file {i+1}: {file_id}")
                gdown.download(f"https://drive.google.com/uc?id={file_id}", 
                             str(output_path), quiet=False)
                downloaded_files.append(output_path)
                
            except Exception as e:
                logger.error(f"Failed to download {link}: {e}")
                continue
        
        return self._process_downloaded_files(downloaded_files)
    
    def load_from_csv(self, file_paths: Dict[str, str]) -> Dict:
        """
        Load data from CSV files.
        
        Parameters
        ----------
        file_paths : Dict[str, str]
            Dictionary mapping data types to file paths
            
        Returns
        -------
        Dict
            Dictionary containing loaded data
        """
        logger.info("Loading data from CSV files")
        
        data = {}
        for data_type, file_path in file_paths.items():
            try:
                df = pd.read_csv(file_path)
                data[data_type] = self._validate_data(df, data_type)
                logger.info(f"Loaded {data_type}: {df.shape}")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        return self._align_data(data)
    
    def load_synthetic_data(self, 
                           n_wells: int = 8,
                           n_layers: int = 5,
                           n_days: int = 1826) -> Dict:
        """
        Generate synthetic reservoir data for testing.
        
        Parameters
        ----------
        n_wells : int
            Number of production wells
        n_layers : int
            Number of reservoir layers
        n_days : int
            Number of days in simulation
            
        Returns
        -------
        Dict
            Synthetic reservoir data
        """
        logger.info(f"Generating synthetic data: {n_wells} wells, {n_layers} layers")
        
        np.random.seed(42)
        time = np.linspace(0, 365*5, n_days)
        
        # Generate production data
        production_data = {}
        for i in range(n_wells):
            base_rate = np.random.uniform(1000, 5000)
            decline_rate = np.random.uniform(0.001, 0.01)
            
            rate = base_rate * np.exp(-decline_rate * time)
            noise = np.random.normal(0, 100, len(time))
            rate = np.maximum(rate + noise, 0)
            production_data[f'Well_{i+1}'] = rate
        
        # Generate pressure data
        initial_pressure = 4500
        pressure_decline = 0.5
        reservoir_pressure = initial_pressure - pressure_decline * (time / 365)
        reservoir_pressure += np.random.normal(0, 50, len(time))
        
        # Generate injection data
        injection_data = {}
        for i in range(3):
            base_injection = np.random.uniform(2000, 4000)
            injection = base_injection * (1 + 0.1 * np.sin(2 * np.pi * time / 365))
            injection_data[f'Inj_{i+1}'] = injection
        
        # Generate petrophysical data
        petrophysical_data = {
            'Porosity': np.random.uniform(0.15, 0.25, n_layers),
            'Permeability': np.random.lognormal(2, 0.5, n_layers),
            'NetThickness': np.random.uniform(10, 50, n_layers),
            'WaterSaturation': np.random.uniform(0.2, 0.4, n_layers)
        }
        
        data = {
            'time': time,
            'production': pd.DataFrame(production_data, index=time),
            'pressure': reservoir_pressure,
            'injection': pd.DataFrame(injection_data, index=time),
            'petrophysical': pd.DataFrame(petrophysical_data),
            'n_wells': n_wells,
            'n_layers': n_layers,
            'metadata': {
                'data_type': 'synthetic',
                'generation_date': pd.Timestamp.now(),
                'parameters': {
                    'n_wells': n_wells,
                    'n_layers': n_layers,
                    'n_days': n_days
                }
            }
        }
        
        logger.info("Synthetic data generation complete")
        return data
    
    def _extract_file_id(self, drive_link: str) -> str:
        """Extract file ID from Google Drive link."""
        # Handle different Google Drive link formats
        if "/file/d/" in drive_link:
            start = drive_link.find("/file/d/") + 8
            end = drive_link.find("/", start)
            return drive_link[start:end]
        elif "id=" in drive_link:
            return drive_link.split("id=")[-1].split("&")[0]
        else:
            raise ValueError(f"Invalid Google Drive link format: {drive_link}")
    
    def _validate_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Validate and clean data based on type."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values based on data type
        if data_type == 'production':
            # Forward fill for production data
            df = df.fillna(method='ffill').fillna(0)
        elif data_type == 'pressure':
            # Interpolate pressure data
            df = df.interpolate(method='linear')
        
        # Validate numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Replace negative values with NaN for certain data types
            if data_type in ['production', 'injection']:
                df[col] = df[col].clip(lower=0)
        
        return df
    
    def _align_data(self, data: Dict) -> Dict:
        """Align all time series data to common time index."""
        # Find common time index
        time_series = []
        for key, df in data.items():
            if isinstance(df, pd.DataFrame) and 'time' in df.columns:
                time_series.append(df['time'])
        
        if time_series:
            # Use the longest time series as reference
            reference_time = max(time_series, key=len)
            
            # Align all dataframes to reference time
            for key, df in data.items():
                if isinstance(df, pd.DataFrame) and 'time' in df.columns:
                    data[key] = self._align_to_reference(df, reference_time)
        
        return data
    
    def _align_to_reference(self, df: pd.DataFrame, reference_time: pd.Series) -> pd.DataFrame:
        """Align dataframe to reference time series."""
        aligned = pd.DataFrame(index=reference_time)
        
        for col in df.columns:
            if col != 'time':
                # Interpolate to align with reference time
                aligned[col] = np.interp(reference_time, df['time'], df[col])
        
        aligned['time'] = reference_time
        return aligned
    
    def _process_downloaded_files(self, file_paths: List[Path]) -> Dict:
        """Process downloaded files into structured data."""
        # This is a placeholder - actual implementation depends on file formats
        data = {}
        
        for file_path in file_paths:
            try:
                # Attempt to detect file type and load accordingly
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                    # Simple detection logic - adjust based on actual data
                    if 'pressure' in df.columns.str.lower().any():
                        data['pressure'] = df
                    elif any('prod' in col.lower() for col in df.columns):
                        data['production'] = df
                    elif any('inj' in col.lower() for col in df.columns):
                        data['injection'] = df
                elif file_path.suffix in ['.xlsx', '.xls']:
                    # Handle Excel files
                    xls = pd.ExcelFile(file_path)
                    for sheet_name in xls.sheet_names:
                        df = pd.read_excel(xls, sheet_name)
                        # Add to data based on sheet name or content
                        data[sheet_name.lower()] = df
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        return self._align_data(data)
    
    def get_data_summary(self) -> pd.DataFrame:
        """Generate summary statistics for loaded data."""
        summary_data = []
        
        for key, value in self.data.items():
            if isinstance(value, pd.DataFrame):
                summary_data.append({
                    'dataset': key,
                    'rows': len(value),
                    'columns': len(value.columns),
                    'missing_values': value.isnull().sum().sum(),
                    'data_types': str(value.dtypes.to_dict())
                })
        
        return pd.DataFrame(summary_data)
