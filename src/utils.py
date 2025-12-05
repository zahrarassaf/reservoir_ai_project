"""
Utility functions for reservoir simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)


def validate_data(data: Dict) -> bool:
    """
    Validate reservoir data
    
    Parameters
    ----------
    data : Dict
        Reservoir data
        
    Returns
    -------
    bool
        True if data is valid
    """
    required_keys = ['time', 'production', 'pressure']
    
    for key in required_keys:
        if key not in data:
            logger.error(f"Missing required data: {key}")
            return False
    
    # Check data types
    if not isinstance(data['time'], np.ndarray):
        logger.error("Time data must be numpy array")
        return False
    
    if not isinstance(data['production'], pd.DataFrame):
        logger.error("Production data must be pandas DataFrame")
        return False
    
    if not isinstance(data['pressure'], np.ndarray):
        logger.error("Pressure data must be numpy array")
        return False
    
    # Check data consistency
    if len(data['time']) != len(data['production']):
        logger.warning("Time and production data have different lengths")
    
    if len(data['time']) != len(data['pressure']):
        logger.warning("Time and pressure data have different lengths")
    
    return True


def clean_production_data(production_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean production data
    
    Parameters
    ----------
    production_df : pd.DataFrame
        Production data
        
    Returns
    -------
    pd.DataFrame
        Cleaned production data
    """
    df = production_df.copy()
    
    # Convert to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Fill missing values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Remove negative values
    df = df.clip(lower=0)
    
    # Remove outliers (values > 3 standard deviations)
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = df[col].clip(lower=0, upper=mean + 3*std)
    
    return df


def clean_pressure_data(pressure: np.ndarray) -> np.ndarray:
    """
    Clean pressure data
    
    Parameters
    ----------
    pressure : np.ndarray
        Pressure data
        
    Returns
    -------
    np.ndarray
        Cleaned pressure data
    """
    # Remove NaN values
    pressure_clean = pressure[~np.isnan(pressure)]
    
    # Remove negative values
    pressure_clean = pressure_clean[pressure_clean > 0]
    
    # Remove outliers
    if len(pressure_clean) > 10:
        mean = np.mean(pressure_clean)
        std = np.std(pressure_clean)
        pressure_clean = pressure_clean[(pressure_clean > mean - 3*std) & 
                                       (pressure_clean < mean + 3*std)]
    
    return pressure_clean


def interpolate_missing_data(data: np.ndarray) -> np.ndarray:
    """
    Interpolate missing data
    
    Parameters
    ----------
    data : np.ndarray
        Data with missing values
        
    Returns
    -------
    np.ndarray
        Interpolated data
    """
    if len(data) == 0:
        return data
    
    # Create mask for valid data
    valid_mask = ~np.isnan(data)
    
    if np.sum(valid_mask) == 0:
        return data
    
    if np.sum(valid_mask) == len(data):
        return data
    
    # Get indices
    indices = np.arange(len(data))
    
    # Interpolate
    interpolated = np.interp(indices, indices[valid_mask], data[valid_mask])
    
    return interpolated


def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistics
    
    Parameters
    ----------
    data : np.ndarray
        Input data
        
    Returns
    -------
    Dict
        Statistics
    """
    if len(data) == 0:
        return {}
    
    data_clean = data[~np.isnan(data)]
    
    if len(data_clean) == 0:
        return {}
    
    return {
        'mean': float(np.mean(data_clean)),
        'std': float(np.std(data_clean)),
        'min': float(np.min(data_clean)),
        'max': float(np.max(data_clean)),
        'median': float(np.median(data_clean)),
        'count': int(len(data_clean))
    }


def export_to_json(data: Dict, filepath: str) -> bool:
    """
    Export data to JSON file
    
    Parameters
    ----------
    data : Dict
        Data to export
    filepath : str
        Output file path
        
    Returns
    -------
    bool
        True if successful
    """
    try:
        def convert_for_json(obj):
            if isinstance(obj, (np.ndarray, pd.Series)):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, (pd.Timestamp, np.datetime64)):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            else:
                return obj
        
        json_data = convert_for_json(data)
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Data exported to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export to JSON: {e}")
        return False


def export_to_csv(data: pd.DataFrame, filepath: str) -> bool:
    """
    Export DataFrame to CSV
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to export
    filepath : str
        Output file path
        
    Returns
    -------
    bool
        True if successful
    """
    try:
        data.to_csv(filepath, index=False)
        logger.info(f"Data exported to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export to CSV: {e}")
        return False


def load_from_json(filepath: str) -> Optional[Dict]:
    """
    Load data from JSON file
    
    Parameters
    ----------
    filepath : str
        Input file path
        
    Returns
    -------
    Dict or None
        Loaded data
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Data loaded from {filepath}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load from JSON: {e}")
        return None


def load_from_pickle(filepath: str) -> Optional[Any]:
    """
    Load data from pickle file
    
    Parameters
    ----------
    filepath : str
        Input file path
        
    Returns
    -------
    Any or None
        Loaded data
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Data loaded from {filepath}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load from pickle: {e}")
        return None


def setup_logging(log_file: Optional[str] = None, 
                 level: str = 'INFO') -> logging.Logger:
    """
    Setup logging configuration
    
    Parameters
    ----------
    log_file : str, optional
        Log file path
    level : str
        Logging level
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    # Create logger
    logger = logging.getLogger('reservoir_simulation')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
