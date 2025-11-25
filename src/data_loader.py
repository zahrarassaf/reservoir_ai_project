"""
Data loading and synthetic data generation for SPE9-like reservoir data
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os

from .config import config

class DataLoader:
    """Handles data loading and generation for reservoir forecasting"""
    
    def __init__(self):
        self.data = None
    
    def generate_spe9_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic data that mimics SPE9 reservoir characteristics
        Based on SPE9 benchmark model specifications
        """
        np.random.seed(config.RANDOM_STATE)
        
        data_records = []
        
        for time_step in range(config.TIME_STEPS):
            # Field-level trends based on reservoir simulation behavior
            field_pressure_trend = 3600 - time_step * 1.5 + 45 * np.sin(0.04 * time_step)
            field_flow_trend = 5000 - time_step * 8 + 200 * np.sin(0.03 * time_step)
            
            for well_id in range(config.N_WELLS):
                # Well-specific variations
                well_bias = np.random.uniform(-15, 15)
                reservoir_quality = np.random.uniform(0.7, 1.3)
                
                # Pressure calculations with reservoir physics
                base_pressure = field_pressure_trend + 60 * np.sin(0.08 * well_id)
                pressure_variation = np.random.normal(0, 25)
                pressure = base_pressure + pressure_variation + well_bias
                
                # Flow rate calculations
                if well_id == 0:  # Injector well
                    flow_rate = 5000 + np.random.normal(0, 150)
                else:  # Producer wells
                    base_flow = field_flow_trend - well_id * 12
                    flow_variation = 8 * np.sin(0.12 * well_id + 0.05 * time_step)
                    flow_rate = max(100, base_flow + flow_variation + np.random.normal(0, 8))
                
                # Petrophysical properties with realistic distributions
                permeability = np.random.lognormal(5.2, 0.4)  # mD, log-normal distribution
                porosity = np.random.beta(2, 5) * 0.25 + 0.05  # Beta distribution for porosity
                saturation = np.clip(
                    0.15 + 0.008 * time_step + 0.006 * well_id + np.random.normal(0, 0.012),
                    0.1, 0.8
                )
                
                # Well type classification
                well_type = 'INJECTOR' if well_id == 0 else 'PRODUCER'
                
                data_records.append([
                    time_step, well_id, pressure, flow_rate, saturation,
                    permeability, porosity, well_type, reservoir_quality
                ])
        
        columns = [
            'Time', 'Well', 'Pressure', 'FlowRate', 'Saturation',
            'Permeability', 'Porosity', 'WellType', 'ReservoirQuality'
        ]
        
        df = pd.DataFrame(data_records, columns=columns)
        
        # Save generated data
        df.to_csv(config.PROCESSED_DIR / "spe9_synthetic_data.csv", index=False)
        
        print(f"Generated synthetic SPE9 data: {df.shape}")
        return df
    
    def load_data(self, use_synthetic: bool = True) -> pd.DataFrame:
        """
        Load data from file or generate synthetic data
        """
        data_path = config.PROCESSED_DIR / "spe9_synthetic_data.csv"
        
        if use_synthetic or not data_path.exists():
            print("Generating synthetic SPE9-like data...")
            return self.generate_spe9_synthetic_data()
        else:
            print(f"Loading data from {data_path}")
            return pd.read_csv(data_path)
