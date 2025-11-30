import os
import pandas as pd
import numpy as np
import torch
from typing import Optional

class OPMDataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_opm_data(self):
        print("Loading synthetic OPM data...")
        
        np.random.seed(42)
        n_blocks = 900
        
        df = pd.DataFrame({
            'GRID_BLOCK': range(n_blocks),
            'PERMX': np.random.lognormal(mean=3, sigma=1.5, size=n_blocks),
            'PORO': np.random.normal(loc=0.2, scale=0.05, size=n_blocks),
            'DEPTH': np.linspace(2000, 2500, n_blocks),
            'REGION': np.random.randint(1, 6, size=n_blocks)
        })
        
        df['PORO'] = np.clip(df['PORO'], 0.01, 0.35)
        df['PERMX'] = np.clip(df['PERMX'], 0.1, 5000)
        
        df['LOG_PERMX'] = np.log10(df['PERMX'])
        df['PERMZ'] = df['PERMX'] * 0.1
        df['POROSITY_FRACTION'] = df['PORO']
        df['TRANSMISSIBILITY'] = df['PERMX'] * df['PORO']
        
        print(f"Created synthetic data with {len(df)} grid blocks")
        return df
