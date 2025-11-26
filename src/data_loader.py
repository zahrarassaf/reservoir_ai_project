"""
MULTI-DATASET RESERVOIR DATA LOADER
INDUSTRY-GRADE DATA GENERATION
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from .config import config

class ReservoirDataLoader:
    """PRODUCTION-GRADE DATA LOADER FOR MULTIPLE DATASETS"""
    
    def __init__(self):
        self.datasets = {}
        
    def generate_physics_based_data(self, dataset_name: str) -> pd.DataFrame:
        """GENERATE PHYSICS-BASED DATA FOR SPECIFIED DATASET"""
        if dataset_name not in config.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        specs = config.DATASETS[dataset_name]
        np.random.seed(config.RANDOM_STATE)
        
        print(f"🔥 GENERATING {dataset_name.upper()} DATA...")
        
        time_points = np.linspace(0, specs['years'], specs['time_steps'])
        production_data = []
        
        for well_id in range(specs['wells']):
            is_producer = well_id < specs['wells'] * 0.6  # 60% producers
            
            for t_idx, years in enumerate(time_points):
                # RESERVOIR PHYSICS-BASED CALCULATIONS
                if is_producer:
                    # PRODUCER WELL - DECLINE CURVE PHYSICS
                    initial_rate = 5000 - well_id * 100
                    decline_rate = 0.15 + well_id * 0.01
                    base_oil_rate = initial_rate * np.exp(-decline_rate * years)
                    
                    # ADD SEASONALITY AND NOISE
                    seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * years)
                    noise = np.random.normal(0, base_oil_rate * 0.08)
                    oil_rate = max(50, base_oil_rate * seasonal_factor + noise)
                    
                    # WATER AND GAS RATES WITH INCREASING TRENDS
                    water_cut = 0.1 + 0.015 * years  # Increasing water cut
                    water_rate = oil_rate * water_cut
                    gor = 800 + 20 * years  # Increasing GOR
                    gas_rate = oil_rate * gor / 1000
                    
                    # PRESSURE DECLINE WITH DEPLETION
                    initial_pressure = 3600 - well_id * 50
                    pressure_decline = 200 + 25 * years
                    pressure = max(500, initial_pressure - pressure_decline)
                    
                else:
                    # INJECTOR WELL - DIFFERENT PHYSICS
                    oil_rate = 0
                    base_injection = 8000 + well_id * 100
                    injection_variation = 0.2 * np.sin(2 * np.pi * years / 2)
                    water_rate = base_injection * (1 + injection_variation)
                    gas_rate = 0
                    
                    # INJECTION PRESSURE PROFILES
                    base_pressure = 4000 + well_id * 30
                    pressure_variation = 300 * np.sin(2 * np.pi * years / 3)
                    pressure = base_pressure + pressure_variation
                
                # CUMULATIVE PRODUCTION/INJECTION
                if t_idx == 0:
                    cum_oil = oil_rate
                    cum_water = water_rate  
                    cum_gas = gas_rate
                else:
                    prev_data = production_data[-specs['wells']:]
                    prev_cum_oil = sum([d['CUMULATIVE_OIL'] for d in prev_data if d['WELL_ID'] == well_id])
                    prev_cum_water = sum([d['CUMULATIVE_WATER'] for d in prev_data if d['WELL_ID'] == well_id])
                    prev_cum_gas = sum([d['CUMULATIVE_GAS'] for d in prev_data if d['WELL_ID'] == well_id])
                    
                    cum_oil = prev_cum_oil + oil_rate if is_producer else 0
                    cum_water = prev_cum_water + abs(water_rate)
                    cum_gas = prev_cum_gas + gas_rate if is_producer else 0
                
                record = {
                    'DATASET': dataset_name,
                    'TIME_INDEX': t_idx,
                    'YEARS': years,
                    'WELL_ID': well_id,
                    'WELL_NAME': f"{'PROD' if is_producer else 'INJ'}_{well_id:03d}",
                    'WELL_TYPE': 'PRODUCER' if is_producer else 'INJECTOR',
                    'BOTTOMHOLE_PRESSURE': pressure,
                    'FLOW_RATE_OIL': oil_rate,
                    'FLOW_RATE_WATER': water_rate,
                    'FLOW_RATE_GAS': gas_rate,
                    'CUMULATIVE_OIL': cum_oil,
                    'CUMULATIVE_WATER': cum_water,
                    'CUMULATIVE_GAS': cum_gas,
                    'X_LOCATION': np.random.uniform(0, 10000),
                    'Y_LOCATION': np.random.uniform(0, 10000),
                    'PERMEABILITY': np.random.lognormal(5.0, 0.3),
                    'POROSITY': np.random.beta(2, 5) * 0.2 + 0.05
                }
                
                production_data.append(record)
        
        df = pd.DataFrame(production_data)
        
        # SAVE PROCESSED DATA
        output_path = config.DATA_PROCESSED / f"{dataset_name}_production_data.csv"
        df.to_csv(output_path, index=False)
        
        print(f"✅ {dataset_name.upper()} GENERATED: {df.shape} | Saved to: {output_path}")
        return df
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """LOAD ALL DATASETS FOR COMPREHENSIVE TRAINING"""
        for dataset_name in config.DATASETS.keys():
            data_path = config.DATA_PROCESSED / f"{dataset_name}_production_data.csv"
            
            if data_path.exists():
                self.datasets[dataset_name] = pd.read_csv(data_path)
                print(f"📁 LOADED {dataset_name.upper()}: {self.datasets[dataset_name].shape}")
            else:
                print(f"🔄 GENERATING {dataset_name.upper()}...")
                self.datasets[dataset_name] = self.generate_physics_based_data(dataset_name)
        
        return self.datasets
    
    def get_combined_dataset(self) -> pd.DataFrame:
        """COMBINE ALL DATASETS FOR ROBUST TRAINING"""
        if not self.datasets:
            self.load_all_datasets()
            
        combined_data = pd.concat(self.datasets.values(), ignore_index=True)
        print(f"🔥 COMBINED DATASET: {combined_data.shape}")
        
        return combined_data
