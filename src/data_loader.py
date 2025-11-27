"""
INDUSTRIAL-GRADE RESERVOIR DATA LOADER
WITH PHYSICS-BASED SYNTHETIC DATA GENERATION
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .config import config

class ReservoirDataLoader:
    """PRODUCTION-READY RESERVOIR DATA MANAGEMENT"""
    
    def __init__(self, dataset_name: str = 'spe9'):
        self.dataset_name = dataset_name
        self.specs = config.DATASET_SPECS[dataset_name]
        self.data = None
        
    def generate_physics_based_data(self) -> pd.DataFrame:
        """GENERATE INDUSTRY-STANDARD RESERVOIR DATA"""
        np.random.seed(config.RANDOM_STATE)
        
        n_wells = self.specs['wells']
        n_timesteps = self.specs['time_steps']
        n_producers = int(n_wells * self.specs['producer_ratio'])
        
        time_days = np.linspace(0, self.specs['simulation_years'] * 365, n_timesteps)
        
        reservoir_data = []
        
        for well_idx in range(n_wells):
            is_producer = well_idx < n_producers
            
            # WELL-SPECIFIC PROPERTIES
            well_properties = self._generate_well_properties(well_idx, is_producer)
            
            for time_idx, days in enumerate(time_days):
                years = days / 365
                
                # RESERVOIR PHYSICS CALCULATIONS
                production_data = self._calculate_production_physics(
                    well_idx, time_idx, years, is_producer, well_properties
                )
                
                # ASSEMBLE RECORD
                record = {
                    'timestamp': days,
                    'years': years,
                    'time_index': time_idx,
                    'well_id': well_idx,
                    'well_name': f"{'PROD' if is_producer else 'INJ'}_{well_idx:03d}",
                    'well_type': 'PRODUCER' if is_producer else 'INJECTOR',
                    **production_data,
                    **well_properties
                }
                
                reservoir_data.append(record)
        
        df = pd.DataFrame(reservoir_data)
        self.data = self._add_advanced_features(df)
        
        # SAVE PROCESSED DATA
        self._save_dataset(df)
        
        print(f"✅ GENERATED {self.dataset_name.upper()} DATA: {df.shape}")
        return df
    
    def _generate_well_properties(self, well_id: int, is_producer: bool) -> Dict:
        """GENERATE REALISTIC WELL PROPERTIES"""
        return {
            'permeability': np.random.lognormal(5.2, 0.35),
            'porosity': np.random.beta(2.5, 6) * 0.22 + 0.06,
            'well_depth': np.random.uniform(8000, 15000),
            'completion_length': np.random.uniform(50, 200),
            'drainage_radius': np.random.uniform(500, 2000),
            'skin_factor': np.random.normal(0, 2)
        }
    
    def _calculate_production_physics(self, well_id: int, time_idx: int, 
                                   years: float, is_producer: bool, 
                                   properties: Dict) -> Dict:
        """CALCULATE RESERVOIR PHYSICS-BASED PRODUCTION"""
        
        if is_producer:
            # PRODUCER WELL PHYSICS
            base_decline = self.specs['initial_oil_rate'] * np.exp(-0.18 * years)
            harmonic_decline = 0.12 * (1 - np.exp(-0.25 * years))
            seasonal_factor = 1 + 0.08 * np.sin(2 * np.pi * years)
            
            oil_rate = base_decline * (1 - harmonic_decline) * seasonal_factor
            oil_rate *= (properties['permeability'] / 200)  # Permeability effect
            
            # WATER AND GAS CALCULATIONS
            water_cut = 0.08 + 0.015 * years
            water_rate = oil_rate * water_cut
            gor = 750 + 25 * years
            gas_rate = oil_rate * gor / 1000
            
            # PRESSURE CALCULATIONS
            pressure_decline = 280 * (1 - np.exp(-0.35 * years))
            base_pressure = self.specs['initial_pressure'] - pressure_decline
            well_effect = 30 * np.sin(0.05 * well_id + 0.1 * years)
            pressure = base_pressure + well_effect
            
        else:
            # INJECTOR WELL PHYSICS
            oil_rate = 0
            base_injection = 7500 + 500 * np.sin(2 * np.pi * years / 3)
            injection_variation = 0.15 * np.cos(2 * np.pi * years / 2)
            water_rate = base_injection * (1 + injection_variation)
            gas_rate = 0
            
            # INJECTION PRESSURE
            base_pressure = self.specs['initial_pressure'] + 550
            pressure_variation = 200 * np.sin(2 * np.pi * years / 4)
            pressure = base_pressure + pressure_variation
        
        # ADD NOISE AND CONSTRAINTS
        oil_rate = max(0, oil_rate + np.random.normal(0, oil_rate * 0.07))
        water_rate = max(0, water_rate + np.random.normal(0, water_rate * 0.05))
        gas_rate = max(0, gas_rate + np.random.normal(0, gas_rate * 0.08))
        pressure = max(100, pressure + np.random.normal(0, 25))
        
        # CUMULATIVE CALCULATIONS
        if time_idx == 0:
            cum_oil, cum_water, cum_gas = oil_rate, water_rate, gas_rate
        else:
            # Simplified cumulative calculation
            time_delta = 1  # Assuming uniform time steps
            cum_oil = oil_rate * time_delta
            cum_water = water_rate * time_delta
            cum_gas = gas_rate * time_delta
        
        return {
            'bottomhole_pressure': pressure,
            'oil_rate': oil_rate,
            'water_rate': water_rate,
            'gas_rate': gas_rate,
            'cumulative_oil': cum_oil,
            'cumulative_water': cum_water,
            'cumulative_gas': cum_gas
        }
    
    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ADD ENGINEERING FEATURES"""
        df = df.copy()
        
        # WELL PRODUCTIVITY INDEX
        df['productivity_index'] = df['oil_rate'] / (df['bottomhole_pressure'] + 1e-8)
        
        # FLUID RATIOS
        df['water_cut'] = df['water_rate'] / (df['oil_rate'] + df['water_rate'] + 1e-8)
        df['gas_oil_ratio'] = df['gas_rate'] / (df['oil_rate'] + 1e-8)
        
        # RECOVERY FACTORS
        df['recovery_factor'] = df['cumulative_oil'] / (df['cumulative_oil'].max() + 1e-8)
        
        return df
    
    def _save_dataset(self, df: pd.DataFrame):
        """SAVE PROCESSED DATASET"""
        output_path = config.DATA_PROCESSED / f"{self.dataset_name}_reservoir_data.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 DATASET SAVED: {output_path}")
    
    def load_dataset(self) -> pd.DataFrame:
        """LOAD EXISTING DATASET"""
        data_path = config.DATA_PROCESSED / f"{self.dataset_name}_reservoir_data.csv"
        
        if data_path.exists():
            self.data = pd.read_csv(data_path)
            print(f"📁 DATASET LOADED: {self.data.shape}")
        else:
            print("🔄 GENERATING NEW DATASET...")
            self.data = self.generate_physics_based_data()
        
        return self.data
