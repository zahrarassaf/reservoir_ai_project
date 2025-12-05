import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
import glob
import logging

logger = logging.getLogger(__name__)


class ReservoirData:
    def __init__(self):
        self.production = pd.DataFrame()
        self.pressure = np.array([])
        self.time = np.array([])
        self.wells = []
        self.metadata = {}
    
    def load_csv(self, filepath: str):
        try:
            df = pd.read_csv(filepath)
            
            if df.empty:
                return False
            
            if len(df.columns) >= 2:
                self.time = np.arange(len(df))
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 1:
                    prod_col = numeric_cols[0]
                    self.production = pd.DataFrame({f'Well_{prod_col}': df[prod_col].values})
                    
                    if len(numeric_cols) >= 2:
                        self.pressure = df[numeric_cols[1]].values
                    else:
                        self.pressure = np.zeros(len(df))
                else:
                    self.production = pd.DataFrame({'Well_1': np.ones(len(df))})
                    self.pressure = np.zeros(len(df))
                
                self.wells = list(self.production.columns)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading CSV {filepath}: {e}")
            return False
    
    def load_multiple_csv(self, directory: str, pattern: str = "*.csv"):
        files = glob.glob(os.path.join(directory, pattern))
        
        if not files:
            return False
        
        all_dfs = []
        
        for file in files:
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    all_dfs.append(df)
            except:
                continue
        
        if not all_dfs:
            return False
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        if combined_df.empty:
            return False
        
        self.time = np.arange(len(combined_df))
        
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            prod_data = {}
            for i, col in enumerate(numeric_cols[:3]):
                prod_data[f'Well_{i+1}'] = combined_df[col].values
            self.production = pd.DataFrame(prod_data)
            
            if len(numeric_cols) > 3:
                self.pressure = combined_df[numeric_cols[3]].values
            else:
                self.pressure = np.random.uniform(3000, 4000, len(combined_df))
        else:
            self.production = pd.DataFrame({'Well_1': np.random.uniform(100, 500, len(combined_df))})
            self.pressure = np.random.uniform(3000, 4000, len(combined_df))
        
        self.wells = list(self.production.columns)
        return True
    
    def create_sample_data(self, n_days: int = 1825, n_wells: int = 6):
        np.random.seed(42)
        
        self.time = np.arange(n_days)
        
        production_data = {}
        
        for i in range(n_wells):
            qi = 800 + np.random.uniform(-200, 200)
            Di = 0.0005 + np.random.uniform(-0.0001, 0.0001)
            rates = qi * np.exp(-Di * self.time)
            noise = np.random.normal(0, 30, n_days)
            rates = np.maximum(10, rates + noise)
            production_data[f'Well_{i+1}'] = rates
        
        self.production = pd.DataFrame(production_data)
        
        initial_pressure = 4000
        pressure_decline = 0.00005 * self.time
        noise = np.random.normal(0, 30, n_days)
        self.pressure = np.maximum(1000, initial_pressure - pressure_decline + noise)
        
        self.wells = list(self.production.columns)
        
        return True
    
    @property
    def has_production_data(self):
        return not self.production.empty and len(self.production) > 0
    
    @property
    def has_pressure_data(self):
        return len(self.pressure) > 0
    
    def summary(self):
        return {
            'wells': len(self.wells),
            'time_points': len(self.time),
            'production_columns': list(self.production.columns),
            'pressure_available': self.has_pressure_data
        }
