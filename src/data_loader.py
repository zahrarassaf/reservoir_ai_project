# src/data_loader.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
import os

class DataLoader:
    def __init__(self):
        self.well_data = None
        self.reservoir_data = None
        
    def load_spe9_data(self, file_path: str) -> bool:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                
                if len(df) > 0:
                    self._process_csv_data(df)
                    return True
                    
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self._process_json_data(data)
                return True
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
        return False
    
    def _process_csv_data(self, df: pd.DataFrame):
        from src.economics import WellProductionData
        
        well_dict = {}
        
        time_col = self._find_time_column(df.columns)
        rate_col = self._find_rate_column(df.columns)
        
        if time_col and rate_col:
            time_points = df[time_col].values
            oil_rate = df[rate_col].values
            
            well_dict['FIELD'] = WellProductionData(
                time_points=time_points,
                oil_rate=oil_rate,
                well_type='PRODUCER'
            )
        
        if not well_dict:
            well_dict = self._generate_synthetic_data()
        
        self.reservoir_data = {
            'wells': well_dict,
            'grid': {
                'dimensions': (24, 25, 15),
                'porosity': np.random.uniform(0.1, 0.25, 100)
            }
        }
    
    def _process_json_data(self, data: Dict):
        from src.economics import WellProductionData
        
        well_dict = {}
        
        if 'wells' in data:
            for well_name, well_info in data['wells'].items():
                if 'time' in well_info and 'oil_rate' in well_info:
                    well_dict[well_name] = WellProductionData(
                        time_points=np.array(well_info['time']),
                        oil_rate=np.array(well_info['oil_rate']),
                        well_type=well_info.get('type', 'PRODUCER')
                    )
        
        if not well_dict:
            well_dict = self._generate_synthetic_data()
        
        self.reservoir_data = {
            'wells': well_dict,
            'grid': data.get('grid', {
                'dimensions': (24, 25, 15),
                'porosity': np.random.uniform(0.1, 0.25, 100)
            })
        }
    
    def _find_time_column(self, columns: List[str]) -> str:
        time_keywords = ['time', 'date', 'days', 'month', 'year']
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in time_keywords):
                return col
        return columns[0] if columns else None
    
    def _find_rate_column(self, columns: List[str]) -> str:
        rate_keywords = ['oil', 'rate', 'prod', 'opr', 'fopr']
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in rate_keywords):
                return col
        
        for col in columns:
            if col != self._find_time_column(columns):
                return col
        return columns[1] if len(columns) > 1 else None
    
    def _generate_synthetic_data(self) -> Dict:
        from src.economics import WellProductionData
        
        time_points = np.linspace(0, 365*5, 60)
        
        return {
            'WELL_001': WellProductionData(
                time_points=time_points,
                oil_rate=1000 * np.exp(-0.05 * np.arange(60)) * (1 + 0.1 * np.random.randn(60)),
                well_type='PRODUCER'
            ),
            'WELL_002': WellProductionData(
                time_points=time_points,
                oil_rate=800 * np.exp(-0.04 * np.arange(60)) * (1 + 0.1 * np.random.randn(60)),
                well_type='PRODUCER'
            ),
            'WELL_003': WellProductionData(
                time_points=time_points,
                oil_rate=600 * np.exp(-0.03 * np.arange(60)) * (1 + 0.1 * np.random.randn(60)),
                well_type='PRODUCER'
            )
        }
    
    def get_reservoir_data(self) -> Dict:
        return self.reservoir_data
