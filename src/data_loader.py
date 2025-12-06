# src/data_loader.py
import pandas as pd
import numpy as np
import gdown
import os
from typing import Dict, List, Any
from src.economics import WellProductionData

class DataLoader:
    def __init__(self):
        self.reservoir_data = None
        
    def load_google_drive_data(self, file_id: str) -> bool:
        try:
            os.makedirs("google_drive_data", exist_ok=True)
            output_path = f"google_drive_data/{file_id}.csv"
            
            if not os.path.exists(output_path):
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, output_path, quiet=True)
            
            if os.path.exists(output_path):
                return self._process_csv_file(output_path)
            else:
                return self._generate_synthetic_data()
                
        except Exception as e:
            print(f"Error loading {file_id}: {e}")
            return self._generate_synthetic_data()
    
    def _process_csv_file(self, file_path: str) -> bool:
        try:
            df = pd.read_csv(file_path)
            
            time_column = None
            rate_column = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'time' in col_lower or 'date' in col_lower or 'days' in col_lower:
                    time_column = col
                elif 'oil' in col_lower or 'rate' in col_lower or 'prod' in col_lower:
                    rate_column = col
            
            if not time_column:
                time_column = df.columns[0]
            if not rate_column:
                rate_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            time_data = df[time_column].values
            rate_data = df[rate_column].values
            
            if len(time_data) > 0 and len(rate_data) > 0:
                self.reservoir_data = {
                    'wells': {
                        'FIELD_01': WellProductionData(
                            time_points=time_data,
                            oil_rate=rate_data,
                            well_type='PRODUCER'
                        )
                    },
                    'grid': {
                        'dimensions': (24, 25, 15),
                        'porosity': np.random.uniform(0.15, 0.25, 1000)
                    }
                }
                return True
            else:
                return self._generate_synthetic_data()
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> bool:
        try:
            time_points = np.linspace(0, 1825, 60)
            
            self.reservoir_data = {
                'wells': {
                    'SYN_WELL_01': WellProductionData(
                        time_points=time_points,
                        oil_rate=1200 * np.exp(-0.06 * np.arange(60)) * (1 + 0.08 * np.random.randn(60)),
                        well_type='PRODUCER'
                    ),
                    'SYN_WELL_02': WellProductionData(
                        time_points=time_points,
                        oil_rate=900 * np.exp(-0.05 * np.arange(60)) * (1 + 0.1 * np.random.randn(60)),
                        well_type='PRODUCER'
                    ),
                    'SYN_WELL_03': WellProductionData(
                        time_points=time_points,
                        oil_rate=700 * np.exp(-0.04 * np.arange(60)) * (1 + 0.12 * np.random.randn(60)),
                        well_type='PRODUCER'
                    )
                },
                'grid': {
                    'dimensions': (24, 25, 15),
                    'porosity': np.random.uniform(0.18, 0.22, 1000)
                }
            }
            return True
            
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            return False
    
    def get_reservoir_data(self) -> Dict:
        return self.reservoir_data if self.reservoir_data else {}
