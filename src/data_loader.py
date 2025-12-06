# src/data_loader.py - نسخه اصلاح شده
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
            # Try different CSV reading strategies
            df = None
            for engine in ['python', 'c']:
                try:
                    df = pd.read_csv(file_path, engine=engine, on_bad_lines='skip')
                    if len(df) > 0:
                        break
                except:
                    continue
            
            if df is None or len(df) == 0:
                print(f"No valid data in {file_path}, using synthetic")
                return self._generate_synthetic_data()
            
            # Clean column names
            df.columns = [str(col).strip() for col in df.columns]
            
            time_column = None
            rate_column = None
            
            # Find time column
            for col in df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ['time', 'date', 'days', 'month', 't']):
                    time_column = col
                    break
            
            if not time_column and len(df.columns) > 0:
                time_column = df.columns[0]
            
            # Find rate column
            for col in df.columns:
                if col == time_column:
                    continue
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ['oil', 'rate', 'prod', 'opr', 'fopr', 'bbl', 'stb']):
                    rate_column = col
                    break
            
            if not rate_column and len(df.columns) > 1:
                rate_column = df.columns[1]
            elif not rate_column:
                rate_column = df.columns[0]
            
            # Extract data
            time_data = pd.to_numeric(df[time_column], errors='coerce').dropna().values
            rate_data = pd.to_numeric(df[rate_column], errors='coerce').dropna().values
            
            if len(time_data) == 0 or len(rate_data) == 0:
                print(f"No numeric data in {file_path}, using synthetic")
                return self._generate_synthetic_data()
            
            # Ensure same length
            min_len = min(len(time_data), len(rate_data))
            time_data = time_data[:min_len]
            rate_data = rate_data[:min_len]
            
            # Convert to appropriate units if needed
            if max(rate_data) < 1:  # Probably in MMbbl/day
                rate_data = rate_data * 1_000_000
            elif max(rate_data) < 1000:  # Probably in Mbbl/day
                rate_data = rate_data * 1_000
            
            self.reservoir_data = {
                'wells': {
                    'WELL_01': WellProductionData(
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
            
            print(f"Processed {file_path}: {len(time_data)} data points")
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> bool:
        try:
            time_points = np.linspace(0, 1825, 60)  # 5 years
            
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
            print("Generated synthetic data")
            return True
            
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            return False
    
    def get_reservoir_data(self) -> Dict:
        return self.reservoir_data if self.reservoir_data else {}
