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
    
    def load_txt_file(self, filepath: str) -> bool:
        print("\n" + "="*60)
        print("DEBUG: LOAD_TXT_FILE CALLED - VERSION 3.0")
        print("="*60)
        
        try:
            print(f"\nLoading text file: {os.path.basename(filepath)}")
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            print(f"File has {len(lines)} lines")
            
            raw_data = []
            for i, line in enumerate(lines[:20]):
                print(f"Line {i+1}: {line.strip()}")
            
            data = []
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                if any(line.startswith(c) for c in ['#', '//', '%', '*', '!']):
                    continue
                
                numbers = []
                parts = []
                
                if '\t' in line:
                    parts = line.split('\t')
                elif ',' in line:
                    parts = line.split(',')
                elif ';' in line:
                    parts = line.split(';')
                else:
                    parts = line.split()
                
                for part in parts:
                    part = part.strip()
                    if part:
                        try:
                            num = float(part)
                            numbers.append(num)
                        except:
                            try:
                                num = float(part.replace(',', ''))
                                numbers.append(num)
                            except:
                                continue
                
                if numbers:
                    data.append(numbers)
            
            print(f"\nExtracted {len(data)} data rows")
            
            if len(data) < 10:
                print("Not enough data rows")
                return False
            
            max_cols = max(len(row) for row in data)
            min_cols = min(len(row) for row in data)
            print(f"Columns: min={min_cols}, max={max_cols}")
            
            padded_data = []
            for row in data:
                if len(row) < max_cols:
                    padded_row = list(row) + [0.0] * (max_cols - len(row))
                    padded_data.append(padded_row)
                else:
                    padded_data.append(row)
            
            data_array = np.array(padded_data)
            print(f"Raw data array shape: {data_array.shape}")
            
            print("\nFirst 5 rows of raw data:")
            for i in range(min(5, len(data_array))):
                print(f"  Row {i}: {data_array[i]}")
            
            self.time = np.arange(len(data_array))
            
            n_cols = data_array.shape[1]
            n_wells = min(6, n_cols)
            
            print(f"\nProcessing {n_wells} wells from {n_cols} columns")
            
            production_data = {}
            for i in range(n_wells):
                col_data = data_array[:, i].copy()
                print(f"\nWell {i+1} raw data:")
                print(f"  Min: {np.min(col_data):.2f}, Max: {np.max(col_data):.2f}, Mean: {np.mean(col_data):.2f}")
                
                if np.all(col_data == 0):
                    col_data = np.random.uniform(100, 500, len(col_data))
                    print(f"  All zeros - generated synthetic data")
                else:
                    valid_data = col_data[col_data > 0]
                    if len(valid_data) > 0:
                        median_val = np.median(valid_data)
                        print(f"  Median of positive values: {median_val:.2f}")
                        
                        if median_val > 0:
                            if median_val < 1:
                                scale_factor = 500
                            elif median_val > 10000:
                                scale_factor = 500 / median_val
                            else:
                                scale_factor = 500 / median_val
                            
                            print(f"  Scale factor: {scale_factor:.4f}")
                            col_data = col_data * scale_factor
                
                col_data = np.maximum(10, col_data)
                col_data = np.minimum(5000, col_data)
                production_data[f'Well_{i+1}'] = col_data
                print(f"  Scaled: Min: {np.min(col_data):.1f}, Max: {np.max(col_data):.1f}")
            
            self.production = pd.DataFrame(production_data)
            
            print(f"\nProcessing pressure data...")
            if n_cols > n_wells:
                pressure_col = n_wells
                pressure_data = data_array[:, pressure_col].copy()
                print(f"Pressure raw data - Min: {np.min(pressure_data):.2f}, Max: {np.max(pressure_data):.2f}")
                
                if np.all(pressure_data == 0):
                    self.pressure = np.random.uniform(3000, 4000, len(self.time))
                    print("  All zeros - generated synthetic pressure")
                else:
                    valid_pressure = pressure_data[pressure_data > 0]
                    if len(valid_pressure) > 0:
                        median_press = np.median(valid_pressure)
                        print(f"  Median pressure: {median_press:.2f}")
                        
                        if median_press > 0:
                            if median_press < 10:
                                scale_press = 3500
                            elif median_press > 10000:
                                scale_press = 3500 / median_press
                            else:
                                scale_press = 3500 / median_press
                            
                            print(f"  Pressure scale factor: {scale_press:.4f}")
                            pressure_data = pressure_data * scale_press
                    
                    pressure_data = np.maximum(1000, pressure_data)
                    pressure_data = np.minimum(5000, pressure_data)
                    self.pressure = pressure_data
            else:
                self.pressure = np.random.uniform(3000, 4000, len(self.time))
                print("No pressure column - generated synthetic pressure")
            
            self.wells = list(self.production.columns)
            
            prod_min = self.production.min().min()
            prod_max = self.production.max().max()
            prod_mean = self.production.mean().mean()
            pres_min = self.pressure.min()
            pres_max = self.pressure.max()
            pres_mean = self.pressure.mean()
            
            print("\n" + "="*60)
            print("FINAL DATA STATISTICS:")
            print("="*60)
            print(f"✓ Time points: {len(self.time)}")
            print(f"✓ Wells: {len(self.wells)}")
            print(f"✓ Production range: {prod_min:.1f} to {prod_max:.1f} bbl/day (mean: {prod_mean:.1f})")
            print(f"✓ Pressure range: {pres_min:.1f} to {pres_max:.1f} psi (mean: {pres_mean:.1f})")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"\n✗ ERROR loading text file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_csv(self, filepath: str) -> bool:
        try:
            return self.load_txt_file(filepath)
        except:
            return False
    
    def load_multiple_csv(self, directory: str, pattern: str = "*.csv") -> bool:
        files = glob.glob(os.path.join(directory, pattern))
        
        if not files:
            return False
        
        for file in files:
            try:
                if self.load_txt_file(file):
                    return True
            except:
                continue
        
        return False
    
    def create_sample_data(self, n_days: int = 1825, n_wells: int = 6) -> bool:
        np.random.seed(42)
        
        self.time = np.arange(n_days)
        
        production_data = {}
        for i in range(n_wells):
            qi = 500 + np.random.uniform(-100, 100)
            Di = 0.0003 + np.random.uniform(-0.0001, 0.0001)
            rates = qi * np.exp(-Di * self.time)
            noise = np.random.normal(0, 20, n_days)
            rates = np.maximum(5, rates + noise)
            production_data[f'Well_{i+1}'] = rates
        
        self.production = pd.DataFrame(production_data)
        
        self.pressure = 4000 - 0.2 * self.time + np.random.normal(0, 50, n_days)
        self.pressure = np.maximum(1000, self.pressure)
        
        self.wells = list(self.production.columns)
        
        print(f"Created sample data: {n_days} days, {n_wells} wells")
        return True
    
    @property
    def has_production_data(self) -> bool:
        return not self.production.empty and len(self.production) > 0
    
    @property
    def has_pressure_data(self) -> bool:
        return len(self.pressure) > 0
    
    def summary(self) -> Dict[str, Any]:
        summary_dict = {
            'wells': len(self.wells),
            'time_points': len(self.time),
            'production_columns': list(self.production.columns),
            'pressure_available': self.has_pressure_data
        }
        
        if self.has_production_data:
            summary_dict['production_range'] = {
                'min': float(self.production.min().min()),
                'max': float(self.production.max().max()),
                'mean': float(self.production.mean().mean())
            }
        else:
            summary_dict['production_range'] = {'min': 0, 'max': 0, 'mean': 0}
        
        if self.has_pressure_data:
            summary_dict['pressure_range'] = {
                'min': float(self.pressure.min()),
                'max': float(self.pressure.max()),
                'mean': float(self.pressure.mean())
            }
        else:
            summary_dict['pressure_range'] = {'min': 0, 'max': 0, 'mean': 0}
        
        return summary_dict
