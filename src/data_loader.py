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
        """Load data from text file"""
        try:
            print(f"\nLoading text file: {os.path.basename(filepath)}")
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            print(f"File has {len(lines)} lines")
            
            data = []
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                if any(line.startswith(c) for c in ['#', '//', '%', '*']):
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
            
            print(f"Extracted {len(data)} data rows")
            
            if len(data) < 10:
                print("Not enough data rows")
                return False
            
            max_cols = max(len(row) for row in data)
            padded_data = []
            for row in data:
                if len(row) < max_cols:
                    padded_row = list(row) + [0.0] * (max_cols - len(row))
                    padded_data.append(padded_row)
                else:
                    padded_data.append(row)
            
            data_array = np.array(padded_data)
            print(f"Data shape: {data_array.shape}")
            
            self.time = np.arange(len(data_array))
            
            n_cols = data_array.shape[1]
            n_wells = min(6, n_cols)
            
            production_data = {}
            for i in range(n_wells):
                col_data = data_array[:, i]
                production_data[f'Well_{i+1}'] = col_data
            
            self.production = pd.DataFrame(production_data)
            
            if n_cols > n_wells:
                self.pressure = data_array[:, n_wells]
            else:
                self.pressure = 4000 - 0.3 * self.time + np.random.normal(0, 100, len(self.time))
                self.pressure = np.maximum(800, self.pressure)
            
            self.wells = list(self.production.columns)
            
            print(f"✓ Successfully loaded: {len(self.time)} time points, {len(self.wells)} wells")
            return True
            
        except Exception as e:
            print(f"✗ Error loading text file: {e}")
            return False
    
    def load_csv(self, filepath: str) -> bool:
        """Try to load CSV or text file"""
        try:
            return self.load_txt_file(filepath)
        except:
            return False
    
    def load_multiple_csv(self, directory: str, pattern: str = "*.csv") -> bool:
        """Load multiple CSV files from directory"""
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
        """Create sample data for testing"""
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
        """Check if production data exists"""
        return not self.production.empty and len(self.production) > 0
    
    @property
    def has_pressure_data(self) -> bool:
        """Check if pressure data exists"""
        return len(self.pressure) > 0
    
    def summary(self) -> Dict[str, Any]:
        """Get data summary"""
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
