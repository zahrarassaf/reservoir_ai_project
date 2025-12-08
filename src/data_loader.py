import pandas as pd
import numpy as np
import gdown
import os
import re
from typing import Dict
from src.economics import WellProductionData

class DataLoader:
    def __init__(self):
        self.reservoir_data = None
        
    def load_google_drive_data(self, file_id: str) -> bool:
        try:
            os.makedirs("google_drive_data", exist_ok=True)
            output_path = f"google_drive_data/{file_id}.txt"
            
            if not os.path.exists(output_path):
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, output_path, quiet=True)
            
            if os.path.exists(output_path):
                print(f"  Processing OPM file: {file_id}")
                return self._parse_opm_file(output_path)
            else:
                print(f"  Download failed, using synthetic")
                return self._generate_synthetic_data()
                
        except Exception as e:
            print(f"  Error: {e}")
            return self._generate_synthetic_data()
    
    def _parse_opm_file(self, file_path: str) -> bool:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            print(f"  File size: {len(content)} chars")
            
            wells = {}
            
            if 'SCHEDULE' in content or 'COMPDAT' in content:
                wells = self._extract_wells_from_schedule(content)
            elif 'SUMMARY' in content:
                wells = self._extract_wells_from_summary(content)
            else:
                print(f"  No schedule/summary found, checking for data")
                wells = self._extract_from_general_content(content)
            
            if wells:
                self.reservoir_data = {
                    'wells': wells,
                    'grid': {
                        'dimensions': (24, 25, 15),
                        'porosity': np.random.uniform(0.15, 0.25, 9000)
                    }
                }
                print(f"  ✓ Created data with {len(wells)} wells")
                return True
            else:
                print(f"  ✗ No wells found, using synthetic")
                return self._generate_synthetic_data()
                
        except Exception as e:
            print(f"  Parse error: {e}")
            return self._generate_synthetic_data()
    
    def _extract_wells_from_schedule(self, content: str) -> Dict:
        wells = {}
        
        compdat_pattern = r'COMPDAT\s*\n(.*?)\n/\s*\n'
        match = re.search(compdat_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            compdat_content = match.group(1)
            lines = compdat_content.strip().split('\n')
            
            well_counter = 1
            for line in lines:
                if line.strip() and not line.strip().startswith('--'):
                    parts = re.split(r'\s+', line.strip())
                    if len(parts) >= 4:
                        well_name = parts[0].strip("'") if "'" in parts[0] else f"Well_{well_counter}"
                        
                        time_points = np.linspace(0, 900, 30)
                        base_rate = 500 + np.random.randn() * 200
                        oil_rate = base_rate * np.exp(-0.05 * np.arange(30)) * (1 + 0.1 * np.random.randn(30))
                        oil_rate = np.maximum(oil_rate, 50)
                        
                        wells[well_name] = WellProductionData(
                            time_points=time_points,
                            oil_rate=oil_rate,
                            well_type='PRODUCER' if 'PROD' in well_name.upper() or 'P' in well_name else 'INJECTOR'
                        )
                        
                        well_counter += 1
        
        if not wells:
            wells = self._create_default_wells()
        
        return wells
    
    def _extract_wells_from_summary(self, content: str) -> Dict:
        wells = {}
        
        well_pattern = r'WOPR\s+([A-Z0-9_]+)\s*\n(.*?)\n/\s*\n'
        matches = re.findall(well_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if matches:
            for well_match in matches:
                well_name = well_match[0]
                data_content = well_match[1]
                
                numbers = re.findall(r'[-+]?\d*\.\d+|\d+', data_content)
                if len(numbers) >= 10:
                    rates = [float(num) for num in numbers[:30]]
                    time_points = np.arange(len(rates)) * 30
                    
                    wells[well_name] = WellProductionData(
                        time_points=time_points,
                        oil_rate=np.array(rates),
                        well_type='PRODUCER'
                    )
        
        if not wells:
            wells = self._create_default_wells()
        
        return wells
    
    def _extract_from_general_content(self, content: str) -> Dict:
        lines = content.strip().split('\n')
        wells = {}
        
        data_found = False
        numbers = []
        
        for line in lines[:100]:
            line_nums = re.findall(r'[-+]?\d*\.\d+|\d+', line)
            if line_nums:
                data_found = True
                numbers.extend([float(num) for num in line_nums])
        
        if data_found and len(numbers) >= 20:
            time_points = np.arange(min(30, len(numbers)//2))
            rates = numbers[:len(time_points)]
            
            wells['DATA_WELL_01'] = WellProductionData(
                time_points=time_points,
                oil_rate=np.array(rates),
                well_type='PRODUCER'
            )
        
        if not wells:
            wells = self._create_default_wells()
        
        return wells
    
    def _create_default_wells(self) -> Dict:
        time_points = np.linspace(0, 900, 30)
        
        return {
            'PROD_01': WellProductionData(
                time_points=time_points,
                oil_rate=800 * np.exp(-0.05 * np.arange(30)) * (1 + 0.1 * np.random.randn(30)),
                well_type='PRODUCER'
            ),
            'PROD_02': WellProductionData(
                time_points=time_points,
                oil_rate=600 * np.exp(-0.04 * np.arange(30)) * (1 + 0.12 * np.random.randn(30)),
                well_type='PRODUCER'
            ),
            'INJ_01': WellProductionData(
                time_points=time_points,
                oil_rate=np.zeros(30),
                water_rate=2000 * np.ones(30) * (1 + 0.05 * np.random.randn(30)),
                well_type='INJECTOR'
            )
        }
    
    def _generate_synthetic_data(self) -> bool:
        try:
            self.reservoir_data = {
                'wells': self._create_default_wells(),
                'grid': {
                    'dimensions': (24, 25, 15),
                    'porosity': np.random.uniform(0.15, 0.25, 9000)
                }
            }
            print("  Generated synthetic SPE9 data")
            return True
        except Exception as e:
            print(f"  Synthetic data error: {e}")
            return False
    
    def get_reservoir_data(self) -> Dict:
        return self.reservoir_data if self.reservoir_data else {}
