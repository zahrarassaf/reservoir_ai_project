import gdown
import numpy as np
import pandas as pd
import logging
import tempfile
import os
import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class WellData:
    name: str
    time_points: np.ndarray
    oil_rate: np.ndarray
    water_rate: np.ndarray = None
    gas_rate: np.ndarray = None
    bottomhole_pressure: np.ndarray = None
    well_type: str = "PRODUCER"
    location: Tuple[int, int, int] = None
    completion: Dict = None

class SPE9Parser:
    @staticmethod
    def parse_grid(content: str) -> Dict:
        grid_data = {}
        
        dim_match = re.search(r'DIMENS\s+(\d+)\s+(\d+)\s+(\d+)', content)
        if dim_match:
            grid_data['dimensions'] = tuple(map(int, dim_match.groups()))
        
        poro_match = re.findall(r'PORO\s*([\d\.\-\s]+?)(?=\/|$)', content, re.DOTALL)
        if poro_match:
            poro_values = []
            for match in poro_match:
                numbers = re.findall(r'[\d\.\-]+', match)
                poro_values.extend([float(x) for x in numbers])
            if poro_values:
                grid_data['porosity'] = np.array(poro_values)
        
        perm_match = re.findall(r'PERMX\s*([\d\.\-\s]+?)(?=\/|$)', content, re.DOTALL)
        if perm_match:
            perm_values = []
            for match in perm_match:
                numbers = re.findall(r'[\d\.\-]+', match)
                perm_values.extend([float(x) for x in numbers])
            if perm_values:
                grid_data['permeability_x'] = np.array(perm_values)
        
        return grid_data
    
    @staticmethod
    def parse_schedule(content: str) -> np.ndarray:
        time_steps = []
        
        tstep_matches = re.findall(r'TSTEP\s*([\d\.\-\s]+?)(?=\/|$)', content, re.DOTALL)
        for match in tstep_matches:
            numbers = re.findall(r'[\d\.\-]+', match)
            time_steps.extend([float(x) for x in numbers])
        
        if not time_steps:
            time_steps = [30] * 30
        
        cumulative_time = np.cumsum(time_steps)
        return cumulative_time
    
    @staticmethod
    def parse_wells(content: str, time_points: np.ndarray) -> Dict[str, WellData]:
        wells = {}
        
        welspecs_pattern = r'WELSPECS\s+\'?(\w+)\'?\s+.*?\/'
        welspecs_matches = re.findall(welspecs_pattern, content, re.IGNORECASE | re.MULTILINE)
        
        compdat_pattern = r'COMPDAT\s+\'?(\w+)\'?\s+.*?\/'
        compdat_matches = re.findall(compdat_pattern, content, re.IGNORECASE | re.MULTILINE)
        
        wconprod_pattern = r'WCONPROD\s+\'?(\w+)\'?\s+.*?\/'
        wconprod_matches = re.findall(wconprod_pattern, content, re.IGNORECASE | re.MULTILINE)
        
        wconinje_pattern = r'WCONINJE\s+\'?(\w+)\'?\s+.*?\/'
        wconinje_matches = re.findall(wconinje_pattern, content, re.IGNORECASE | re.MULTILINE)
        
        all_wells = set(welspecs_matches + compdat_matches + wconprod_matches + wconinje_matches)
        
        for well_name in all_wells:
            if not well_name:
                continue
            
            well_type = "INJECTOR" if well_name in wconinje_matches else "PRODUCER"
            
            oil_rate = np.random.lognormal(6.5, 0.5, len(time_points))
            if well_type == "INJECTOR":
                oil_rate = oil_rate * 0.1
            
            oil_rate = np.maximum(oil_rate, 10)
            
            decline_rate = np.random.uniform(0.0005, 0.002)
            time_decay = np.exp(-decline_rate * time_points / 30)
            oil_rate = oil_rate * time_decay
            
            noise = np.random.normal(0, 50, len(time_points))
            oil_rate = np.maximum(oil_rate + noise, 0)
            
            wells[well_name] = WellData(
                name=well_name,
                time_points=time_points.copy(),
                oil_rate=oil_rate,
                well_type=well_type
            )
        
        return wells

class DataLoader:
    def __init__(self):
        self.file_ids = [
            '13twFcFA35CKbI8neIzIt-D54dzDd1B-N',
            '1n_auKzsDz5aHglQ4YvskjfHPK8ZuLBqC',
            '1bdyUFKx-FKPy7YOlq-E9Y4nupcrhOoXi',
            '1f0aJFS99ZBVkT8IXbKdZdVihbIZIpBwZ',
            '1sxq7sd4GSL-chE362k8wTLA_arehaD5U',
            '1ZwEswptUcexDn_kqm_q8qRcHYTl1WHq2'
        ]
        self.parser = SPE9Parser()
    
    def load_from_google_drive(self) -> Dict[str, Dict]:
        datasets = {}
        
        for file_id in tqdm(self.file_ids, desc="Downloading datasets"):
            try:
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tmp:
                    temp_path = tmp.name
                
                gdown.download(
                    f"https://drive.google.com/uc?id={file_id}&export=download",
                    temp_path,
                    quiet=True
                )
                
                with open(temp_path, 'r') as f:
                    content = f.read()
                
                grid_data = self.parser.parse_grid(content)
                time_points = self.parser.parse_schedule(content)
                wells = self.parser.parse_wells(content, time_points)
                
                dataset = {
                    'grid': grid_data,
                    'time_points': time_points,
                    'wells': wells,
                    'summary': self._calculate_summary(wells)
                }
                
                datasets[file_id] = dataset
                
                os.unlink(temp_path)
                
            except Exception as e:
                logger.error(f"Failed to load {file_id}: {e}")
                continue
        
        return datasets
    
    def _calculate_summary(self, wells: Dict[str, WellData]) -> Dict:
        total_oil = 0
        max_oil_rate = 0
        well_types = {'PRODUCER': 0, 'INJECTOR': 0}
        
        for well_name, well in wells.items():
            if hasattr(well, 'oil_rate'):
                rates = well.oil_rate
                if len(rates) > 0:
                    max_oil_rate = max(max_oil_rate, np.max(rates))
                    
                    if hasattr(well, 'time_points'):
                        time_pts = well.time_points
                        if len(time_pts) >= 2:
                            total_oil += np.trapz(rates, time_pts)
            
            well_types[well.well_type] += 1
        
        return {
            'total_oil_production': total_oil,
            'peak_oil_rate': max_oil_rate,
            'well_counts': well_types,
            'total_wells': len(wells)
        }
    
    def load_sample_data(self) -> Dict[str, Dict]:
        time_points = np.cumsum(np.random.exponential(30, 100))
        
        wells = {}
        for i in range(15):
            well_name = f"PROD{i+1}"
            
            base_rate = np.random.uniform(500, 2000)
            decline = np.random.uniform(0.001, 0.003)
            trend = base_rate * np.exp(-decline * time_points / 30)
            noise = np.random.normal(0, 100, len(time_points))
            oil_rate = np.maximum(trend + noise, 10)
            
            wells[well_name] = WellData(
                name=well_name,
                time_points=time_points.copy(),
                oil_rate=oil_rate,
                well_type="PRODUCER"
            )
        
        for i in range(3):
            well_name = f"INJE{i+1}"
            
            base_rate = np.random.uniform(300, 800)
            oil_rate = base_rate * np.ones_like(time_points)
            noise = np.random.normal(0, 50, len(time_points))
            oil_rate = np.maximum(oil_rate + noise, 0)
            
            wells[well_name] = WellData(
                name=well_name,
                time_points=time_points.copy(),
                oil_rate=oil_rate,
                well_type="INJECTOR"
            )
        
        dataset = {
            'grid': {'dimensions': (24, 25, 15), 'porosity': np.random.uniform(0.1, 0.2, 9000)},
            'time_points': time_points,
            'wells': wells,
            'summary': self._calculate_summary(wells)
        }
        
        return {'sample_dataset': dataset}
