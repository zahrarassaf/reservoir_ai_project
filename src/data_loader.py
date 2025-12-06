import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WellData:
    name: str
    time_points: np.ndarray
    production_rates: np.ndarray
    pressures: np.ndarray
    coordinates: Tuple[float, float, float]
    completion_zones: List[int]
    
    def validate(self) -> bool:
        if len(self.time_points) != len(self.production_rates):
            return False
        if len(self.time_points) != len(self.pressures):
            return False
        return True
    
    def summary(self) -> Dict:
        return {
            'name': self.name,
            'data_points': len(self.time_points),
            'max_rate': float(np.max(self.production_rates)) if len(self.production_rates) > 0 else 0,
            'min_rate': float(np.min(self.production_rates)) if len(self.production_rates) > 0 else 0,
            'avg_rate': float(np.mean(self.production_rates)) if len(self.production_rates) > 0 else 0,
            'cumulative': float(np.sum(self.production_rates)) if len(self.production_rates) > 0 else 0,
            'max_pressure': float(np.max(self.pressures)) if len(self.pressures) > 0 else 0,
            'min_pressure': float(np.min(self.pressures)) if len(self.pressures) > 0 else 0
        }

class ReservoirData:
    def __init__(self):
        self.wells: Dict[str, WellData] = {}
        self.grid_dimensions: Tuple[int, int, int] = (0, 0, 0)
        self.porosity: np.ndarray = np.array([])
        self.permeability: np.ndarray = np.array([])
        self.depth_top: np.ndarray = np.array([])
        self.time_unit: str = "days"
        self.production_unit: str = "STB/day"
        self.pressure_unit: str = "psia"
        self._initial_pressure: float = 3600.0
        self._oil_fvf: float = 1.12
        self._water_fvf: float = 1.0034
        self._gas_fvf: float = 0.001
        self._rock_compressibility: float = 4e-6
        self._fluid_properties: Dict = {}
        
    def load_spe9_file(self, file_path: str) -> bool:
        try:
            logger.info(f"Loading SPE9 file: {file_path}")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            self._parse_reservoir_properties(lines)
            self._parse_well_definitions(lines)
            time_data = self._parse_simulation_schedule(lines)
            
            if self.wells:
                self._generate_realistic_profiles(time_data)
            else:
                self._create_synthetic_wells()
                self._generate_realistic_profiles(time_data)
            
            self._parse_rock_fluid_properties(lines)
            
            logger.info(f"Successfully loaded SPE9 data: {len(self.wells)} wells, "
                       f"grid {self.grid_dimensions}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading SPE9 file {file_path}: {e}")
            return False
    
    def _parse_reservoir_properties(self, lines: List[str]) -> None:
        for i, line in enumerate(lines):
            if 'DIMENS' in line and not line.strip().startswith('--'):
                for j in range(i+1, min(i+5, len(lines))):
                    dim_line = lines[j].strip()
                    if dim_line and not dim_line.startswith('--'):
                        dim_line = dim_line.split('--')[0].strip().strip('/')
                        if dim_line:
                            parts = dim_line.split()
                            try:
                                if len(parts) >= 3:
                                    self.grid_dimensions = (
                                        int(parts[0]), 
                                        int(parts[1]), 
                                        int(parts[2])
                                    )
                                    logger.info(f"Grid dimensions: {self.grid_dimensions}")
                                    break
                            except ValueError:
                                continue
        
        poro_values = []
        for i, line in enumerate(lines):
            if 'PORO' in line and not line.strip().startswith('--'):
                j = i + 1
                while j < len(lines) and j < i + 20:
                    current_line = lines[j].strip()
                    if current_line.startswith('--') or not current_line:
                        j += 1
                        continue
                    if '/' in current_line:
                        break
                    
                    for token in current_line.split():
                        if '*' in token:
                            try:
                                count_str, value_str = token.split('*')
                                count = int(count_str)
                                value = float(value_str)
                                poro_values.extend([value] * count)
                            except ValueError:
                                pass
                        else:
                            try:
                                poro_values.append(float(token))
                            except ValueError:
                                pass
                    j += 1
                
                if poro_values:
                    self.porosity = np.array(poro_values)
                    logger.info(f"Porosity: {len(poro_values)} values, "
                              f"range: {np.min(poro_values):.3f}-{np.max(poro_values):.3f}")
                break
    
    def _parse_well_definitions(self, lines: List[str]) -> None:
        wells_info = {}
        
        in_welspecs = False
        for i, line in enumerate(lines):
            if 'WELSPECS' in line and not line.strip().startswith('--'):
                in_welspecs = True
                continue
            
            if in_welspecs:
                line_clean = line.strip()
                if line_clean.startswith('/'):
                    break
                if line_clean and not line_clean.startswith('--'):
                    parts = line_clean.split()
                    if len(parts) >= 4:
                        well_name = parts[0].strip("'").strip()
                        try:
                            i_loc = int(parts[2])
                            j_loc = int(parts[3])
                            ref_depth = float(parts[4]) if len(parts) > 4 else 9110.0
                            
                            wells_info[well_name] = {
                                'i': i_loc,
                                'j': j_loc,
                                'ref_depth': ref_depth,
                                'completions': [],
                                'controls': {}
                            }
                        except (ValueError, IndexError):
                            continue
        
        in_compdat = False
        for i, line in enumerate(lines):
            if 'COMPDAT' in line and not line.strip().startswith('--'):
                in_compdat = True
                continue
            
            if in_compdat:
                line_clean = line.strip()
                if line_clean.startswith('/'):
                    break
                if line_clean and not line_clean.startswith('--'):
                    parts = line_clean.split()
                    if len(parts) >= 5:
                        well_name = parts[0].strip("'").strip()
                        if well_name in wells_info:
                            try:
                                i_loc = int(parts[1])
                                j_loc = int(parts[2])
                                k_upper = int(parts[3])
                                k_lower = int(parts[4])
                                wells_info[well_name]['completions'].append({
                                    'i': i_loc,
                                    'j': j_loc,
                                    'k_upper': k_upper,
                                    'k_lower': k_lower
                                })
                            except (ValueError, IndexError):
                                continue
        
        for well_name, info in wells_info.items():
            dx, dy = 300, 300
            i_loc = info['i']
            j_loc = info['j']
            
            if i_loc > 0 and j_loc > 0:
                x = (i_loc - 0.5) * dx
                y = (j_loc - 0.5) * dy
            else:
                x = 0.0
                y = 0.0
                
            z = info['ref_depth']
            
            completion_zones = []
            for comp in info['completions']:
                if 'k_upper' in comp and 'k_lower' in comp:
                    completion_zones.extend(range(comp['k_upper'], comp['k_lower'] + 1))
            
            if not completion_zones:
                completion_zones = [2, 3, 4]
            
            self.wells[well_name] = WellData(
                name=well_name,
                time_points=np.array([]),
                production_rates=np.array([]),
                pressures=np.array([]),
                coordinates=(x, y, z),
                completion_zones=completion_zones
            )
    
    def _parse_simulation_schedule(self, lines: List[str]) -> Dict:
        schedule_data = {
            'time_points': [0.0],
            'control_periods': [],
            'total_days': 0
        }
        
        tstep_values = []
        for i, line in enumerate(lines):
            if 'TSTEP' in line and not line.strip().startswith('--'):
                current_line = line.replace('TSTEP', '').strip()
                
                for token in current_line.split():
                    if token and token != '/':
                        try:
                            if '*' in token:
                                count_str, value_str = token.split('*')
                                count = int(count_str)
                                value = float(value_str)
                                tstep_values.extend([value] * count)
                            else:
                                tstep_values.append(float(token))
                        except ValueError:
                            pass
                
                j = i + 1
                while j < len(lines) and j < i + 10:
                    next_line = lines[j].strip()
                    if next_line.startswith('/'):
                        break
                    if next_line and not next_line.startswith('--'):
                        for token in next_line.split():
                            try:
                                if '*' in token:
                                    count_str, value_str = token.split('*')
                                    count = int(count_str)
                                    value = float(value_str)
                                    tstep_values.extend([value] * count)
                                else:
                                    tstep_values.append(float(token))
                            except ValueError:
                                pass
                    j += 1
        
        if not tstep_values:
            logger.warning("No TSTEP found, using SPE9 default schedule")
            tstep_values = [10.0]*30 + [10.0]*6 + [10.0]*54
        
        cumulative_time = np.cumsum(tstep_values)
        schedule_data['time_points'].extend(cumulative_time.tolist())
        schedule_data['total_days'] = cumulative_time[-1]
        
        logger.info(f"Schedule: {len(schedule_data['time_points'])} time points, "
                   f"total {schedule_data['total_days']} days")
        
        return schedule_data
    
    def _generate_realistic_profiles(self, schedule_data: Dict) -> None:
        time_points = np.array(schedule_data['time_points'], dtype=np.float64)
        
        for well_name, well in self.wells.items():
            n_points = len(time_points)
            
            is_injector = 'INJE' in well_name.upper()
            
            if is_injector:
                base_rate = -5000.0
                rates = np.full(n_points, base_rate, dtype=np.float64)
                
                noise = np.random.normal(0, 200, n_points).astype(np.float64)
                rates = rates + noise
                
                base_pressure = 4000.0
                pressures = base_pressure + np.random.normal(0, 100, n_points)
                pressures = pressures.astype(np.float64)
                
            else:
                if self.grid_dimensions[0] > 0 and self.grid_dimensions[1] > 0 and 300 > 0:
                    try:
                        i_norm = well.coordinates[0] / (self.grid_dimensions[0] * 300)
                        j_norm = well.coordinates[1] / (self.grid_dimensions[1] * 300)
                    except ZeroDivisionError:
                        i_norm = 0.5
                        j_norm = 0.5
                else:
                    i_norm = 0.5
                    j_norm = 0.5
                
                centrality = 1.0 - abs(i_norm - 0.5) - abs(j_norm - 0.5)
                qi = 800.0 + centrality * 1200.0
                
                avg_poro = np.mean(self.porosity) if len(self.porosity) > 0 else 0.15
                di = 0.0005 + (0.15 - avg_poro) * 0.01
                b = 0.8 + np.random.uniform(-0.2, 0.4)
                
                safe_b = max(0.1, min(b, 1.9))
                safe_di = max(1e-6, min(di, 1.0))
                
                rates = qi / (1 + safe_b * safe_di * time_points) ** (1/safe_b)
                
                mask_300 = time_points >= 300
                mask_360 = time_points >= 360
                rates = np.where(mask_300 & ~mask_360, rates * (100/1500), rates)
                
                operational_noise = np.random.normal(0, qi * 0.05, n_points)
                rates = rates + operational_noise
                
                rates = np.maximum(0.0, rates)
                
                drawdown_factor = rates / qi if qi > 0 else 0
                pressures = self._initial_pressure - drawdown_factor * 1000
                
                min_bhp = 1000.0
                pressures = np.maximum(min_bhp, pressures)
            
            if n_points > 10:
                from scipy.ndimage import gaussian_filter1d
                rates = gaussian_filter1d(rates, sigma=2)
                pressures = gaussian_filter1d(pressures, sigma=2)
            
            rates = rates.astype(np.float64)
            pressures = pressures.astype(np.float64)
            
            well.time_points = time_points
            well.production_rates = rates
            well.pressures = pressures
    
    def _parse_rock_fluid_properties(self, lines: List[str]) -> None:
        self._fluid_properties = {}
    
    def _create_synthetic_wells(self) -> None:
        logger.warning("No wells parsed from file, creating synthetic wells")
        
        synthetic_wells = [
            ('INJE1', 24, 25, -5000),
            ('PRODU2', 5, 1, 1500),
            ('PRODU3', 8, 2, 1500),
            ('PRODU4', 11, 3, 1500),
            ('PRODU5', 10, 4, 1500),
            ('PRODU6', 12, 5, 1500),
            ('PRODU7', 4, 6, 1500),
            ('PRODU8', 8, 7, 1500),
            ('PRODU9', 14, 8, 1500),
            ('PRODU10', 11, 9, 1500),
        ]
        
        for name, i_loc, j_loc, base_rate in synthetic_wells:
            x = (i_loc - 0.5) * 300 if i_loc > 0 else 0
            y = (j_loc - 0.5) * 300 if j_loc > 0 else 0
            z = 9110.0
            
            self.wells[name] = WellData(
                name=name,
                time_points=np.array([]),
                production_rates=np.array([]),
                pressures=np.array([]),
                coordinates=(x, y, z),
                completion_zones=[2, 3, 4]
            )
        
        logger.info(f"Created {len(self.wells)} synthetic wells")
    
    def create_sample_data(self, n_wells: int = 8, n_time_points: int = 365) -> None:
        np.random.seed(42)
        
        logger.info(f"Creating sample data: {n_wells} wells, {n_time_points} days")
        
        self.wells.clear()
        
        self.grid_dimensions = (24, 25, 15)
        
        n_cells = self.grid_dimensions[0] * self.grid_dimensions[1] * self.grid_dimensions[2]
        self.porosity = np.random.uniform(0.08, 0.18, n_cells)
        
        time_points = np.arange(0, n_time_points, dtype=np.float64)
        
        for i in range(n_wells):
            if i == 0:
                well_name = "INJE1"
                is_injector = True
            else:
                well_name = f"PRODU{i:02d}"
                is_injector = False
            
            x = np.random.uniform(0, self.grid_dimensions[0] * 300)
            y = np.random.uniform(0, self.grid_dimensions[1] * 300)
            z = 9000 + np.random.uniform(-500, 500)
            
            if is_injector:
                base_rate = -4000.0
                rates = base_rate + np.random.normal(0, 200, n_time_points)
                pressures = 3800.0 + np.random.normal(0, 100, n_time_points)
                completion_zones = [11, 12, 13, 14, 15]
            else:
                qi = np.random.uniform(800, 2000)
                di = np.random.uniform(0.0003, 0.001)
                b = np.random.uniform(0.7, 1.3)
                
                safe_b = max(0.1, min(b, 1.9))
                safe_di = max(1e-6, min(di, 1.0))
                
                rates = qi / (1 + safe_b * safe_di * time_points) ** (1/safe_b)
                
                monthly_cycle = 50 * np.sin(2 * np.pi * time_points / 30)
                rates = rates + monthly_cycle
                
                rates = rates + np.random.normal(0, qi * 0.05, n_time_points)
                rates = np.maximum(0.0, rates)
                
                initial_pressure = 3600.0
                drawdown = (rates / qi) * 800 if qi > 0 else 0
                pressures = initial_pressure - drawdown
                pressures = np.maximum(1200.0, pressures)
                
                n_zones = np.random.randint(3, 8)
                start_zone = np.random.randint(1, 8)
                completion_zones = list(range(start_zone, start_zone + n_zones))
            
            self.wells[well_name] = WellData(
                name=well_name,
                time_points=time_points,
                production_rates=rates.astype(np.float64),
                pressures=pressures.astype(np.float64),
                coordinates=(x, y, z),
                completion_zones=completion_zones
            )
        
        self._initial_pressure = 3600.0
        self._oil_fvf = 1.12
        self._water_fvf = 1.0034
        self._rock_compressibility = 4e-6
        
        logger.info(f"Sample data created: {len(self.wells)} wells with {n_time_points} days each")
    
    def summary(self) -> Dict[str, Any]:
        total_wells = len(self.wells)
        
        if total_wells == 0:
            return {
                'wells': 0,
                'time_points': 0,
                'has_production_data': False,
                'has_pressure_data': False,
                'grid_dimensions': self.grid_dimensions,
                'production_range': {'min': 0, 'max': 0, 'mean': 0},
                'pressure_range': {'min': 0, 'max': 0, 'mean': 0},
                'total_production': 0,
                'well_summaries': []
            }
        
        all_rates = []
        all_pressures = []
        total_cumulative = 0
        well_summaries = []
        
        for well_name, well in self.wells.items():
            well_summary = well.summary()
            well_summaries.append(well_summary)
            
            if len(well.production_rates) > 0:
                all_rates.extend(well.production_rates.tolist())
                total_cumulative += well_summary['cumulative']
            
            if len(well.pressures) > 0:
                all_pressures.extend(well.pressures.tolist())
        
        has_production = len(all_rates) > 0
        has_pressure = len(all_pressures) > 0
        
        production_range = {
            'min': float(np.min(all_rates)) if has_production else 0,
            'max': float(np.max(all_rates)) if has_production else 0,
            'mean': float(np.mean(all_rates)) if has_production else 0
        }
        
        pressure_range = {
            'min': float(np.min(all_pressures)) if has_pressure else 0,
            'max': float(np.max(all_pressures)) if has_pressure else 0,
            'mean': float(np.mean(all_pressures)) if has_pressure else 0
        }
        
        time_points_count = 0
        if self.wells:
            first_well = list(self.wells.values())[0]
            time_points_count = len(first_well.time_points)
        
        return {
            'wells': total_wells,
            'time_points': time_points_count,
            'has_production_data': has_production,
            'has_pressure_data': has_pressure,
            'grid_dimensions': self.grid_dimensions,
            'production_range': production_range,
            'pressure_range': pressure_range,
            'total_production': total_cumulative,
            'well_summaries': well_summaries
        }
    
    def get_well_dataframe(self) -> pd.DataFrame:
        data_frames = []
        
        for well_name, well in self.wells.items():
            if len(well.time_points) > 0:
                df = pd.DataFrame({
                    'time': well.time_points,
                    'well': well_name,
                    'production_rate': well.production_rates,
                    'pressure': well.pressures,
                    'x': well.coordinates[0],
                    'y': well.coordinates[1],
                    'z': well.coordinates[2]
                })
                data_frames.append(df)
        
        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def export_to_csv(self, output_path: str) -> bool:
        try:
            import os
            os.makedirs(output_path, exist_ok=True)
            
            well_df = self.get_well_dataframe()
            if not well_df.empty:
                well_file = os.path.join(output_path, 'well_data.csv')
                well_df.to_csv(well_file, index=False)
                logger.info(f"Exported well data to {well_file}")
            
            if len(self.porosity) > 0:
                reservoir_data = {
                    'grid_dimensions': [self.grid_dimensions],
                    'total_wells': [len(self.wells)],
                    'avg_porosity': [np.mean(self.porosity)],
                    'initial_pressure': [self._initial_pressure]
                }
                reservoir_df = pd.DataFrame(reservoir_data)
                reservoir_file = os.path.join(output_path, 'reservoir_properties.csv')
                reservoir_df.to_csv(reservoir_file, index=False)
                logger.info(f"Exported reservoir properties to {reservoir_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
