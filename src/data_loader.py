import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

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
        """Validate well data consistency."""
        if len(self.time_points) != len(self.production_rates):
            return False
        if np.any(self.production_rates < 0):
            logger.warning(f"Well {self.name} has negative production rates")
        return True

class ReservoirData:
    def __init__(self):
        self.wells: Dict[str, WellData] = {}
        self.grid_dimensions: Tuple[int, int, int] = (0, 0, 0)
        self.porosity: np.ndarray = np.array([])
        self.permeability: np.ndarray = np.array([])
        self.depth_top: np.ndarray = np.array([])
        self.time_unit: str = "days"
        self.production_unit: str = "bbl/day"
        self.pressure_unit: str = "psi"
        self._initial_pressure: float = 3600.0  # psia from SPE9
        
    def parse_spe9_data(self, file_content: str) -> bool:
        """
        Parse SPE9 reservoir simulation data file.
        
        Args:
            file_content: Raw content of SPE9.DATA file
            
        Returns:
            bool: True if parsing successful
        """
        try:
            lines = file_content.split('\n')
            
            # Parse DIMENS
            for i, line in enumerate(lines):
                if 'DIMENS' in line and not line.strip().startswith('--'):
                    # Get next line with dimensions
                    dim_line = lines[i+1].strip().strip('/')
                    dims = [int(x) for x in dim_line.split()]
                    if len(dims) >= 3:
                        self.grid_dimensions = tuple(dims[:3])
                        logger.info(f"Grid dimensions: {self.grid_dimensions}")
            
            # Parse porosities
            for i, line in enumerate(lines):
                if 'PORO' in line and not line.strip().startswith('--'):
                    poro_values = []
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().startswith('--'):
                        if '/' in lines[j]:
                            break
                        # Parse values like "600*0.087"
                        for val in lines[j].split():
                            if '*' in val:
                                count, value = val.split('*')
                                poro_values.extend([float(value)] * int(count))
                            else:
                                try:
                                    poro_values.append(float(val))
                                except:
                                    pass
                        j += 1
                    if poro_values:
                        self.porosity = np.array(poro_values)
                        logger.info(f"Porosity loaded: {len(poro_values)} values, "
                                  f"mean: {np.mean(poro_values):.3f}")
            
            # Parse PVTO data for fluid properties
            pvto_data = self._parse_pvto_section(lines)
            if pvto_data:
                logger.info(f"PVTO data parsed with {len(pvto_data)} rows")
            
            # Parse well data from COMPDAT and WCONPROD
            self._parse_well_data(lines)
            
            # Generate time series data based on TSTEP
            time_points = self._parse_time_steps(lines)
            if time_points.size > 0:
                self._generate_production_profiles(time_points)
            
            return True
            
        except Exception as e:
            logger.error(f"Error parsing SPE9 data: {e}")
            return False
    
    def _parse_well_data(self, lines: List[str]) -> None:
        """Parse well completion and control data."""
        well_locations = {}
        
        # Parse COMPDAT for well locations
        for i, line in enumerate(lines):
            if 'COMPDAT' in line and not line.strip().startswith('--'):
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith('--'):
                    if '/' in lines[j]:
                        break
                    parts = lines[j].split()
                    if len(parts) >= 5:
                        well_name = parts[0].strip("'")
                        i_idx, j_idx, k_upper, k_lower = map(int, parts[1:5])
                        
                        if well_name not in well_locations:
                            well_locations[well_name] = {
                                'i': i_idx,
                                'j': j_idx,
                                'k_range': (k_upper, k_lower),
                                'completions': []
                            }
                        well_locations[well_name]['completions'].append(
                            (i_idx, j_idx, k_upper, k_lower)
                        )
                    j += 1
        
        # Parse WCONPROD for production controls
        for i, line in enumerate(lines):
            if 'WCONPROD' in line and not line.strip().startswith('--'):
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith('--'):
                    if '/' in lines[j]:
                        break
                    parts = lines[j].split()
                    if len(parts) >= 2:
                        well_name = parts[0].strip("'")
                        if '*' in well_name:  # Handle wildcards
                            continue
                        if well_name in well_locations:
                            # Extract control mode and rate
                            control_mode = parts[2]
                            try:
                                oil_rate = float(parts[3]) if len(parts) > 3 else 1500.0
                                bhp_limit = float(parts[8]) if len(parts) > 8 else 1000.0
                                well_locations[well_name].update({
                                    'control_mode': control_mode,
                                    'oil_rate': oil_rate,
                                    'bhp_limit': bhp_limit
                                })
                            except (ValueError, IndexError):
                                pass
                    j += 1
        
        # Create WellData objects
        for well_name, data in well_locations.items():
            self.wells[well_name] = WellData(
                name=well_name,
                time_points=np.array([]),  # Will be populated later
                production_rates=np.array([]),
                pressures=np.array([]),
                coordinates=(data['i'] * 300, data['j'] * 300, 0),  # Using DX=300 from SPE9
                completion_zones=list(range(data['k_range'][0], data['k_range'][1] + 1))
            )
    
    def _parse_time_steps(self, lines: List[str]) -> np.ndarray:
        """Parse TSTEP data to get simulation time points."""
        time_points = [0.0]  # Start at time 0
        
        for i, line in enumerate(lines):
            if 'TSTEP' in line and not line.strip().startswith('--'):
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith('--'):
                    if '/' in lines[j]:
                        break
                    parts = lines[j].split()
                    for part in parts:
                        if part.endswith('*'):
                            try:
                                count = int(part.rstrip('*'))
                                value = float(parts[parts.index(part) + 1])
                                time_points.extend([value] * count)
                            except:
                                pass
                        else:
                            try:
                                time_points.append(float(part))
                            except:
                                pass
                    j += 1
        
        # Convert to cumulative time
        cumulative_time = np.cumsum(time_points)
        logger.info(f"Time steps parsed: {len(cumulative_time)} points, "
                   f"total {cumulative_time[-1]:.1f} days")
        return cumulative_time
    
    def _generate_production_profiles(self, time_points: np.ndarray) -> None:
        """
        Generate production profiles based on reservoir decline curve analysis.
        This should be replaced with actual reservoir simulation.
        """
        for well_name, well in self.wells.items():
            n_points = len(time_points)
            
            # Generate production profile using Arps decline
            qi = np.random.uniform(500, 2000)  # Initial rate
            di = np.random.uniform(0.01, 0.05)  # Decline rate
            b = np.random.uniform(0.5, 1.5)     # Decline exponent
            
            # Hyperbolic decline
            production = qi / (1 + b * di * time_points) ** (1/b)
            
            # Add some noise
            noise = np.random.normal(0, 50, n_points)
            production = np.maximum(0, production + noise)
            
            # Generate pressure profile
            initial_pressure = self._initial_pressure
            pressure_drop = (production / qi) * 500  # Simple correlation
            pressure = np.maximum(1000, initial_pressure - pressure_drop)
            
            well.time_points = time_points
            well.production_rates = production
            well.pressures = pressure
    
    def _parse_pvto_section(self, lines: List[str]) -> List[Dict]:
        """Parse PVTO section for oil properties."""
        pvto_data = []
        in_pvto = False
        
        for line in lines:
            if 'PVTO' in line and not line.strip().startswith('--'):
                in_pvto = True
                continue
            
            if in_pvto:
                if line.strip() == '/':
                    break
                if not line.strip().startswith('--'):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            rs = float(parts[0])  # Solution GOR
                            pb = float(parts[1])  # Bubble point pressure
                            bo = float(parts[2])  # Oil FVF
                            visc = float(parts[3])  # Viscosity
                            pvto_data.append({
                                'rs': rs,
                                'pb': pb,
                                'bo': bo,
                                'viscosity': visc
                            })
                        except ValueError:
                            continue
        
        return pvto_data
    
    def summary(self) -> Dict:
        """Generate comprehensive data summary."""
        total_wells = len(self.wells)
        total_production = 0
        max_production = 0
        min_production = float('inf')
        
        for well in self.wells.values():
            if well.production_rates.size > 0:
                total_production += np.sum(well.production_rates)
                max_production = max(max_production, np.max(well.production_rates))
                min_production = min(min_production, np.min(well.production_rates))
        
        return {
            'wells': total_wells,
            'grid_dimensions': self.grid_dimensions,
            'total_production': total_production,
            'max_production_rate': max_production if max_production != 0 else 0,
            'min_production_rate': min_production if min_production != float('inf') else 0,
            'average_porosity': np.mean(self.porosity) if self.porosity.size > 0 else 0,
            'has_production_data': any(w.production_rates.size > 0 for w in self.wells.values()),
            'has_pressure_data': any(w.pressures.size > 0 for w in self.wells.values()),
            'time_points': len(self.wells[list(self.wells.keys())[0]].time_points) 
                          if self.wells else 0
        }
    
    def create_sample_data(self, n_wells: int = 5, n_time_points: int = 365) -> None:
        """Create realistic sample data for testing."""
        np.random.seed(42)
        
        for i in range(n_wells):
            well_name = f"WELL_{i+1:03d}"
            
            # Time points (daily for one year)
            time_points = np.arange(n_time_points)
            
            # Generate production using decline curve
            qi = np.random.uniform(800, 2000)
            di = np.random.uniform(0.001, 0.01)
            b = 0.8  # Hyperbolic decline exponent
            
            production = qi / (1 + b * di * time_points) ** (1/b)
            production += np.random.normal(0, 50, n_time_points)
            production = np.maximum(0, production)
            
            # Generate pressure profile
            initial_pressure = 3500
            pressure = initial_pressure - (production / qi) * 1000
            pressure = np.maximum(1500, pressure)
            
            self.wells[well_name] = WellData(
                name=well_name,
                time_points=time_points,
                production_rates=production,
                pressures=pressure,
                coordinates=(
                    np.random.uniform(0, 10000),
                    np.random.uniform(0, 10000),
                    np.random.uniform(5000, 10000)
                ),
                completion_zones=list(range(1, np.random.randint(3, 8)))
            )
        
        self.grid_dimensions = (24, 25, 15)
        self.porosity = np.random.uniform(0.08, 0.18, 9000)  # 24*25*15 = 9000
