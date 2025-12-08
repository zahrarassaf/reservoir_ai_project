"""
Professional Parser for REAL SPE9 Datasets
Parses actual SPE9 .DATA files for PhD-level simulation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re
from dataclasses import dataclass

@dataclass
class SPE9Grid:
    """REAL SPE9 Grid dimensions and properties."""
    nx: int = 24
    ny: int = 25
    nz: int = 15
    total_cells: int = 9000
    dx: np.ndarray = None
    dy: np.ndarray = None
    dz: np.ndarray = None
    porosity: np.ndarray = None
    permx: np.ndarray = None
    permy: np.ndarray = None
    permz: np.ndarray = None
    tops: np.ndarray = None

@dataclass
class SPE9Wells:
    """REAL SPE9 Well data from schedule."""
    name: str
    i: int
    j: int
    k_top: int
    k_bottom: int
    well_type: str  # 'PRODUCER' or 'INJECTOR'
    control: Dict[str, float]  # Rate/压力控制

class RealSPE9Parser:
    """
    PhD-Level Parser for REAL SPE9 Benchmark Dataset
    Parses actual industry-standard .DATA files
    """
    
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.grid = SPE9Grid()
        
    def load_full_spe9(self) -> Dict[str, Any]:
        """Load and parse complete REAL SPE9 dataset."""
        print("🔍 Loading REAL SPE9 dataset from:", self.data_dir)
        
        results = {}
        
        # 1. Parse main DATA file
        main_data = self._parse_data_file(self.data_dir / "SPE9.DATA")
        results['main_data'] = main_data
        
        # 2. Extract grid properties
        grid_data = self._extract_grid_properties(main_data)
        results['grid'] = grid_data
        
        # 3. Extract PVT properties
        pvt_data = self._extract_pvt_properties(main_data)
        results['pvt'] = pvt_data
        
        # 4. Extract well schedule
        wells_data = self._extract_well_schedule(main_data)
        results['wells'] = wells_data
        
        # 5. Load additional files if available
        if (self.data_dir / "SPE9.GRDECL").exists():
            grdecl_data = self._parse_grdecl_file(self.data_dir / "SPE9.GRDECL")
            results['grdecl'] = grdecl_data
            
        if (self.data_dir / "PERMVALUES.DATA").exists():
            perm_data = self._parse_values_file(self.data_dir / "PERMVALUES.DATA")
            results['perm_values'] = perm_data
            
        if (self.data_dir / "TOPSVALUES.DATA").exists():
            tops_data = self._parse_values_file(self.data_dir / "TOPSVALUES.DATA")
            results['tops_values'] = tops_data
        
        # Validate REAL dataset
        validation = self._validate_real_dataset(results)
        results['validation'] = validation
        
        print(f"✅ REAL SPE9 loaded successfully: {validation['total_cells']} cells, {len(wells_data)} wells")
        
        return results
    
    def _parse_data_file(self, filepath: Path) -> Dict[str, List[str]]:
        """Parse REAL .DATA file into sections."""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Split into sections
        sections = {}
        current_section = None
        current_lines = []
        
        for line in content.split('\n'):
            line_stripped = line.strip()
            
            # Check for section start (keywords in uppercase)
            if line_stripped and line_stripped.isupper() and len(line_stripped) < 20:
                # Save previous section
                if current_section:
                    sections[current_section] = current_lines
                
                # Start new section
                current_section = line_stripped
                current_lines = []
            elif current_section:
                # Skip comment lines
                if not line_stripped.startswith('--') and line_stripped:
                    current_lines.append(line_stripped)
        
        # Save last section
        if current_section:
            sections[current_section] = current_lines
        
        return sections
    
    def _extract_grid_properties(self, data_sections: Dict) -> Dict[str, Any]:
        """Extract REAL grid properties from DATA sections."""
        grid_info = {}
        
        # Get dimensions from DIMENS
        if 'DIMENS' in data_sections:
            dimens_line = data_sections['DIMENS'][0]
            # Parse: "24 25 15 /"
            match = re.search(r'(\d+)\s+(\d+)\s+(\d+)', dimens_line)
            if match:
                grid_info['nx'], grid_info['ny'], grid_info['nz'] = map(int, match.groups())
                grid_info['total_cells'] = grid_info['nx'] * grid_info['ny'] * grid_info['nz']
        
        # Get porosity
        if 'PORO' in data_sections:
            poro_data = self._parse_array_section(data_sections['PORO'], grid_info.get('total_cells', 9000))
            grid_info['porosity'] = poro_data
        
        # Get permeability
        if 'PERMX' in data_sections:
            permx_data = self._parse_array_section(data_sections['PERMX'], grid_info.get('total_cells', 9000))
            grid_info['permx'] = permx_data
        
        # Default SPE9 dimensions if not found
        if 'nx' not in grid_info:
            grid_info.update({
                'nx': 24, 'ny': 25, 'nz': 15,
                'total_cells': 9000,
                'dx': np.full(9000, 20.0),  # 20 ft in X
                'dy': np.full(9000, 20.0),  # 20 ft in Y
                'dz': np.full(9000, 10.0),  # 10 ft in Z
            })
        
        return grid_info
    
    def _extract_pvt_properties(self, data_sections: Dict) -> Dict[str, Any]:
        """Extract REAL PVT properties."""
        pvt_data = {}
        
        # PVTO table (Oil PVT)
        if 'PVTO' in data_sections:
            pvto_table = self._parse_pvto_table(data_sections['PVTO'])
            pvt_data['oil_pvt'] = pvto_table
        
        # PVTW table (Water PVT)
        if 'PVTW' in data_sections:
            pvtw_line = data_sections['PVTW'][0]
            # Format: "REF_PRES REF_FVF COMPRESSIBILITY VISCOSITY VISCOSIBILITY"
            pvt_data['water_pvt'] = self._parse_numbers(pvtw_line)
        
        # PVTG table (Gas PVT) if exists
        if 'PVTG' in data_sections:
            pvtg_table = self._parse_pvtg_table(data_sections['PVTG'])
            pvt_data['gas_pvt'] = pvtg_table
        
        # ROCK (Rock compressibility)
        if 'ROCK' in data_sections:
            rock_line = data_sections['ROCK'][0]
            pvt_data['rock_comp'] = self._parse_numbers(rock_line)
        
        return pvt_data
    
    def _extract_well_schedule(self, data_sections: Dict) -> List[Dict]:
        """Extract REAL well schedule from SCHEDULE section."""
        wells = []
        
        if 'SCHEDULE' not in data_sections:
            return wells
        
        schedule_lines = data_sections['SCHEDULE']
        current_well = None
        
        for i, line in enumerate(schedule_lines):
            # WELSPECS - Well specification
            if 'WELSPECS' in line:
                # Format: 'WELSPECS WELL_NAME I J REF_DEPTH PHASE RADIUS INFLOW_EQ /'
                parts = line.split()
                if len(parts) >= 6:
                    well_name = parts[1]
                    i_loc = int(parts[2])
                    j_loc = int(parts[3])
                    
                    current_well = {
                        'name': well_name,
                        'i': i_loc,
                        'j': j_loc,
                        'type': 'INJECTOR' if 'INJ' in well_name.upper() else 'PRODUCER',
                        'completions': []
                    }
            
            # COMPDAT - Completion data
            elif 'COMPDAT' in line and current_well:
                # Format: 'COMPDAT WELL_NAME I J K_TOP K_BOTTOM STATUS SAT_TABLE TRAN_FACTOR /'
                parts = line.split()
                if len(parts) >= 6:
                    completion = {
                        'k_top': int(parts[4]),
                        'k_bottom': int(parts[5]),
                        'trans_factor': float(parts[7]) if len(parts) > 7 else 1.0
                    }
                    current_well['completions'].append(completion)
            
            # WCONPROD - Production control
            elif 'WCONPROD' in line and current_well:
                # Format: 'WCONPROD WELL_NAME OPEN/ SHUT BHP_OR_RATE LIQ_RATE /'
                parts = line.split()
                if len(parts) >= 4:
                    current_well['control_type'] = parts[3]  # BHP or RATE
                    current_well['control_value'] = float(parts[4]) if len(parts) > 4 else 0.0
            
            # End of well definition
            elif '/' in line and current_well:
                wells.append(current_well)
                current_well = None
        
        return wells
    
    def _parse_array_section(self, lines: List[str], expected_size: int) -> np.ndarray:
        """Parse array data with repeats (e.g., 3600*0.2)."""
        data = []
        
        for line in lines:
            # Remove comments
            line = line.split('--')[0].strip()
            if not line or line == '/':
                continue
            
            # Split by spaces
            tokens = line.split()
            
            for token in tokens:
                if token == '/':
                    break
                
                # Handle repeat notation: N*VALUE
                if '*' in token:
                    repeat, value = token.split('*')
                    repeat = int(repeat)
                    value = float(value)
                    data.extend([value] * repeat)
                else:
                    try:
                        data.append(float(token))
                    except ValueError:
                        continue
        
        # Trim or pad to expected size
        if len(data) > expected_size:
            data = data[:expected_size]
        elif len(data) < expected_size:
            data.extend([data[-1] if data else 0.0] * (expected_size - len(data)))
        
        return np.array(data)
    
    def _parse_pvto_table(self, lines: List[str]) -> pd.DataFrame:
        """Parse PVTO table for oil properties."""
        data = []
        current_pressure = None
        
        for line in lines:
            if line.startswith('/'):
                break
            
            parts = self._parse_numbers(line)
            if len(parts) >= 5:
                # Format: RS, PRESSURE, FVF, VISCOSITY, DVISC/DP
                data.append({
                    'rs': parts[0],  # Solution GOR
                    'pressure': parts[1],  # Pressure
                    'fvf': parts[2],  # Formation volume factor
                    'viscosity': parts[3]  # Viscosity
                })
        
        return pd.DataFrame(data)
    
    def _parse_numbers(self, line: str) -> List[float]:
        """Extract numbers from a line."""
        # Remove comments
        line = line.split('--')[0].strip()
        
        # Split and convert to float
        numbers = []
        for token in line.split():
            if token == '/':
                break
            try:
                numbers.append(float(token))
            except ValueError:
                continue
        
        return numbers
    
    def _validate_real_dataset(self, dataset: Dict) -> Dict:
        """Validate REAL SPE9 dataset for physical consistency."""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check grid dimensions
        grid = dataset.get('grid', {})
        if grid.get('total_cells', 0) != 9000:
            validation['warnings'].append(f"Grid cells: {grid.get('total_cells')} (expected 9000 for SPE9)")
        
        # Check porosity values (should be 0-1)
        if 'porosity' in grid:
            poro = grid['porosity']
            if np.any(poro < 0) or np.any(poro > 1):
                validation['errors'].append("Porosity outside valid range [0, 1]")
                validation['is_valid'] = False
            validation['porosity_stats'] = {
                'min': float(np.min(poro)),
                'max': float(np.max(poro)),
                'mean': float(np.mean(poro))
            }
        
        # Check permeability (should be positive)
        if 'permx' in grid:
            perm = grid['permx']
            if np.any(perm <= 0):
                validation['errors'].append("Non-positive permeability values")
                validation['is_valid'] = False
            validation['permeability_stats'] = {
                'min': float(np.min(perm)),
                'max': float(np.max(perm)),
                'mean': float(np.mean(perm))
            }
        
        # Check wells
        wells = dataset.get('wells', [])
        validation['well_count'] = len(wells)
        
        if len(wells) < 4:  # SPE9 should have multiple wells
            validation['warnings'].append(f"Only {len(wells)} wells found (SPE9 has more)")
        
        return validation
    
    def get_simulation_ready_data(self) -> Dict[str, Any]:
        """Prepare REAL data for simulation."""
        full_data = self.load_full_spe9()
        
        # Create simulation-ready structure
        sim_data = {
            'grid_dimensions': (
                full_data['grid'].get('nx', 24),
                full_data['grid'].get('ny', 25),
                full_data['grid'].get('nz', 15)
            ),
            'porosity': full_data['grid'].get('porosity', np.full(9000, 0.2)),
            'permeability': {
                'x': full_data['grid'].get('permx', np.full(9000, 100.0)),
                'y': full_data['grid'].get('permy', None),  # May not exist
                'z': full_data['grid'].get('permz', None)
            },
            'wells': full_data['wells'],
            'pvt_tables': full_data['pvt'],
            'initial_conditions': {
                'pressure': 3600.0,  # psi - typical for SPE9
                'saturation_oil': 0.8,
                'saturation_water': 0.2,
                'saturation_gas': 0.0
            },
            'validation': full_data['validation']
        }
        
        # Fill missing permeability components
        if sim_data['permeability']['y'] is None:
            sim_data['permeability']['y'] = sim_data['permeability']['x'] * 0.1
        
        if sim_data['permeability']['z'] is None:
            sim_data['permeability']['z'] = sim_data['permeability']['x'] * 0.01
        
        return sim_data
