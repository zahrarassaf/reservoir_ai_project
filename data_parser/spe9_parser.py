"""
Enhanced SPE9 Parser - Reads ALL real data
"""

import numpy as np
import re
from pathlib import Path

class EnhancedSPE9Parser:
    """Parse ALL SPE9 data files correctly."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
    
    def parse_all(self):
        """Parse ALL SPE9 files."""
        data = {}
        
        # 1. Parse main DATA file
        data.update(self._parse_spe9_data())
        
        # 2. Parse grid properties
        data.update(self._parse_grid_files())
        
        # 3. Parse rock properties
        data.update(self._parse_rock_properties())
        
        # 4. Parse fluid properties
        data.update(self._parse_fluid_properties())
        
        return data
    
    def _parse_spe9_data(self):
        """Parse SPE9.DATA - the main simulation file."""
        data_file = self.data_dir / "SPE9.DATA"
        
        if not data_file.exists():
            raise FileNotFoundError(f"SPE9.DATA not found: {data_file}")
        
        with open(data_file, 'r') as f:
            content = f.read()
        
        parsed = {}
        
        # Extract grid dimensions
        dim_match = re.search(r'DIMENS\s+(\d+)\s+(\d+)\s+(\d+)', content)
        if dim_match:
            parsed['grid_dimensions'] = (
                int(dim_match.group(1)),
                int(dim_match.group(2)),
                int(dim_match.group(3))
            )
        
        # Extract wells
        wells = []
        well_pattern = r"'(\w+)'\s+['\"]?(\w+)['\"]?\s+(\d+)\s+(\d+)\s+([\d\.]+)\s+['\"]?(\w+)['\"]?"
        for match in re.finditer(well_pattern, content):
            wells.append({
                'name': match.group(1),
                'type': 'INJECTOR' if 'INJ' in match.group(1).upper() else 'PRODUCER',
                'i': int(match.group(3)),
                'j': int(match.group(4)),
                'k': float(match.group(5)),
                'fluid': match.group(6)
            })
        
        parsed['wells'] = wells
        
        return parsed
    
    def _parse_grid_files(self):
        """Parse grid geometry files."""
        parsed = {}
        
        # Parse GRID.INC
        grid_file = self.data_dir / "SPE9_GRID.INC"
        if grid_file.exists():
            with open(grid_file, 'r') as f:
                grid_data = f.read()
            # Parse DX, DY, DZ arrays
            parsed['cell_dimensions'] = self._extract_arrays(grid_data)
        
        # Parse TOPS
        tops_file = self.data_dir / "TOPSVALUES.DATA"
        if tops_file.exists():
            with open(tops_file, 'r') as f:
                tops_data = f.read()
            parsed['tops'] = self._parse_numbers(tops_data)
        
        return parsed
    
    def _parse_rock_properties(self):
        """Parse rock properties."""
        parsed = {}
        
        # Parse porosity
        poro_file = self.data_dir / "SPE9_PORO.INC"
        if poro_file.exists():
            with open(poro_file, 'r') as f:
                poro_data = f.read()
            parsed['porosity'] = self._parse_numbers(poro_data)
        
        # Parse permeability
        perm_file = self.data_dir / "PERMVALUES.DATA"
        if perm_file.exists():
            with open(perm_file, 'r') as f:
                perm_data = f.read()
            parsed['permeability'] = self._parse_numbers(perm_data)
        
        return parsed
    
    def _parse_fluid_properties(self):
        """Parse fluid properties."""
        parsed = {}
        
        # Parse PVT
        pvt_file = self.data_dir / "SPE9_PVT.INC"
        if pvt_file.exists():
            parsed['pvt_tables'] = self._parse_pvt_tables(pvt_file)
        
        # Parse saturation tables
        sat_file = self.data_dir / "SPE9_SATURATION_TABLES.INC"
        if sat_file.exists():
            parsed['saturation_tables'] = self._parse_saturation_tables(sat_file)
        
        return parsed
    
    def _parse_numbers(self, text):
        """Extract numbers from text."""
        # Remove comments
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            if '--' in line:
                line = line.split('--')[0]
            line = line.strip()
            if line:
                cleaned.append(line)
        
        text = ' '.join(cleaned)
        
        # Extract numbers
        numbers = []
        for match in re.finditer(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text):
            try:
                numbers.append(float(match.group()))
            except:
                pass
        
        return np.array(numbers)
    
    def _extract_arrays(self, text):
        """Extract array definitions."""
        arrays = {}
        
        # Find array definitions like DX, DY, DZ
        array_pattern = r'(\w+)\s*\n\s*(\d+\*\s*[\d\.]+(?:\s+\d+\*\s*[\d\.]+)*)'
        
        for match in re.finditer(array_pattern, text, re.IGNORECASE):
            array_name = match.group(1).strip()
            array_values = match.group(2)
            arrays[array_name] = self._parse_numbers(array_values)
        
        return arrays
    
    def _parse_pvt_tables(self, filepath):
        """Parse PVT tables."""
        with open(filepath, 'r') as f:
            content = f.read()
        
        tables = {}
        
        # Look for PVTO (Oil PVT) and PVDG (Gas PVT) tables
        pvto_match = re.search(r'PVTO\s*(.*?)\s*/\s*/\s*', content, re.DOTALL)
        if pvto_match:
            tables['oil'] = self._parse_pvto_table(pvto_match.group(1))
        
        pvdg_match = re.search(r'PVDG\s*(.*?)\s*/\s*', content, re.DOTALL)
        if pvdg_match:
            tables['gas'] = self._parse_pvdg_table(pvdg_match.group(1))
        
        pvtw_match = re.search(r'PVTW\s*(.*?)\s*/\s*', content, re.DOTALL)
        if pvtw_match:
            tables['water'] = self._parse_numbers(pvtw_match.group(1))
        
        return tables
    
    def _parse_saturation_tables(self, filepath):
        """Parse saturation tables."""
        with open(filepath, 'r') as f:
            content = f.read()
        
        tables = {}
        
        # Look for SWOF and SGOF tables
        swof_match = re.search(r'SWOF\s*(.*?)\s*/\s*', content, re.DOTALL)
        if swof_match:
            tables['swof'] = self._parse_numbers(swof_match.group(1))
        
        sgof_match = re.search(r'SGOF\s*(.*?)\s*/\s*', content, re.DOTALL)
        if sgof_match:
            tables['sgof'] = self._parse_numbers(sgof_match.group(1))
        
        return tables
