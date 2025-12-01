"""
Validate simulation input data
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate SPE9 simulation input data"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def validate_all(self, data_dir: str = "data") -> Tuple[bool, List[str]]:
        """Validate all input files"""
        self.errors.clear()
        self.warnings.clear()
        
        validation_methods = [
            self.validate_main_data_file,
            self.validate_grid_files,
            self.validate_pvt_data,
            self.validate_saturation_tables,
            self.validate_well_data,
        ]
        
        for method in validation_methods:
            try:
                method(data_dir)
            except Exception as e:
                self.errors.append(f"Validation error in {method.__name__}: {e}")
                
        if self.errors:
            logger.error(f"Validation failed with {len(self.errors)} errors")
            return False, self.errors + self.warnings
        else:
            logger.info("All data validation passed successfully")
            return True, self.warnings
            
    def validate_main_data_file(self, data_dir: str) -> None:
        """Validate main SPE9.DATA file"""
        data_file = Path(data_dir) / "SPE9.DATA"
        
        if not data_file.exists():
            self.errors.append(f"Main data file not found: {data_file}")
            return
            
        with open(data_file, 'r') as f:
            content = f.read()
            
        # Check for required sections
        required_sections = ['RUNSPEC', 'GRID', 'PROPS', 'SOLUTION', 'SCHEDULE']
        for section in required_sections:
            if section not in content:
                self.errors.append(f"Missing required section: {section}")
                
        # Check grid dimensions
        dims_match = re.search(r'DIMENS\s*\n\s*(\d+)\s+(\d+)\s+(\d+)', content)
        if dims_match:
            nx, ny, nz = map(int, dims_match.groups())
            if not (nx == 24 and ny == 25 and nz == 15):
                self.errors.append(f"Grid dimensions incorrect: {nx}x{ny}x{nz}, expected 24x25x15")
        else:
            self.errors.append("Grid dimensions not found")
            
        # Check for INCLUDE statements
        includes = re.findall(r'INCLUDE\s*\n\s*\'([^\']+)\'', content)
        for inc in includes:
            if not (Path(data_dir) / inc).exists():
                self.errors.append(f"INCLUDE file not found: {inc}")
                
        logger.info(f"Main data file validated: {len(includes)} includes found")
        
    def validate_grid_files(self, data_dir: str) -> None:
        """Validate grid-related files"""
        grid_files = [
            "SPE9_GRID.INC",
            "SPE9_PORO.INC",
            "TOPSVALUES.DATA",
            "PERMVALUES.DATA"
        ]
        
        for file_name in grid_files:
            file_path = Path(data_dir) / file_name
            if not file_path.exists():
                self.errors.append(f"Grid file not found: {file_name}")
                continue
                
            # File-specific validations
            if file_name == "SPE9_PORO.INC":
                self._validate_porosity_file(file_path)
            elif file_name == "PERMVALUES.DATA":
                self._validate_permeability_file(file_path)
                
    def _validate_porosity_file(self, file_path: Path) -> None:
        """Validate porosity distribution file"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Count porosity values
        poro_values = []
        for line in lines:
            if line.strip() and not line.strip().startswith('--'):
                # Extract numbers like "600*0.087"
                matches = re.findall(r'(\d+)\*([\d.]+)', line)
                for count, value in matches:
                    poro_values.extend([float(value)] * int(count))
                    
        if len(poro_values) != 9000:
            self.errors.append(f"Porosity file should have 9000 values, found {len(poro_values)}")
            
        # Check porosity range
        for value in poro_values:
            if not (0.01 <= value <= 0.35):
                self.warnings.append(f"Porosity value {value} outside typical range")
                
    def _validate_permeability_file(self, file_path: Path) -> None:
        """Validate permeability data file"""
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Count PERMX values (simplified check)
        perm_values = re.findall(r'[\d.]+', content)
        if len(perm_values) < 1000:  # Should have many values
            self.warnings.append(f"Permeability file may be incomplete, found {len(perm_values)} values")
            
    def validate_pvt_data(self, data_dir: str) -> None:
        """Validate PVT data"""
        pvt_file = Path(data_dir) / "SPE9_PVT.INC"
        if not pvt_file.exists():
            self.errors.append("PVT data file not found")
            return
            
        with open(pvt_file, 'r') as f:
            content = f.read()
            
        # Check for required PVT tables
        if 'PVTO' not in content:
            self.errors.append("PVTO table missing in PVT data")
        if 'PVDG' not in content:
            self.errors.append("PVDG table missing in PVT data")
        if 'PVTW' not in content:
            self.errors.append("PVTW table missing in PVT data")
            
    def validate_saturation_tables(self, data_dir: str) -> None:
        """Validate saturation tables"""
        sat_file = Path(data_dir) / "SPE9_SATURATION_TABLES.INC"
        if not sat_file.exists():
            self.errors.append("Saturation tables file not found")
            return
            
        with open(sat_file, 'r') as f:
            content = f.read()
            
        # Check for SGOF and SWOF tables
        if 'SGOF' not in content:
            self.errors.append("SGOF table missing")
        if 'SWOF' not in content:
            self.errors.append("SWOF table missing")
            
        # Validate table endpoints
        self._validate_table_endpoints(content)
        
    def _validate_table_endpoints(self, content: str) -> None:
        """Validate saturation table endpoints"""
        # Check SGOF endpoints
        sgof_match = re.search(r'SGOF\s*\n([\s\S]*?)\s*/', content)
        if sgof_match:
            sgof_data = sgof_match.group(1)
            lines = [l.strip() for l in sgof_data.split('\n') if l.strip()]
            if lines:
                first_line = lines[0].split()
                last_line = lines[-1].split()
                
                if float(first_line[0]) != 0.0:
                    self.warnings.append("SGOF table should start at Sg=0")
                if float(last_line[0]) < 0.8:
                    self.warnings.append("SGOF table maximum Sg seems low")
                    
    def validate_well_data(self, data_dir: str) -> None:
        """Validate well data in main file"""
        data_file = Path(data_dir) / "SPE9.DATA"
        if not data_file.exists():
            return
            
        with open(data_file, 'r') as f:
            content = f.read()
            
        # Count wells in WELSPECS
        wells_match = re.findall(r"'([^']+)'\s+'[^']*'\s+\d+\s+\d+\s+\d+', content)
        if wells_match:
            if len(wells_match) != 26:
                self.errors.append(f"Expected 26 wells in WELSPECS, found {len(wells_match)}")
                
        # Check for injector and producers
        if "'INJE1'" not in content:
            self.errors.append("Injector INJE1 not found")
            
    def get_validation_summary(self) -> Dict[str, List[str]]:
        """Get summary of validation results"""
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings)
        }
