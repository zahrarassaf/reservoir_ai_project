"""
Validate simulation input data
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate SPE9 simulation input data"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate_all(self, data_dir: str = "data") -> Tuple[bool, List[str]]:
        """Validate all input files"""
        self.errors.clear()
        self.warnings.clear()
        
        logger.info("🔍 Starting comprehensive data validation...")
        
        validation_methods = [
            (self.validate_main_data_file, "Main data file"),
            (self.validate_grid_files, "Grid files"),
            (self.validate_pvt_data, "PVT data"),
            (self.validate_saturation_tables, "Saturation tables"),
            (self.validate_well_data, "Well data"),
        ]
        
        for method, description in validation_methods:
            try:
                logger.info(f"  Validating {description}...")
                method(data_dir)
            except Exception as e:
                self.errors.append(f"Validation error in {description}: {str(e)}")
                logger.error(f"  ❌ {description} validation failed: {e}")
            else:
                logger.info(f"  ✅ {description} validation passed")
                
        if self.errors:
            logger.error(f"❌ Data validation failed with {len(self.errors)} errors")
            return False, self.errors + self.warnings
        else:
            logger.info(f"✅ All data validation passed successfully")
            if self.warnings:
                logger.warning(f"⚠️  Found {len(self.warnings)} warnings")
            return True, self.warnings
            
    def validate_main_data_file(self, data_dir: str) -> None:
        """Validate main SPE9.DATA file"""
        data_file = Path(data_dir) / "SPE9.DATA"
        
        self._check_file_exists(data_file, "Main SPE9.DATA file")
        if self.errors and "SPE9.DATA" in self.errors[-1]:
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
            self.errors.append("Grid dimensions (DIMENS keyword) not found")
            
        # Check for INCLUDE statements
        includes = re.findall(r'INCLUDE\s*\n\s*[\'"]?([^\'"\n]+)[\'"]?', content)
        logger.info(f"  Found {len(includes)} INCLUDE statements")
        
        for inc in includes:
            inc_path = Path(data_dir) / inc
            if not inc_path.exists():
                self.errors.append(f"INCLUDE file not found: {inc}")
            else:
                logger.info(f"    ✓ {inc}")
                
    def validate_grid_files(self, data_dir: str) -> None:
        """Validate grid-related files"""
        grid_files = [
            "SPE9_GRID.INC",
            "SPE9_PORO.INC",
            "TOPSVALUES.DATA",
            "PERMVALUES.DATA"
        ]
        
        logger.info(f"  Checking {len(grid_files)} grid files...")
        
        for file_name in grid_files:
            file_path = Path(data_dir) / file_name
            self._check_file_exists(file_path, f"Grid file: {file_name}")
            
            if file_path.exists():
                # File-specific validations
                if file_name == "SPE9_PORO.INC":
                    self._validate_porosity_file(file_path)
                elif file_name == "PERMVALUES.DATA":
                    self._validate_permeability_file(file_path)
                elif file_name == "TOPSVALUES.DATA":
                    self._validate_tops_file(file_path)
                    
    def _validate_porosity_file(self, file_path: Path) -> None:
        """Validate porosity distribution file"""
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Count porosity values
        poro_values = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('--'):
                # Extract numbers like "600*0.087"
                matches = re.findall(r'(\d+)\*([\d.]+)', line)
                for count, value in matches:
                    try:
                        poro_values.extend([float(value)] * int(count))
                    except ValueError:
                        self.warnings.append(f"Invalid porosity value in {file_path.name}: {value}")
                        
        if len(poro_values) != 9000:
            self.errors.append(f"Porosity file should have 9000 values, found {len(poro_values)}")
        else:
            logger.info(f"    ✓ Porosity file has correct number of values: {len(poro_values)}")
            
        # Check porosity range
        invalid_values = [v for v in poro_values if not (0.01 <= v <= 0.35)]
        if invalid_values:
            self.warnings.append(f"Porosity values outside typical range in {file_path.name}")
            
    def _validate_permeability_file(self, file_path: Path) -> None:
        """Validate permeability data file"""
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Count PERMX values (simplified check)
        # Look for numeric patterns
        perm_values = re.findall(r'\b\d+\.?\d*\b', content)
        
        # Remove layer and row markers
        filtered_values = [v for v in perm_values if not any(marker in content.split('\n')[i] 
                           for i, line in enumerate(content.split('\n')) 
                           for marker in ['LAYER', 'ROW'] if marker in line)]
        
        if len(filtered_values) < 8000:  # Should have many values (9000 total)
            self.warnings.append(f"Permeability file may be incomplete, found {len(filtered_values)} values")
        else:
            logger.info(f"    ✓ Permeability file has sufficient values: {len(filtered_values)}")
            
    def _validate_tops_file(self, file_path: Path) -> None:
        """Validate TOPS values file"""
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Check for TOPS keyword
        if 'TOPS' not in content:
            self.warnings.append(f"TOPS keyword not found in {file_path.name}")
            
        # Count values after TOPS keyword
        tops_section = content.split('TOPS')[-1] if 'TOPS' in content else content
        values = re.findall(r'\b\d+\.?\d*\b', tops_section)
        
        if len(values) < 600:  # 24x25 = 600 values for top layer
            self.warnings.append(f"TOPS file may be incomplete, found {len(values)} values")
        else:
            logger.info(f"    ✓ TOPS file has {len(values)} depth values")
            
    def validate_pvt_data(self, data_dir: str) -> None:
        """Validate PVT data"""
        pvt_file = Path(data_dir) / "SPE9_PVT.INC"
        self._check_file_exists(pvt_file, "PVT data file")
        
        if not pvt_file.exists():
            return
            
        with open(pvt_file, 'r') as f:
            content = f.read()
            
        # Check for required PVT tables
        required_tables = ['PVTW', 'PVTO', 'PVDG']
        missing_tables = [table for table in required_tables if table not in content]
        
        if missing_tables:
            self.errors.append(f"Missing PVT tables: {', '.join(missing_tables)}")
        else:
            logger.info("    ✓ All required PVT tables present")
            
        # Check DENSITY keyword
        if 'DENSITY' not in content:
            self.warnings.append("DENSITY keyword missing in PVT data")
            
        # Check ROCK keyword
        if 'ROCK' not in content:
            self.warnings.append("ROCK keyword missing in PVT data")
            
    def validate_saturation_tables(self, data_dir: str) -> None:
        """Validate saturation tables"""
        sat_file = Path(data_dir) / "SPE9_SATURATION_TABLES.INC"
        self._check_file_exists(sat_file, "Saturation tables file")
        
        if not sat_file.exists():
            return
            
        with open(sat_file, 'r') as f:
            content = f.read()
            
        # Check for SGOF and SWOF tables
        if 'SGOF' not in content:
            self.errors.append("SGOF table missing")
        else:
            logger.info("    ✓ SGOF table present")
            
        if 'SWOF' not in content:
            self.errors.append("SWOF table missing")
        else:
            logger.info("    ✓ SWOF table present")
            
        # Validate table endpoints
        self._validate_table_endpoints(content)
        
    def _validate_table_endpoints(self, content: str) -> None:
        """Validate saturation table endpoints"""
        # Check SGOF endpoints
        sgof_section = re.search(r'SGOF\s*\n([\s\S]*?)\s*/', content)
        if sgof_section:
            sgof_data = sgof_section.group(1)
            lines = [l.strip() for l in sgof_data.split('\n') if l.strip()]
            
            if lines:
                try:
                    first_values = [float(x) for x in lines[0].split()]
                    last_values = [float(x) for x in lines[-1].split()]
                    
                    if first_values[0] != 0.0:
                        self.warnings.append("SGOF table should start at Sg=0")
                    if last_values[0] < 0.8:
                        self.warnings.append("SGOF table maximum Sg seems low")
                except (ValueError, IndexError):
                    self.warnings.append("Could not parse SGOF table values")
                    
    def validate_well_data(self, data_dir: str) -> None:
        """Validate well data in main file"""
        data_file = Path(data_dir) / "SPE9.DATA"
        if not data_file.exists():
            return
            
        with open(data_file, 'r') as f:
            content = f.read()
            
        # Count wells in WELSPECS
        wells_section = re.search(r'WELSPECS\s*\n([\s\S]*?)\s*/', content)
        if wells_section:
            wells_content = wells_section.group(1)
            # Count lines with well definitions
            well_lines = [line for line in wells_content.split('\n') 
                         if line.strip() and not line.strip().startswith('--')]
            
            if len(well_lines) != 26:
                self.errors.append(f"Expected 26 wells in WELSPECS, found {len(well_lines)}")
            else:
                logger.info(f"    ✓ Correct number of wells in WELSPECS: {len(well_lines)}")
                
        # Check for injector and producers
        if "'INJE1'" not in content:
            self.errors.append("Injector INJE1 not found")
        else:
            logger.info("    ✓ Injector INJE1 found")
            
        # Check COMPDAT sections
        compdat_sections = re.findall(r"COMPDAT\s*\n([\s\S]*?)\s*/", content)
        if not compdat_sections:
            self.warnings.append("COMPDAT sections not found")
            
    def _check_file_exists(self, file_path: Path, description: str) -> None:
        """Check if file exists and log appropriately"""
        if not file_path.exists():
            self.errors.append(f"{description} not found: {file_path}")
            logger.error(f"    ❌ {description}: {file_path.name} - NOT FOUND")
        else:
            file_size = file_path.stat().st_size
            logger.info(f"    ✓ {description}: {file_path.name} ({file_size:,} bytes)")
            
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results"""
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "is_valid": len(self.errors) == 0
        }
