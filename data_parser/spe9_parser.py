"""
SPE9.DATA Parser - Professional grade reservoir simulation data parser.
Handles Eclipse format with robust error handling and validation.
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WellSpecification:
    """Well specification data structure."""
    name: str
    i_location: int
    j_location: int
    k_location: int
    phase: str
    well_type: str
    drainage_radius: float = 0.0
    productivity_index: float = 1.0
    
    def __str__(self):
        return f"Well(name={self.name}, location=({self.i_location},{self.j_location},{self.k_location}), type={self.well_type})"

class SPE9DataParser:
    """
    Parser for SPE9.DATA files in Eclipse format.
    
    Key features:
    - Robust section parsing with regex patterns
    - Data validation and integrity checks
    - Support for multiple data formats
    - Comprehensive error handling
    """
    
    # Regular expressions for parsing
    SECTION_PATTERN = re.compile(r'(\b[A-Z]+\b)\s*(.*?)(?=\s*\b[A-Z]+\b\s*$|\Z)', re.DOTALL | re.MULTILINE)
    COMMENT_PATTERN = re.compile(r'--.*$', re.MULTILINE)
    NUMBER_PATTERN = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?')
    
    def __init__(self, data_file_path: str):
        self.data_file_path = Path(data_file_path)
        self.raw_content: Optional[str] = None
        self.parsed_sections: Dict[str, str] = {}
        self.grid_dimensions: Optional[Tuple[int, int, int]] = None
        self.wells: List[WellSpecification] = []
        self.permeability_data: Optional[np.ndarray] = None
        self.tops_data: Optional[np.ndarray] = None
        self.sgof_table: Optional[np.ndarray] = None
        
        if not self.data_file_path.exists():
            raise FileNotFoundError(f"SPE9 data file not found: {data_file_path}")
    
    def parse(self) -> 'SPE9DataParser':
        """Main parsing method with full validation."""
        logger.info(f"Parsing SPE9 data file: {self.data_file_path}")
        
        # Read and clean file
        self._read_file()
        self._clean_content()
        
        # Parse sections
        self._parse_all_sections()
        
        # Extract and validate data
        self._extract_grid_dimensions()
        self._extract_well_specifications()
        self._extract_permeability_data()
        self._extract_tops_data()
        self._extract_sgof_table()
        
        # Validate data integrity
        self._validate_data_integrity()
        
        logger.info(f"Successfully parsed {len(self.wells)} wells")
        logger.info(f"Grid dimensions: {self.grid_dimensions}")
        logger.info(f"Permeability data shape: {self.permeability_data.shape if self.permeability_data is not None else 'None'}")
        
        return self
    
    def _read_file(self):
        """Read file with encoding detection."""
        try:
            with open(self.data_file_path, 'r', encoding='utf-8') as f:
                self.raw_content = f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'ascii']:
                try:
                    with open(self.data_file_path, 'r', encoding=encoding) as f:
                        self.raw_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Cannot decode file: {self.data_file_path}")
    
    def _clean_content(self):
        """Remove comments and clean up content."""
        if self.raw_content is None:
            raise ValueError("No content to clean")
        
        # Remove comments
        cleaned = self.COMMENT_PATTERN.sub('', self.raw_content)
        
        # Normalize line endings and remove empty lines
        cleaned = '\n'.join(line.strip() for line in cleaned.splitlines() if line.strip())
        
        self.raw_content = cleaned
    
    def _parse_all_sections(self):
        """Parse all sections from the data file."""
        sections = self.SECTION_PATTERN.findall(self.raw_content)
        
        for keyword, content in sections:
            keyword = keyword.strip().upper()
            content = content.strip()
            
            # Remove trailing slash if present
            if content.endswith('/'):
                content = content[:-1].strip()
            
            self.parsed_sections[keyword] = content
        
        if not self.parsed_sections:
            logger.warning("No sections found in data file")
    
    def _extract_grid_dimensions(self):
        """Extract grid dimensions from DIMENS section."""
        dimens_content = self.parsed_sections.get('DIMENS')
        if not dimens_content:
            # SPE9 default dimensions: 24x25x15
            self.grid_dimensions = (24, 25, 15)
            logger.warning("DIMENS section not found, using default SPE9 dimensions (24, 25, 15)")
            return
        
        # Extract numbers from DIMENS
        numbers = self.NUMBER_PATTERN.findall(dimens_content)
        
        if len(numbers) >= 3:
            try:
                self.grid_dimensions = (
                    int(float(numbers[0])),
                    int(float(numbers[1])),
                    int(float(numbers[2]))
                )
                logger.info(f"Extracted grid dimensions: {self.grid_dimensions}")
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing grid dimensions: {e}")
                self.grid_dimensions = (24, 25, 15)
        else:
            self.grid_dimensions = (24, 25, 15)
            logger.warning("Insufficient data in DIMENS section, using defaults")
    
    def _extract_well_specifications(self):
        """Extract well specifications from WELSPECS section."""
        welspecs_content = self.parsed_sections.get('WELSPECS')
        if not welspecs_content:
            logger.warning("WELSPECS section not found")
            return
        
        lines = welspecs_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Split and parse well data
            parts = line.split()
            
            if len(parts) >= 5:
                try:
                    well = WellSpecification(
                        name=parts[0].strip("'").strip('"'),
                        i_location=int(parts[1]),
                        j_location=int(parts[2]),
                        k_location=int(parts[3]),
                        phase=parts[4].strip("'").strip('"').upper(),
                        well_type=self._determine_well_type(parts[0])
                    )
                    self.wells.append(well)
                except (ValueError, IndexError) as e:
                    logger.error(f"Error parsing well data: {line} - {e}")
    
    def _determine_well_type(self, well_name: str) -> str:
        """Determine well type based on naming convention."""
        name = well_name.upper()
        
        if 'INJ' in name or 'I' == name[0]:
            return 'INJECTOR'
        elif 'PROD' in name or 'P' == name[0]:
            return 'PRODUCER'
        else:
            # Default based on SPE9 convention
            if 'W' in name:
                return 'INJECTOR'
            else:
                return 'PRODUCER'
    
    def _extract_permeability_data(self):
        """Extract permeability data from PERMX section."""
        permx_content = self.parsed_sections.get('PERMX')
        
        if not permx_content:
            # Try alternative names
            for key in ['PERMY', 'PERMZ', 'PERM']:
                if key in self.parsed_sections:
                    permx_content = self.parsed_sections[key]
                    break
        
        if not permx_content:
            logger.warning("No permeability data found")
            return
        
        # Parse all numbers
        numbers = []
        for match in self.NUMBER_PATTERN.finditer(permx_content):
            try:
                numbers.append(float(match.group()))
            except ValueError:
                continue
        
        if numbers:
            self.permeability_data = np.array(numbers)
            logger.info(f"Extracted {len(self.permeability_data)} permeability values")
    
    def _extract_tops_data(self):
        """Extract TOPS data."""
        tops_content = self.parsed_sections.get('TOPS')
        
        if not tops_content:
            logger.warning("TOPS section not found")
            return
        
        # Parse TOPS numbers
        numbers = []
        for match in self.NUMBER_PATTERN.finditer(tops_content):
            try:
                numbers.append(float(match.group()))
            except ValueError:
                continue
        
        if numbers:
            self.tops_data = np.array(numbers)
            logger.info(f"Extracted {len(self.tops_data)} TOPS values")
    
    def _extract_sgof_table(self):
        """Extract SGOF table data."""
        sgof_content = self.parsed_sections.get('SGOF')
        
        if not sgof_content:
            logger.warning("SGOF section not found")
            return
        
        # Parse SGOF table
        table_data = []
        lines = sgof_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            numbers = []
            for match in self.NUMBER_PATTERN.finditer(line):
                try:
                    numbers.append(float(match.group()))
                except ValueError:
                    continue
            
            if len(numbers) >= 4:  # Sg, krg, krog, Pc
                table_data.append(numbers[:4])
        
        if table_data:
            self.sgof_table = np.array(table_data)
            logger.info(f"Extracted SGOF table with {len(self.sgof_table)} rows")
    
    def _validate_data_integrity(self):
        """Validate parsed data for consistency."""
        issues = []
        
        # Check grid dimensions
        if self.grid_dimensions:
            total_cells = self.grid_dimensions[0] * self.grid_dimensions[1] * self.grid_dimensions[2]
            
            # Check permeability data size
            if self.permeability_data is not None:
                if len(self.permeability_data) != total_cells:
                    issues.append(f"Permeability data size mismatch: {len(self.permeability_data)} != {total_cells}")
            
            # Check TOPS data size
            if self.tops_data is not None:
                tops_expected = self.grid_dimensions[0] * self.grid_dimensions[1]
                if len(self.tops_data) != tops_expected:
                    issues.append(f"TOPS data size mismatch: {len(self.tops_data)} != {tops_expected}")
        
        # Check wells
        if not self.wells:
            issues.append("No wells parsed")
        
        # Check SGOF table
        if self.sgof_table is not None:
            if len(self.sgof_table) == 0:
                issues.append("SGOF table is empty")
        
        # Log issues
        for issue in issues:
            logger.warning(f"Data validation issue: {issue}")
        
        return len(issues) == 0
    
    def export_to_project_format(self, output_dir: str = 'data') -> Dict[str, Path]:
        """
        Export parsed data to project format.
        
        Returns:
            Dictionary mapping file types to paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # 1. Export well data
        wells_file = output_path / 'well_locations.txt'
        with open(wells_file, 'w') as f:
            f.write("# Well specifications exported from SPE9.DATA\n")
            f.write("# Format: WellID, Name, I, J, K, Type, Phase\n")
            for i, well in enumerate(self.wells, 1):
                f.write(f"{i}, {well.name}, {well.i_location}, {well.j_location}, "
                       f"{well.k_location}, {well.well_type}, {well.phase}\n")
        exported_files['wells'] = wells_file
        
        # 2. Export permeability data
        if self.permeability_data is not None:
            perm_file = output_path / 'permeability.txt'
            np.savetxt(perm_file, self.permeability_data, header='Permeability values (mD)')
            exported_files['permeability'] = perm_file
        
        # 3. Export TOPS data
        if self.tops_data is not None:
            tops_file = output_path / 'grid_tops.txt'
            np.savetxt(tops_file, self.tops_data, header='Grid tops elevation (ft)')
            exported_files['tops'] = tops_file
        
        # 4. Export SGOF table
        if self.sgof_table is not None:
            sgof_file = output_path / 'sgof_table.txt'
            np.savetxt(sgof_file, self.sgof_table, 
                      header='Sg, krg, krog, Pc (Gas-oil relative permeability table)')
            exported_files['sgof'] = sgof_file
        
        # 5. Export grid information
        if self.grid_dimensions:
            grid_file = output_path / 'grid_info.txt'
            with open(grid_file, 'w') as f:
                f.write(f"Grid dimensions: {self.grid_dimensions[0]} x {self.grid_dimensions[1]} x {self.grid_dimensions[2]}\n")
                f.write(f"Total cells: {self.grid_dimensions[0] * self.grid_dimensions[1] * self.grid_dimensions[2]}\n")
                f.write(f"Number of wells: {len(self.wells)}\n")
            exported_files['grid_info'] = grid_file
        
        logger.info(f"Exported {len(exported_files)} data files to {output_path}")
        
        return exported_files

# Factory function for convenience
def create_spe9_parser(data_file_path: str) -> SPE9DataParser:
    """Factory function to create and parse SPE9 data."""
    parser = SPE9DataParser(data_file_path)
    return parser.parse()
