"""
SPE9.DATA Parser - Optimized for your project structure.
Handles multiple data files: SPE9.DATA, PERMVALUES.DATA, TOPSVALUES.DATA, etc.
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import yaml
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WellSpecification:
    """Well specification for SPE9 dataset."""
    name: str
    i_location: int
    j_location: int
    k_location: int
    well_type: str  # 'INJECTOR' or 'PRODUCER'
    phase: str = 'OIL'
    control_mode: str = 'RATE'
    
    def to_dict(self):
        return {
            'name': self.name,
            'i': self.i_location,
            'j': self.j_location,
            'k': self.k_location,
            'type': self.well_type,
            'phase': self.phase,
            'control': self.control_mode
        }

class SPE9ProjectParser:
    """
    Parser for SPE9 project with multiple data files.
    
    Handles:
    - SPE9.DATA (main simulation deck)
    - PERMVALUES.DATA (permeability values)
    - TOPSVALUES.DATA (grid tops)
    - INC files (grid, porosity, PVT, saturation tables)
    - JSON/YAML config files
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.spe9_data_path = self.data_dir / "SPE9.DATA"
        
        # Check which files actually exist
        self.files = {
            'grid_inc': self.data_dir / "SPE9_GRID.INC",
            'poro_inc': self.data_dir / "SPE9_PORO.INC",
            'pvt_inc': self.data_dir / "SPE9_PVT.INC",
            'sat_inc': self.data_dir / "SPE9_SATURATION_TABLES.INC",
            'perm_values': self.data_dir / "PERMVALUES.DATA",
            'tops_values': self.data_dir / "TOPSVALUES.DATA"
        }
        
        # Configuration paths
        self.config_dir = Path("config")
        self.config_files = {
            'grid': self.config_dir / "grid_parameters.json",
            'sim': self.config_dir / "simulation_config.yaml",
            'wells': self.config_dir / "well_controls.json"
        }
        
        # Parsed data
        self.wells: List[WellSpecification] = []
        self.grid_dimensions: Tuple[int, int, int] = (24, 25, 15)  # Default SPE9
        self.permeability: Optional[np.ndarray] = None
        self.porosity: Optional[np.ndarray] = None
        self.tops: Optional[np.ndarray] = None
        self.pvt_tables: Dict[str, Any] = {}
        self.saturation_tables: Dict[str, Any] = {}
        
        # Load configurations
        self.configs = self._load_configs()
    
    def _load_configs(self) -> Dict[str, Any]:
        """Load all configuration files."""
        configs = {}
        
        for name, path in self.config_files.items():
            if path.exists():
                try:
                    if path.suffix == '.json':
                        with open(path, 'r') as f:
                            configs[name] = json.load(f)
                    elif path.suffix in ['.yaml', '.yml']:
                        with open(path, 'r') as f:
                            configs[name] = yaml.safe_load(f)
                    logger.info(f"Loaded config: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load config {path}: {e}")
            else:
                logger.warning(f"Config file not found: {path}")
        
        return configs
    
    def parse_all(self) -> 'SPE9ProjectParser':
        """Parse all available data files."""
        logger.info("Starting SPE9 project parsing...")
        
        # Parse main SPE9.DATA file
        self._parse_spe9_data()
        
        # Parse other files if they exist
        for file_type, path in self.files.items():
            if path.exists():
                try:
                    if 'grid' in file_type:
                        self._parse_grid_inc(path)
                    elif 'poro' in file_type:
                        self._parse_porosity_inc(path)
                    elif 'pvt' in file_type:
                        self._parse_pvt_inc(path)
                    elif 'sat' in file_type:
                        self._parse_saturation_tables_inc(path)
                    elif 'perm' in file_type:
                        self._parse_permeability_values(path)
                    elif 'tops' in file_type:
                        self._parse_tops_values(path)
                except Exception as e:
                    logger.warning(f"Failed to parse {path}: {e}")
        
        # Set defaults if data missing
        self._set_defaults_if_needed()
        
        # Validate
        self._validate_data_consistency()
        
        logger.info(f"Parsing complete: {len(self.wells)} wells")
        return self
    
    def _parse_spe9_data(self):
        """Parse main SPE9.DATA file."""
        if not self.spe9_data_path.exists():
            logger.error(f"SPE9.DATA not found at {self.spe9_data_path}")
            return
        
        logger.info(f"Parsing {self.spe9_data_path}")
        
        try:
            with open(self.spe9_data_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract DIMENS
            dimens_match = re.search(r'DIMENS\s+(\d+)\s+(\d+)\s+(\d+)', content, re.IGNORECASE)
            if dimens_match:
                self.grid_dimensions = (
                    int(dimens_match.group(1)),
                    int(dimens_match.group(2)),
                    int(dimens_match.group(3))
                )
                logger.info(f"Grid dimensions: {self.grid_dimensions}")
            
            # Extract WELSPECS
            welspecs_pattern = r'WELSPECS\s*(.*?)(?=\s*/\s*|\n\s*[A-Z]+\s*$|\Z)'
            welspecs_match = re.search(welspecs_pattern, content, re.DOTALL | re.IGNORECASE)
            if welspecs_match:
                self._parse_welspecs(welspecs_match.group(1))
            
            # Extract other sections if needed
            self._extract_other_sections(content)
            
        except Exception as e:
            logger.error(f"Error parsing SPE9.DATA: {e}")
    
    def _parse_welspecs(self, content: str):
        """Parse WELSPECS section."""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            # Remove comments
            if '--' in line:
                line = line.split('--')[0].strip()
            
            # Split by whitespace, handling quoted strings
            parts = []
            current = ''
            in_quotes = False
            quote_char = ''
            
            for char in line:
                if char in ['\'', '"']:
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                        current += char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = ''
                        current += char
                    else:
                        current += char
                elif char == ' ' and not in_quotes:
                    if current:
                        parts.append(current)
                        current = ''
                else:
                    current += char
            
            if current:
                parts.append(current)
            
            if len(parts) >= 5:
                # Remove quotes from well name
                well_name = parts[0].strip('\'"')
                
                # Determine well type
                well_type = 'PRODUCER'
                if well_name.upper().startswith(('I', 'INJ', 'W')):
                    well_type = 'INJECTOR'
                
                try:
                    well = WellSpecification(
                        name=well_name,
                        i_location=int(float(parts[1])),
                        j_location=int(float(parts[2])),
                        k_location=int(float(parts[3])),
                        well_type=well_type,
                        phase=parts[4].strip('\'"').upper()
                    )
                    
                    self.wells.append(well)
                    logger.debug(f"Parsed well: {well.name} ({well.well_type}) at ({well.i_location},{well.j_location},{well.k_location})")
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping invalid well data: {line} - {e}")
    
    def _extract_other_sections(self, content: str):
        """Extract other important sections."""
        # Add extraction for PERMX, PORO, etc. if in main file
        pass
    
    def _parse_grid_inc(self, path: Path):
        """Parse grid include file."""
        logger.info(f"Parsing grid from {path}")
        
        with open(path, 'r') as f:
            content = f.read()
        
        # Look for SPECGRID or COORD
        specgrid_match = re.search(r'SPECGRID\s+(\d+)\s+(\d+)\s+(\d+)', content, re.IGNORECASE)
        if specgrid_match:
            self.grid_dimensions = (
                int(specgrid_match.group(1)),
                int(specgrid_match.group(2)),
                int(specgrid_match.group(3))
            )
    
    def _parse_porosity_inc(self, path: Path):
        """Parse porosity data."""
        logger.info(f"Parsing porosity from {path}")
        
        values = self._parse_eclipse_data_file(path)
        if values:
            self.porosity = np.array(values)
            logger.info(f"Loaded porosity: {self.porosity.shape}")
    
    def _parse_pvt_inc(self, path: Path):
        """Parse PVT tables."""
        logger.info(f"Parsing PVT from {path}")
        
        with open(path, 'r') as f:
            content = f.read()
        
        # Simplified parsing - extract all tables
        tables = ['PVTO', 'PVTG', 'PVTW', 'PVDG']
        for table in tables:
            pattern = rf'{table}\s*(.*?)(?=\s*/\s*|\n\s*[A-Z]+\s*$|\Z)'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                self.pvt_tables[table] = self._parse_table_data(match.group(1))
    
    def _parse_saturation_tables_inc(self, path: Path):
        """Parse saturation tables."""
        logger.info(f"Parsing saturation tables from {path}")
        
        with open(path, 'r') as f:
            content = f.read()
        
        tables = ['SGOF', 'SWOF', 'SLGOF', 'SOF3']
        for table in tables:
            pattern = rf'{table}\s*(.*?)(?=\s*/\s*|\n\s*[A-Z]+\s*$|\Z)'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                data = self._parse_table_data(match.group(1))
                if len(data) > 0 and all(len(row) >= 4 for row in data):
                    self.saturation_tables[table] = np.array(data)
    
    def _parse_permeability_values(self, path: Path):
        """Parse permeability values."""
        logger.info(f"Parsing permeability from {path}")
        
        values = self._parse_eclipse_data_file(path)
        if values:
            self.permeability = np.array(values)
            logger.info(f"Loaded permeability: {self.permeability.shape}")
    
    def _parse_tops_values(self, path: Path):
        """Parse TOPS values."""
        logger.info(f"Parsing TOPS from {path}")
        
        values = self._parse_eclipse_data_file(path)
        if values:
            self.tops = np.array(values)
            logger.info(f"Loaded TOPS: {self.tops.shape}")
    
    def _parse_eclipse_data_file(self, path: Path) -> List[float]:
        """Generic parser for Eclipse data files."""
        values = []
        
        try:
            with open(path, 'r') as f:
                for line in f:
                    # Remove comments
                    if '--' in line:
                        line = line.split('--')[0]
                    
                    # Parse numbers, handling Eclipse format (e.g., 4*0.25)
                    tokens = line.strip().split()
                    i = 0
                    while i < len(tokens):
                        token = tokens[i]
                        
                        # Check for multiplier pattern
                        if '*' in token and not token.startswith('*'):
                            try:
                                mult_str, val_str = token.split('*')
                                multiplier = int(mult_str)
                                value = float(val_str)
                                values.extend([value] * multiplier)
                            except ValueError:
                                # Try to parse as single number
                                try:
                                    values.append(float(token))
                                except ValueError:
                                    pass  # Skip non-numeric
                        else:
                            try:
                                values.append(float(token))
                            except ValueError:
                                pass
                        i += 1
            
        except Exception as e:
            logger.error(f"Error parsing {path}: {e}")
        
        return values
    
    def _parse_table_data(self, content: str) -> List[List[float]]:
        """Parse table data."""
        table = []
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            if '--' in line:
                line = line.split('--')[0]
            
            numbers = []
            tokens = line.split()
            for token in tokens:
                try:
                    numbers.append(float(token))
                except ValueError:
                    # Skip non-numeric
                    continue
            
            if numbers:
                table.append(numbers)
        
        return table
    
    def _set_defaults_if_needed(self):
        """Set default values for missing data."""
        total_cells = self.grid_dimensions[0] * self.grid_dimensions[1] * self.grid_dimensions[2]
        
        # Default permeability if missing
        if self.permeability is None:
            logger.warning("No permeability data, using defaults")
            self.permeability = np.random.lognormal(mean=2.0, sigma=1.0, size=total_cells)
        
        # Default porosity if missing
        if self.porosity is None:
            logger.warning("No porosity data, using defaults")
            self.porosity = np.random.uniform(0.1, 0.3, size=total_cells)
        
        # Default TOPS if missing
        if self.tops is None:
            logger.warning("No TOPS data, using defaults")
            num_tops = self.grid_dimensions[0] * self.grid_dimensions[1]
            self.tops = np.linspace(8000, 8500, num_tops)
        
        # Default wells if none parsed
        if not self.wells:
            logger.warning("No wells parsed, adding defaults")
            self.wells = [
                WellSpecification(name='INJ1', i_location=5, j_location=5, k_location=1, well_type='INJECTOR'),
                WellSpecification(name='PROD1', i_location=20, j_location=20, k_location=15, well_type='PRODUCER')
            ]
    
    def _validate_data_consistency(self):
        """Validate data consistency."""
        total_cells = self.grid_dimensions[0] * self.grid_dimensions[1] * self.grid_dimensions[2]
        
        if self.permeability is not None and len(self.permeability) != total_cells:
            logger.warning(f"Permeability size mismatch: {len(self.permeability)} != {total_cells}")
        
        if self.porosity is not None and len(self.porosity) != total_cells:
            logger.warning(f"Porosity size mismatch: {len(self.porosity)} != {total_cells}")
        
        expected_tops = self.grid_dimensions[0] * self.grid_dimensions[1]
        if self.tops is not None and len(self.tops) != expected_tops:
            logger.warning(f"TOPS size mismatch: {len(self.tops)} != {expected_tops}")
    
    def export_for_simulation(self, output_dir: str = "data/processed") -> Dict[str, Path]:
        """Export data for simulation."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Export wells
        wells_csv = output_path / "wells.csv"
        with open(wells_csv, 'w') as f:
            f.write("name,i,j,k,type,phase,control\n")
            for well in self.wells:
                f.write(f"{well.name},{well.i_location},{well.j_location},{well.k_location},"
                       f"{well.well_type},{well.phase},{well.control_mode}\n")
        files['wells'] = wells_csv
        
        # Export arrays
        if self.permeability is not None:
            perm_npy = output_path / "permeability.npy"
            np.save(perm_npy, self.permeability)
            files['permeability'] = perm_npy
        
        if self.porosity is not None:
            poro_npy = output_path / "porosity.npy"
            np.save(poro_npy, self.porosity)
            files['porosity'] = poro_npy
        
        if self.tops is not None:
            tops_npy = output_path / "tops.npy"
            np.save(tops_npy, self.tops)
            files['tops'] = tops_npy
        
        # Export config summary
        summary = {
            'grid': self.grid_dimensions,
            'well_count': len(self.wells),
            'has_permeability': self.permeability is not None,
            'has_porosity': self.porosity is not None,
            'has_tops': self.tops is not None,
            'pvt_tables': list(self.pvt_tables.keys()),
            'saturation_tables': list(self.saturation_tables.keys())
        }
        
        summary_json = output_path / "summary.json"
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        files['summary'] = summary_json
        
        logger.info(f"Exported {len(files)} files to {output_path}")
        return files
    
    def get_simulation_data(self) -> Dict[str, Any]:
        """Get all parsed data."""
        return {
            'wells': [well.to_dict() for well in self.wells],
            'grid_dimensions': self.grid_dimensions,
            'permeability': self.permeability,
            'porosity': self.porosity,
            'tops': self.tops,
            'pvt_tables': self.pvt_tables,
            'saturation_tables': self.saturation_tables,
            'configs': self.configs
        }
