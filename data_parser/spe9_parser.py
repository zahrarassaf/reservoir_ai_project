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
        self.grid_inc_path = self.data_dir / "SPE9_GRID.INC"
        self.poro_inc_path = self.data_dir / "SPE9_PORO.INC"
        self.pvt_inc_path = self.data_dir / "SPE9_PVT.INC"
        self.sat_inc_path = self.data_dir / "SPE9_SATURATION_TABLES.INC"
        self.perm_values_path = self.data_dir / "PERMVALUES.DATA"
        self.tops_values_path = self.data_dir / "TOPSVALUES.DATA"
        
        # Configuration paths
        self.config_dir = Path("config")
        self.grid_params_path = self.config_dir / "grid_parameters.json"
        self.sim_config_path = self.config_dir / "simulation_config.yaml"
        self.well_controls_path = self.config_dir / "well_controls.json"
        
        # Parsed data
        self.wells: List[WellSpecification] = []
        self.grid_dimensions: Tuple[int, int, int] = (24, 25, 15)  # Default SPE9
        self.permeability: Optional[np.ndarray] = None
        self.porosity: Optional[np.ndarray] = None
        self.tops: Optional[np.ndarray] = None
        self.pvt_tables: Dict[str, Any] = {}
        self.saturation_tables: Dict[str, Any] = {}
        
        # Load configurations
        self.grid_params = self._load_json_config(self.grid_params_path)
        self.sim_config = self._load_yaml_config(self.sim_config_path)
        self.well_controls = self._load_json_config(self.well_controls_path)
        
    def _load_json_config(self, path: Path) -> Dict:
        """Load JSON configuration file."""
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_yaml_config(self, path: Path) -> Dict:
        """Load YAML configuration file."""
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def parse_all(self) -> 'SPE9ProjectParser':
        """Parse all data files in the project."""
        logger.info("Starting comprehensive SPE9 project parsing...")
        
        # Parse main SPE9.DATA file
        self._parse_spe9_data()
        
        # Parse INC files if they exist
        if self.grid_inc_path.exists():
            self._parse_grid_inc()
        
        if self.poro_inc_path.exists():
            self._parse_porosity_inc()
        
        if self.pvt_inc_path.exists():
            self._parse_pvt_inc()
        
        if self.sat_inc_path.exists():
            self._parse_saturation_tables_inc()
        
        # Parse separate data files
        if self.perm_values_path.exists():
            self._parse_permeability_values()
        
        if self.tops_values_path.exists():
            self._parse_tops_values()
        
        # Validate and cross-check data
        self._validate_data_consistency()
        
        logger.info(f"Parsing complete: {len(self.wells)} wells, grid {self.grid_dimensions}")
        return self
    
    def _parse_spe9_data(self):
        """Parse main SPE9.DATA file."""
        if not self.spe9_data_path.exists():
            raise FileNotFoundError(f"SPE9.DATA not found at {self.spe9_data_path}")
        
        logger.info(f"Parsing {self.spe9_data_path}")
        
        with open(self.spe9_data_path, 'r') as f:
            content = f.read()
        
        # Extract DIMENS
        dimens_match = re.search(r'DIMENS\s+(\d+)\s+(\d+)\s+(\d+)', content)
        if dimens_match:
            self.grid_dimensions = (
                int(dimens_match.group(1)),
                int(dimens_match.group(2)),
                int(dimens_match.group(3))
            )
        
        # Extract WELSPECS
        welspecs_section = self._extract_section(content, 'WELSPECS')
        if welspecs_section:
            self._parse_welspecs(welspecs_section)
        
        # Extract COMPDAT for completion data
        compdat_section = self._extract_section(content, 'COMPDAT')
        if compdat_section:
            self._parse_compdat(compdat_section)
        
        # Extract SCHEDULE for well controls
        schedule_section = self._extract_section(content, 'SCHEDULE')
        if schedule_section:
            self._parse_schedule(schedule_section)
    
    def _extract_section(self, content: str, keyword: str) -> Optional[str]:
        """Extract a section from Eclipse data file."""
        pattern = rf'\{keyword}(.*?)(?=\s*/\s*|\Z)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def _parse_welspecs(self, content: str):
        """Parse WELSPECS section."""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            # Remove comments
            if '--' in line:
                line = line.split('--')[0].strip()
            
            parts = line.split()
            if len(parts) >= 5:
                well_name = parts[0].strip("'").strip('"')
                
                # Determine well type from name or default
                well_type = 'PRODUCER'
                if well_name.upper().startswith('I') or 'INJ' in well_name.upper():
                    well_type = 'INJECTOR'
                
                well = WellSpecification(
                    name=well_name,
                    i_location=int(parts[1]),
                    j_location=int(parts[2]),
                    k_location=int(parts[3]),
                    well_type=well_type,
                    phase=parts[4].strip("'").strip('"').upper()
                )
                
                self.wells.append(well)
                logger.debug(f"Parsed well: {well}")
    
    def _parse_compdat(self, content: str):
        """Parse COMPDAT section for completion data."""
        # This would add completion data to existing wells
        pass  # Implement based on your needs
    
    def _parse_schedule(self, content: str):
        """Parse SCHEDULE section for well controls."""
        # This would update well control modes
        pass  # Implement based on your needs
    
    def _parse_grid_inc(self):
        """Parse GRID include file."""
        logger.info(f"Parsing grid from {self.grid_inc_path}")
        
        with open(self.grid_inc_path, 'r') as f:
            content = f.read()
        
        # Extract SPECGRID for dimensions
        specgrid_match = re.search(r'SPECGRID\s+(\d+)\s+(\d+)\s+(\d+)', content)
        if specgrid_match:
            self.grid_dimensions = (
                int(specgrid_match.group(1)),
                int(specgrid_match.group(2)),
                int(specgrid_match.group(3))
            )
    
    def _parse_porosity_inc(self):
        """Parse porosity data from include file."""
        logger.info(f"Parsing porosity from {self.poro_inc_path}")
        
        with open(self.poro_inc_path, 'r') as f:
            content = f.read()
        
        # Find PORO keyword and extract values
        poro_match = re.search(r'PORO\s*(.*?)(?=\s*/\s*)', content, re.DOTALL)
        if poro_match:
            values_text = poro_match.group(1)
            values = self._parse_numbers(values_text)
            self.porosity = np.array(values)
            logger.info(f"Loaded porosity data: {self.porosity.shape}")
    
    def _parse_pvt_inc(self):
        """Parse PVT tables from include file."""
        logger.info(f"Parsing PVT tables from {self.pvt_inc_path}")
        
        with open(self.pvt_inc_path, 'r') as f:
            content = f.read()
        
        # Extract PVTO tables (oil PVT)
        pvto_match = re.search(r'PVTO\s*(.*?)(?=\s*/\s*)', content, re.DOTALL)
        if pvto_match:
            self.pvt_tables['PVTO'] = self._parse_pvt_table(pvto_match.group(1))
        
        # Extract PVTG tables (gas PVT) if present
        pvtg_match = re.search(r'PVTG\s*(.*?)(?=\s*/\s*)', content, re.DOTALL)
        if pvtg_match:
            self.pvt_tables['PVTG'] = self._parse_pvt_table(pvtg_match.group(1))
    
    def _parse_saturation_tables_inc(self):
        """Parse saturation tables from include file."""
        logger.info(f"Parsing saturation tables from {self.sat_inc_path}")
        
        with open(self.sat_inc_path, 'r') as f:
            content = f.read()
        
        # Extract SGOF table
        sgof_match = re.search(r'SGOF\s*(.*?)(?=\s*/\s*)', content, re.DOTALL)
        if sgof_match:
            self.saturation_tables['SGOF'] = self._parse_saturation_table(sgof_match.group(1))
        
        # Extract SWOF table
        swof_match = re.search(r'SWOF\s*(.*?)(?=\s*/\s*)', content, re.DOTALL)
        if swof_match:
            self.saturation_tables['SWOF'] = self._parse_saturation_table(swof_match.group(1))
    
    def _parse_permeability_values(self):
        """Parse permeability values from separate file."""
        logger.info(f"Parsing permeability from {self.perm_values_path}")
        
        with open(self.perm_values_path, 'r') as f:
            content = f.read()
        
        # Parse all numbers from the file
        numbers = self._parse_numbers(content)
        self.permeability = np.array(numbers)
        
        total_cells = self.grid_dimensions[0] * self.grid_dimensions[1] * self.grid_dimensions[2]
        
        if len(self.permeability) != total_cells:
            logger.warning(
                f"Permeability data size mismatch: {len(self.permeability)} != {total_cells}. "
                f"Reshaping or truncating may be needed."
            )
    
    def _parse_tops_values(self):
        """Parse TOPS values from separate file."""
        logger.info(f"Parsing TOPS from {self.tops_values_path}")
        
        with open(self.tops_values_path, 'r') as f:
            content = f.read()
        
        numbers = self._parse_numbers(content)
        self.tops = np.array(numbers)
        
        expected_tops = self.grid_dimensions[0] * self.grid_dimensions[1]
        if len(self.tops) != expected_tops:
            logger.warning(
                f"TOPS data size mismatch: {len(self.tops)} != {expected_tops}"
            )
    
    def _parse_numbers(self, text: str) -> List[float]:
        """Parse numbers from text, handling Eclipse format."""
        # Remove comments
        text = re.sub(r'--.*$', '', text, flags=re.MULTILINE)
        
        # Handle Eclipse multipliers (e.g., 4*0.15 means 0.15 repeated 4 times)
        numbers = []
        tokens = text.split()
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Check for multiplier pattern: N*VALUE
            if '*' in token and not token.startswith('*'):
                try:
                    multiplier_str, value_str = token.split('*')
                    multiplier = int(multiplier_str)
                    value = float(value_str)
                    numbers.extend([value] * multiplier)
                    i += 1
                except ValueError:
                    # Not a multiplier, try to parse as single number
                    try:
                        numbers.append(float(token))
                        i += 1
                    except ValueError:
                        i += 1  # Skip invalid token
            else:
                # Single number
                try:
                    numbers.append(float(token))
                except ValueError:
                    pass  # Skip non-numeric tokens
                i += 1
        
        return numbers
    
    def _parse_pvt_table(self, content: str) -> List[List[float]]:
        """Parse PVT table data."""
        table = []
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            # Remove comments
            if '--' in line:
                line = line.split('--')[0]
            
            numbers = self._parse_numbers(line)
            if numbers:
                table.append(numbers)
        
        return table
    
    def _parse_saturation_table(self, content: str) -> np.ndarray:
        """Parse saturation table data."""
        table_data = []
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            if '--' in line:
                line = line.split('--')[0]
            
            numbers = self._parse_numbers(line)
            if len(numbers) >= 4:  # Sg/Sw, krg/krw, krog/krow, Pc
                table_data.append(numbers[:4])
        
        return np.array(table_data) if table_data else np.array([])
    
    def _validate_data_consistency(self):
        """Validate consistency between different data sources."""
        issues = []
        
        # Check grid size consistency
        total_cells = self.grid_dimensions[0] * self.grid_dimensions[1] * self.grid_dimensions[2]
        
        if self.permeability is not None:
            if len(self.permeability) != total_cells:
                issues.append(
                    f"Permeability size {len(self.permeability)} != grid cells {total_cells}"
                )
        
        if self.porosity is not None:
            if len(self.porosity) != total_cells:
                issues.append(
                    f"Porosity size {len(self.porosity)} != grid cells {total_cells}"
                )
        
        # Check TOPS size
        expected_tops = self.grid_dimensions[0] * self.grid_dimensions[1]
        if self.tops is not None and len(self.tops) != expected_tops:
            issues.append(f"TOPS size {len(self.tops)} != expected {expected_tops}")
        
        # Check well locations are within grid bounds
        for well in self.wells:
            if (well.i_location > self.grid_dimensions[0] or 
                well.j_location > self.grid_dimensions[1] or
                well.k_location > self.grid_dimensions[2]):
                issues.append(
                    f"Well {well.name} location ({well.i_location}, {well.j_location}, "
                    f"{well.k_location}) outside grid bounds {self.grid_dimensions}"
                )
        
        # Log issues
        for issue in issues:
            logger.warning(f"Data consistency issue: {issue}")
        
        if not issues:
            logger.info("All data consistency checks passed")
        
        return len(issues) == 0
    
    def export_for_simulation(self, output_dir: str = "data/processed") -> Dict[str, Path]:
        """Export data in format ready for simulation."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # 1. Export well data
        wells_file = output_path / "simulation_wells.csv"
        with open(wells_file, 'w') as f:
            f.write("WellID,Name,I,J,K,Type,Phase,Control\n")
            for i, well in enumerate(self.wells, 1):
                f.write(f"{i},{well.name},{well.i_location},{well.j_location},"
                       f"{well.k_location},{well.well_type},{well.phase},{well.control_mode}\n")
        exported_files['wells'] = wells_file
        
        # 2. Export permeability
        if self.permeability is not None:
            perm_file = output_path / "permeability.npy"
            np.save(perm_file, self.permeability)
            exported_files['permeability'] = perm_file
            
            # Also export as text for inspection
            perm_txt = output_path / "permeability.txt"
            np.savetxt(perm_txt, self.permeability, fmt='%.6f')
        
        # 3. Export porosity
        if self.porosity is not None:
            poro_file = output_path / "porosity.npy"
            np.save(poro_file, self.porosity)
            exported_files['porosity'] = poro_file
        
        # 4. Export TOPS
        if self.tops is not None:
            tops_file = output_path / "grid_tops.npy"
            np.save(tops_file, self.tops)
            exported_files['tops'] = tops_file
            
            tops_txt = output_path / "grid_tops.txt"
            np.savetxt(tops_txt, self.tops, fmt='%.2f')
        
        # 5. Export SGOF table
        if 'SGOF' in self.saturation_tables:
            sgof_file = output_path / "sgof_table.csv"
            np.savetxt(sgof_file, self.saturation_tables['SGOF'], 
                      delimiter=',', header='Sg,krg,krog,Pc')
            exported_files['sgof'] = sgof_file
        
        # 6. Export configuration summary
        config_summary = {
            'grid_dimensions': self.grid_dimensions,
            'well_count': len(self.wells),
            'data_files': {k: str(v) for k, v in exported_files.items()},
            'parsed_tables': list(self.pvt_tables.keys()) + list(self.saturation_tables.keys())
        }
        
        summary_file = output_path / "simulation_config_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(config_summary, f, indent=2)
        exported_files['config_summary'] = summary_file
        
        logger.info(f"Exported {len(exported_files)} files for simulation")
        
        return exported_files
    
    def get_simulation_data(self) -> Dict[str, Any]:
        """Get all data ready for simulation."""
        return {
            'wells': [well.to_dict() for well in self.wells],
            'grid_dimensions': self.grid_dimensions,
            'permeability': self.permeability,
            'porosity': self.porosity,
            'tops': self.tops,
            'pvt_tables': self.pvt_tables,
            'saturation_tables': self.saturation_tables,
            'config': {
                'grid': self.grid_params,
                'simulation': self.sim_config,
                'well_controls': self.well_controls
            }
        }
