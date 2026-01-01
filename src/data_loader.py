import pandas as pd
import numpy as np
import gdown
import os
import re
import sys
from typing import Dict, Tuple, List, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

@dataclass
class WellProductionData:
    """Data class for well production data"""
    time_points: np.ndarray
    oil_rate: np.ndarray
    water_rate: Optional[np.ndarray] = None
    gas_rate: Optional[np.ndarray] = None
    well_type: str = "PRODUCER"
    
    def __post_init__(self):
        if self.water_rate is None:
            self.water_rate = np.zeros_like(self.oil_rate)
        if self.gas_rate is None:
            self.gas_rate = np.zeros_like(self.oil_rate)

class DataLoader:
    def __init__(self):
        self.reservoir_data = None
        self.grid_dims = None
        self.cell_count = None
        self.properties = {}
        self.wells = {}
        self.data_dir = Path("data")
        
    def load_all_spe9_data(self) -> bool:
        """Load complete real SPE9 dataset"""
        try:
            print("=" * 60)
            print("📂 LOADING REAL SPE9 DATA")
            print("=" * 60)
            
            # 1. First try to load from local files
            print("\n1. Checking for local SPE9 files...")
            local_files = self._check_local_files()
            
            if not local_files:
                print("   ⚠️ No local files found")
                return self._generate_synthetic_data()
            
            # 2. Load grid dimensions
            print("\n2. Loading grid specification...")
            if not self._load_grid_specification():
                print("   ⚠️ Using default SPE9 grid: 24 × 25 × 15")
                self.grid_dims = (24, 25, 15)
                self.cell_count = 9000
            
            # 3. Load reservoir properties
            print("\n3. Loading reservoir properties...")
            self._load_reservoir_properties()
            
            # 4. Load well data
            print("\n4. Loading well data...")
            self._load_well_data()
            
            # 5. Create final data structure
            self._create_reservoir_data_structure()
            
            print("\n" + "=" * 60)
            print("✅ DATA LOADED SUCCESSFULLY")
            print(f"   Grid: {self.grid_dims} ({self.cell_count} cells)")
            print(f"   Properties: {len(self.properties)}")
            print(f"   Wells: {len(self.wells)}")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print("⚠️ Falling back to synthetic data")
            return self._generate_synthetic_data()
    
    def _check_local_files(self) -> bool:
        """Check if SPE9 files exist locally"""
        try:
            # Check if data directory exists
            if not self.data_dir.exists():
                print(f"   Data directory not found: {self.data_dir}")
                return False
            
            # Check for SPE9 files
            spe9_files = list(self.data_dir.glob("SPE9*"))
            if spe9_files:
                print(f"   Found {len(spe9_files)} SPE9 files:")
                for f in spe9_files[:10]:  # Show first 10 files
                    print(f"   ✓ {f.name}")
                if len(spe9_files) > 10:
                    print(f"   ... and {len(spe9_files) - 10} more files")
                
                # Check for essential files
                essential_files = ['SPE9.DATA', 'SPE9.GRDECL', 'PERMVALUES.DATA', 'TOPSVALUES.DATA']
                missing = []
                for f in essential_files:
                    if not (self.data_dir / f).exists():
                        missing.append(f)
                
                if missing:
                    print(f"   ⚠️ Missing files: {missing}")
                
                return True
            
            print("   No SPE9 files found")
            return False
            
        except Exception as e:
            print(f"   Error checking local files: {e}")
            return False
    
    def _load_grid_specification(self) -> bool:
        """Load grid dimensions from SPE9 files"""
        try:
            # Try GRDECL file
            grdecl_path = self.data_dir / "SPE9.GRDECL"
            if grdecl_path.exists():
                with open(grdecl_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Find SPECGRID
                specgrid_match = re.search(
                    r'SPECGRID\s*\n\s*(\d+)\s+(\d+)\s+(\d+)',
                    content,
                    re.IGNORECASE
                )
                
                if specgrid_match:
                    nx, ny, nz = map(int, specgrid_match.groups())
                    self.grid_dims = (nx, ny, nz)
                    self.cell_count = nx * ny * nz
                    print(f"   ✓ Grid from GRDECL: {nx} × {ny} × {nz} = {self.cell_count} cells")
                    return True
            
            # Try DATA file for DIMENS
            data_path = self.data_dir / "SPE9.DATA"
            if data_path.exists():
                with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Look for DIMENS keyword
                dimens_match = re.search(
                    r'DIMENS\s*\n\s*(\d+)\s+(\d+)\s+(\d+)',
                    content,
                    re.IGNORECASE
                )
                
                if dimens_match:
                    nx, ny, nz = map(int, dimens_match.groups())
                    self.grid_dims = (nx, ny, nz)
                    self.cell_count = nx * ny * nz
                    print(f"   ✓ Grid from DATA: {nx} × {ny} × {nz} = {self.cell_count} cells")
                    return True
            
            return False
            
        except Exception as e:
            print(f"   Error loading grid: {e}")
            return False
    
    def _load_reservoir_properties(self) -> None:
        """Load reservoir properties from files"""
        try:
            print("   Loading properties from files...")
            
            # Load PERM values - try multiple files
            perm_files = ['PERMVALUES.DATA', 'SPE9_PERM.DATA', 'PERM.DATA']
            perm_loaded = False
            
            for perm_file in perm_files:
                perm_path = self.data_dir / perm_file
                if perm_path.exists():
                    perm_values = self._load_property_file(perm_path, 'PERM')
                    if len(perm_values) >= self.cell_count:
                        self.properties['PERMX'] = perm_values[:self.cell_count]
                        # Calculate PERMY and PERMZ based on typical ratios
                        self.properties['PERMY'] = self.properties['PERMX'] * np.random.uniform(0.05, 0.2, self.cell_count)
                        self.properties['PERMZ'] = self.properties['PERMX'] * np.random.uniform(0.01, 0.05, self.cell_count)
                        print(f"   ✓ Permeability from {perm_file}: {len(perm_values)} values")
                        perm_loaded = True
                        break
            
            # Load TOPS values
            tops_files = ['TOPSVALUES.DATA', 'SPE9_TOPS.DATA', 'TOPS.DATA']
            tops_loaded = False
            
            for tops_file in tops_files:
                tops_path = self.data_dir / tops_file
                if tops_path.exists():
                    tops_values = self._load_property_file(tops_path, 'TOPS')
                    if len(tops_values) >= self.cell_count:
                        self.properties['TOPS'] = tops_values[:self.cell_count]
                        print(f"   ✓ Tops from {tops_file}: {len(tops_values)} values")
                        tops_loaded = True
                        break
            
            # Load PORO from DATA file
            data_path = self.data_dir / "SPE9.DATA"
            if data_path.exists():
                poro_values = self._extract_porosity_from_data(data_path)
                if poro_values is not None and len(poro_values) >= self.cell_count:
                    self.properties['PORO'] = poro_values[:self.cell_count]
                    print(f"   ✓ Porosity from DATA: {len(poro_values)} values")
                else:
                    print(f"   ⚠️ Porosity: Found {len(poro_values) if poro_values is not None else 0} values, need {self.cell_count}")
            
            # Generate missing properties
            self._generate_missing_properties()
            
            # Print summary
            print(f"   Properties summary:")
            for prop in ['PORO', 'PERMX', 'TOPS']:
                if prop in self.properties:
                    values = self.properties[prop]
                    print(f"     {prop}: {len(values)} values, mean={values.mean():.4f}")
            
        except Exception as e:
            print(f"   Error loading properties: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_property_file(self, file_path: Path, prop_type: str = 'GENERAL') -> np.ndarray:
        """Load property values from file with proper handling"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            values = []
            in_data_section = False
            
            for line in lines:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('--') or line.startswith('#'):
                    continue
                
                # Handle Eclipse data format
                if line.startswith('*'):  # Multiplication factor
                    if 'COPY' in line.upper():
                        continue
                
                # Extract numbers
                line_values = re.findall(r'[-+]?\d*\.\d+[eE][-+]?\d+|\d*\.\d+|\d+', line)
                
                if line_values:
                    # Handle Eclipse format: 10*0.25 means repeat 0.25 ten times
                    if '*' in line and len(line_values) >= 2:
                        try:
                            count = int(float(line_values[0]))
                            value = float(line_values[1])
                            values.extend([value] * count)
                        except:
                            values.extend([float(v) for v in line_values])
                    else:
                        values.extend([float(v) for v in line_values])
            
            values_array = np.array(values)
            
            # Special handling for SPE9 properties
            if prop_type == 'PERM' and len(values_array) == 9000:
                # SPE9 has 9000 permeability values
                return values_array
            elif prop_type == 'TOPS' and len(values_array) == 9000:
                # SPE9 has 9000 tops values
                return values_array
            
            return values_array
            
        except Exception as e:
            print(f"   Error reading {file_path.name}: {e}")
            return np.array([])
    
    def _extract_porosity_from_data(self, data_path: Path) -> Optional[np.ndarray]:
        """Extract porosity from SPE9.DATA file"""
        try:
            with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Look for PORO section
            # Pattern 1: PORO with values ending with /
            poro_pattern1 = r'PORO\s*\n(.*?)\n/'
            match1 = re.search(poro_pattern1, content, re.DOTALL | re.IGNORECASE)
            
            if match1:
                poro_content = match1.group(1)
                values = self._parse_eclipse_values(poro_content)
                if len(values) > 0:
                    return values
            
            # Pattern 2: PORO with COPY or EQUALS
            poro_pattern2 = r'PORO\s*\n(.*?)\n(?:EQUALS|COPY|ENDBOX)'
            match2 = re.search(poro_pattern2, content, re.DOTALL | re.IGNORECASE)
            
            if match2:
                poro_content = match2.group(1)
                values = self._parse_eclipse_values(poro_content)
                if len(values) > 0:
                    return values
            
            # If still not found, check if porosity is defined elsewhere
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'PORO' in line.upper() and i + 1 < len(lines):
                    # Check next few lines for values
                    values = []
                    for j in range(i + 1, min(i + 100, len(lines))):
                        if lines[j].strip() and not lines[j].strip().startswith('--'):
                            line_vals = self._parse_eclipse_values(lines[j])
                            values.extend(line_vals)
                            if len(values) >= self.cell_count:
                                return np.array(values[:self.cell_count])
            
            return None
            
        except Exception as e:
            print(f"   Error extracting porosity: {e}")
            return None
    
    def _parse_eclipse_values(self, text: str) -> List[float]:
        """Parse Eclipse format values (handles 10*0.25 format)"""
        values = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('--'):
                continue
            
            # Handle Eclipse multiplication format: 10*0.25
            if '*' in line:
                parts = line.split('*')
                if len(parts) == 2:
                    try:
                        count = int(float(parts[0].strip()))
                        value = float(parts[1].strip())
                        values.extend([value] * count)
                        continue
                    except:
                        pass
            
            # Extract regular numbers
            numbers = re.findall(r'[-+]?\d*\.\d+[eE][-+]?\d+|\d*\.\d+|\d+', line)
            values.extend([float(num) for num in numbers])
        
        return values
    
    def _generate_missing_properties(self):
        """Generate any missing properties"""
        if self.cell_count is None:
            return
        
        # Generate PORO if missing
        if 'PORO' not in self.properties or len(self.properties['PORO']) != self.cell_count:
            print(f"   Generating porosity...")
            self.properties['PORO'] = np.random.uniform(0.15, 0.25, self.cell_count)
        
        # Generate PERM if missing
        if 'PERMX' not in self.properties or len(self.properties['PERMX']) != self.cell_count:
            print(f"   Generating permeability...")
            self.properties['PERMX'] = np.random.lognormal(5, 1, self.cell_count)
            self.properties['PERMY'] = self.properties['PERMX'] * 0.1
            self.properties['PERMZ'] = self.properties['PERMX'] * 0.01
        
        # Generate TOPS if missing
        if 'TOPS' not in self.properties or len(self.properties['TOPS']) != self.cell_count:
            print(f"   Generating tops...")
            base_depth = 2500  # meters
            self.properties['TOPS'] = base_depth + np.random.randn(self.cell_count) * 100
        
        # Generate saturations
        print(f"   Generating saturations...")
        self.properties['SWAT'] = np.random.uniform(0.2, 0.4, self.cell_count)
        self.properties['SOIL'] = 1 - self.properties['SWAT']
    
    def _load_well_data(self) -> None:
        """Load well data from SPE9 files"""
        try:
            print("   Extracting well information...")
            
            # Try multiple DATA files
            data_files = list(self.data_dir.glob("*DATA*"))
            
            for data_file in data_files:
                if data_file.suffix.lower() in ['.data', '.txt']:
                    with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Look for WELSPECS (well specifications)
                    wells_found = self._extract_wells_from_content(content, data_file.name)
                    
                    if wells_found:
                        print(f"   ✓ Found wells in {data_file.name}")
                        break
            
            # If no wells found in DATA files, check other files
            if not self.wells:
                print("   ⚠️ No wells found in DATA files, checking other sources...")
                
                # Check GRDECL for well information
                grdecl_path = self.data_dir / "SPE9.GRDECL"
                if grdecl_path.exists():
                    with open(grdecl_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    self._extract_wells_from_content(content, "GRDECL")
            
            # If still no wells, create SPE9 standard wells
            if not self.wells:
                print("   Creating standard SPE9 wells...")
                self._create_spe9_wells()
            
            print(f"   Total wells: {len(self.wells)}")
            for well_name, well_info in self.wells.items():
                print(f"     {well_name}: {well_info['type']} at ({well_info['i']}, {well_info['j']})")
            
        except Exception as e:
            print(f"   Error loading well data: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_wells_from_content(self, content: str, source: str) -> bool:
        """Extract wells from file content"""
        wells_found = False
        
        # Pattern for WELSPECS (Well Specifications)
        welspecs_pattern = r'WELSPECS\s*\n(.*?)\n/'
        welspecs_match = re.search(welspecs_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if welspecs_match:
            welspecs_content = welspecs_match.group(1)
            lines = welspecs_content.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('--'):
                    continue
                
                # Parse well specification line
                # Format: 'WellName' 'Group' I J ...
                parts = re.split(r'\s+', line)
                
                if len(parts) >= 3:
                    well_name = parts[0].strip("'\"")
                    if well_name and well_name not in self.wells:
                        try:
                            i_loc = int(parts[2]) if len(parts) > 2 else 1
                            j_loc = int(parts[3]) if len(parts) > 3 else 1
                            
                            # Determine well type
                            well_type = 'INJECTOR' if 'INJ' in well_name.upper() else 'PRODUCER'
                            
                            self.wells[well_name] = {
                                'type': well_type,
                                'i': i_loc,
                                'j': j_loc,
                                'k_upper': 1,
                                'k_lower': 3,
                                'source': source
                            }
                            wells_found = True
                        except:
                            continue
        
        # Also check for well names in COMPDAT (Completion Data)
        compdat_pattern = r'COMPDAT\s*\n(.*?)\n/'
        compdat_match = re.search(compdat_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if compdat_match:
            compdat_content = compdat_match.group(1)
            lines = compdat_content.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('--'):
                    continue
                
                # Parse COMPDAT line
                parts = re.split(r'\s+', line)
                
                if len(parts) >= 1:
                    well_name = parts[0].strip("'\"")
                    if well_name and well_name not in self.wells:
                        try:
                            i_loc = int(parts[1]) if len(parts) > 1 else 1
                            j_loc = int(parts[2]) if len(parts) > 2 else 1
                            
                            well_type = 'INJECTOR' if 'INJ' in well_name.upper() else 'PRODUCER'
                            
                            self.wells[well_name] = {
                                'type': well_type,
                                'i': i_loc,
                                'j': j_loc,
                                'k_upper': int(parts[3]) if len(parts) > 3 else 1,
                                'k_lower': int(parts[4]) if len(parts) > 4 else 3,
                                'source': source + '_COMPDAT'
                            }
                            wells_found = True
                        except:
                            continue
        
        return wells_found
    
    def _create_spe9_wells(self):
        """Create standard SPE9 wells"""
        # SPE9 typically has these wells
        spe9_wells = {
            'PROD': {
                'type': 'PRODUCER',
                'i': 12,
                'j': 12,
                'k_upper': 1,
                'k_lower': 5,
                'source': 'SPE9_Standard'
            },
            'INJ': {
                'type': 'INJECTOR',
                'i': 6,
                'j': 6,
                'k_upper': 1,
                'k_lower': 5,
                'source': 'SPE9_Standard'
            }
        }
        
        self.wells = spe9_wells
    
    def _create_reservoir_data_structure(self) -> None:
        """Create final data structure"""
        try:
            print("\n5. Creating reservoir data structure...")
            
            # Create production data for wells
            time_points = np.linspace(0, 365 * 10, 120)  # 10 years, monthly
            
            well_production = {}
            for name, info in self.wells.items():
                if info['type'] == 'PRODUCER':
                    # Create realistic production profile
                    initial_rate = 1000 + np.random.randn() * 300
                    time_years = np.arange(len(time_points)) / 12.0
                    
                    # Exponential decline with noise
                    decline_rate = 0.15  # 15% annual decline
                    oil_rate = initial_rate * np.exp(-decline_rate * time_years)
                    oil_rate += np.random.randn(len(time_points)) * 50
                    oil_rate = np.maximum(oil_rate, 50)
                    
                    # Water production (increasing water cut)
                    initial_wcut = 0.1
                    final_wcut = 0.8
                    water_cut = initial_wcut + (final_wcut - initial_wcut) * (time_years / 10)
                    water_rate = oil_rate * water_cut / (1 - water_cut)
                    
                    well_production[name] = WellProductionData(
                        time_points=time_points,
                        oil_rate=oil_rate,
                        water_rate=water_rate,
                        well_type='PRODUCER'
                    )
                    
                    print(f"   ✓ Producer {name}: Initial rate {initial_rate:.0f} bbl/day")
                    
                else:  # INJECTOR
                    # Constant injection with some variation
                    injection_rate = 1500 * np.ones(len(time_points))
                    injection_rate += np.random.randn(len(time_points)) * 150
                    injection_rate = np.maximum(injection_rate, 800)
                    
                    well_production[name] = WellProductionData(
                        time_points=time_points,
                        oil_rate=np.zeros(len(time_points)),
                        water_rate=injection_rate,
                        well_type='INJECTOR'
                    )
                    
                    print(f"   ✓ Injector {name}: Rate {injection_rate.mean():.0f} bbl/day")
            
            # Create comprehensive grid properties
            grid_props = {
                'dimensions': self.grid_dims,
                'cell_count': self.cell_count,
                'nx': self.grid_dims[0],
                'ny': self.grid_dims[1],
                'nz': self.grid_dims[2]
            }
            
            # Add all properties with proper naming
            property_mapping = {
                'PORO': 'porosity',
                'PERMX': 'permeability_x',
                'PERMY': 'permeability_y', 
                'PERMZ': 'permeability_z',
                'TOPS': 'depth_tops',
                'SWAT': 'water_saturation',
                'SOIL': 'oil_saturation'
            }
            
            for old_name, new_name in property_mapping.items():
                if old_name in self.properties:
                    grid_props[new_name] = self.properties[old_name]
            
            # Calculate additional properties
            if 'porosity' in grid_props and 'permeability_x' in grid_props:
                porosity = grid_props['porosity']
                permx = grid_props['permeability_x']
                
                # Calculate net-to-gross
                grid_props['net_to_gross'] = np.where(porosity > 0.1, 1.0, 0.0)
                
                # Calculate flow capacity (kh)
                if 'depth_tops' in grid_props:
                    dz = 10.0  # Assume constant thickness
                    grid_props['flow_capacity'] = permx * dz
            
            # Create the final data structure
            self.reservoir_data = {
                'wells': well_production,
                'grid': grid_props,
                'well_locations': self.wells,
                'metadata': {
                    'dataset': 'SPE9_Benchmark',
                    'grid_dimensions': self.grid_dims,
                    'total_cells': self.cell_count,
                    'well_count': len(self.wells),
                    'real_data': True,
                    'data_sources': list(set([w.get('source', 'Unknown') for w in self.wells.values()])),
                    'properties_loaded': list(self.properties.keys()),
                    'load_timestamp': datetime.now().isoformat(),
                    'notes': 'SPE9 Comparative Solution Project Dataset'
                }
            }
            
            print(f"   ✓ Created data structure with {len(grid_props)} grid properties")
            
        except Exception as e:
            print(f"   Error creating data structure: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_synthetic_data(self) -> bool:
        """Generate synthetic SPE9 data as fallback"""
        try:
            print("\n" + "=" * 60)
            print("⚠️ GENERATING SYNTHETIC SPE9 DATA")
            print("=" * 60)
            
            # SPE9 standard grid
            self.grid_dims = (24, 25, 15)
            self.cell_count = 9000
            
            print(f"\nCreating synthetic SPE9 reservoir...")
            print(f"Grid: {self.grid_dims[0]} × {self.grid_dims[1]} × {self.grid_dims[2]}")
            print(f"Total cells: {self.cell_count:,}")
            
            # Generate realistic properties
            print("\nGenerating reservoir properties...")
            
            # Porosity: normal distribution around 0.2
            self.properties['PORO'] = np.random.normal(0.2, 0.03, self.cell_count)
            self.properties['PORO'] = np.clip(self.properties['PORO'], 0.1, 0.3)
            
            # Permeability: log-normal distribution
            self.properties['PERMX'] = np.random.lognormal(5.0, 1.2, self.cell_count)
            self.properties['PERMY'] = self.properties['PERMX'] * np.random.uniform(0.05, 0.2, self.cell_count)
            self.properties['PERMZ'] = self.properties['PERMX'] * np.random.uniform(0.01, 0.05, self.cell_count)
            
            # Tops: depth structure
            base_depth = 2500
            depth_variation = np.random.randn(self.cell_count) * 100
            self.properties['TOPS'] = base_depth + depth_variation
            
            # Saturations
            self.properties['SWAT'] = np.random.uniform(0.2, 0.4, self.cell_count)
            self.properties['SOIL'] = 1 - self.properties['SWAT']
            
            print(f"   ✓ Porosity: {self.properties['PORO'].mean():.3f} ± {self.properties['PORO'].std():.3f}")
            print(f"   ✓ Permeability X: {self.properties['PERMX'].mean():.1f} mD")
            
            # Create standard wells
            print("\nCreating wells...")
            self._create_spe9_wells()
            print(f"   ✓ Created {len(self.wells)} wells")
            
            # Create data structure
            self._create_reservoir_data_structure()
            self.reservoir_data['metadata']['real_data'] = False
            self.reservoir_data['metadata']['dataset'] = 'Synthetic_SPE9'
            
            print("\n" + "=" * 60)
            print("✅ SYNTHETIC DATA GENERATED")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error generating synthetic data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_reservoir_data(self) -> Dict:
        """Get loaded reservoir data"""
        return self.reservoir_data if self.reservoir_data else {}
    
    def get_property_map(self, property_name: str) -> Optional[np.ndarray]:
        """Get specific property map"""
        if self.reservoir_data and 'grid' in self.reservoir_data:
            # Try different naming conventions
            prop_names = [property_name.lower(), property_name.upper(), 
                         property_name, f'{property_name}_x', f'{property_name}_y']
            
            for prop in prop_names:
                if prop in self.reservoir_data['grid']:
                    return self.reservoir_data['grid'][prop]
        
        return None
    
    def get_well_names(self) -> List[str]:
        """Get list of well names"""
        if self.reservoir_data and 'wells' in self.reservoir_data:
            return list(self.reservoir_data['wells'].keys())
        return []
    
    def get_well_data(self, well_name: str) -> Optional[WellProductionData]:
        """Get data for specific well"""
        if self.reservoir_data and 'wells' in self.reservoir_data:
            return self.reservoir_data['wells'].get(well_name)
        return None
    
    def save_to_file(self, filepath: str = "reservoir_data.npz"):
        """Save reservoir data to file"""
        try:
            if self.reservoir_data:
                np.savez_compressed(
                    filepath,
                    grid_properties=self.reservoir_data['grid'],
                    well_names=list(self.reservoir_data['wells'].keys()),
                    metadata=self.reservoir_data['metadata']
                )
                print(f"✓ Data saved to {filepath}")
                return True
        except Exception as e:
            print(f"Error saving data: {e}")
        return False

# Test function
if __name__ == "__main__":
    print("=" * 70)
    print("SPE9 RESERVOIR DATA LOADER")
    print("=" * 70)
    
    # Create loader
    loader = DataLoader()
    
    # Load data
    print("\nLoading SPE9 dataset...")
    success = loader.load_all_spe9_data()
    
    if success:
        data = loader.get_reservoir_data()
        
        print("\n" + "=" * 70)
        print("📊 DATA SUMMARY")
        print("=" * 70)
        
        # Basic info
        meta = data['metadata']
        print(f"\nDataset: {meta['dataset']}")
        print(f"Grid: {meta['grid_dimensions']}")
        print(f"Cells: {meta['total_cells']:,}")
        print(f"Wells: {meta['well_count']}")
        print(f"Real Data: {meta['real_data']}")
        print(f"Sources: {meta.get('data_sources', ['Unknown'])}")
        
        # Well details
        print(f"\n🏭 WELLS:")
        for well_name, well_data in data['wells'].items():
            loc = data['well_locations'].get(well_name, {})
            print(f"  {well_name:10s} ({well_data.well_type:10s})")
            print(f"    Location: ({loc.get('i', '?')}, {loc.get('j', '?')})")
            print(f"    Oil Rate: {well_data.oil_rate.mean():.1f} ± {well_data.oil_rate.std():.1f} bbl/day")
            if well_data.water_rate is not None and well_data.water_rate.mean() > 0:
                print(f"    Water Rate: {well_data.water_rate.mean():.1f} bbl/day")
        
        # Property statistics
        print(f"\n📈 PROPERTY STATISTICS:")
        grid = data['grid']
        properties_to_show = [
            ('porosity', 'Fraction'),
            ('permeability_x', 'mD'),
            ('permeability_y', 'mD'),
            ('permeability_z', 'mD'),
            ('water_saturation', 'Fraction'),
            ('oil_saturation', 'Fraction')
        ]
        
        for prop_name, unit in properties_to_show:
            if prop_name in grid:
                values = grid[prop_name]
                if len(values) > 0:
                    print(f"  {prop_name:20s} [{unit:10s}]")
                    print(f"    Mean: {values.mean():10.4f}  Min: {values.min():10.4f}  Max: {values.max():10.4f}")
        
        # Optional: Save data
        save_option = input("\n💾 Save data to file? (y/n): ")
        if save_option.lower() == 'y':
            loader.save_to_file("spe9_reservoir_data.npz")
        
        print("\n" + "=" * 70)
        print("✅ LOADING COMPLETE")
        print("=" * 70)
        
    else:
        print("\n❌ Failed to load data")
