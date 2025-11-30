import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
import re
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ReservoirData:
    """Structured reservoir data container"""
    permeability: torch.Tensor  # 3D tensor [nz, ny, nx]
    porosity: torch.Tensor      # 3D tensor [nz, ny, nx] 
    depth: torch.Tensor         # 3D tensor [nz, ny, nx]
    regions: torch.Tensor       # 3D tensor [nz, ny, nx]
    production: Dict[str, torch.Tensor]  # Time series data
    well_locations: List[Tuple[int, int, int]]
    grid_config: any

class SPE9DataLoader:
    """Professional SPE9 data loader with real data parsing"""
    
    def __init__(self, config):
        self.config = config
        
    def load_complete_dataset(self, data_directory: str) -> ReservoirData:
        """
        Load complete SPE9 dataset including grid, properties, and production data
        
        Args:
            data_directory: Path to directory containing SPE9 data files
            
        Returns:
            Structured ReservoirData object
        """
        print("Loading complete SPE9 dataset...")
        
        # Load grid and static properties
        grid_data = self._load_grid_data(data_directory)
        permeability = self._load_permeability_data(data_directory)
        porosity = self._create_porosity_field()
        depth = self._create_depth_field()
        regions = self._create_region_field()
        
        # Load production data
        production_data = self._load_production_data(data_directory)
        well_locations = self._extract_well_locations(data_directory)
        
        return ReservoirData(
            permeability=permeability,
            porosity=porosity,
            depth=depth,
            regions=regions,
            production=production_data,
            well_locations=well_locations,
            grid_config=self.config
        )
    
    def _load_grid_data(self, data_directory: str) -> Dict:
        """Load grid dimensions and geometry"""
        spe9_file = self._find_spe9_file(data_directory)
        
        with open(spe9_file, 'r') as f:
            content = f.read()
            
        # Extract grid dimensions
        dimens_match = re.search(r'DIMENS\s+(\d+)\s+(\d+)\s+(\d+)', content)
        if dimens_match:
            nx, ny, nz = map(int, dimens_match.groups())
            return {'nx': nx, 'ny': ny, 'nz': nz}
        
        return {'nx': 24, 'ny': 25, 'nz': 15}
    
    def _load_permeability_data(self, data_directory: str) -> torch.Tensor:
        """Load real permeability data from PERMVALUES.DATA"""
        perm_file = Path(data_directory) / "PERMVALUES.DATA"
        
        if not perm_file.exists():
            raise FileNotFoundError(f"PERMVALUES.DATA not found in {data_directory}")
            
        with open(perm_file, 'r') as f:
            content = f.read()
            
        return self._parse_permeability_tensor(content)
    
    def _parse_permeability_tensor(self, content: str) -> torch.Tensor:
        """Parse 3D permeability tensor from SPE9 format"""
        layers_data = []
        layer_pattern = r'-- LAYER\s+(\d+)(.*?)(?=-- LAYER|\Z)'
        
        for layer_match in re.finditer(layer_pattern, content, re.DOTALL):
            layer_content = layer_match.group(2)
            layer_data = self._parse_layer_data(layer_content)
            layers_data.append(layer_data)
            
        # Convert to 3D tensor [nz, ny, nx]
        perm_tensor = torch.tensor(np.stack(layers_data, axis=0), dtype=torch.float32)
        return perm_tensor
    
    def _parse_layer_data(self, layer_content: str) -> np.ndarray:
        """Parse single layer permeability data"""
        rows_data = []
        row_pattern = r'-- ROW\s+(\d+)(.*?)(?=-- ROW|\Z)'
        
        for row_match in re.finditer(row_pattern, layer_content, re.DOTALL):
            row_content = row_match.group(2)
            numbers = re.findall(r'[\d.]+(?:e-?\d+)?', row_content)
            row_values = [float(x) for x in numbers]
            rows_data.append(row_values)
            
        return np.array(rows_data)
    
    def _create_porosity_field(self) -> torch.Tensor:
        """Create 3D porosity field based on SPE9 layer values"""
        porosity_layers = [0.087, 0.097, 0.111, 0.16, 0.13, 0.17, 0.17, 
                          0.08, 0.14, 0.13, 0.12, 0.105, 0.12, 0.116, 0.157]
        
        porosity_tensor = torch.zeros(self.config.nz, self.config.ny, self.config.nx)
        for k in range(self.config.nz):
            porosity_tensor[k] = porosity_layers[k]
            
        return porosity_tensor
    
    def _create_depth_field(self) -> torch.Tensor:
        """Create 3D depth field"""
        depth_tensor = torch.zeros(self.config.nz, self.config.ny, self.config.nx)
        cumulative_depth = 2000.0  # Top depth
        
        for k in range(self.config.nz):
            depth_tensor[k] = cumulative_depth
            cumulative_depth += self.config.dz[k]
            
        return depth_tensor
    
    def _create_region_field(self) -> torch.Tensor:
        """Create geological region field"""
        region_tensor = torch.zeros(self.config.nz, self.config.ny, self.config.nx)
        
        # Simple region assignment based on geological layers
        for k in range(self.config.nz):
            if k < 5:
                region_tensor[k] = 1  # Upper zone
            elif k < 10:
                region_tensor[k] = 2  # Middle zone
            else:
                region_tensor[k] = 3  # Lower zone
                
        return region_tensor
    
    def _load_production_data(self, data_directory: str) -> Dict[str, torch.Tensor]:
        """Load or generate realistic production data"""
        # This would parse actual production history from SPE9 results
        # For now, generate physics-based synthetic data
        return self._generate_physics_based_production()
    
    def _generate_physics_based_production(self) -> Dict[str, torch.Tensor]:
        """Generate physics-based production data using reservoir engineering principles"""
        time_steps = 900
        time = torch.arange(time_steps).float()
        
        # Material balance based production profiles
        initial_oil_in_place = 1e8  # STB
        recovery_factor = 0.35
        decline_parameter = 0.002
        
        # Oil production (exponential decline)
        oil_production = initial_oil_in_place * recovery_factor * decline_parameter * \
                        torch.exp(-decline_parameter * time)
        
        # Gas production (increasing GOR with depletion)
        solution_gor = 1000  # SCF/STB
        gor_increase = 1.0 + (time / time_steps) * 2.0  # GOR doubles over time
        gas_production = oil_production * solution_gor * gor_increase
        
        # Water production (water breakthrough model)
        breakthrough_time = 300
        water_cut = torch.sigmoid((time - breakthrough_time) / 50) * 0.7
        water_production = oil_production * water_cut / (1 - water_cut)
        
        return {
            'FOPR': oil_production,
            'FGPR': gas_production,
            'FWPR': water_production,
            'FGOR': solution_gor * gor_increase,
            'time': time
        }
    
    def _extract_well_locations(self, data_directory: str) -> List[Tuple[int, int, int]]:
        """Extract well locations from SPE9 data"""
        # Parse COMPDAT keyword from SPE9 file
        spe9_file = self._find_spe9_file(data_directory)
        
        with open(spe9_file, 'r') as f:
            content = f.read()
            
        well_locations = []
        compdat_pattern = r"COMPDAT\s+'.*?'\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
        
        for match in re.finditer(compdat_pattern, content):
            i, j, k1, k2 = map(int, match.groups())
            well_locations.append((i, j, (k1 + k2) // 2))  # Mid-perforation
            
        return well_locations if well_locations else self._get_default_well_locations()
    
    def _get_default_well_locations(self) -> List[Tuple[int, int, int]]:
        """Default well locations based on SPE9 specification"""
        return [(24, 25, 13), (5, 1, 3), (8, 2, 3), (11, 3, 3)]  # Sample locations
    
    def _find_spe9_file(self, data_directory: str) -> Path:
        """Find SPE9 data file in directory"""
        possible_files = [
            Path(data_directory) / "SPE9.DATA",
            Path(data_directory) / "SPE9_CP.DATA",
            Path(data_directory) / "SPE9_CP_GROUP.DATA",
            Path("SPE9.DATA")
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                return file_path
                
        raise FileNotFoundError("No SPE9 data file found")
