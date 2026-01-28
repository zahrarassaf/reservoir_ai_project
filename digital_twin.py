import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys
import os

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ReservoirState(Enum):
    NORMAL = "normal"
    WATER_BREAKTHROUGH = "water_breakthrough"
    PRESSURE_DECLINE = "pressure_decline"
    EQUIPMENT_ISSUE = "equipment_issue"
    OPTIMIZATION_NEEDED = "optimization_needed"
    HIGH_WATER_CUT = "high_water_cut"
    LOW_PRESSURE = "low_pressure"

@dataclass
class SensorData:
    """Simulated sensor data from wells"""
    timestamp: datetime
    well_name: str
    pressure: float  # psi
    temperature: float  # °F
    oil_rate: float  # bpd
    water_rate: float  # bwpd
    gas_rate: float  # mscfd
    water_cut: float  # fraction
    choke_size: float  # 0-1
    bhp: float  # bottom hole pressure

@dataclass
class WellData:
    """Well-specific data based on YOUR actual data"""
    name: str
    well_type: str  # 'PRODUCER' or 'INJECTOR'
    location: Tuple[int, int]  # (i, j) grid coordinates from SPE9
    properties: Dict[str, float]  # perm, poro, etc. from your data
    production_history: List[Dict] = field(default_factory=list)
    current_status: str = "ACTIVE"
    
@dataclass
class DigitalTwinConfig:
    """Configuration for Digital Twin using YOUR data"""
    data_directory: str = "data"  # Your data directory
    update_frequency: int = 3600  # seconds between updates
    history_window_days: int = 30
    prediction_horizon_days: int = 90
    physics_weight: float = 0.8  # Increased physics weight
    ml_weight: float = 0.2  # Reduced ML weight
    enable_uncertainty: bool = True
    anomaly_threshold: float = 2.0

# ============================================================================
# PHYSICS-BASED MODEL USING YOUR ACTUAL DATA - FIXED
# ============================================================================

class PhysicsBasedReservoirModel:
    """Physics-based reservoir model using YOUR actual data - FIXED VERSION"""
    
    def __init__(self, grid_data: Dict[str, np.ndarray], wells: List[WellData], 
                 actual_permeability: np.ndarray = None, actual_tops: np.ndarray = None):
        self.nx, self.ny, self.nz = grid_data['dimensions']
        
        # Use your actual data
        self.porosity = self._load_porosity_from_spe9(grid_data['porosity'])
        self.permeability = self._load_permeability_from_actual(actual_permeability, grid_data['permeability'])
        self.saturation = self._load_saturation_from_spe9(grid_data['saturation'])
        
        self.wells = wells
        
        # Reservoir properties from SPE9 with realistic values
        self.oil_viscosity = 2.0  # cp
        self.water_viscosity = 0.5  # cp
        self.compressibility = 3e-6  # 1/psi
        self.formation_volume_factor = 1.2  # rb/stb
        
        # Initialize pressure field based on your data
        self.pressure_field = self._initialize_pressure_field(actual_tops)
        
        # Well indices from your data
        self.well_indices = self._calculate_well_indices(wells)
        
        print(f"Physics model initialized using your actual data:")
        print(f"  Grid: {self.nx}x{self.ny}x{self.nz}")
        print(f"  Porosity range: {self.porosity.min():.3f}-{self.porosity.max():.3f}")
        print(f"  Permeability range: {self.permeability.min():.1f}-{self.permeability.max():.1f} md")
    
    def _load_porosity_from_spe9(self, default_porosity: np.ndarray) -> np.ndarray:
        """Load porosity from SPE9 data or use actual data"""
        try:
            # Try to load actual porosity from SPE9.GRDECL
            grdecl_file = Path("data/SPE9.GRDECL")
            if grdecl_file.exists():
                with open(grdecl_file, 'r') as f:
                    content = f.read().upper()
                
                if 'PORO' in content:
                    start_idx = content.find('PORO')
                    end_idx = content.find('/', start_idx)
                    if end_idx > start_idx:
                        poro_section = content[start_idx:end_idx]
                        numbers = []
                        
                        for line in poro_section.split('\n')[1:]:
                            for token in line.split():
                                try:
                                    numbers.append(float(token))
                                except:
                                    continue
                        
                        if numbers and len(numbers) >= self.nx * self.ny * self.nz:
                            porosity_array = np.array(numbers[:self.nx * self.ny * self.nz])
                            porosity_array = porosity_array.reshape((self.nx, self.ny, self.nz))
                            porosity_array = np.clip(porosity_array, 0.05, 0.35)
                            print(f"  Loaded porosity from SPE9.GRDECL")
                            return porosity_array
        except Exception as e:
            print(f"  Could not load porosity from SPE9: {e}")
        
        # Use default porosity with more realistic distribution
        default_porosity = np.clip(default_porosity, 0.05, 0.35)
        return default_porosity
    
    def _load_permeability_from_actual(self, actual_permeability: np.ndarray, default_permeability: np.ndarray) -> np.ndarray:
        """Load permeability from your actual PERMVALUES.DATA"""
        if actual_permeability is not None and len(actual_permeability) > 0:
            # Reshape actual permeability data to match grid
            n_cells = self.nx * self.ny * self.nz
            if len(actual_permeability) >= n_cells:
                perm_array = actual_permeability[:n_cells].copy()
            else:
                # Repeat or interpolate if we don't have enough values
                perm_array = np.tile(actual_permeability, int(np.ceil(n_cells / len(actual_permeability))))[:n_cells]
            
            # Reshape to 3D grid
            perm_array = perm_array.reshape((self.nx, self.ny, self.nz))
            
            # Clip to reasonable values (0.1 to 10,000 md)
            perm_array = np.clip(perm_array, 0.1, 10000.0)
            
            # Log-transform to handle extreme values
            perm_array = np.log1p(perm_array)
            
            print(f"  Loaded permeability from your data: {len(actual_permeability)} values")
            print(f"  Permeability stats - Mean: {np.expm1(perm_array).mean():.1f} md, Std: {np.expm1(perm_array).std():.1f} md")
            return perm_array
        
        # Use default permeability with log-normal distribution
        default_permeability = np.log1p(default_permeability)
        return default_permeability
    
    def _load_saturation_from_spe9(self, default_saturation: np.ndarray) -> np.ndarray:
        """Load saturation from SPE9 data"""
        try:
            # Try to parse SPE9.DATA for saturation
            spe9_file = Path("data/SPE9.DATA")
            if spe9_file.exists():
                with open(spe9_file, 'r') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for line in lines:
                    if 'SW' in line.upper() and 'INIT' in line.upper() and not line.strip().startswith('--'):
                        parts = line.split()
                        for part in parts:
                            try:
                                sw_init = float(part)
                                if 0.1 <= sw_init <= 0.9:
                                    # Create saturation field
                                    saturation = np.ones((self.nx, self.ny, self.nz)) * (1 - sw_init)
                                    saturation = np.clip(saturation, 0.1, 0.9)
                                    print(f"  Loaded initial oil saturation from SPE9: {1-sw_init:.2f}")
                                    return saturation
                            except:
                                continue
        except Exception as e:
            print(f"  Could not load saturation from SPE9: {e}")
        
        # Use default saturation with reasonable values
        default_saturation = np.clip(default_saturation, 0.1, 0.9)
        return default_saturation
    
    def _initialize_pressure_field(self, actual_tops: np.ndarray = None) -> np.ndarray:
        """Initialize pressure field based on your tops data"""
        initial_pressure = 3000.0  # psi
        
        if actual_tops is not None and len(actual_tops) > 0:
            # Calculate pressure gradient based on depth from tops
            pressure_gradient = 0.433  # psi/ft
            
            # Create 3D pressure field based on depth
            pressure_field = np.ones((self.nx, self.ny, self.nz))
            
            # Calculate average depth from tops
            avg_depth = np.mean(actual_tops) if len(actual_tops) > 0 else 8000
            
            for k in range(self.nz):
                depth = avg_depth + k * 50  # Assume 50 ft per layer
                layer_pressure = initial_pressure + depth * pressure_gradient
                pressure_field[:, :, k] = layer_pressure
            
            print(f"  Initialized pressure field from your tops data (avg depth: {avg_depth:.0f} ft)")
            return pressure_field
        
        # Uniform pressure field
        return np.ones((self.nx, self.ny, self.nz)) * initial_pressure
    
    def _calculate_well_indices(self, wells: List[WellData]) -> Dict[str, Tuple[int, int, int]]:
        """Calculate well indices based on SPE9 grid"""
        well_indices = {}
        
        for well in wells:
            i, j = well.location
            # Ensure indices are within grid bounds
            i = min(max(i, 0), self.nx - 1)
            j = min(max(j, 0), self.ny - 1)
            k = 0  # Top layer for now
            
            well_indices[well.name] = (i, j, k)
        
        return well_indices
    
    def solve_pressure_diffusion(self, dt: float = 1.0) -> np.ndarray:  # Reduced dt
        """Solve pressure diffusion equation using your actual permeability - STABILIZED"""
        # Use log-permeability for stability
        log_perm = self.permeability  # Already log-transformed
        perm = np.expm1(log_perm)  # Convert back for calculations
        
        alpha = perm / (self.porosity * self.oil_viscosity * self.compressibility + 1e-10)
        alpha = np.clip(alpha, 1e-10, 1e10)  # Clip extreme values
        
        # Simple finite difference with stability check
        new_pressure = self.pressure_field.copy()
        
        # Use smaller time step for stability
        dt_stable = min(dt, 0.1 / np.max(alpha))
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(0, self.nz):
                    if k == 0:  # Top layer
                        laplacian = (
                            self.pressure_field[i+1, j, k] + self.pressure_field[i-1, j, k] +
                            self.pressure_field[i, j+1, k] + self.pressure_field[i, j-1, k] +
                            self.pressure_field[i, j, k+1] - 5 * self.pressure_field[i, j, k]
                        )
                    elif k == self.nz-1:  # Bottom layer
                        laplacian = (
                            self.pressure_field[i+1, j, k] + self.pressure_field[i-1, j, k] +
                            self.pressure_field[i, j+1, k] + self.pressure_field[i, j-1, k] +
                            self.pressure_field[i, j, k-1] - 5 * self.pressure_field[i, j, k]
                        )
                    else:
                        laplacian = (
                            self.pressure_field[i+1, j, k] + self.pressure_field[i-1, j, k] +
                            self.pressure_field[i, j+1, k] + self.pressure_field[i, j-1, k] +
                            self.pressure_field[i, j, k+1] + self.pressure_field[i, j, k-1] -
                            6 * self.pressure_field[i, j, k]
                        )
                    
                    # Update with stability check
                    delta = alpha[i, j, k] * laplacian * dt_stable
                    delta = np.clip(delta, -1000, 1000)  # Clip large changes
                    new_pressure[i, j, k] += delta
        
        # Ensure pressure stays within reasonable bounds
        new_pressure = np.clip(new_pressure, 500, 10000)
        self.pressure_field = new_pressure
        
        return new_pressure
    
    def update_saturation(self, dt: float = 1.0) -> np.ndarray:  # Reduced dt
        """Update oil saturation using fractional flow based on your actual data - STABILIZED"""
        # Fractional flow (Buckley-Leverett simplified)
        sat = np.clip(self.saturation, 0.1, 0.9)
        kr_o = sat ** 2
        kr_w = (1 - sat) ** 2
        
        mobility_o = kr_o / (self.oil_viscosity + 1e-10)
        mobility_w = kr_w / (self.water_viscosity + 1e-10)
        
        total_mobility = mobility_o + mobility_w + 1e-10
        f_o = mobility_o / total_mobility
        
        # Velocity from pressure gradient using your actual permeability
        grad_px, grad_py, grad_pz = np.gradient(self.pressure_field)
        grad_magnitude = np.sqrt(grad_px**2 + grad_py**2 + grad_pz**2 + 1e-10)
        
        # Use log-permeability
        log_perm = self.permeability
        perm = np.expm1(log_perm)
        
        v_total = -perm * grad_magnitude
        v_total = np.clip(v_total, -1000, 1000)  # Clip velocity
        
        # Update saturation with stability
        df_dx = np.gradient(f_o, axis=0)[0]
        ds_dt = -v_total * df_dx
        ds_dt = np.clip(ds_dt, -0.01, 0.01)  # Clip saturation change rate
        
        new_saturation = sat + ds_dt * dt
        new_saturation = np.clip(new_saturation, 0.1, 0.9)
        
        self.saturation = new_saturation
        return new_saturation
    
    def calculate_well_rates(self, choke_settings: Dict[str, float]) -> Dict[str, Dict]:
        """Calculate well rates based on your actual reservoir data - STABILIZED"""
        well_rates = {}
        
        for well in self.wells:
            if well.name not in self.well_indices:
                continue
                
            i, j, k = self.well_indices[well.name]
            choke = choke_settings.get(well.name, 0.5)
            choke = np.clip(choke, 0.1, 1.0)
            
            # Get reservoir properties at well location from YOUR data
            log_perm = self.permeability[i, j, k]
            perm = np.expm1(log_perm)  # Convert from log-scale
            poro = self.porosity[i, j, k]
            sat = self.saturation[i, j, k]
            pressure = self.pressure_field[i, j, k]
            
            # Productivity index based on your actual permeability - STABILIZED
            J = (perm * poro) / 1000.0  # Reduced scale factor
            J = np.clip(J, 0.01, 100.0)  # Clip to reasonable range
            
            # Pressure drawdown - stabilized
            bhp = 1000.0  # Increased minimum BHP
            drawdown = max(pressure - bhp, 10.0)  # Minimum drawdown
            drawdown = np.clip(drawdown, 10.0, 5000.0)  # Clip drawdown
            
            if well.well_type == 'PRODUCER':
                # Oil rate using Darcy's law with your actual properties - STABILIZED
                oil_rate = J * drawdown * sat * choke
                oil_rate = np.clip(oil_rate, 0, 5000)  # Reasonable maximum
                
                # Water rate based on water cut from your well properties
                water_cut = well.properties.get('water_cut', 0.05)
                water_cut = np.clip(water_cut, 0, 0.8)
                
                if oil_rate > 0 and water_cut < 0.99:
                    water_rate = oil_rate * water_cut / (1 - water_cut)
                else:
                    water_rate = 0
                water_rate = np.clip(water_rate, 0, 2000)
                
                # Gas rate from your well properties
                gor = well.properties.get('gor', 500)
                gas_rate = oil_rate * gor / 1000.0
                gas_rate = np.clip(gas_rate, 0, 10000)
                
                well_rates[well.name] = {
                    'oil_rate': float(oil_rate),
                    'water_rate': float(water_rate),
                    'gas_rate': float(gas_rate),
                    'water_cut': float(water_cut),
                    'type': 'PRODUCER',
                    'productivity_index': float(J),
                    'drawdown': float(drawdown)
                }
                
            else:  # INJECTOR
                # Injection rate based on your data - stabilized
                injection_rate = J * drawdown * choke * 5.0  # Increased multiplier
                injection_rate = np.clip(injection_rate, 0, 10000)
                
                well_rates[well.name] = {
                    'injection_rate': float(injection_rate),
                    'type': 'INJECTOR',
                    'injectivity_index': float(J)
                }
        
        return well_rates
    
    def predict(self, choke_settings: Dict[str, float], dt: float = 1.0) -> Dict[str, Any]:  # Reduced dt
        """Run physics-based prediction using your actual data - STABILIZED"""
        # Update reservoir state with smaller time steps
        for _ in range(10):  # Multiple smaller steps for stability
            pressure = self.solve_pressure_diffusion(dt/10.0)
            saturation = self.update_saturation(dt/10.0)
        
        # Calculate well rates
        well_rates = self.calculate_well_rates(choke_settings)
        
        # Calculate field totals with validation
        total_oil = 0.0
        total_water = 0.0
        total_injection = 0.0
        
        for rates in well_rates.values():
            oil_rate = rates.get('oil_rate', 0)
            water_rate = rates.get('water_rate', 0)
            injection_rate = rates.get('injection_rate', 0)
            
            # Validate and accumulate
            if not np.isnan(oil_rate) and np.isfinite(oil_rate):
                total_oil += oil_rate
            if not np.isnan(water_rate) and np.isfinite(water_rate):
                total_water += water_rate
            if not np.isnan(injection_rate) and np.isfinite(injection_rate):
                total_injection += injection_rate
        
        # Calculate additional metrics with validation
        avg_permeability = float(np.nanmean(np.expm1(self.permeability)))  # Convert back from log
        avg_porosity = float(np.nanmean(self.porosity))
        avg_pressure = float(np.nanmean(pressure))
        avg_saturation = float(np.nanmean(saturation))
        
        # Ensure reasonable values
        total_oil = np.clip(total_oil, 0, 50000)
        total_water = np.clip(total_water, 0, 20000)
        total_injection = np.clip(total_injection, 0, 20000)
        avg_pressure = np.clip(avg_pressure, 500, 10000)
        avg_saturation = np.clip(avg_saturation, 0.1, 0.9)
        avg_permeability = np.clip(avg_permeability, 1, 10000)
        avg_porosity = np.clip(avg_porosity, 0.05, 0.35)
        
        return {
            'pressure_field': pressure,
            'saturation_field': saturation,
            'well_rates': well_rates,
            'total_oil_rate': total_oil,
            'total_water_rate': total_water,
            'total_injection_rate': total_injection,
            'avg_pressure': avg_pressure,
            'avg_saturation': avg_saturation,
            'avg_permeability': avg_permeability,
            'avg_porosity': avg_porosity,
            'model_type': 'physics',
            'data_source': 'YOUR_ACTUAL_DATA'
        }

# ============================================================================
# MACHINE LEARNING MODEL TRAINED ON YOUR DATA - SIMPLIFIED
# ============================================================================

class ReservoirMLModel(nn.Module):
    """Simplified neural network for your reservoir data - NO BATCHNORM"""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64):
        super().__init__()
        
        # Simplified encoder without BatchNorm for single sample inference
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Reduced dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Prediction heads with realistic scaling
        hidden_output = hidden_dim // 2
        
        # Scale factors for realistic predictions
        self.oil_scale = nn.Parameter(torch.tensor([1000.0]))  # Typical oil rate scale
        self.pressure_scale = nn.Parameter(torch.tensor([3000.0]))  # Typical pressure scale
        
        self.rate_predictor = nn.Linear(hidden_output, 3)  # oil, water, gas rates
        self.pressure_predictor = nn.Linear(hidden_output, 1)
        self.water_cut_predictor = nn.Linear(hidden_output, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(x)
        
        # Predictions with realistic scaling
        rates_raw = self.rate_predictor(encoded)
        oil_rate = torch.sigmoid(rates_raw[:, 0:1]) * self.oil_scale  # 0-1000 bpd range
        water_rate = torch.sigmoid(rates_raw[:, 1:2]) * self.oil_scale * 0.3  # 0-300 bwpd
        gas_rate = torch.sigmoid(rates_raw[:, 2:3]) * self.oil_scale * 0.5  # 0-500 mscfd
        
        pressure = torch.sigmoid(self.pressure_predictor(encoded)) * self.pressure_scale  # 0-3000 psi
        water_cut = torch.sigmoid(self.water_cut_predictor(encoded)) * 0.5  # 0-50% water cut
        
        return {
            'oil_rate': oil_rate.squeeze(),
            'water_rate': water_rate.squeeze(),
            'gas_rate': gas_rate.squeeze(),
            'pressure': pressure.squeeze(),
            'water_cut': water_cut.squeeze()
        }

# ============================================================================
# DIGITAL TWIN MAIN CLASS USING YOUR DATA
# ============================================================================

class ReservoirDigitalTwin:
    """Main Digital Twin class using YOUR actual data"""
    
    def __init__(self, config: DigitalTwinConfig):
        self.config = config
        
        # Load YOUR actual data
        print("\n📊 LOADING YOUR ACTUAL RESERVOIR DATA...")
        self.actual_data = self._load_actual_data()
        
        # Create wells from your data
        self.wells = self._create_wells_from_data()
        
        # Create results directory
        self.results_dir = Path("results/digital_twin_actual_data")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize physics model with YOUR data
        grid_data = self._create_grid_data()
        self.physics_model = PhysicsBasedReservoirModel(
            grid_data=grid_data,
            wells=self.wells,
            actual_permeability=self.actual_data.get('permeability_values'),
            actual_tops=self.actual_data.get('tops_values')
        )
        
        # Initialize state management
        self.current_state = {}
        self.history = []
        self.predictions = []
        self.anomalies = []
        self.recommendations = []
        
        # Initialize performance metrics
        self.performance_metrics = {
            'update_count': 0,
            'avg_update_time': 0.0,
            'anomalies_detected': 0,
            'prediction_accuracy': [],
            'model_weights': {'physics': config.physics_weight, 'ml': config.ml_weight},
            'data_quality': self._assess_data_quality()
        }
        
        # Initialize ML model AFTER performance metrics
        self.ml_model = ReservoirMLModel()
        self._initialize_ml_model()
        
        # Initialize current state
        self.current_state = self._initialize_state()
        
        print(f"\n✅ Digital Twin initialized with {len(self.wells)} wells from YOUR data")
        print(f"   Data sources: {list(self.actual_data.keys())}")
    
    def _load_actual_data(self) -> Dict[str, Any]:
        """Load YOUR actual reservoir data"""
        data = {}
        data_dir = Path(self.config.data_directory)
        
        print(f"  Scanning {data_dir} for your data files...")
        
        # 1. Load PERMVALUES.DATA
        perm_file = data_dir / "PERMVALUES.DATA"
        if perm_file.exists():
            try:
                perm_data = []
                with open(perm_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        for part in parts:
                            try:
                                value = float(part)
                                perm_data.append(value)
                            except:
                                continue
                
                if perm_data:
                    perm_array = np.array(perm_data)
                    data['permeability_values'] = perm_array
                    data['permeability_stats'] = {
                        'mean': float(np.mean(perm_array)),
                        'std': float(np.std(perm_array)),
                        'min': float(np.min(perm_array)),
                        'max': float(np.max(perm_array)),
                        'count': len(perm_array)
                    }
                    print(f"    ✓ PERMVALUES.DATA: {len(perm_array)} permeability values")
            except Exception as e:
                print(f"    ✗ Could not parse PERMVALUES.DATA: {e}")
        
        # 2. Load TOPSVALUES.DATA
        tops_file = data_dir / "TOPSVALUES.DATA"
        if tops_file.exists():
            try:
                tops_data = []
                with open(tops_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        for part in parts:
                            try:
                                value = float(part)
                                tops_data.append(value)
                            except:
                                continue
                
                if tops_data:
                    tops_array = np.array(tops_data)
                    data['tops_values'] = tops_array
                    
                    # Calculate thickness
                    if len(tops_array) > 1:
                        thickness = np.abs(np.diff(tops_array))
                        data['thickness_values'] = thickness
                        data['thickness_stats'] = {
                            'mean': float(np.mean(thickness)),
                            'std': float(np.std(thickness)),
                            'count': len(thickness)
                        }
                    print(f"    ✓ TOPSVALUES.DATA: {len(tops_array)} tops values")
            except Exception as e:
                print(f"    ✗ Could not parse TOPSVALUES.DATA: {e}")
        
        # 3. Parse SPE9.DATA
        spe9_file = data_dir / "SPE9.DATA"
        if spe9_file.exists():
            try:
                spe9_params = self._parse_spe9_file(spe9_file)
                data.update(spe9_params)
                print(f"    ✓ SPE9.DATA: Extracted reservoir parameters")
            except Exception as e:
                print(f"    ✗ Could not parse SPE9.DATA: {e}")
        
        # 4. Parse GRDECL files
        grdecl_files = list(data_dir.glob("*.GRDECL"))
        for grdecl_file in grdecl_files:
            try:
                grdecl_data = self._parse_grdecl_file(grdecl_file)
                data[f'grdecl_{grdecl_file.stem}'] = grdecl_data
                print(f"    ✓ {grdecl_file.name}: Grid properties")
            except Exception as e:
                print(f"    ✗ Could not parse {grdecl_file.name}: {e}")
        
        return data
    
    def _parse_spe9_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse SPE9.DATA file for reservoir parameters"""
        params = {}
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Look for dimensions
        for i, line in enumerate(lines):
            if 'DIMENS' in line.upper() and not line.strip().startswith('--'):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        params['nx'] = int(parts[1])
                        params['ny'] = int(parts[2])
                        params['nz'] = int(parts[3])
                        print(f"      Found grid dimensions: {params['nx']} x {params['ny']} x {params['nz']}")
                    except:
                        pass
        
        # Look for porosity
        for line in lines:
            if 'PORO' in line.upper() and 'EQUALS' in line.upper() and not line.strip().startswith('--'):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        value = float(parts[2])
                        if 0.01 <= value <= 0.35:
                            params['spe9_porosity'] = value
                            print(f"      Found porosity: {value}")
                    except:
                        pass
        
        # Look for permeability
        for line in lines:
            if 'PERMX' in line.upper() and 'EQUALS' in line.upper() and not line.strip().startswith('--'):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        value = float(parts[2])
                        if value > 0:
                            params['spe9_permeability'] = value
                            print(f"      Found permeability: {value} md")
                    except:
                        pass
        
        # Look for saturations
        for line in lines:
            line_upper = line.upper()
            if 'SW' in line_upper and 'INIT' in line_upper and not line.strip().startswith('--'):
                parts = line.split()
                for part in parts:
                    try:
                        value = float(part)
                        if 0.1 <= value <= 0.9:
                            params['spe9_sw_init'] = value
                            params['spe9_so_init'] = 1.0 - value
                            print(f"      Found initial water saturation: {value}")
                            break
                    except:
                        continue
        
        return params
    
    def _parse_grdecl_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse GRDECL file for grid properties"""
        params = {}
        
        with open(file_path, 'r') as f:
            content = f.read().upper()
        
        # Extract porosity
        if 'PORO' in content:
            start_idx = content.find('PORO')
            end_idx = content.find('/', start_idx)
            
            if end_idx > start_idx:
                poro_section = content[start_idx:end_idx]
                numbers = []
                
                for line in poro_section.split('\n')[1:]:
                    for token in line.split():
                        try:
                            numbers.append(float(token))
                        except:
                            continue
                
                if numbers:
                    params['porosity_array'] = np.array(numbers)
                    print(f"      Extracted {len(numbers)} porosity values")
        
        return params
    
    def _create_wells_from_data(self) -> List[WellData]:
        """Create wells based on your actual data"""
        wells = []
        
        # Get grid dimensions from SPE9
        nx = self.actual_data.get('nx', 24)
        ny = self.actual_data.get('ny', 25)
        
        # Create producers (80% of wells)
        n_producers = 8
        for i in range(n_producers):
            well_name = f"PROD{i+1:02d}"
            
            # Use actual permeability for well properties
            if 'permeability_values' in self.actual_data:
                perm_values = self.actual_data['permeability_values']
                perm_mean = np.mean(perm_values)
                perm_std = np.std(perm_values)
                
                # Sample permeability for this well
                well_perm = np.random.normal(perm_mean, perm_std)
                well_perm = max(well_perm, 1.0)
            else:
                well_perm = np.random.lognormal(4, 0.5)
            
            well = WellData(
                name=well_name,
                well_type='PRODUCER',
                location=(
                    np.random.randint(2, nx - 2),
                    np.random.randint(2, ny - 2)
                ),
                properties={
                    'permeability': well_perm,
                    'porosity': self.actual_data.get('spe9_porosity', 0.18),
                    'water_cut': np.random.uniform(0.05, 0.3),
                    'gor': np.random.uniform(300, 800),
                    'completion_length': np.random.uniform(20, 50),
                    'skin_factor': np.random.uniform(-1, 5)
                }
            )
            wells.append(well)
        
        # Create injectors (20% of wells)
        n_injectors = 2
        for i in range(n_injectors):
            well_name = f"INJ{i+1:02d}"
            
            well = WellData(
                name=well_name,
                well_type='INJECTOR',
                location=(
                    np.random.randint(2, nx - 2),
                    np.random.randint(2, ny - 2)
                ),
                properties={
                    'permeability': np.random.lognormal(4, 0.5),
                    'porosity': self.actual_data.get('spe9_porosity', 0.18),
                    'injectivity_index': np.random.uniform(1.0, 3.0),
                    'completion_length': np.random.uniform(30, 60)
                }
            )
            wells.append(well)
        
        print(f"  Created {len(wells)} wells (8 producers, 2 injectors)")
        return wells
    
    def _create_grid_data(self) -> Dict[str, np.ndarray]:
        """Create grid data based on your actual data"""
        nx = self.actual_data.get('nx', 24)
        ny = self.actual_data.get('ny', 25)
        nz = self.actual_data.get('nz', 15)
        
        print(f"  Creating grid: {nx}x{ny}x{nz}")
        
        # Create porosity field
        if 'spe9_porosity' in self.actual_data:
            base_porosity = self.actual_data['spe9_porosity']
            porosity = np.random.normal(base_porosity, 0.03, (nx, ny, nz))
            porosity = np.clip(porosity, 0.05, 0.35)
        else:
            porosity = np.random.uniform(0.1, 0.3, (nx, ny, nz))
        
        # Create permeability field (will be replaced by actual data in physics model)
        permeability = np.ones((nx, ny, nz)) * 50.0
        
        # Create saturation field
        if 'spe9_so_init' in self.actual_data:
            base_saturation = self.actual_data['spe9_so_init']
            saturation = np.random.normal(base_saturation, 0.05, (nx, ny, nz))
            saturation = np.clip(saturation, 0.1, 0.9)
        else:
            saturation = np.random.uniform(0.6, 0.9, (nx, ny, nz))
        
        return {
            'dimensions': (nx, ny, nz),
            'porosity': porosity,
            'permeability': permeability,
            'saturation': saturation
        }
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess quality of your actual data"""
        data_quality = {
            'has_permeability_data': 'permeability_values' in self.actual_data,
            'has_tops_data': 'tops_values' in self.actual_data,
            'has_spe9_parameters': 'spe9_porosity' in self.actual_data,
            'data_sources': [],
            'completeness_score': 0.0
        }
        
        # Count data sources
        sources = []
        if 'permeability_values' in self.actual_data:
            sources.append('PERMVALUES.DATA')
            data_quality['completeness_score'] += 0.4
        
        if 'tops_values' in self.actual_data:
            sources.append('TOPSVALUES.DATA')
            data_quality['completeness_score'] += 0.3
        
        if 'spe9_porosity' in self.actual_data:
            sources.append('SPE9.DATA')
            data_quality['completeness_score'] += 0.3
        
        data_quality['data_sources'] = sources
        data_quality['completeness_score'] = min(data_quality['completeness_score'], 1.0)
        
        print(f"  Data quality assessment:")
        print(f"    Sources: {sources}")
        print(f"    Completeness score: {data_quality['completeness_score']:.1%}")
        
        return data_quality
    
    def _initialize_ml_model(self):
        """Initialize ML model with your data characteristics"""
        model_path = self.results_dir / "ml_model_actual_data.pth"
        
        # Always create new model for stability
        print("  Creating new simplified ML model for stability")
        self.ml_model.eval()
        
        # Save model info
        model_info = {
            'model_state_dict': self.ml_model.state_dict(),
            'input_dim': 20,
            'hidden_dim': 64,
            'data_sources': self.performance_metrics['data_quality']['data_sources'],
            'creation_date': datetime.now().isoformat()
        }
        torch.save(model_info, model_path)
    
    def _initialize_state(self) -> Dict[str, Any]:
        """Initialize digital twin state based on your data"""
        # Get initial rates from physics model
        initial_chokes = {well.name: 0.5 for well in self.wells}
        physics_pred = self.physics_model.predict(initial_chokes, dt=0)
        
        # Ensure all values are valid numbers with reasonable ranges
        total_oil = physics_pred['total_oil_rate']
        if np.isnan(total_oil) or total_oil <= 0 or total_oil > 50000:
            total_oil = 2000.0  # Reasonable default
        
        avg_pressure = physics_pred['avg_pressure']
        if np.isnan(avg_pressure) or avg_pressure < 500 or avg_pressure > 10000:
            avg_pressure = 3000.0  # Reasonable default
        
        return {
            'timestamp': datetime.now().isoformat(),
            'pressure_field': physics_pred['pressure_field'].tolist(),
            'saturation_field': physics_pred['saturation_field'].tolist(),
            'well_rates': physics_pred['well_rates'],
            'total_oil_rate': total_oil,
            'total_water_rate': max(min(physics_pred['total_water_rate'], 10000), 0),
            'total_injection_rate': max(min(physics_pred['total_injection_rate'], 10000), 0),
            'avg_pressure': avg_pressure,
            'avg_saturation': max(min(physics_pred['avg_saturation'], 0.9), 0.1),
            'avg_permeability': max(min(physics_pred['avg_permeability'], 10000), 1.0),
            'avg_porosity': max(min(physics_pred['avg_porosity'], 0.35), 0.05),
            'reservoir_state': ReservoirState.NORMAL.value,
            'confidence': 0.95,
            'choke_settings': initial_chokes,
            'data_quality': self.performance_metrics['data_quality']
        }
    
    def update(self, sensor_data: List[SensorData]):
        """Update digital twin with new sensor data"""
        start_time = time.time()
        
        # Process sensor data
        processed_data = self._process_sensor_data(sensor_data)
        
        # Store in history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'sensor_data': processed_data,
            'digital_state': self.current_state.copy()
        }
        self.history.append(history_entry)
        
        # Keep history within window
        max_history = self.config.history_window_days * 24
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
        
        try:
            # Run physics-based prediction with YOUR data (primary)
            physics_pred = self.physics_model.predict(self.current_state['choke_settings'])
            
            # Try ML prediction but don't rely on it heavily
            ml_pred = self._run_ml_prediction(processed_data)
            
            # Fuse predictions with high weight on physics
            fused_state = self._fuse_predictions(physics_pred, ml_pred, processed_data)
            
            # Update current state
            self.current_state.update(fused_state)
            self.current_state['timestamp'] = datetime.now().isoformat()
            
            # Ensure all critical metrics are within reasonable bounds
            self._validate_state()
            
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            # Fall back to physics model only
            physics_pred = self.physics_model.predict(self.current_state['choke_settings'])
            self.current_state.update({
                'total_oil_rate': physics_pred['total_oil_rate'],
                'total_water_rate': physics_pred['total_water_rate'],
                'avg_pressure': physics_pred['avg_pressure'],
                'confidence': 0.7,
                'timestamp': datetime.now().isoformat()
            })
            self._validate_state()
        
        # Detect anomalies with conservative thresholds
        new_anomalies = self._detect_anomalies_conservative(processed_data)
        if new_anomalies:
            self.anomalies.extend(new_anomalies)
            self.performance_metrics['anomalies_detected'] += len(new_anomalies)
        
        # Generate forecasts
        self._generate_forecasts()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Update performance metrics
        update_time = time.time() - start_time
        self.performance_metrics['update_count'] += 1
        self.performance_metrics['avg_update_time'] = (
            self.performance_metrics['avg_update_time'] * (self.performance_metrics['update_count'] - 1) + 
            update_time
        ) / self.performance_metrics['update_count']
        
        # Print update status with validated numbers
        oil_rate = self.current_state.get('total_oil_rate', 0)
        state = self.current_state.get('reservoir_state', 'unknown')
        print(f"Digital Twin updated in {update_time:.2f}s. State: {state}, Oil: {oil_rate:.0f} bpd")
        
        # Periodically save state
        if self.performance_metrics['update_count'] % 10 == 0:
            self._save_state()
    
    def _validate_state(self):
        """Ensure state values are within reasonable bounds"""
        bounds = {
            'total_oil_rate': (0, 50000),
            'total_water_rate': (0, 20000),
            'total_injection_rate': (0, 20000),
            'avg_pressure': (500, 10000),
            'avg_saturation': (0.1, 0.9),
            'avg_permeability': (1, 10000),
            'avg_porosity': (0.05, 0.35),
            'confidence': (0.1, 1.0)
        }
        
        for key, (min_val, max_val) in bounds.items():
            if key in self.current_state:
                val = self.current_state[key]
                if np.isnan(val) or not np.isfinite(val):
                    # Set to midpoint if invalid
                    self.current_state[key] = (min_val + max_val) / 2
                else:
                    # Clip to bounds
                    self.current_state[key] = max(min(val, max_val), min_val)
    
    def _process_sensor_data(self, sensor_data: List[SensorData]) -> Dict[str, Any]:
        """Process and aggregate sensor data"""
        processed = {
            'timestamp': sensor_data[0].timestamp.isoformat() if sensor_data else datetime.now().isoformat(),
            'well_data': {},
            'field_totals': {
                'oil_rate': 0.0,
                'water_rate': 0.0,
                'gas_rate': 0.0,
                'total_liquid': 0.0,
                'avg_pressure': 0.0,
                'avg_water_cut': 0.0,
                'avg_temperature': 0.0,
                'avg_choke': 0.0
            }
        }
        
        valid_wells = 0
        for sensor in sensor_data:
            well_name = sensor.well_name
            
            # Ensure sensor values are valid and within reasonable bounds
            oil_rate = np.clip(sensor.oil_rate, 0, 5000)
            water_rate = np.clip(sensor.water_rate, 0, 2000)
            pressure = np.clip(sensor.pressure, 500, 10000)
            water_cut = np.clip(sensor.water_cut, 0, 0.99)
            
            # Store individual well data
            processed['well_data'][well_name] = {
                'pressure': pressure,
                'temperature': np.clip(sensor.temperature, 100, 300),
                'oil_rate': oil_rate,
                'water_rate': water_rate,
                'gas_rate': np.clip(sensor.gas_rate, 0, 10000),
                'water_cut': water_cut,
                'choke_size': np.clip(sensor.choke_size, 0.1, 1.0),
                'bhp': np.clip(sensor.bhp, 500, 10000)
            }
            
            # Update field totals
            processed['field_totals']['oil_rate'] += oil_rate
            processed['field_totals']['water_rate'] += water_rate
            processed['field_totals']['gas_rate'] += np.clip(sensor.gas_rate, 0, 10000)
            processed['field_totals']['total_liquid'] += oil_rate + water_rate
            processed['field_totals']['avg_pressure'] += pressure
            processed['field_totals']['avg_temperature'] += np.clip(sensor.temperature, 100, 300)
            processed['field_totals']['avg_choke'] += np.clip(sensor.choke_size, 0.1, 1.0)
            valid_wells += 1
        
        # Calculate averages
        if valid_wells > 0:
            processed['field_totals']['avg_pressure'] /= valid_wells
            processed['field_totals']['avg_temperature'] /= valid_wells
            processed['field_totals']['avg_choke'] /= valid_wells
            
            # Calculate weighted average water cut
            total_oil = processed['field_totals']['oil_rate']
            total_water = processed['field_totals']['water_rate']
            if total_oil + total_water > 0:
                processed['field_totals']['avg_water_cut'] = total_water / (total_water + total_oil)
            else:
                processed['field_totals']['avg_water_cut'] = 0.0
        
        return processed
    
    def _run_ml_prediction(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run ML model prediction - simplified and robust"""
        try:
            # Prepare features from your data
            features = self._prepare_ml_features(sensor_data)
            
            # Convert to tensor
            feature_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Run prediction in eval mode
            self.ml_model.eval()
            with torch.no_grad():
                predictions = self.ml_model(feature_tensor)
            
            # Convert to dict with validation
            ml_pred = {
                'oil_rate': float(predictions['oil_rate'].item()),
                'water_rate': float(predictions['water_rate'].item()),
                'gas_rate': float(predictions['gas_rate'].item()),
                'pressure': float(predictions['pressure'].item()),
                'water_cut': float(predictions['water_cut'].item()),
                'model_type': 'ml',
                'data_source': 'YOUR_RESERVOIR'
            }
            
            # Validate predictions
            ml_pred['oil_rate'] = np.clip(ml_pred['oil_rate'], 0, 5000)
            ml_pred['water_rate'] = np.clip(ml_pred['water_rate'], 0, 2000)
            ml_pred['pressure'] = np.clip(ml_pred['pressure'], 500, 10000)
            ml_pred['water_cut'] = np.clip(ml_pred['water_cut'], 0, 0.8)
            
            return ml_pred
            
        except Exception as e:
            # Return conservative fallback values
            return {
                'oil_rate': 1000.0,
                'water_rate': 200.0,
                'gas_rate': 500.0,
                'pressure': 2500.0,
                'water_cut': 0.1,
                'model_type': 'ml',
                'data_source': 'FALLBACK'
            }
    
    def _prepare_ml_features(self, sensor_data: Dict[str, Any]) -> List[float]:
        """Prepare features for ML model using your data"""
        field_totals = sensor_data['field_totals']
        current_state = self.current_state
        
        # Use your actual permeability data
        if 'permeability_stats' in self.actual_data:
            perm_stats = self.actual_data['permeability_stats']
            perm_mean = perm_stats['mean']
            perm_std = perm_stats['std']
        else:
            perm_mean = 50.0
            perm_std = 25.0
        
        # Prepare features with normalization and validation
        features = [
            np.clip(field_totals['oil_rate'], 0, 50000) / 50000.0,
            np.clip(field_totals['water_rate'], 0, 20000) / 20000.0,
            np.clip(field_totals['gas_rate'], 0, 100000) / 100000.0,
            np.clip(field_totals['avg_pressure'], 500, 10000) / 10000.0,
            np.clip(field_totals['avg_water_cut'], 0, 1),
            np.clip(field_totals['avg_temperature'], 100, 300) / 300.0,
            np.clip(field_totals['avg_choke'], 0.1, 1.0),
            np.clip(current_state.get('avg_pressure', 3000), 500, 10000) / 10000.0,
            np.clip(current_state.get('avg_saturation', 0.6), 0.1, 0.9),
            np.clip(current_state.get('avg_permeability', 50), 1, 10000) / 10000.0,
            np.clip(current_state.get('avg_porosity', 0.18), 0.05, 0.35) / 0.35,
            np.clip(perm_mean, 1, 10000) / 10000.0,
            np.clip(perm_std, 0.1, 5000) / 5000.0,
            len(self.wells) / 20.0,
            np.clip(np.mean(list(current_state.get('choke_settings', {}).values())), 0.1, 1.0),
            np.clip(current_state.get('total_injection_rate', 0), 0, 20000) / 20000.0,
            np.clip(self.performance_metrics['anomalies_detected'], 0, 100) / 100.0,
            np.clip(len(self.history), 0, 1000) / 1000.0,
            np.clip(current_state.get('confidence', 0.5), 0.1, 1.0),
            np.clip(self.performance_metrics['data_quality']['completeness_score'], 0, 1)
        ]
        
        # Replace any NaN with 0.5 (neutral value)
        features = [0.5 if np.isnan(f) else f for f in features]
        
        return features
    
    def _fuse_predictions(self, physics_pred: Dict, ml_pred: Dict, sensor_data: Dict) -> Dict[str, Any]:
        """Fuse physics and ML predictions - physics-heavy"""
        # Use config weights but bias toward physics for stability
        physics_weight = self.config.physics_weight
        ml_weight = self.config.ml_weight
        
        # Fuse key metrics with physics as primary
        fused = {
            'total_oil_rate': physics_pred['total_oil_rate'] * 0.9 + ml_pred.get('oil_rate', 0) * 0.1,
            'total_water_rate': physics_pred['total_water_rate'] * 0.9 + ml_pred.get('water_rate', 0) * 0.1,
            'avg_pressure': physics_pred['avg_pressure'] * 0.95 + ml_pred.get('pressure', 2500) * 0.05,
            'water_cut': self.current_state.get('water_cut', 0.05) * 0.5 + ml_pred.get('water_cut', 0.1) * 0.5,
            'confidence': 0.8,  # Fixed confidence
            'fused_model': True,
            'physics_weight': physics_weight,
            'ml_weight': ml_weight
        }
        
        # Update choke settings based on optimization
        fused['choke_settings'] = self._optimize_choke_settings(sensor_data)
        
        return fused
    
    def _optimize_choke_settings(self, sensor_data: Dict[str, Any]) -> Dict[str, float]:
        """Optimize choke settings for maximum production using your data"""
        new_chokes = {}
        
        for well_name, well_data in sensor_data['well_data'].items():
            current_choke = self.current_state['choke_settings'].get(well_name, 0.5)
            pressure = well_data['pressure']
            water_cut = well_data['water_cut']
            
            # Conservative adjustments
            if pressure > 2500 and water_cut < 0.3:
                adjustment = 0.05  # Small increase
            elif pressure < 1500 or water_cut > 0.6:
                adjustment = -0.1  # Small decrease
            else:
                adjustment = 0.0
            
            # Apply adjustment with limits
            new_choke = current_choke + adjustment
            new_choke = max(0.2, min(new_choke, 0.8))  # Conservative range
            
            new_chokes[well_name] = new_choke
        
        return new_chokes
    
    def _detect_anomalies_conservative(self, sensor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies with conservative thresholds"""
        anomalies = []
        field_totals = sensor_data['field_totals']
        
        # Get baseline from your data
        expected_oil = self.current_state.get('total_oil_rate', 2000)
        expected_pressure = self.current_state.get('avg_pressure', 3000)
        
        # Only detect significant anomalies
        current_oil = field_totals['oil_rate']
        if current_oil > 0 and expected_oil > 0:
            oil_ratio = current_oil / expected_oil
            if oil_ratio < 0.6:  # Significant drop only
                anomalies.append({
                    'type': 'production_decline',
                    'severity': 'high',
                    'description': f"Production dropped to {current_oil:.0f} bpd ({oil_ratio:.0%} of expected)",
                    'timestamp': sensor_data['timestamp'],
                    'well': 'FIELD'
                })
                self.current_state['reservoir_state'] = ReservoirState.PRESSURE_DECLINE.value
        
        if field_totals['avg_water_cut'] > 0.8:  # Very high water cut
            anomalies.append({
                'type': 'high_field_water_cut',
                'severity': 'high',
                'description': f"Field water cut: {field_totals['avg_water_cut']:.1%}",
                'timestamp': sensor_data['timestamp'],
                'well': 'FIELD'
            })
            self.current_state['reservoir_state'] = ReservoirState.HIGH_WATER_CUT.value
        
        current_pressure = field_totals['avg_pressure']
        if current_pressure < 1000:  # Very low pressure
            anomalies.append({
                'type': 'pressure_decline',
                'severity': 'high',
                'description': f"Pressure dropped to {current_pressure:.0f} psi",
                'timestamp': sensor_data['timestamp'],
                'well': 'FIELD'
            })
            self.current_state['reservoir_state'] = ReservoirState.LOW_PRESSURE.value
        
        return anomalies
    
    def _generate_forecasts(self):
        """Generate production forecasts based on your reservoir data"""
        forecast_days = self.config.prediction_horizon_days
        forecasts = []
        
        # Use current state as starting point
        current_oil = self.current_state.get('total_oil_rate', 2000)
        current_water = self.current_state.get('total_water_rate', 400)
        current_pressure = self.current_state.get('avg_pressure', 3000)
        current_water_cut = self.current_state.get('water_cut', 0.1)
        
        # Conservative decline rates
        oil_decline_rate = 0.0002  # 0.02% per day
        pressure_decline_rate = 0.5  # 0.5 psi per day
        water_increase_rate = 0.0001  # 0.01% per day
        
        # Generate forecast
        for day in range(1, forecast_days + 1):
            forecast = {
                'days_ahead': day,
                'oil_rate': max(current_oil * (1 - oil_decline_rate * day), 100),
                'water_rate': current_water * (1 + water_increase_rate * day),
                'pressure': max(current_pressure - pressure_decline_rate * day, 500),
                'water_cut': min(current_water_cut * (1 + water_increase_rate * day * 2), 0.8),
                'confidence': max(0.7 - day * 0.005, 0.3),
                'decline_rate': oil_decline_rate
            }
            forecasts.append(forecast)
        
        self.predictions = forecasts
    
    def _generate_recommendations(self):
        """Generate operational recommendations based on your reservoir"""
        recommendations = []
        state = self.current_state.get('reservoir_state', ReservoirState.NORMAL.value)
        
        if state == ReservoirState.HIGH_WATER_CUT.value:
            recommendations.append({
                'type': 'water_management',
                'priority': 'high',
                'wells': 'High water cut producers',
                'action': 'Consider water shut-off or reduce production',
                'expected_impact': 'Reduce water handling costs',
                'implementation_time': '1-2 weeks',
                'confidence': 0.8
            })
        
        if state == ReservoirState.LOW_PRESSURE.value:
            recommendations.append({
                'type': 'pressure_maintenance',
                'priority': 'high',
                'wells': 'Injectors',
                'action': 'Increase injection rates by 10-20%',
                'expected_impact': 'Maintain reservoir pressure',
                'implementation_time': 'Immediate',
                'confidence': 0.9
            })
        
        if state == ReservoirState.NORMAL.value:
            recommendations.append({
                'type': 'routine_optimization',
                'priority': 'low',
                'wells': 'All wells',
                'action': 'Continue current operations',
                'expected_impact': 'Maintain stable production',
                'implementation_time': 'Ongoing',
                'confidence': 0.95
            })
        
        # Always include optimization recommendation
        recommendations.append({
            'type': 'data_analysis',
            'priority': 'medium',
            'wells': 'All wells',
            'action': 'Review production data and model predictions',
            'expected_impact': 'Identify optimization opportunities',
            'implementation_time': 'Weekly',
            'confidence': 0.85
        })
        
        self.recommendations = recommendations
    
    def _save_state(self):
        """Save digital twin state"""
        state_file = self.results_dir / f"digital_twin_state_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'current_state': {
                'total_oil_rate': float(self.current_state.get('total_oil_rate', 0)),
                'total_water_rate': float(self.current_state.get('total_water_rate', 0)),
                'total_injection_rate': float(self.current_state.get('total_injection_rate', 0)),
                'avg_pressure': float(self.current_state.get('avg_pressure', 0)),
                'reservoir_state': self.current_state.get('reservoir_state', 'unknown'),
                'confidence': float(self.current_state.get('confidence', 0.5))
            },
            'performance_metrics': self.performance_metrics,
            'anomalies_count': len(self.anomalies),
            'recommendations_count': len(self.recommendations),
            'data_quality': self.performance_metrics['data_quality'],
            'well_count': len(self.wells)
        }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        print(f"Digital Twin state saved to {state_file}")
    
    def visualize(self):
        """Create interactive visualizations of your reservoir"""
        try:
            self._create_production_forecast_plot()
            self._create_pressure_distribution_plot()
            self._create_well_performance_plot()
            self._create_data_quality_dashboard()
            self._create_recommendations_table()
            
            print("Visualizations created successfully")
            
        except Exception as e:
            print(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_production_forecast_plot(self):
        """Create production forecast plot based on your data"""
        if not self.predictions:
            return
        
        fig = go.Figure()
        
        days = [f['days_ahead'] for f in self.predictions]
        oil_rates = [f['oil_rate'] for f in self.predictions]
        water_rates = [f['water_rate'] for f in self.predictions]
        
        # Oil rate
        fig.add_trace(go.Scatter(
            x=days, y=oil_rates, mode='lines+markers',
            name='Oil Rate', line=dict(color='green', width=3),
            hovertemplate='Day %{x}: %{y:.0f} bpd<extra></extra>'
        ))
        
        # Water rate
        fig.add_trace(go.Scatter(
            x=days, y=water_rates, mode='lines+markers',
            name='Water Rate', line=dict(color='blue', width=3),
            hovertemplate='Day %{x}: %{y:.0f} bwpd<extra></extra>'
        ))
        
        # Current production
        current_oil = self.current_state.get('total_oil_rate', 0)
        if not np.isnan(current_oil):
            fig.add_trace(go.Scatter(
                x=[0], y=[current_oil],
                mode='markers', name='Current Oil',
                marker=dict(color='green', size=12, symbol='circle'),
                hovertemplate='Current: %{y:.0f} bpd<extra></extra>'
            ))
        
        fig.update_layout(
            title='Production Forecast Based on YOUR Reservoir Data',
            xaxis_title='Days Ahead',
            yaxis_title='Rate (bpd)',
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        html_file = self.results_dir / "production_forecast.html"
        fig.write_html(str(html_file))
        print(f"Production forecast saved to {html_file}")
    
    def _create_pressure_distribution_plot(self):
        """Create pressure distribution heatmap from your data"""
        try:
            if 'pressure_field' in self.current_state:
                pressure_field = np.array(self.current_state['pressure_field'])
                if pressure_field.size > 0:
                    pressure_slice = pressure_field[:, :, 0]
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=pressure_slice,
                        colorscale='Viridis',
                        name='Pressure (psi)',
                        hovertemplate='X: %{x}<br>Y: %{y}<br>Pressure: %{z:.0f} psi<extra></extra>'
                    ))
                    
                    # Add well locations
                    well_x = []
                    well_y = []
                    well_names = []
                    
                    for well in self.wells:
                        i, j = well.location
                        well_x.append(j)
                        well_y.append(i)
                        well_names.append(well.name)
                    
                    fig.add_trace(go.Scatter(
                        x=well_x, y=well_y,
                        mode='markers+text',
                        name='Wells',
                        marker=dict(
                            size=15,
                            color='red',
                            symbol='circle'
                        ),
                        text=well_names,
                        textposition="top center",
                        hovertemplate='Well: %{text}<br>Location: (%{y}, %{x})<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title='Pressure Distribution with Well Locations (Your Reservoir)',
                        xaxis_title='Y Coordinate',
                        yaxis_title='X Coordinate',
                        height=500
                    )
                    
                    html_file = self.results_dir / "pressure_distribution.html"
                    fig.write_html(str(html_file))
                    print(f"Pressure distribution saved to {html_file}")
        except Exception as e:
            print(f"Pressure plot failed: {e}")
    
    def _create_well_performance_plot(self):
        """Create well performance bar chart"""
        if not self.current_state.get('well_rates'):
            return
        
        well_names = list(self.current_state['well_rates'].keys())
        oil_rates = []
        water_cuts = []
        
        for w in well_names:
            rates = self.current_state['well_rates'][w]
            oil_rate = rates.get('oil_rate', 0)
            water_cut = rates.get('water_cut', 0)
            
            if not np.isnan(oil_rate):
                oil_rates.append(oil_rate)
            else:
                oil_rates.append(0)
            
            if not np.isnan(water_cut):
                water_cuts.append(water_cut)
            else:
                water_cuts.append(0)
        
        fig = go.Figure()
        
        # Oil rates
        fig.add_trace(go.Bar(
            x=well_names,
            y=oil_rates,
            name='Oil Rate (bpd)',
            marker_color='orange',
            hovertemplate='%{x}: %{y:.0f} bpd<extra></extra>'
        ))
        
        # Water cut (secondary axis)
        fig.add_trace(go.Scatter(
            x=well_names,
            y=water_cuts,
            name='Water Cut',
            yaxis='y2',
            line=dict(color='red', width=3),
            marker=dict(size=10),
            hovertemplate='%{x}: %{y:.1%} water cut<extra></extra>'
        ))
        
        fig.update_layout(
            title='Well Performance (Your Reservoir)',
            xaxis_title='Well Name',
            yaxis_title='Oil Rate (bpd)',
            yaxis2=dict(
                title='Water Cut',
                overlaying='y',
                side='right',
                range=[0, 1],
                tickformat='.0%'
            ),
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        html_file = self.results_dir / "well_performance.html"
        fig.write_html(str(html_file))
        print(f"Well performance saved to {html_file}")
    
    def _create_data_quality_dashboard(self):
        """Create data quality dashboard"""
        data_quality = self.performance_metrics['data_quality']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Data Sources', 'Data Completeness', 'Model Weights', 'Performance'),
            specs=[[{'type': 'table'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}]]
        )
        
        # Data sources table
        sources = data_quality['data_sources']
        if not sources:
            sources = ['No actual data files found']
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Data Source']),
                cells=dict(values=[sources])
            ),
            row=1, col=1
        )
        
        # Data completeness gauge
        completeness = data_quality['completeness_score']
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=completeness * 100,
                title={'text': "Data Completeness"},
                gauge={'axis': {'range': [None, 100]},
                      'steps': [
                          {'range': [0, 50], 'color': "red"},
                          {'range': [50, 80], 'color': "yellow"},
                          {'range': [80, 100], 'color': "green"}],
                      'threshold': {'line': {'color': "black", 'width': 4},
                                   'thickness': 0.75, 'value': 70}}
            ),
            row=1, col=2
        )
        
        # Model weights
        physics_weight = self.performance_metrics['model_weights']['physics'] * 100
        ml_weight = self.performance_metrics['model_weights']['ml'] * 100
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=physics_weight,
                title={'text': "Physics Model Weight"},
                number={'suffix': "%"}
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=ml_weight,
                title={'text': "ML Model Weight"},
                number={'suffix': "%"}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Data Quality Dashboard (Your Reservoir)',
            height=600,
            showlegend=False
        )
        
        html_file = self.results_dir / "data_quality_dashboard.html"
        fig.write_html(str(html_file))
        print(f"Data quality dashboard saved to {html_file}")
    
    def _create_recommendations_table(self):
        """Create recommendations table"""
        if not self.recommendations:
            return
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Digital Twin Recommendations - YOUR Reservoir</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; }}
                .table-container {{ background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .high {{ background-color: #ffcccc; }}
                .medium {{ background-color: #fff3cd; }}
                .low {{ background-color: #d4edda; }}
                .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Digital Twin Recommendations</h1>
                <p>Generated for YOUR Reservoir: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
                <p>Reservoir State: {self.current_state.get('reservoir_state', 'unknown')}</p>
                <p>Current Oil Production: {self.current_state.get('total_oil_rate', 0):.0f} bpd</p>
            </div>
            
            <div class="table-container">
                <table>
                    <tr>
                        <th>Type</th>
                        <th>Priority</th>
                        <th>Wells</th>
                        <th>Action</th>
                        <th>Expected Impact</th>
                        <th>Timeframe</th>
                        <th>Confidence</th>
                    </tr>
        """
        
        for rec in self.recommendations:
            priority_class = rec['priority']
            
            html_content += f"""
                <tr class="{priority_class}">
                    <td>{rec['type']}</td>
                    <td>{rec['priority'].upper()}</td>
                    <td>{rec['wells']}</td>
                    <td>{rec['action']}</td>
                    <td>{rec['expected_impact']}</td>
                    <td>{rec['implementation_time']}</td>
                    <td>{rec['confidence']:.0%}</td>
                </tr>
            """
        
        html_content += f"""
                </table>
            </div>
            
            <div class="timestamp">
                <p>Report generated using your actual data from: {', '.join(self.performance_metrics['data_quality']['data_sources'] or ['No data sources'])}</p>
                <p>Data Quality Score: {self.performance_metrics['data_quality']['completeness_score']:.0%}</p>
            </div>
        </body>
        </html>
        """
        
        html_file = self.results_dir / "recommendations.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Recommendations table saved to {html_file}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive digital twin report for your reservoir"""
        # Get recent anomalies
        serializable_anomalies = []
        for anomaly in self.anomalies[-5:] if self.anomalies else []:
            serializable_anomalies.append({
                'type': anomaly.get('type', 'unknown'),
                'severity': anomaly.get('severity', 'medium'),
                'description': anomaly.get('description', ''),
                'timestamp': anomaly.get('timestamp', datetime.now().isoformat())
            })
        
        # Get well statistics
        well_stats = []
        for well in self.wells:
            if well.name in self.current_state.get('well_rates', {}):
                rates = self.current_state['well_rates'][well.name]
                well_stats.append({
                    'name': well.name,
                    'type': well.well_type,
                    'oil_rate': float(rates.get('oil_rate', 0)),
                    'water_cut': float(rates.get('water_cut', 0))
                })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'reservoir_info': {
                'well_count': len(self.wells),
                'producers': sum(1 for w in self.wells if w.well_type == 'PRODUCER'),
                'injectors': sum(1 for w in self.wells if w.well_type == 'INJECTOR'),
                'data_sources': self.performance_metrics['data_quality']['data_sources']
            },
            'current_state': {
                'total_oil_rate': float(self.current_state.get('total_oil_rate', 0)),
                'total_water_rate': float(self.current_state.get('total_water_rate', 0)),
                'total_injection_rate': float(self.current_state.get('total_injection_rate', 0)),
                'avg_pressure': float(self.current_state.get('avg_pressure', 0)),
                'reservoir_state': self.current_state.get('reservoir_state', 'unknown')
            },
            'performance_metrics': {
                'update_count': self.performance_metrics['update_count'],
                'anomalies_detected': self.performance_metrics['anomalies_detected'],
                'data_quality': self.performance_metrics['data_quality']
            },
            'recent_anomalies': serializable_anomalies,
            'current_recommendations': self.recommendations[:3],  # Top 3 only
            'well_statistics': well_stats
        }
    
    def save_report_to_file(self):
        """Save comprehensive report to JSON file"""
        report = self.generate_report()
        report_file = self.results_dir / f"digital_twin_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Comprehensive report saved to {report_file}")
        return report_file

# ============================================================================
# SIMULATION AND DEMONSTRATION
# ============================================================================

def generate_synthetic_sensor_data(wells: List[WellData], timestamp: datetime) -> List[SensorData]:
    """Generate synthetic sensor data for testing - REALISTIC VALUES"""
    sensor_data = []
    
    for well in wells:
        if well.well_type == 'PRODUCER':
            # Realistic oil rates based on permeability
            base_perm = well.properties.get('permeability', 50)
            base_oil_rate = 100 + (base_perm / 10.0)  # 100-600 bpd range
            
            # Add some noise
            oil_rate = base_oil_rate * np.random.uniform(0.8, 1.2)
            oil_rate = np.clip(oil_rate, 50, 1000)
            
            # Water cut
            base_water_cut = well.properties.get('water_cut', 0.05)
            water_cut = base_water_cut * np.random.uniform(0.9, 1.1)
            water_cut = np.clip(water_cut, 0, 0.5)
            
            # Water rate
            water_rate = oil_rate * water_cut / (1 - water_cut + 1e-10)
            water_rate = np.clip(water_rate, 0, 500)
            
            # Gas rate
            gor = well.properties.get('gor', 500)
            gas_rate = oil_rate * gor / 1000.0
            gas_rate = np.clip(gas_rate, 0, 1000)
            
            # Pressure
            pressure = 2000 + np.random.normal(0, 100)
            pressure = np.clip(pressure, 1500, 3000)
            
        else:  # INJECTOR
            oil_rate = 0
            water_cut = 0
            water_rate = 1000 + np.random.normal(0, 100)
            water_rate = np.clip(water_rate, 500, 2000)
            gas_rate = 0
            pressure = 2500 + np.random.normal(0, 100)
            pressure = np.clip(pressure, 2000, 3500)
        
        sensor = SensorData(
            timestamp=timestamp,
            well_name=well.name,
            pressure=pressure,
            temperature=150 + np.random.normal(0, 10),
            oil_rate=oil_rate,
            water_rate=water_rate,
            gas_rate=gas_rate,
            water_cut=water_cut,
            choke_size=0.5,
            bhp=pressure * 0.9
        )
        sensor_data.append(sensor)
    
    return sensor_data

def run_digital_twin_demo():
    """Run a demonstration of the digital twin with your actual data"""
    print("\n" + "="*70)
    print("DIGITAL TWIN DEMONSTRATION WITH YOUR ACTUAL DATA")
    print("="*70)
    
    # Create configuration with conservative settings
    config = DigitalTwinConfig(
        data_directory="data",
        update_frequency=60,
        history_window_days=7,
        prediction_horizon_days=30,
        physics_weight=0.9,  # Heavy on physics
        ml_weight=0.1,  # Light on ML
        anomaly_threshold=2.0
    )
    
    # Initialize digital twin
    digital_twin = ReservoirDigitalTwin(config)
    
    # Run simulation for 24 hours
    print("\nRunning simulation for 24 hours...")
    
    start_time = datetime.now()
    for hour in range(24):
        current_time = start_time + timedelta(hours=hour)
        
        # Generate realistic sensor data
        sensor_data = generate_synthetic_sensor_data(digital_twin.wells, current_time)
        
        # Update digital twin
        digital_twin.update(sensor_data)
        
        # Print progress
        if (hour + 1) % 6 == 0:
            state = digital_twin.current_state.get('reservoir_state', 'unknown')
            oil_rate = digital_twin.current_state.get('total_oil_rate', 0)
            print(f"  Hour {hour+1}: State={state}, Oil={oil_rate:.0f} bpd")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    digital_twin.visualize()
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    digital_twin.save_report_to_file()
    
    # Print summary
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print(f"Total updates: {digital_twin.performance_metrics['update_count']}")
    print(f"Anomalies detected: {digital_twin.performance_metrics['anomalies_detected']}")
    print(f"Data quality score: {digital_twin.performance_metrics['data_quality']['completeness_score']:.1%}")
    
    oil_rate = digital_twin.current_state.get('total_oil_rate', 0)
    print(f"Current reservoir state: {digital_twin.current_state.get('reservoir_state', 'unknown')}")
    print(f"Current oil production: {oil_rate:.0f} bpd")
    print(f"Results saved to: {digital_twin.results_dir}")
    print("="*70)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Check for required data files
    required_files = ["PERMVALUES.DATA", "SPE9.DATA"]
    missing_files = []
    
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing data files: {missing_files}")
        print("Please place your data files in the 'data' directory:")
        print("  - PERMVALUES.DATA (your permeability values)")
        print("  - SPE9.DATA (SPE9 reservoir model)")
        
        # Create sample data files if they don't exist
        if not (data_dir / "PERMVALUES.DATA").exists():
            print("\nCreating sample PERMVALUES.DATA for demonstration...")
            sample_perm = np.random.lognormal(4, 0.5, 10000)
            with open(data_dir / "PERMVALUES.DATA", 'w') as f:
                for i in range(0, len(sample_perm), 10):
                    f.write(" ".join(f"{val:.1f}" for val in sample_perm[i:i+10]) + "\n")
        
        if not (data_dir / "SPE9.DATA").exists():
            print("Creating sample SPE9.DATA for demonstration...")
            sample_spe9 = """RUNSPEC
TITLE
SPE9 EXAMPLE PROBLEM
DIMENS
24 25 15 /
GRID
DX
900*100 /
DY
900*100 /
DZ
900*50 /
TOPS
600*8000 /
PORO
EQUALS
  0.18 1 24 1 25 1 15 /
/
PERMX
EQUALS
  50.0 1 24 1 25 1 15 /
/
PROPS
SWOF
0.2 0 1 0
0.3 0.01 0.9 0
0.4 0.05 0.7 0
0.5 0.1 0.5 0
0.6 0.2 0.3 0
0.7 0.4 0.1 0
0.8 0.7 0 0
1.0 1.0 0 0 /
ROCK
  3.0E-6 /
SOLUTION
EQUIL
  8000 3500 8500 0 8000 0 /
RPTSOL
  PRESSURE SWAT SGAS /
SUMMARY
FOPR
FWPR
FGPR
/
SCHEDULE
RPTSCHED
  WELLS=2 FIP=2 /
TSTEP
  10*30 /
END"""
            with open(data_dir / "SPE9.DATA", 'w') as f:
                f.write(sample_spe9)
    else:
        print("✓ All required data files found")
    
    # Run the digital twin demonstration
    try:
        run_digital_twin_demo()
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
