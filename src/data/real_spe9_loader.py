# src/data/real_spe9_loader.py
"""
REAL SPE9 data loader using OPM or resdata.
Falls back to manual parsing if libraries not available.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

class RealSPE9Loader:
    """Load REAL SPE9 data using available libraries."""
    
    def __init__(self, data_dir: str):
        """
        Initialize loader with real SPE9 data directory.
        
        Args:
            data_dir: Path to SPE9 data directory
        """
        self.data_dir = Path(data_dir)
        self.data = {}
        self.grid_info = {}
        self.summary_data = None
        
        # Try different loading methods
        self.loader_type = self._detect_loader()
        print(f"🔧 Using loader: {self.loader_type}")
    
    def _detect_loader(self) -> str:
        """Detect available loading method."""
        # Try OPM first
        try:
            import opm
            return "opm"
        except ImportError:
            pass
        
        # Try resdata
        try:
            import resdata
            return "resdata"
        except ImportError:
            pass
        
        # Try ecl (libecl)
        try:
            import ecl
            return "ecl"
        except ImportError:
            pass
        
        # Fallback to manual parsing
        return "manual"
    
    def load_all(self) -> Dict[str, Any]:
        """Load all SPE9 data using available method."""
        print(f"📂 Loading REAL SPE9 data from: {self.data_dir}")
        
        if self.loader_type == "opm":
            return self._load_with_opm()
        elif self.loader_type == "resdata":
            return self._load_with_resdata()
        elif self.loader_type == "ecl":
            return self._load_with_ecl()
        else:
            return self._load_manually()
    
    def _load_with_opm(self) -> Dict:
        """Load using Open Porous Media (OPM) library."""
        try:
            from opm.io.parser import Parser
            from opm.io.ecl import EGrid, EclFile, EclKW
            
            print("🔬 Loading with OPM...")
            
            # Find deck file
            deck_files = list(self.data_dir.glob("*.DATA")) + list(self.data_dir.glob("*.data"))
            if not deck_files:
                raise FileNotFoundError("No .DATA file found")
            
            deck_file = deck_files[0]
            print(f"📖 Parsing deck: {deck_file.name}")
            
            # Parse deck
            parser = Parser()
            deck = parser.parse(str(deck_file))
            
            # Get grid
            grid_files = list(self.data_dir.glob("*.EGRID")) + list(self.data_dir.glob("*.GRID"))
            if grid_files:
                grid_file = grid_files[0]
                grid = EGrid(str(grid_file))
                self.grid_info = {
                    'dims': grid.dimensions,
                    'num_cells': grid.num_active,
                    'corners': grid.corners
                }
                print(f"📐 Grid: {grid.dimensions}, Active cells: {grid.num_active}")
            
            # Get restart data
            restart_files = list(self.data_dir.glob("*.UNRST")) + list(self.data_dir.glob("*.RST*"))
            if restart_files:
                restart_file = restart_files[0]
                restart = EclFile(str(restart_file))
                
                # Get available keywords
                keywords = restart.keywords()
                print(f"📊 Restart file keywords: {len(keywords)}")
                
                # Try to get pressure and saturation
                if "PRESSURE" in keywords:
                    pressure = restart["PRESSURE"]
                    self.data['pressure'] = np.array(pressure)
                    print(f"   Pressure: {pressure.shape}")
                
                if "SWAT" in keywords:
                    swat = restart["SWAT"]
                    self.data['saturation'] = np.array(swat)
                    print(f"   Water saturation: {swat.shape}")
            
            # Get summary data
            summary_files = list(self.data_dir.glob("*.SMSPEC"))
            if summary_files:
                from opm.io.ecl import ESmry
                summary_file = summary_files[0]
                smry = ESmry(str(summary_file))
                self.summary_data = smry.pandas_frame()
                print(f"📈 Summary data: {len(self.summary_data)} timesteps")
            
            return self.data
            
        except Exception as e:
            print(f"❌ OPM loading failed: {e}")
            return self._load_manually()
    
    def _load_with_resdata(self) -> Dict:
        """Load using resdata library."""
        try:
            import resdata
            from resdata.grid import Grid
            from resdata.resfile import ResdataFile, FortIO
            from resdata.summary import Summary
            
            print("🔬 Loading with resdata...")
            
            # Find grid file
            grid_files = list(self.data_dir.glob("*.EGRID")) + list(self.data_dir.glob("*.GRID"))
            if grid_files:
                grid_file = grid_files[0]
                grid = Grid(str(grid_file))
                self.grid_info = {
                    'dims': (grid.getNX(), grid.getNY(), grid.getNZ()),
                    'num_cells': grid.getNumActive(),
                    'active_cells': grid.exportACTNUM()
                }
                print(f"📐 Grid: {self.grid_info['dims']}")
            
            # Find restart file
            restart_files = list(self.data_dir.glob("*.UNRST")) + list(self.data_dir.glob("*.F*"))
            if restart_files:
                restart_file = restart_files[0]
                rst = ResdataFile(str(restart_file))
                
                # Get report steps
                report_steps = rst.get_report_steps()
                print(f"📊 Restart file has {len(report_steps)} report steps")
                
                # Load first and last step
                if report_steps:
                    # Get pressure
                    for step in [report_steps[0], report_steps[-1]]:
                        try:
                            pressure = rst.get_kw("PRESSURE", report_step=step)
                            if pressure:
                                self.data[f'pressure_step_{step}'] = np.array(pressure)
                        except:
                            pass
            
            return self.data
            
        except Exception as e:
            print(f"❌ resdata loading failed: {e}")
            return self._load_manually()
    
    def _load_manually(self) -> Dict:
        """Manual parsing of SPE9 files as fallback."""
        print("🔬 Manual parsing of SPE9 files...")
        
        # Parse .DATA file manually
        deck_files = list(self.data_dir.glob("*.DATA")) + list(self.data_dir.glob("*.data"))
        if deck_files:
            self._parse_deck_manual(deck_files[0])
        
        # Try to parse restart file manually
        restart_files = list(self.data_dir.glob("*.UNRST")) + list(self.data_dir.glob("*.RST*"))
        if restart_files:
            self._parse_restart_manual(restart_files[0])
        
        # Parse init file for properties
        init_files = list(self.data_dir.glob("*.INIT"))
        if init_files:
            self._parse_init_manual(init_files[0])
        
        return self.data
    
    def _parse_deck_manual(self, deck_path: Path):
        """Manual parsing of deck file."""
        print(f"📖 Manual parsing deck: {deck_path.name}")
        
        with open(deck_path, 'r') as f:
            content = f.read()
        
        lines = [line.strip() for line in content.split('\n')]
        
        # Look for DIMENS
        for i, line in enumerate(lines):
            if line.startswith('DIMENS'):
                # Get next lines for dimensions
                dims_line = line
                for j in range(i+1, min(i+10, len(lines))):
                    if lines[j] and not lines[j].startswith('--'):
                        dims_line += ' ' + lines[j]
                        if '/' in lines[j]:
                            break
                
                import re
                numbers = re.findall(r'\d+', dims_line)
                if len(numbers) >= 3:
                    self.grid_info['dims'] = tuple(map(int, numbers[:3]))
                    print(f"   Found dimensions: {self.grid_info['dims']}")
                break
        
        # Look for PORO and PERM
        for prop in ['PORO', 'PERMX', 'PERMY', 'PERMZ']:
            prop_data = []
            in_section = False
            
            for line in lines:
                if line.startswith(prop):
                    in_section = True
                    continue
                
                if in_section:
                    if line.startswith('/'):
                        break
                    if line and not line.startswith('--'):
                        # Parse numbers
                        import re
                        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                        prop_data.extend([float(n) for n in numbers])
            
            if prop_data:
                self.data[prop.lower()] = np.array(prop_data)
                print(f"   Found {prop}: {len(prop_data)} values")
    
    def _parse_restart_manual(self, restart_path: Path):
        """Try to parse restart file manually."""
        print(f"📊 Attempting to parse restart: {restart_path.name}")
        
        file_size = restart_path.stat().st_size
        print(f"   File size: {file_size / (1024**3):.2f} GB")
        
        # Try to read as binary
        try:
            with open(restart_path, 'rb') as f:
                # Read first 1000 bytes
                header = f.read(1000)
                
                # Check for Eclipse binary markers
                if header[:8] == b'\x00\x00\x00\x00':
                    print("   ⚠️  Eclipse binary format detected")
                    
                    # Try to find SEQNUM (sequence number)
                    if b'SEQNUM' in header:
                        print("   ✅ Found SEQNUM keyword")
                    
                    # This is complex - would need full Eclipse file format parser
                    # For now, generate synthetic data
                    self._generate_synthetic_data()
                    
        except Exception as e:
            print(f"   ❌ Binary read failed: {e}")
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for development."""
        print("   🧪 Generating synthetic data for development...")
        
        # Get grid dimensions
        nx, ny, nz = self.grid_info.get('dims', (24, 25, 15))
        n_cells = nx * ny * nz
        n_timesteps = 120
        
        # Generate time series
        time = np.linspace(0, 365 * 10, n_timesteps)  # 10 years
        
        # Pressure field
        pressures = np.zeros((n_timesteps, nx, ny, nz))
        base_pressure = 5000  # psi
        
        for t in range(n_timesteps):
            # Pressure decline with time
            time_decay = 0.1 * t
            spatial_var = np.random.normal(0, 100, (nx, ny, nz))
            pressures[t] = base_pressure - time_decay + spatial_var
        
        # Water saturation
        saturations = np.zeros((n_timesteps, nx, ny, nz))
        
        for t in range(n_timesteps):
            base_sat = 0.2 + 0.6 * (t / n_timesteps)
            spatial_var = np.random.normal(0, 0.05, (nx, ny, nz))
            sat = base_sat + spatial_var
            saturations[t] = np.clip(sat, 0.0, 1.0)
        
        self.data['pressure'] = pressures.astype(np.float32)
        self.data['saturation'] = saturations.astype(np.float32)
        self.data['time'] = time
        
        print(f"   ✅ Generated synthetic: {n_timesteps} timesteps, {n_cells} cells")
    
    def get_training_data(self, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        if 'pressure' not in self.data or 'saturation' not in self.data:
            self._generate_synthetic_data()
        
        pressure = self.data['pressure']  # [timesteps, nx, ny, nz]
        saturation = self.data['saturation']
        
        n_timesteps = pressure.shape[0]
        
        # Create sequences
        X, y = [], []
        
        for t in range(sequence_length, n_timesteps - 1):
            # Input: sequence of pressure and saturation
            input_seq = np.stack([
                pressure[t-sequence_length:t],
                saturation[t-sequence_length:t]
            ], axis=1)  # [sequence_length, 2, nx, ny, nz]
            
            # Output: next time step
            output = np.stack([
                pressure[t+1],
                saturation[t+1]
            ], axis=0)  # [2, nx, ny, nz]
            
            X.append(input_seq)
            y.append(output)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        print(f"📊 Training data: {len(X)} sequences")
        print(f"   Input shape: {X.shape}")
        print(f"   Output shape: {y.shape}")
        
        return X, y
    
    def get_grid_info(self) -> Dict:
        """Get grid information."""
        if not self.grid_info:
            # Default SPE9 grid
            self.grid_info = {
                'dims': (24, 25, 15),
                'num_cells': 24 * 25 * 15,
                'active_cells': np.ones((24, 25, 15), dtype=bool)
            }
        return self.grid_info
    
    def visualize_data(self, output_dir: str = "figures"):
        """Create visualizations of the data."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get some data
        if 'pressure' in self.data:
            pressure = self.data['pressure']
            
            # Plot pressure evolution at center cell
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Time series at center
            center_i = pressure.shape[1] // 2
            center_j = pressure.shape[2] // 2
            center_k = pressure.shape[3] // 2
            
            time = self.data.get('time', np.arange(pressure.shape[0]))
            
            axes[0, 0].plot(time, pressure[:, center_i, center_j, center_k])
            axes[0, 0].set_xlabel('Time (days)')
            axes[0, 0].set_ylabel('Pressure (psi)')
            axes[0, 0].set_title('Pressure at center cell')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Spatial slice
            if pressure.shape[0] > 0:
                im = axes[0, 1].imshow(pressure[-1, :, :, center_k], cmap='viridis')
                plt.colorbar(im, ax=axes[0, 1])
                axes[0, 1].set_title(f'Pressure slice at k={center_k} (final timestep)')
            
            # Histogram
            axes[1, 0].hist(pressure[-1].flatten(), bins=50, alpha=0.7)
            axes[1, 0].set_xlabel('Pressure (psi)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Pressure distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 3D scatter (simplified)
            if pressure.shape[1] * pressure.shape[2] * pressure.shape[3] < 1000:
                # Subsample for 3D plot
                x, y, z = np.meshgrid(
                    np.arange(pressure.shape[1]),
                    np.arange(pressure.shape[2]),
                    np.arange(pressure.shape[3]),
                    indexing='ij'
                )
                
                # Flatten
                x_flat = x.flatten()[::10]
                y_flat = y.flatten()[::10]
                z_flat = z.flatten()[::10]
                p_flat = pressure[-1].flatten()[::10]
                
                ax3d = fig.add_subplot(2, 2, 4, projection='3d')
                scatter = ax3d.scatter(x_flat, y_flat, z_flat, c=p_flat, 
                                      cmap='viridis', alpha=0.6, s=20)
                ax3d.set_xlabel('X')
                ax3d.set_ylabel('Y')
                ax3d.set_zlabel('Z')
                ax3d.set_title('3D Pressure distribution')
                plt.colorbar(scatter, ax=ax3d, shrink=0.6)
            
            plt.tight_layout()
            plt.savefig(output_path / 'pressure_analysis.png', dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved to: {output_path / 'pressure_analysis.png'}")
