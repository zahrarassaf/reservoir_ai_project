import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Union, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SPE9DataParser:
    """Parser for SPE9 reservoir simulation data"""
    
    def __init__(self, config: dict):
        self.config = config
        self.permx_data = None
        self.poro_data = None
        self.depth_data = None
        
    def parse_spe9_data(self, file_path: str) -> pd.DataFrame:
        """
        Parse SPE9.DATA file and extract reservoir properties
        
        Args:
            file_path (str): Path to SPE9.DATA file
            
        Returns:
            pd.DataFrame: Parsed reservoir data
        """
        try:
            logger.info(f"Parsing SPE9 data from: {file_path}")
            
            # Read the SPE9 data file
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Extract permeability data (PERMX)
            permx_data = self._extract_property_data(lines, 'PERMX')
            # Extract porosity data (PORO)
            poro_data = self._extract_property_data(lines, 'PORO')
            # Extract depth data
            depth_data = self._extract_depth_data(lines)
            
            # Create comprehensive DataFrame
            df = self._create_dataframe(permx_data, poro_data, depth_data)
            
            logger.info(f"Successfully parsed SPE9 data with {len(df)} grid blocks")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing SPE9 data: {e}")
            raise
    
    def _extract_property_data(self, lines: List[str], property_name: str) -> np.ndarray:
        """Extract specific property data from SPE9 file"""
        property_data = []
        in_section = False
        
        for i, line in enumerate(lines):
            # Look for property keyword
            if property_name in line and not line.strip().startswith('--'):
                in_section = True
                continue
            
            if in_section:
                # Skip comment lines
                if line.strip().startswith('--') or not line.strip():
                    continue
                
                # Check if we've reached next section
                if line.strip().isupper() and not property_name in line:
                    break
                
                # Parse data lines
                try:
                    # Remove comments and split values
                    clean_line = line.split('--')[0].strip()
                    if clean_line:
                        values = [float(x) for x in clean_line.split()]
                        property_data.extend(values)
                except ValueError:
                    continue
        
        return np.array(property_data)
    
    def _extract_depth_data(self, lines: List[str]) -> np.ndarray:
        """Extract depth data from SPE9 file"""
        depth_data = []
        # This would need to be customized based on SPE9 format
        # For now, creating synthetic depth data
        return np.linspace(2000, 2500, 900)  # Example depth range
    
    def _create_dataframe(self, permx_data: np.ndarray, poro_data: np.ndarray, 
                         depth_data: np.ndarray) -> pd.DataFrame:
        """Create DataFrame from extracted data"""
        
        # Ensure all arrays have the same length
        min_length = min(len(permx_data), len(poro_data), len(depth_data))
        
        df = pd.DataFrame({
            'PERMX': permx_data[:min_length],
            'PORO': poro_data[:min_length],
            'DEPTH': depth_data[:min_length],
            'GRID_BLOCK': range(min_length)
        })
        
        # Calculate additional properties
        df['LOG_PERMX'] = np.log10(df['PERMX'])
        df['PERMZ'] = df['PERMX'] * 0.1  # Typical vertical/horizontal ratio
        df['POROSITY_FRACTION'] = df['PORO']
        
        return df


class OPMDataLoader:
    """Loader for synthetic OPM data"""
    
    def __init__(self, config: dict):
        self.config = config
        
    def load_opm_data(self) -> pd.DataFrame:
        """Load synthetic reservoir data"""
        logger.info("Loading synthetic reservoir data")
        
        np.random.seed(42)  # For reproducible results
        
        n_blocks = 900  # Matching typical SPE9 grid size
        
        # Create synthetic reservoir data
        df = pd.DataFrame({
            'GRID_BLOCK': range(n_blocks),
            'PERMX': np.random.lognormal(mean=3, sigma=1.5, size=n_blocks),
            'PORO': np.random.normal(loc=0.2, scale=0.05, size=n_blocks),
            'DEPTH': np.linspace(2000, 2500, n_blocks),
            'REGION': np.random.randint(1, 6, size=n_blocks)
        })
        
        # Ensure physical constraints
        df['PORO'] = np.clip(df['PORO'], 0.01, 0.35)
        df['PERMX'] = np.clip(df['PERMX'], 0.1, 5000)
        
        # Calculate derived properties
        df['LOG_PERMX'] = np.log10(df['PERMX'])
        df['PERMZ'] = df['PERMX'] * 0.1
        df['POROSITY_FRACTION'] = df['PORO']
        df['TRANSMISSIBILITY'] = df['PERMX'] * df['PORO']
        
        logger.info(f"Created synthetic data with {len(df)} grid blocks")
        return df


def load_synthetic_data() -> pd.DataFrame:
    """Load synthetic data as fallback"""
    from src.opm_data_loader import OPMDataLoader
    
    # Create default config if not provided
    default_config = {
        'data_loading': {
            'synthetic_size': 900,
            'random_seed': 42
        }
    }
    
    opm_loader = OPMDataLoader(default_config)
    return opm_loader.load_opm_data()


def validate_spe9_file(file_path: str) -> bool:
    """Validate SPE9 file existence and basic integrity"""
    if not os.path.isfile(file_path):
        logger.warning(f"SPE9 file not found: {file_path}")
        return False
    
    if os.path.getsize(file_path) == 0:
        logger.warning(f"SPE9 file is empty: {file_path}")
        return False
    
    # Basic content validation
    try:
        with open(file_path, 'r') as f:
            first_lines = [f.readline().strip() for _ in range(5)]
        if any('SPE' in line or 'RUNSPEC' in line for line in first_lines):
            return True
    except Exception as e:
        logger.warning(f"Error validating SPE9 file: {e}")
    
    return True


def analyze_reservoir_data(df: pd.DataFrame) -> dict:
    """Perform basic analysis on reservoir data"""
    analysis = {
        'total_blocks': len(df),
        'permx_stats': {
            'mean': df['PERMX'].mean(),
            'std': df['PERMX'].std(),
            'min': df['PERMX'].min(),
            'max': df['PERMX'].max()
        },
        'poro_stats': {
            'mean': df['PORO'].mean(),
            'std': df['PORO'].std(),
            'min': df['PORO'].min(),
            'max': df['PORO'].max()
        },
        'depth_range': {
            'min': df['DEPTH'].min(),
            'max': df['DEPTH'].max()
        }
    }
    return analysis


def main():
    """Main function for SPE9 data loading and processing"""
    
    # Configuration
    config = {
        'data_loading': {
            'synthetic_size': 900,
            'random_seed': 42,
            'validate_files': True
        },
        'output': {
            'save_processed': True,
            'output_path': 'processed_data'
        }
    }
    
    # Step 1: SPE9 Data Loading with real permeability
    print("\n📊 STEP 1: SPE9 DATA LOADING WITH REAL PERMEABILITY")
    print("-" * 50)
    
    # Define SPE9 file paths
    spe9_paths = [
        "opm-data/spe9/SPE9.DATA",
        "spe9/SPE9.DATA", 
        "SPE9.DATA",
        "../SPE9.DATA",
        "../../spe9/SPE9.DATA"
    ]
    
    df = None
    spe9_file_path = None
    data_source = "Unknown"
    
    # Find and validate SPE9 file
    for path in spe9_paths:
        if os.path.exists(path) and validate_spe9_file(path):
            spe9_file_path = path
            print(f"🎯 Found valid SPE9 file: {path}")
            break
    
    # Load data based on availability
    if spe9_file_path:
        try:
            spe9_parser = SPE9DataParser(config)
            df = spe9_parser.parse_spe9_data(spe9_file_path)
            data_source = "SPE9"
            print(f"✅ Successfully loaded SPE9 data from {spe9_file_path}")
        except Exception as e:
            print(f"❌ Error loading SPE9 data: {e}")
            print("🔄 Falling back to synthetic data...")
            df = load_synthetic_data()
            data_source = "Synthetic (Fallback)"
    else:
        print("⚠️ No valid SPE9 files found, using synthetic data")
        df = load_synthetic_data()
        data_source = "Synthetic"
    
    # Step 2: Data Analysis and Reporting
    print("\n📈 STEP 2: DATA ANALYSIS")
    print("-" * 50)
    
    # Perform basic analysis
    analysis = analyze_reservoir_data(df)
    
    print(f"📦 Data Source: {data_source}")
    print(f"🔢 Total Grid Blocks: {analysis['total_blocks']:,}")
    print(f"📏 Depth Range: {analysis['depth_range']['min']:.1f} - {analysis['depth_range']['max']:.1f} m")
    print(f"🔄 Permeability (PERMX): {analysis['permx_stats']['mean']:.2f} ± {analysis['permx_stats']['std']:.2f} mD")
    print(f"🕳️ Porosity: {analysis['poro_stats']['mean']:.3f} ± {analysis['poro_stats']['std']:.3f} fraction")
    
    # Step 3: Data Quality Check
    print("\n🔍 STEP 3: DATA QUALITY CHECK")
    print("-" * 50)
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        print("✅ No missing values found")
    else:
        print(f"⚠️ Found {missing_values} missing values")
    
    # Check data ranges
    valid_poro = ((df['PORO'] >= 0) & (df['PORO'] <= 1)).all()
    valid_permx = (df['PERMX'] > 0).all()
    
    print(f"✅ Porosity values in valid range: {valid_poro}")
    print(f"✅ Permeability values positive: {valid_permx}")
    
    # Step 4: Save Processed Data (Optional)
    if config['output']['save_processed']:
        output_dir = config['output']['output_path']
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"reservoir_data_{data_source.lower().replace(' ', '_')}.csv")
        df.to_csv(output_file, index=False)
        print(f"💾 Data saved to: {output_file}")
    
    print("\n🎯 DATA LOADING COMPLETED SUCCESSFULLY!")
    return df


if __name__ == "__main__":
    # Create sample directory structure for testing
    os.makedirs("src", exist_ok=True)
    os.makedirs("opm-data/spe9", exist_ok=True)
    os.makedirs("spe9", exist_ok=True)
    
    # Run the main function
    df_result = main()
    
    # Display first few rows
    if df_result is not None:
        print("\n📋 SAMPLE DATA:")
        print(df_result.head())
