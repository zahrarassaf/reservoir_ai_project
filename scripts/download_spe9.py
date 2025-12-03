# scripts/download_spe9.py
"""
Download real SPE9 benchmark dataset.
"""

import requests
import tarfile
import zipfile
import os
import shutil
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SPE9Downloader:
    """Download SPE9 benchmark dataset from official sources."""
    
    # Official OPM repository
    OPM_REPO = "https://github.com/OPM/opm-data"
    SPE9_DATA_URL = f"{OPM_REPO}/raw/master/spe9/SPE9_CP.DATA"
    
    # Alternative sources
    ALTERNATIVE_URLS = [
        "https://www.spe.org/web/csp/datasets/spe9.zip",
        "https://data.norge.no/datasets/opm/spe9",
    ]
    
    def __init__(self, download_dir: Path = Path("data/spe9")):
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.raw_dir = self.download_dir / "raw"
        self.processed_dir = self.download_dir / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
    
    def download_from_opm(self) -> bool:
        """Download from OPM GitHub repository."""
        try:
            logger.info(f"Downloading SPE9 from OPM: {self.SPE9_DATA_URL}")
            
            response = requests.get(self.SPE9_DATA_URL, stream=True, timeout=30)
            response.raise_for_status()
            
            output_file = self.raw_dir / "SPE9_CP.DATA"
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Downloaded to {output_file}")
            
            # Download additional files if available
            self._download_additional_files()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download from OPM: {e}")
            return False
    
    def _download_additional_files(self):
        """Download additional SPE9 files."""
        additional_files = [
            "SPE9.INIT",
            "SPE9.UNSMRY", 
            "SPE9.SMSPEC",
            "SPE9.EGRID",
            "SPE9.ECLEND",
        ]
        
        base_url = "https://github.com/OPM/opm-data/raw/master/spe9/"
        
        for filename in additional_files:
            try:
                url = f"{base_url}{filename}"
                response = requests.get(url, stream=True, timeout=30)
                
                if response.status_code == 200:
                    output_file = self.raw_dir / filename
                    
                    with open(output_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    logger.info(f"Downloaded {filename}")
                    
            except Exception as e:
                logger.debug(f"Could not download {filename}: {e}")
    
    def download_from_spe(self) -> bool:
        """Download from SPE website (requires authentication)."""
        # Note: SPE website may require login
        try:
            logger.info("Attempting to download from SPE website...")
            
            # This is a placeholder - actual implementation would need
            # to handle SPE website authentication
            
            raise NotImplementedError(
                "SPE website download requires authentication. "
                "Please download manually from: "
                "https://www.spe.org/web/csp/datasets/"
            )
            
        except Exception as e:
            logger.error(f"Failed to download from SPE: {e}")
            return False
    
    def use_local_files(self, local_path: Path) -> bool:
        """Use locally available SPE9 files."""
        if not local_path.exists():
            logger.error(f"Local path does not exist: {local_path}")
            return False
        
        try:
            # Copy files to raw directory
            for file_path in local_path.glob("*"):
                if file_path.is_file():
                    dest_path = self.raw_dir / file_path.name
                    shutil.copy2(file_path, dest_path)
                    logger.info(f"Copied {file_path.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy local files: {e}")
            return False
    
    def verify_download(self) -> Dict[str, bool]:
        """Verify downloaded files."""
        expected_files = {
            'DATA': ['SPE9_CP.DATA', 'SPE9.DATA'],
            'INIT': ['SPE9.INIT'],
            'SUMMARY': ['SPE9.UNSMRY', 'SPE9.SMSPEC'],
            'GRID': ['SPE9.EGRID'],
        }
        
        verification = {}
        
        for file_type, filenames in expected_files.items():
            found = False
            
            for filename in filenames:
                file_path = self.raw_dir / filename
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    
                    # Check file size (basic validation)
                    if file_size > 1000:  # At least 1KB
                        found = True
                        verification[f"{file_type}_file"] = {
                            'filename': filename,
                            'size_bytes': file_size,
                            'path': str(file_path),
                        }
                        break
            
            verification[f"has_{file_type.lower()}"] = found
        
        # Check if we have minimum required files
        verification['has_minimum_data'] = (
            verification.get('has_data', False) and 
            (verification.get('has_summary', False) or verification.get('has_init', False))
        )
        
        return verification
    
    def download(self, force: bool = False, source: str = 'opm') -> bool:
        """
        Download SPE9 dataset.
        
        Args:
            force: Force re-download even if files exist
            source: 'opm', 'spe', or 'local'
            
        Returns:
            True if successful
        """
        # Check if already downloaded
        if not force and self.raw_dir.exists() and any(self.raw_dir.iterdir()):
            logger.info(f"Data already exists in {self.raw_dir}")
            
            verification = self.verify_download()
            if verification.get('has_minimum_data', False):
                logger.info("Existing data verified successfully")
                return True
        
        # Download based on source
        success = False
        
        if source == 'opm':
            success = self.download_from_opm()
        elif source == 'spe':
            success = self.download_from_spe()
        elif source == 'local':
            # Ask user for local path
            local_path = input("Enter path to local SPE9 files: ").strip()
            success = self.use_local_files(Path(local_path))
        else:
            raise ValueError(f"Unknown source: {source}")
        
        if success:
            # Verify download
            verification = self.verify_download()
            
            if verification.get('has_minimum_data', False):
                logger.info("✅ SPE9 dataset downloaded and verified successfully")
                
                # Print verification results
                for key, value in verification.items():
                    if isinstance(value, dict):
                        logger.info(f"  {key}: {value.get('filename')} ({value.get('size_bytes')} bytes)")
                    else:
                        logger.info(f"  {key}: {value}")
                
                return True
            else:
                logger.error("❌ Downloaded files are incomplete")
                return False
        else:
            logger.error("❌ Failed to download SPE9 dataset")
            return False
    
    def create_synthetic_if_missing(self) -> Path:
        """
        Create synthetic SPE9 data if real data is not available.
        Returns path to data directory.
        """
        verification = self.verify_download()
        
        if verification.get('has_minimum_data', False):
            logger.info("Real SPE9 data is available")
            return self.raw_dir
        
        logger.warning("Real SPE9 data not available, creating synthetic data")
        
        # Create synthetic DATA file
        synthetic_data = self._create_synthetic_spe9_data()
        synthetic_file = self.raw_dir / "SYNTHETIC_SPE9.DATA"
        
        with open(synthetic_file, 'w') as f:
            f.write(synthetic_data)
        
        logger.info(f"Created synthetic SPE9 data: {synthetic_file}")
        
        # Also create summary file for compatibility
        summary_file = self.raw_dir / "SYNTHETIC_SPE9_SUMMARY.csv"
        
        from src.data.spe9_parser import SPE9Parser
        parser = SPE9Parser()
        synthetic_df = parser._create_simulated_spe9_data()
        synthetic_df.to_csv(summary_file, index=False)
        
        logger.info(f"Created synthetic summary: {summary_file}")
        
        return self.raw_dir
    
    def _create_synthetic_spe9_data(self) -> str:
        """Create synthetic SPE9 DATA file content."""
        return """
-- Synthetic SPE9 Data File
-- Created for Reservoir AI Project
-- Based on SPE9 Benchmark specifications

RUNSPEC
TITLE
SYNTHETIC SPE9 BENCHMARK
/
DIMENS
 24 25 15 /
OIL
WATER
GAS
DISGAS
METRIC
START
 1 'JAN' 2020 /
WELLDIMS
 9 20 9 9 /
TABDIMS
 1 20 20 1 20 /
/

GRID
INCLUDE
 'SYNTHETIC_GRID.INC' /
/

PROPS
INCLUDE
 'SYNTHETIC_PROPS.INC' /
/

REGIONS
INCLUDE
 'SYNTHETIC_REGIONS.INC' /
/

SOLUTION
EQUIL
 8500 3250 8600 0 8600 0 1 0 1 /
/

SUMMARY
FOPR
FWPR
FGPR
FOPT
FWPT
FGPT
FWIR
FGIR
FWIT
FGIT
WBHP
/

SCHEDULE
RPTSCHED
 'PRES' 'SGAS' 'SWAT' /
/

TSTEP
 365*1 /
/

END
"""
