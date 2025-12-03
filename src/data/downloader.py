"""
Data downloader for SPE9 benchmark dataset.
"""

import requests
import tarfile
import zipfile
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SPE9DataDownloader:
    """Downloader for SPE9 benchmark dataset."""
    
    BASE_URL = "https://github.com/OPM/opm-data/raw/master"
    SPE9_FILES = {
        "data": "spe9/SPE9_CP.DATA",
        "summary": "spe9/SPE9_SUMMARY.DATA",
        "include": "spe9/SPE9_INCLUDE.DATA",
    }
    
    def __init__(self, data_dir: str = "data/spe9"):
        """
        Initialize downloader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
    
    def download_file(self, url: str, filename: str) -> bool:
        """
        Download a single file with progress bar.
        
        Args:
            url: File URL
            filename: Local filename
            
        Returns:
            True if successful
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f, tqdm(
                desc=Path(filename).name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def download_all(self, force: bool = False) -> Dict[str, bool]:
        """
        Download all SPE9 files.
        
        Args:
            force: Force re-download even if files exist
            
        Returns:
            Dictionary of download results
        """
        results = {}
        
        for file_type, file_path in self.SPE9_FILES.items():
            url = f"{self.BASE_URL}/{file_path}"
            local_path = self.raw_dir / Path(file_path).name
            
            if local_path.exists() and not force:
                logger.info(f"File already exists: {local_path}")
                results[file_type] = True
                continue
            
            logger.info(f"Downloading {file_type} from {url}")
            success = self.download_file(url, local_path)
            results[file_type] = success
            
            if success:
                logger.info(f"Downloaded {local_path}")
            else:
                logger.error(f"Failed to download {file_type}")
        
        return results
    
    def validate_files(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate downloaded files.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        for file_type, file_path in self.SPE9_FILES.items():
            local_path = self.raw_dir / Path(file_path).name
            
            result = {
                'exists': local_path.exists(),
                'size': local_path.stat().st_size if local_path.exists() else 0,
                'valid': False
            }
            
            if result['exists']:
                # Basic validation - check file size and content
                if result['size'] > 0:
                    try:
                        with open(local_path, 'r') as f:
                            first_line = f.readline()
                            result['valid'] = True
                            result['first_line'] = first_line.strip()
                    except:
                        result['valid'] = False
            
            validation_results[file_type] = result
        
        return validation_results
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        # Remove processed directory
        if self.processed_dir.exists():
            shutil.rmtree(self.processed_dir)
        
        # Recreate empty directory
        self.processed_dir.mkdir(exist_ok=True)
        
        logger.info("Cleaned up processed data")
