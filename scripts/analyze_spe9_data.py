
import tarfile
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def extract_and_analyze_spe9(data_path: str):
    """Extract and analyze the SPE9 dataset."""
    print("🔍 Analyzing SPE9 dataset structure...")
    
    # Extract if tar.gz
    if data_path.endswith('.tar.gz'):
        with tarfile.open(data_path, 'r:gz') as tar:
            tar.extractall(path='data/spe9_raw')
            data_dir = 'data/spe9_raw'
    else:
        data_dir = data_path
    
    data_dir = Path(data_dir)
    
    # List all files
    files = list(data_dir.rglob('*'))
    print(f"\n📁 Found {len(files)} files:")
    
    file_types = {}
    for file in files:
        ext = file.suffix
        file_types[ext] = file_types.get(ext, 0) + 1
    
    for ext, count in sorted(file_types.items()):
        print(f"  {ext}: {count} files")
    
    # Check for key SPE9 files
    key_files = {
        '.DATA': 'Main simulation deck',
        '.GRID': 'Grid geometry',
        '.INIT': 'Initial conditions',
        '.UNRST': 'Restart/unified results',
        '.SMSPEC': 'Summary specifications',
        '.ESMRY': 'Summary results'
    }
    
    print("\n🔑 Looking for key SPE9 files:")
    for file in files:
        if file.suffix in key_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  ✓ {file.name} ({size_mb:.2f} MB) - {key_files[file.suffix]}")
    
    return data_dir, files

if __name__ == "__main__":
    # Update this path to your downloaded file
    data_path = "spe9_dataset.tar.gz"  # یا مسیر فایل شما
    data_dir, files = extract_and_analyze_spe9(data_path)
