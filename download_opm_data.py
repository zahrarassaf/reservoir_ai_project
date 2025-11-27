#!/usr/bin/env python3
"""
AUTOMATIC OPM DATA DOWNLOADER
"""
import requests
import zipfile
import os
from pathlib import Path
import sys

def download_opm_data():
    """Download real OPM data from GitHub"""
    print("🌐 DOWNLOADING REAL OPM DATA...")
    
    opm_url = "https://github.com/OPM/opm-data/archive/refs/heads/master.zip"
    download_path = Path("opm-data.zip")
    extract_path = Path("opm-data")
    
    try:
        # Download zip file
        print("📥 Downloading OPM data...")
        response = requests.get(opm_url, stream=True)
        response.raise_for_status()
        
        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract zip file
        print("📦 Extracting files...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Check successful extraction
        spe9_path = extract_path / "opm-data-master" / "spe9"
        if spe9_path.exists():
            files = list(spe9_path.glob("*"))
            print(f"✅ OPM DATA DOWNLOADED: {len(files)} files in spe9/")
            for file in files[:5]:
                print(f"   📄 {file.name}")
            return True
        else:
            print("❌ SPE9 folder not found in downloaded data")
            return False
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    download_opm_data()
