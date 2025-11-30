import subprocess
import sys
from pathlib import Path

def download_opm_data():
    opm_dir = Path("../../opm-data")
    
    if not opm_dir.exists():
        print("Cloning OPM data repository...")
        try:
            subprocess.run([
                "git", "clone", 
                "https://github.com/OPM/opm-data.git",
                str(opm_dir)
            ], check=True)
            print("OPM data downloaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download OPM data: {e}")
            sys.exit(1)
    else:
        print("OPM data already exists")

if __name__ == "__main__":
    download_opm_data()
