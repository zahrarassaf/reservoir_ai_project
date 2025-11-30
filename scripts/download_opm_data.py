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
            print("✅ OPM data downloaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download OPM data: {e}")
            sys.exit(1)
    else:
        print("✅ OPM data already exists")
    
    # Verify SPE9 case exists
    spe9_dir = opm_dir / "spe9"
    if spe9_dir.exists():
        print(f"✅ SPE9 case found at {spe9_dir}")
        required_files = ["SPE9.GRID", "SPE9.INIT", "SPE9.DATA", "SPE9.UNRST"]
        for file in required_files:
            if (spe9_dir / file).exists():
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} - MISSING")
    else:
        print(f"❌ SPE9 case not found at {spe9_dir}")

if __name__ == "__main__":
    download_opm_data()
