import numpy as np
from pathlib import Path
import sys

def generate_missing_spe9_files():
    """Generate missing SPE9 files for development"""
    opm_dir = Path("../../opm-data/spe9")
    opm_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate SPE9.GRID file
    grid_content = """GDFILE
SPE9 GRID FILE

DIMENS
  24 25 15 /

GRIDUNIT
  METRES /

COORD
  9000*0.0 /

ZCORN
  27000*0.0 /

ACTNUM
  9000*1 /

PORO
  9000*0.2 /

PERMX
  9000*100.0 /

PERMY
  9000*100.0 /

PERMZ
  9000*10.0 /

ENDFILE
"""
    
    # Generate SPE9.INIT file
    init_content = """INIT
SPE9 INITIALIZATION FILE

PORO
  9000*0.2 /

PERMX
  9000*100.0 /

PERMY  
  9000*100.0 /

PERMZ
  9000*10.0 /

SATNUM
  9000*1 /

PVTNUM
  9000*1 /

ENDFILE
"""
    
    # Write files
    with open(opm_dir / "SPE9.GRID", "w") as f:
        f.write(grid_content)
    
    with open(opm_dir / "SPE9.INIT", "w") as f:
        f.write(init_content)
        
    # Create empty UNRST file
    with open(opm_dir / "SPE9.UNRST", "w") as f:
        f.write("UNRST\nENDFILE\n")
    
    print("✅ Generated missing SPE9 files for development")
    print("📁 Files created:")
    print("   - SPE9.GRID")
    print("   - SPE9.INIT") 
    print("   - SPE9.UNRST")

if __name__ == "__main__":
    generate_missing_spe9_files()
