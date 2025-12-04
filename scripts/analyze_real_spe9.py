# scripts/analyze_real_spe9.py
import tarfile
import gzip
import zipfile
import os
import sys
import requests
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

def download_from_drive(file_id: str, output_path: str):
    """Download file from Google Drive using gdown."""
    try:
        import gdown
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
        print(f"✅ Downloaded to: {output_path}")
        return True
    except ImportError:
        print("❌ Install gdown: pip install gdown")
        return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def analyze_spe9_archive(file_path: str) -> Dict:
    """Analyze the SPE9 archive file structure."""
    print(f"🔍 Analyzing: {file_path}")
    
    file_ext = Path(file_path).suffix.lower()
    file_size = Path(file_path).stat().st_size / (1024**3)  # GB
    print(f"📦 File size: {file_size:.2f} GB")
    
    extracted_files = []
    
    # Try different archive formats
    try:
        if file_ext in ['.tar', '.tar.gz', '.tgz']:
            with tarfile.open(file_path, 'r:*') as tar:
                tar.extractall('data/spe9_raw')
                extracted_files = tar.getnames()
                print(f"📁 Extracted {len(extracted_files)} files from tar")
                
        elif file_ext == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall('data/spe9_raw')
                extracted_files = zip_ref.namelist()
                print(f"📁 Extracted {len(extracted_files)} files from zip")
                
        else:
            # Assume it's a directory or single file
            if Path(file_path).is_dir():
                extracted_files = list(Path(file_path).rglob('*'))
                # Copy to data directory
                import shutil
                shutil.copytree(file_path, 'data/spe9_raw', dirs_exist_ok=True)
            else:
                # Single file
                Path('data/spe9_raw').mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, 'data/spe9_raw/')
                extracted_files = [Path(file_path).name]
    
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return {'error': str(e)}
    
    # Analyze extracted files
    data_dir = Path('data/spe9_raw')
    all_files = list(data_dir.rglob('*'))
    
    analysis = {
        'total_files': len(all_files),
        'file_types': {},
        'eclipse_files': [],
        'sizes': {},
        'structure': []
    }
    
    # Categorize files
    eclipse_extensions = ['.DATA', '.GRID', '.INIT', '.UNRST', 
                         '.SMSPEC', '.ESMRY', '.EGRID', '.INIT',
                         '.RSM', '.RST', '.DBG', '.PRT']
    
    for file_path in all_files:
        if file_path.is_file():
            # File type
            ext = file_path.suffix.upper()
            analysis['file_types'][ext] = analysis['file_types'].get(ext, 0) + 1
            
            # Size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            analysis['sizes'][file_path.name] = size_mb
            
            # Eclipse files
            if ext in eclipse_extensions:
                analysis['eclipse_files'].append({
                    'name': file_path.name,
                    'size_mb': size_mb,
                    'path': str(file_path)
                })
            
            # Structure
            rel_path = file_path.relative_to(data_dir)
            analysis['structure'].append(str(rel_path))
    
    # Print summary
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   Total files: {analysis['total_files']}")
    print(f"   File types: {dict(sorted(analysis['file_types'].items()))}")
    
    if analysis['eclipse_files']:
        print(f"\n🔑 ECLIPSE FILES FOUND:")
        for ef in sorted(analysis['eclipse_files'], key=lambda x: x['size_mb'], reverse=True)[:10]:
            print(f"   ✓ {ef['name']:30} {ef['size_mb']:8.2f} MB")
    
    # Check for specific SPE9 files
    spe9_key_files = ['SPE9', 'spe9', 'Spe9']
    found_spe9 = []
    for file_path in all_files:
        if any(keyword in file_path.name.upper() for keyword in [f.upper() for f in spe9_key_files]):
            found_spe9.append(file_path.name)
    
    if found_spe9:
        print(f"\n🎯 SPE9 FILES IDENTIFIED:")
        for f in found_spe9[:10]:
            print(f"   • {f}")
    
    return analysis

def parse_eclipse_deck(deck_path: str) -> Dict:
    """Parse Eclipse deck file manually."""
    print(f"\n📖 Parsing Eclipse deck: {deck_path}")
    
    with open(deck_path, 'r') as f:
        lines = f.readlines()
    
    sections = {}
    current_section = None
    current_data = []
    
    for line_num, line in enumerate(lines[:1000]):  # First 1000 lines
        line = line.strip()
        
        # Skip comments
        if line.startswith('--') or not line:
            continue
        
        # Check for keywords (uppercase words at start of line)
        words = line.split()
        if words and words[0].isupper() and len(words[0]) > 2:
            # New section
            if current_section and current_data:
                sections[current_section] = current_data
            
            current_section = words[0]
            current_data = [line]
        else:
            if current_section:
                current_data.append(line)
    
    # Add last section
    if current_section and current_data:
        sections[current_section] = current_data
    
    print(f"✅ Found {len(sections)} sections")
    for section in list(sections.keys())[:10]:
        print(f"   • {section}: {len(sections[section])} lines")
    
    return sections

def inspect_binary_file(file_path: str, num_bytes: int = 1000):
    """Inspect binary file header."""
    print(f"\n🔬 Inspecting binary file: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            header = f.read(min(num_bytes, Path(file_path).stat().st_size))
        
        # Check for Eclipse binary markers
        if header[:4] in [b'\x00\x00\x00\x00', b'\x00\x00\x00\x01']:
            print("   ⚠️  Possible Eclipse unified restart file")
        
        # Check for text content
        text_part = header[:200].decode('ascii', errors='ignore')
        if 'SEQNUM' in text_part or 'INTEHEAD' in text_part:
            print("   ✅ Eclipse binary format detected")
        
        # File signature
        signatures = {
            b'\x89HDF': 'HDF5 file',
            b'CDF': 'NetCDF file',
            b'PARAVIEW': 'ParaView file',
            b'VARIABLE': 'Eclipse summary',
        }
        
        for sig, desc in signatures.items():
            if sig in header:
                print(f"   ✅ {desc} detected")
        
        return header[:200]
    
    except Exception as e:
        print(f"❌ Inspection failed: {e}")
        return None

def main():
    """Main analysis function."""
    print("="*70)
    print("🔬 REAL SPE9 DATA ANALYSIS - GOOGLE DRIVE VERSION")
    print("="*70)
    
    # Option 1: Direct path if you already downloaded
    data_path = input("Enter path to your SPE9 data file/directory (or press Enter to use gdown): ").strip()
    
    if not data_path:
        # Option 2: Download from Google Drive
        file_id = "1Ue_EHX8w2h8WlT9kGdL3jFjF1b3yLnfL"  # From your link
        output_path = "data/spe9_archive.tar.gz"
        
        print(f"\n📥 Downloading from Google Drive...")
        print(f"   File ID: {file_id}")
        
        if download_from_drive(file_id, output_path):
            data_path = output_path
        else:
            print("❌ Please download manually and provide path")
            return
    
    # Analyze the data
    analysis = analyze_spe9_archive(data_path)
    
    # Look for key files to parse
    data_dir = Path('data/spe9_raw')
    
    # Find .DATA file
    data_files = list(data_dir.rglob('*.DATA')) + list(data_dir.rglob('*.data'))
    
    if data_files:
        print(f"\n🎯 PARSING MAIN DECK FILES:")
        for data_file in data_files[:3]:  # Parse first 3
            if data_file.stat().st_size < 10 * 1024 * 1024:  # < 10MB
                deck_sections = parse_eclipse_deck(str(data_file))
                
                # Look for grid dimensions
                if 'DIMENS' in deck_sections:
                    dims_line = deck_sections['DIMENS'][0]
                    import re
                    numbers = re.findall(r'\d+', dims_line)
                    if len(numbers) >= 3:
                        print(f"   Grid dimensions: {numbers[0]} x {numbers[1]} x {numbers[2]}")
                
                # Look for properties
                for prop in ['PORO', 'PERMX', 'PERMY', 'PERMZ']:
                    if prop in deck_sections:
                        print(f"   Found {prop} section")
    
    # Find binary files
    binary_extensions = ['.UNRST', '.RST', '.SMSPEC', '.ESMRY', '.EGRID']
    for ext in binary_extensions:
        bin_files = list(data_dir.rglob(f'*{ext}'))
        for bin_file in bin_files[:2]:  # Inspect first 2
            if bin_file.stat().st_size > 0:
                inspect_binary_file(str(bin_file))
    
    # Generate report
    print(f"\n📋 GENERATING ANALYSIS REPORT...")
    
    report_path = 'data/analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("SPE9 DATA ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Analysis date: {pd.Timestamp.now()}\n")
        f.write(f"Data source: {data_path}\n\n")
        
        f.write("FILE STRUCTURE:\n")
        f.write("-"*30 + "\n")
        for file_type, count in sorted(analysis['file_types'].items()):
            f.write(f"{file_type}: {count} files\n")
        
        f.write("\nKEY FILES:\n")
        f.write("-"*30 + "\n")
        for ef in analysis.get('eclipse_files', []):
            f.write(f"{ef['name']}: {ef['size_mb']:.1f} MB\n")
    
    print(f"✅ Report saved to: {report_path}")
    
    # Create visualization
    if analysis['eclipse_files']:
        create_visualization(analysis)
    
    print(f"\n🎯 NEXT STEPS:")
    print("   1. Install OPM tools: conda install -c conda-forge opm-simulators")
    print("   2. Run: python -c \"import opm; print('OPM installed successfully')\"")
    print("   3. Parse with: from opm.io.parser import Parser")
    print("   4. Or use resdata if OPM doesn't work: pip install resdata")

def create_visualization(analysis: Dict):
    """Create visualization of data structure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # File types pie chart
    if analysis['file_types']:
        labels = list(analysis['file_types'].keys())
        sizes = list(analysis['file_types'].values())
        
        axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('File Types Distribution')
    
    # File sizes bar chart
    if analysis['eclipse_files']:
        files = [ef['name'][:20] + '...' if len(ef['name']) > 20 else ef['name'] 
                for ef in analysis['eclipse_files'][:10]]
        sizes = [ef['size_mb'] for ef in analysis['eclipse_files'][:10]]
        
        axes[1].barh(files, sizes)
        axes[1].set_xlabel('Size (MB)')
        axes[1].set_title('Largest Eclipse Files')
        axes[1].tick_params(axis='y', labelsize=8)
    
    plt.tight_layout()
    plt.savefig('data/file_analysis.png', dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved to: data/file_analysis.png")

if __name__ == "__main__":
    main()
