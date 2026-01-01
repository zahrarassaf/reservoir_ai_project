#!/usr/bin/env python3
"""
Automated pipeline for reservoir simulation
"""

import subprocess
import sys
from pathlib import Path

def run_pipeline():
    """Run complete simulation pipeline"""
    
    steps = [
        ("📂 Loading data", "python main_refactored.py --phase=data"),
        ("💰 Economic simulation", "python main_refactored.py --phase=economic"),
        ("🤖 ML training", "python main_refactored.py --phase=ml"),
        ("📊 Visualization", "python main_refactored.py --phase=viz")
    ]
    
    for step_name, command in steps:
        print(f"\n{step_name}...")
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   ✅ {step_name} completed")
            else:
                print(f"   ❌ {step_name} failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    return True

if __name__ == "__main__":
    success = run_pipeline()
    
    if success:
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nOutput files in 'results/' folder:")
        
        results_dir = Path("results")
        for file in results_dir.glob("*"):
            print(f"  • {file.name}")
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)
