"""
Final execution script for the complete project.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_project_structure():
    """Check if project structure is complete."""
    required_dirs = [
        'data',
        'config',
        'analysis',
        'data_parser',
        'src',
        'tests',
        'docs'
    ]
    
    required_files = [
        'run_simulation.py',
        'requirements.txt',
        'README.md',
        'setup.py'
    ]
    
    print("Checking project structure...")
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ Directory exists: {dir_name}")
        else:
            print(f"✗ Missing directory: {dir_name}")
    
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"✓ File exists: {file_name}")
        else:
            print(f"✗ Missing file: {file_name}")
    
    print("\nProject structure check complete.")

def run_simulation():
    """Run the main simulation."""
    print("\n" + "="*70)
    print("RUNNING RESERVOIR SIMULATION")
    print("="*70)
    
    try:
        # Import and run main simulation
        from run_simulation import main
        return main()
    except Exception as e:
        print(f"Error running simulation: {e}")
        return 1

def run_tests():
    """Run project tests."""
    print("\n" + "="*70)
    print("RUNNING TESTS")
    print("="*70)
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"],
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

def main():
    """Main execution function."""
    print("RESERVOIR SIMULATION PROJECT - FINAL VERSION")
    print("="*70)
    
    # Check structure
    check_project_structure()
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Run simulation")
    print("2. Run tests")
    print("3. Check project structure")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        return run_simulation()
    elif choice == "2":
        return run_tests()
    elif choice == "3":
        check_project_structure()
        return 0
    elif choice == "4":
        print("Exiting...")
        return 0
    else:
        print("Invalid choice")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
