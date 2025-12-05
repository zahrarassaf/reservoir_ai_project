"""
Google Drive integration example
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_loader import GoogleDriveLoader
from simulator import ReservoirSimulator
from visualizer import ReservoirVisualizer


def main():
    """Run Google Drive integration example"""
    print("=" * 60)
    print("Google Drive Integration Example")
    print("=" * 60)
    
    # Your Google Drive links
    DRIVE_LINKS = [
        "https://drive.google.com/file/d/1ZwEswptUcexDn_kqm_q8qRcHYTl1WHq2/view?usp=sharing",
        "https://drive.google.com/file/d/1sxq7sd4GSL-chE362k8wTLA_arehaD5U/view?usp=sharing",
        "https://drive.google.com/file/d/1f0aJFS99ZBVkT8IXbKdZdVihbIZIpBwZ/view?usp=sharing",
        "https://drive.google.com/file/d/1bdyUFKx-FKPy7YOlq-E9Y4nupcrhOoXi/view?usp=sharing",
        "https://drive.google.com/file/d/1n_auKzsDz5aHglQ4YvskjfHPK8ZuLBqC/view?usp=sharing",
        "https://drive.google.com/file/d/13twFcFA35CKbI8neIzIt-D54dzDd1B-N/view?usp=sharing"
    ]
    
    # Note: You need credentials.json file for Google Drive API
    # Place it in the project root directory
    
    try:
        # 1. Load data from Google Drive
        print("\n1. Loading data from Google Drive...")
        print("   Note: Requires credentials.json in project root")
        
        # Uncomment the following lines when you have credentials.json
        # loader = GoogleDriveLoader(credentials_path='credentials.json')
        # data = loader.load_from_drive(DRIVE_LINKS)
        
        # For demonstration, use sample data
        print("   Using sample data for demonstration")
        from data_loader import create_sample_data
        data = create_sample_data()
        
        print(f"   Loaded data with {data.production.shape[1]} wells")
        
        # 2. Run simulation
        print("\n2. Running reservoir simulation...")
        simulator = ReservoirSimulator(data)
        results = simulator.run_comprehensive_simulation()
        
        # 3. Create visualizations
        print("\n3. Creating visualizations...")
        visualizer = ReservoirVisualizer(data, results)
        
        # Static dashboard
        visualizer.create_dashboard(save_path='google_drive_dashboard.png')
        
        # Interactive dashboard
        visualizer.create_interactive_dashboard(
            save_path='google_drive_interactive.html'
        )
        
        # 4. Export results
        print("\n4. Exporting results...")
        export_files = simulator.export_results('./outputs/google_drive_example')
        
        print("\n✅ Google Drive example completed!")
        print("\nTo use your real Google Drive data:")
        print("1. Get credentials.json from Google Cloud Console")
        print("2. Place it in the project root")
        print("3. Uncomment lines 44-45 in this file")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
