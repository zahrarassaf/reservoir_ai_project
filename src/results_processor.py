"""
Results Processor Module
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

class ResultsProcessor:
    """Process and export simulation results."""
    
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
    
    def export_to_json(self, results, filename=None):
        """Export results to JSON."""
        self.output_dir.mkdir(exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        return filepath
    
    def export_to_csv(self, results, filename=None):
        """Export time series data to CSV."""
        csv_dir = self.output_dir / "csv_data"
        csv_dir.mkdir(exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_{timestamp}.csv"
        
        # Create DataFrame from results
        data = {}
        
        if 'time_series' in results and 'time_steps' in results['time_series']:
            data['Time'] = results['time_series']['time_steps']
        
        if 'production' in results:
            for phase, values in results['production'].items():
                data[f'{phase}_rate'] = values
        
        if data:
            df = pd.DataFrame(data)
            filepath = csv_dir / filename
            df.to_csv(filepath, index=False)
            return filepath
        
        return None
