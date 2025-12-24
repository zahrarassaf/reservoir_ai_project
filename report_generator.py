"""
Report generation module
"""
import json
import pandas as pd
from datetime import datetime

class ReportGenerator:
    def __init__(self):
        pass
    
    def save_report(self, data, filename):
        """Save analysis report to JSON file"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return filename
    
    def generate_summary_table(self, technical_data, economic_data, ml_data):
        """Generate summary table for display"""
        summary = {
            "Parameter": [
                "Grid Dimensions",
                "Total Cells",
                "Simulation Period",
                "Peak Production (bpd)",
                "Total Oil Recovered (MMbbl)",
                "NPV (Million $)",
                "IRR (%)",
                "ROI (%)",
                "Payback Period (years)",
                "CNN R² Score",
                "Economic Model R² (NPV)"
            ],
            "Value": [
                technical_data.get('grid_dimensions', 'N/A'),
                f"{technical_data.get('total_cells', 0):,}",
                technical_data.get('simulation_period', 'N/A'),
                f"{technical_data.get('peak_production', 0):.0f}",
                f"{technical_data.get('total_oil_recovered', 0):.2f}",
                f"${economic_data.get('net_present_value', 0):.2f}",
                f"{economic_data.get('internal_rate_of_return', 0):.1f}",
                f"{economic_data.get('return_on_investment', 0):.1f}",
                f"{economic_data.get('payback_period', 0):.1f}",
                f"{ml_data.get('cnn_property_prediction', {}).get('r2_score', 0):.3f}",
                f"{ml_data.get('economic_forecasting', {}).get('npv_r2', 0):.3f}"
            ]
        }
        return pd.DataFrame(summary)
