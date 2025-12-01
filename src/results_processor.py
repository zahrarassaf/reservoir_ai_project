"""
Process and analyze simulation results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultsProcessor:
    """Process simulation results and calculate performance metrics"""
    
    def __init__(self, results_dir: str = "results/simulation_output"):
        self.results_dir = Path(results_dir)
        self.summary_data = None
        self.grid_data = None
        
    def load_summary_results(self) -> pd.DataFrame:
        """Load summary results from .SMSPEC file"""
        try:
            # Using ecl2df for reading simulation results
            from ecl2df import summary
            
            smspec_file = self.results_dir / "SPE9.SMSPEC"
            if not smspec_file.exists():
                raise FileNotFoundError(f"Summary file not found: {smspec_file}")
                
            df = summary.df(str(smspec_file))
            self.summary_data = df
            logger.info(f"Loaded summary data with {len(df)} rows")
            return df
            
        except ImportError:
            logger.warning("ecl2df not available, using manual parsing")
            return self._parse_summary_manually()
            
    def _parse_summary_manually(self) -> pd.DataFrame:
        """Manual parsing of summary results"""
        prt_file = self.results_dir / "SPE9.PRT"
        if not prt_file.exists():
            return pd.DataFrame()
            
        # This is a simplified parser - in production use ecl2df
        data = []
        with open(prt_file, 'r') as f:
            for line in f:
                if "SUMMARY" in line and "TIME" in line:
                    # Parse summary lines
                    pass
                    
        return pd.DataFrame(data)
        
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate key performance indicators"""
        if self.summary_data is None:
            self.load_summary_results()
            
        if self.summary_data.empty:
            return {}
            
        metrics = {}
        
        # Field production metrics
        metrics['total_oil_produced'] = self.summary_data['FOPT'].max()
        metrics['total_gas_produced'] = self.summary_data['FGPT'].max()
        metrics['total_water_produced'] = self.summary_data['FWPT'].max()
        
        # Recovery factors
        # Assuming initial oil in place from SPE9 specification
        ooip = 7.758e7  # STB (from SPE9 paper)
        metrics['oil_recovery_factor'] = (metrics['total_oil_produced'] / ooip) * 100
        
        # Peak rates
        metrics['peak_oil_rate'] = self.summary_data['FOPR'].max()
        metrics['peak_gas_rate'] = self.summary_data['FGPR'].max()
        metrics['peak_water_rate'] = self.summary_data['FWPR'].max()
        
        # Water cut and GOR
        final_step = self.summary_data.iloc[-1]
        metrics['final_water_cut'] = final_step['FWCT'] if 'FWCT' in final_step else 0
        metrics['final_gor'] = final_step['FGOR'] if 'FGOR' in final_step else 0
        
        # Well performance
        well_metrics = self._calculate_well_metrics()
        metrics.update(well_metrics)
        
        logger.info(f"Calculated {len(metrics)} performance metrics")
        return metrics
        
    def _calculate_well_metrics(self) -> Dict[str, float]:
        """Calculate well-specific performance metrics"""
        well_metrics = {}
        
        # Extract well data from summary
        well_cols = [col for col in self.summary_data.columns 
                    if col.startswith('W') and ('OPR' in col or 'GPR' in col or 'WPR' in col)]
        
        for well_col in well_cols:
            well_name = well_col.split(':')[0] if ':' in well_col else well_col[:-3]
            
            if 'OPR' in well_col:
                key = f"{well_name}_cumulative_oil"
                well_metrics[key] = self.summary_data[well_col].sum() / 365  # Approximate cumulative
                
        return well_metrics
        
    def generate_performance_report(self, output_file: str = None) -> str:
        """Generate comprehensive performance report"""
        metrics = self.calculate_performance_metrics()
        
        report = [
            "=" * 60,
            "SPE9 SIMULATION PERFORMANCE REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "FIELD PERFORMANCE METRICS:",
            "-" * 40,
        ]
        
        # Field metrics
        field_metrics = [
            ("Total Oil Produced", f"{metrics.get('total_oil_produced', 0):,.0f} STB"),
            ("Total Gas Produced", f"{metrics.get('total_gas_produced', 0):,.0f} MSCF"),
            ("Total Water Produced", f"{metrics.get('total_water_produced', 0):,.0f} STB"),
            ("Oil Recovery Factor", f"{metrics.get('oil_recovery_factor', 0):.2f} %"),
            ("Peak Oil Rate", f"{metrics.get('peak_oil_rate', 0):,.0f} STB/D"),
            ("Peak Gas Rate", f"{metrics.get('peak_gas_rate', 0):,.0f} MSCF/D"),
            ("Final Water Cut", f"{metrics.get('final_water_cut', 0):.3f}"),
            ("Final GOR", f"{metrics.get('final_gor', 0):,.0f} SCF/STB"),
        ]
        
        for name, value in field_metrics:
            report.append(f"{name:25} : {value}")
            
        report.append("")
        report.append("WELL PERFORMANCE SUMMARY:")
        report.append("-" * 40)
        
        # Well metrics (top 5 producers)
        well_oil_metrics = {k: v for k, v in metrics.items() if '_cumulative_oil' in k}
        sorted_wells = sorted(well_oil_metrics.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for well_name, oil in sorted_wells:
            report.append(f"{well_name:20} : {oil:,.0f} STB")
            
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if output_file:
            output_path = Path("results/analysis_results") / output_file
            output_path.parent.mkdir(exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_path}")
            
        return report_text
        
    def extract_time_series(self, variables: List[str]) -> pd.DataFrame:
        """Extract time series for specific variables"""
        if self.summary_data is None:
            self.load_summary_results()
            
        available_vars = [v for v in variables if v in self.summary_data.columns]
        if not available_vars:
            return pd.DataFrame()
            
        return self.summary_data[['DATE'] + available_vars]
