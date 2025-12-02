"""
Process and analyze simulation results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
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
            # Try to use ecl2df for reading simulation results
            from ecl2df import summary
            
            smspec_file = self.results_dir / "SPE9.SMSPEC"
            if not smspec_file.exists():
                raise FileNotFoundError(f"❌ Summary file not found: {smspec_file}")
                
            logger.info(f"📊 Loading summary data from: {smspec_file}")
            df = summary.df(str(smspec_file))
            self.summary_data = df
            logger.info(f"✅ Loaded summary data with {len(df)} rows, {len(df.columns)} columns")
            
            # Log available columns
            field_cols = [col for col in df.columns if col.startswith('F')]
            well_cols = [col for col in df.columns if col.startswith('W')]
            logger.info(f"📈 Field columns: {len(field_cols)}, Well columns: {len(well_cols)}")
            
            return df
            
        except ImportError:
            logger.warning("⚠️ ecl2df not available, attempting manual parsing")
            return self._parse_summary_manually()
        except Exception as e:
            logger.error(f"❌ Failed to load summary data: {e}")
            return pd.DataFrame()
            
    def _parse_summary_manually(self) -> pd.DataFrame:
        """Manual parsing of summary results (fallback)"""
        prt_file = self.results_dir / "SPE9.PRT"
        if not prt_file.exists():
            logger.error(f"❌ PRT file not found: {prt_file}")
            return pd.DataFrame()
            
        logger.info("📄 Parsing PRT file manually...")
        
        data = []
        current_date = None
        
        with open(prt_file, 'r') as f:
            for line in f:
                if "REPORT" in line and "TIME" in line:
                    # Extract date from line like: "REPORT      1     31 JAN 2015 /"
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            day = int(parts[2])
                            month = parts[3]
                            year = int(parts[4])
                            current_date = datetime(year, self._month_to_num(month), day)
                        except:
                            current_date = None
                            
                elif current_date and "FIELD" in line and "TOTALS" in line:
                    # Parse production totals
                    next_line = f.readline()
                    if next_line and "STB" in next_line:
                        values = next_line.strip().split()
                        if len(values) >= 5:
                            try:
                                oil_rate = float(values[0])
                                water_rate = float(values[1])
                                gas_rate = float(values[2])
                                
                                data.append({
                                    'DATE': current_date,
                                    'FOPR': oil_rate,
                                    'FWPR': water_rate,
                                    'FGPR': gas_rate
                                })
                            except:
                                pass
                                
        df = pd.DataFrame(data)
        if not df.empty:
            df['FOPT'] = df['FOPR'].cumsum()
            df['FWPT'] = df['FWPR'].cumsum()
            df['FGPT'] = df['FGPR'].cumsum()
            
        self.summary_data = df
        logger.info(f"📊 Manually parsed {len(df)} records")
        return df
        
    def _month_to_num(self, month: str) -> int:
        """Convert month abbreviation to number"""
        months = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
            'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
            'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        return months.get(month.upper(), 1)
        
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate key performance indicators"""
        if self.summary_data is None or self.summary_data.empty:
            logger.warning("⚠️ No summary data available for metrics calculation")
            return {}
            
        logger.info("📈 Calculating performance metrics...")
        metrics = {}
        
        try:
            # Field production metrics
            if 'FOPT' in self.summary_data.columns:
                metrics['total_oil_produced'] = float(self.summary_data['FOPT'].iloc[-1])
            if 'FGPT' in self.summary_data.columns:
                metrics['total_gas_produced'] = float(self.summary_data['FGPT'].iloc[-1])
            if 'FWPT' in self.summary_data.columns:
                metrics['total_water_produced'] = float(self.summary_data['FWPT'].iloc[-1])
            
            # Recovery factors
            ooip = 7.758e7  # STB (from SPE9 paper)
            if 'total_oil_produced' in metrics:
                metrics['oil_recovery_factor'] = (metrics['total_oil_produced'] / ooip) * 100
            
            # Peak rates
            if 'FOPR' in self.summary_data.columns:
                metrics['peak_oil_rate'] = float(self.summary_data['FOPR'].max())
            if 'FGPR' in self.summary_data.columns:
                metrics['peak_gas_rate'] = float(self.summary_data['FGPR'].max())
            if 'FWPR' in self.summary_data.columns:
                metrics['peak_water_rate'] = float(self.summary_data['FWPR'].max())
            
            # Average rates
            if 'FOPR' in self.summary_data.columns:
                metrics['avg_oil_rate'] = float(self.summary_data['FOPR'].mean())
            if 'FGPR' in self.summary_data.columns:
                metrics['avg_gas_rate'] = float(self.summary_data['FGPR'].mean())
            
            # Final values
            if not self.summary_data.empty:
                final_step = self.summary_data.iloc[-1]
                if 'FOPR' in final_step:
                    metrics['final_oil_rate'] = float(final_step['FOPR'])
                if 'FWCT' in final_step:
                    metrics['final_water_cut'] = float(final_step['FWCT'])
                if 'FGOR' in final_step:
                    metrics['final_gor'] = float(final_step['FGOR'])
            
            # Well performance metrics
            well_metrics = self._calculate_well_metrics()
            metrics.update(well_metrics)
            
            logger.info(f"✅ Calculated {len(metrics)} performance metrics")
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics: {e}")
            
        return metrics
        
    def _calculate_well_metrics(self) -> Dict[str, float]:
        """Calculate well-specific performance metrics"""
        well_metrics = {}
        
        if self.summary_data is None:
            return well_metrics
            
        # Extract well data from summary
        well_oil_cols = [col for col in self.summary_data.columns 
                        if 'WOPR' in col or ('WOPR:' in col if ':' in col else False)]
        
        for well_col in well_oil_cols[:10]:  # Limit to first 10 wells
            well_name = well_col.split(':')[1] if ':' in well_col else well_col.replace('WOPR', '')
            
            if well_col in self.summary_data.columns:
                cumulative_oil = self.summary_data[well_col].sum() / 30  # Approximate monthly
                key = f"{well_name}_cumulative_oil"
                well_metrics[key] = float(cumulative_oil)
                
        return well_metrics
        
    def generate_performance_report(self, output_file: str = "performance_report.txt") -> str:
        """Generate comprehensive performance report"""
        metrics = self.calculate_performance_metrics()
        
        report = [
            "=" * 70,
            "SPE9 SIMULATION PERFORMANCE REPORT",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Source: {self.results_dir}",
            "",
            "FIELD PERFORMANCE METRICS:",
            "-" * 40,
        ]
        
        # Field metrics
        field_metrics = [
            ("Total Oil Produced", "total_oil_produced", "STB"),
            ("Total Gas Produced", "total_gas_produced", "MSCF"),
            ("Total Water Produced", "total_water_produced", "STB"),
            ("Oil Recovery Factor", "oil_recovery_factor", "%"),
            ("Peak Oil Rate", "peak_oil_rate", "STB/D"),
            ("Peak Gas Rate", "peak_gas_rate", "MSCF/D"),
            ("Average Oil Rate", "avg_oil_rate", "STB/D"),
            ("Final Oil Rate", "final_oil_rate", "STB/D"),
            ("Final Water Cut", "final_water_cut", "fraction"),
            ("Final GOR", "final_gor", "SCF/STB"),
        ]
        
        for display_name, metric_key, unit in field_metrics:
            if metric_key in metrics:
                value = metrics[metric_key]
                if 'recovery' in metric_key:
                    report.append(f"{display_name:30} : {value:8.2f} {unit}")
                elif 'rate' in metric_key:
                    report.append(f"{display_name:30} : {value:8,.0f} {unit}")
                elif 'cut' in metric_key or 'gor' in metric_key:
                    report.append(f"{display_name:30} : {value:8.3f} {unit}")
                else:
                    report.append(f"{display_name:30} : {value:8,.0f} {unit}")
                    
        report.append("")
        report.append("SIMULATION INFORMATION:")
        report.append("-" * 40)
        
        if self.summary_data is not None:
            report.append(f"Simulation Period       : {len(self.summary_data)} time steps")
            if 'DATE' in self.summary_data.columns:
                start_date = self.summary_data['DATE'].iloc[0]
                end_date = self.summary_data['DATE'].iloc[-1]
                report.append(f"Time Range             : {start_date} to {end_date}")
                
        report.append("")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        # Save to file
        if output_file:
            output_path = Path("results/analysis_results") / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"📋 Performance report saved to: {output_path}")
            
        return report_text
        
    def extract_time_series(self, variables: List[str]) -> pd.DataFrame:
        """Extract time series for specific variables"""
        if self.summary_data is None:
            self.load_summary_results()
            
        if self.summary_data.empty:
            return pd.DataFrame()
            
        available_vars = [v for v in variables if v in self.summary_data.columns]
        if not available_vars:
            logger.warning(f"⚠️ None of the requested variables found: {variables}")
            return pd.DataFrame()
            
        logger.info(f"📊 Extracting time series for {len(available_vars)} variables")
        return self.summary_data[['DATE'] + available_vars]
