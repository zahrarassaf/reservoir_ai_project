"""
Performance regression checker.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceChecker:
    """Check for performance regressions."""
    
    def __init__(self, benchmark_dir: Path = Path(".benchmarks")):
        self.benchmark_dir = benchmark_dir
        
    def load_benchmarks(self) -> List[Dict[str, Any]]:
        """Load all benchmark results."""
        benchmark_files = list(self.benchmark_dir.glob("*.json"))
        
        benchmarks = []
        for file in benchmark_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    benchmarks.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {file}: {e}")
        
        return benchmarks
    
    def check_regressions(self, current_benchmark: Dict[str, Any], 
                         baseline_benchmark: Dict[str, Any],
                         threshold: float = 0.1) -> Dict[str, Any]:
        """Check for performance regressions."""
        regressions = []
        
        # Compare benchmarks
        for bench_name, current_data in current_benchmark.get('benchmarks', {}).items():
            if bench_name in baseline_benchmark.get('benchmarks', {}):
                baseline_data = baseline_benchmark['benchmarks'][bench_name]
                
                # Compare statistics
                for stat in ['mean', 'median', 'std', 'min', 'max']:
                    if stat in current_data['stats'] and stat in baseline_data['stats']:
                        current_val = current_data['stats'][stat]
                        baseline_val = baseline_data['stats'][stat]
                        
                        if baseline_val > 0:  # Avoid division by zero
                            change = (current_val - baseline_val) / baseline_val
                            
                            if abs(change) > threshold:
                                regressions.append({
                                    'benchmark': bench_name,
                                    'statistic': stat,
                                    'current': current_val,
                                    'baseline': baseline_val,
                                    'change': change,
                                    'regression': change > threshold,
                                })
        
        return {
            'regressions': regressions,
            'total_benchmarks': len(current_benchmark.get('benchmarks', {})),
            'regression_count': len([r for r in regressions if r['regression']]),
        }
    
    def run(self):
        """Run performance check."""
        benchmarks = self.load_benchmarks()
        
        if len(benchmarks) < 2:
            logger.info("Need at least 2 benchmark runs for comparison")
            return
        
        # Sort by timestamp
        benchmarks.sort(key=lambda x: x.get('datetime', ''))
        
        # Compare latest with previous
        current = benchmarks[-1]
        baseline = benchmarks[-2]
        
        results = self.check_regressions(current, baseline)
        
        # Report
        logger.info(f"Performance check: {results['regression_count']} regressions found")
        
        for regression in results['regressions']:
            if regression['regression']:
                logger.warning(
                    f"REGRESSION: {regression['benchmark']} - {regression['statistic']}: "
                    f"{regression['change']:.1%} increase "
                    f"({regression['baseline']:.3e} -> {regression['current']:.3e})"
                )
            else:
                logger.info(
                    f"IMPROVEMENT: {regression['benchmark']} - {regression['statistic']}: "
                    f"{abs(regression['change']):.1%} decrease"
                )
        
        # Exit with error if significant regressions
        if results['regression_count'] > 0:
            logger.error("Performance regressions detected!")
            sys.exit(1)
        else:
            logger.info("No significant performance regressions detected")


if __name__ == "__main__":
    checker = PerformanceChecker()
    checker.run()
