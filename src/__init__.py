"""
SPE9 Reservoir Simulation Package
Professional implementation of SPE9 benchmark
"""

__version__ = "1.0.0"
__author__ = "SPE9 Project Team"
__email__ = "your-email@example.com"

from .simulation_runner import SimulationRunner
from .results_processor import ResultsProcessor
from .data_validator import DataValidator

__all__ = [
    "SimulationRunner",
    "ResultsProcessor", 
    "DataValidator"
]
