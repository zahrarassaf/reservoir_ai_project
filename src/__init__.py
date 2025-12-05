"""
Reservoir Simulation Package

A comprehensive reservoir engineering toolkit for production forecasting,
reservoir characterization, and economic analysis.
"""

__version__ = "1.0.0"
__author__ = "Reservoir Engineering Team"
__email__ = "contact@example.com"

from .data_loader import ReservoirDataLoader
from .eda import ReservoirEDA
from .simulator import ReservoirSimulator
from .visualization import ReservoirVisualizer
from .report_generator import ReportGenerator
from .utils import (
    calculate_npv,
    estimate_eur,
    material_balance,
    decline_curve_analysis,
)

__all__ = [
    "ReservoirDataLoader",
    "ReservoirEDA",
    "ReservoirSimulator",
    "ReservoirVisualizer",
    "ReportGenerator",
    "calculate_npv",
    "estimate_eur",
    "material_balance",
    "decline_curve_analysis",
]
