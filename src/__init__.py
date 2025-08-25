"""
Comprehensive ROI-Based Asymmetry Analysis for FreeSurfer ASL Perfusion Data

This package provides tools for analyzing brain asymmetry from FreeSurfer ASL perfusion data.
"""

__version__ = "1.0.0"
__author__ = "FreeSurfer Asymmetry Analysis Team"
__email__ = "your-email@example.com"

from .freesurfer_parser import FreeSurferASLParser
from .calculator import AsymmetryCalculator

__all__ = [
    "FreeSurferASLParser",
    "AsymmetryCalculator",
]
