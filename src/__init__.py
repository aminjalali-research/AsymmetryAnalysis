"""
ROI-based asymmetry analysis for ASL perfusion data (FreeSurfer parcellations).

Core modules: ``calculator`` (15 asymmetry indices) and ``thresholding``
(13 voxelwise thresholding/clustering methods).
"""

__version__ = "1.0.0"

from .calculator import AsymmetryCalculator

__all__ = [
    "AsymmetryCalculator",
]
