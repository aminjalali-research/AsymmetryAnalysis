"""
Unit Tests for Asymmetry Analysis Components

This module provides basic unit tests to verify the functionality of the
asymmetry analysis components.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Add the src directory to the path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from calculator import AsymmetryCalculator
from freesurfer_parser import FreeSurferASLParser


class TestAsymmetryCalculator(unittest.TestCase):
    """Test cases for AsymmetryCalculator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.calculator = AsymmetryCalculator()

    def test_laterality_index(self):
        """Test laterality index calculation"""
        # Perfect symmetry
        self.assertEqual(self.calculator.laterality_index(50, 50), 0)

        # Left > Right
        self.assertAlmostEqual(self.calculator.laterality_index(60, 40), 0.2)

        # Right > Left
        self.assertAlmostEqual(self.calculator.laterality_index(40, 60), -0.2)

        # Zero sum case
        self.assertEqual(self.calculator.laterality_index(0, 0), 0)

    def test_asymmetry_index(self):
        """Test asymmetry index calculation"""
        # Perfect symmetry
        self.assertEqual(self.calculator.asymmetry_index(50, 50), 0)

        # 20% asymmetry
        self.assertAlmostEqual(self.calculator.asymmetry_index(60, 40), 40)

        # Negative asymmetry
        self.assertAlmostEqual(self.calculator.asymmetry_index(40, 60), -40)

    def test_percentage_difference(self):
        """Test percentage difference calculation"""
        # 25% difference
        self.assertAlmostEqual(self.calculator.percentage_difference(80, 60), 25)

        # Negative difference
        self.assertAlmostEqual(
            self.calculator.percentage_difference(60, 80), -33.333333, places=5
        )

        # Zero left value
        self.assertEqual(self.calculator.percentage_difference(0, 50), 0)

    def test_log_ratio(self):
        """Test log ratio calculation"""
        # Equal values
        self.assertAlmostEqual(self.calculator.log_ratio(50, 50), 0)

        # Left > Right
        self.assertAlmostEqual(self.calculator.log_ratio(100, 50), np.log(2))

        # Invalid inputs
        self.assertTrue(np.isnan(self.calculator.log_ratio(0, 50)))
        self.assertTrue(np.isnan(self.calculator.log_ratio(50, 0)))

    def test_calculate_all_indices(self):
        """Test calculation of all asymmetry indices"""
        # Create sample bilateral data
        sample_data = pd.DataFrame(
            {
                "region": ["Thalamus", "Caudate"],
                "left_volume": [1200, 1300],
                "right_volume": [1250, 1280],
                "left_mean_perfusion": [45.0, 38.0],
                "right_mean_perfusion": [40.0, 42.0],
                "left_std": [8.0, 7.0],
                "right_std": [7.5, 7.5],
                "left_nvoxels": [1200, 1300],
                "right_nvoxels": [1250, 1280],
            }
        )

        result = self.calculator.calculate_all_indices(sample_data)

        # Check that all expected columns are present
        expected_columns = [
            "laterality_index",
            "asymmetry_index",
            "percentage_difference",
            "normalized_asymmetry",
            "log_ratio",
            "volume_weighted_asymmetry",
        ]

        for col in expected_columns:
            self.assertIn(col, result.columns)

        # Check specific calculations for first row
        self.assertAlmostEqual(
            result.iloc[0]["laterality_index"], (45 - 40) / (45 + 40)
        )
        self.assertAlmostEqual(
            result.iloc[0]["asymmetry_index"], (45 - 40) / ((45 + 40) / 2) * 100
        )


class TestFreeSurferASLParser(unittest.TestCase):
    """Test cases for FreeSurferASLParser class"""

    def setUp(self):
        """Set up test fixtures"""
        self.parser = FreeSurferASLParser()

    def create_sample_stats_file(self):
        """Create a temporary sample stats file for testing"""
        sample_content = """# Title Segmentation Statistics 
# generating_program mri_segstats
# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1234567.000000, mm^3
# ColHeaders Index SegId NVoxels Volume_mm3 StructName Mean StdDev Min Max Range
  1  10    1234  1234.0  Left-Thalamus-Proper      45.2   8.1  25.0  65.0  40.0
  2  11    1345  1345.0  Left-Caudate             38.7   7.5  20.0  58.0  38.0
  3  49    1289  1289.0  Right-Thalamus-Proper    47.8   8.4  26.0  68.0  42.0
  4  50    1356  1356.0  Right-Caudate            40.2   7.9  22.0  59.0  37.0"""

        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        temp_file.write(sample_content)
        temp_file.close()
        return temp_file.name

    def test_parse_segstats_file(self):
        """Test parsing of FreeSurfer stats file"""
        temp_file = self.create_sample_stats_file()

        try:
            measurements, metadata = self.parser.parse_segstats_file(temp_file)

            # Check that data was parsed correctly
            self.assertEqual(len(measurements), 4)
            self.assertIn("BrainSegVol", metadata)
            self.assertEqual(metadata["BrainSegVol"], 1234567.0)

            # Check column types
            self.assertTrue(pd.api.types.is_numeric_dtype(measurements["Mean"]))
            self.assertTrue(pd.api.types.is_numeric_dtype(measurements["Volume_mm3"]))

        finally:
            os.unlink(temp_file)

    def test_extract_bilateral_pairs(self):
        """Test extraction of bilateral region pairs"""
        temp_file = self.create_sample_stats_file()

        try:
            measurements, metadata = self.parser.parse_segstats_file(temp_file)
            bilateral_df = self.parser.extract_bilateral_pairs()

            # Should find 2 bilateral pairs (Thalamus and Caudate)
            self.assertEqual(len(bilateral_df), 2)

            # Check that regions are correctly identified
            regions = set(bilateral_df["region"])
            expected_regions = {"Thalamus-Proper", "Caudate"}
            self.assertEqual(regions, expected_regions)

            # Check that perfusion values are correctly assigned
            thalamus_row = bilateral_df[
                bilateral_df["region"] == "Thalamus-Proper"
            ].iloc[0]
            self.assertEqual(thalamus_row["left_mean_perfusion"], 45.2)
            self.assertEqual(thalamus_row["right_mean_perfusion"], 47.8)

        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
