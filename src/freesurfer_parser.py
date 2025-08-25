"""
FreeSurfer ASL Parser Module

This module provides functionality to parse FreeSurfer mri_segstats output
for ASL perfusion analysis and extract bilateral region pairs.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path


class FreeSurferASLParser:
    """Parse FreeSurfer mri_segstats output for ASL perfusion analysis"""

    def __init__(self, excluded_regions=None):
        self.metadata = {}
        self.measurements = pd.DataFrame()
        self.logger = self._setup_logging()
        self.excluded_regions = excluded_regions or []
        
        # Default exclusions based on common problematic regions
        self.default_excluded_regions = [
            'Unknown', 'CSF', 'WM-hypointensities', 'Optic-Chiasm',
            'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 
            'CC_Mid_Anterior', 'CC_Anterior', 'Brain-Stem',
            '3rd-Ventricle', '4th-Ventricle', 'vessel', 'choroid-plexus'
        ]

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def parse_segstats_file(self, stats_file_path):
        """Parse mri_segstats output file"""
        with open(stats_file_path, "r") as f:
            lines = f.readlines()

        data_rows = []
        headers = None

        for line in lines:
            # Extract metadata from comment lines
            if line.startswith("# Measure"):
                parts = line.strip().split(",")
                if len(parts) >= 5:
                    measure_name = parts[1].strip()
                    try:
                        value = float(parts[4].strip())
                        self.metadata[measure_name] = value
                    except ValueError:
                        continue

            # Extract column headers
            elif line.startswith("# ColHeaders"):
                headers = line.replace("# ColHeaders", "").strip().split()

            # Extract data rows
            elif not line.startswith("#") and line.strip():
                values = line.strip().split()
                if headers and len(values) >= len(headers):
                    data_rows.append(values[: len(headers)])

        if headers and data_rows:
            self.measurements = pd.DataFrame(data_rows, columns=headers)
            # Convert numeric columns
            numeric_cols = [
                "Index",
                "SegId",
                "NVoxels",
                "Volume_mm3",
                "Mean",
                "StdDev",
                "Min",
                "Max",
                "Range",
            ]
            for col in numeric_cols:
                if col in self.measurements.columns:
                    self.measurements[col] = pd.to_numeric(
                        self.measurements[col], errors="coerce"
                    )

        return self.measurements, self.metadata

    def extract_bilateral_pairs(self, min_volume=100, min_mean_perfusion=10):
        """Extract and match left-right hemisphere pairs for both subcortical and cortical regions"""
        if self.measurements.empty:
            raise ValueError("No measurements loaded. Run parse_segstats_file first.")

        # Filter by quality thresholds
        filtered_data = self.measurements[
            (self.measurements["Volume_mm3"] >= min_volume)
            & (self.measurements["Mean"] >= min_mean_perfusion)
            & (self.measurements["Mean"].notna())
        ]

        # Combine default exclusions with user-specified exclusions
        all_excluded = set(self.default_excluded_regions + self.excluded_regions)
        
        bilateral_pairs = []
        
        # Process subcortical regions (Left-/Right- pattern)
        bilateral_pairs.extend(self._extract_subcortical_pairs(filtered_data, all_excluded))
        
        # Process cortical regions (ctx-lh-/ctx-rh- pattern)  
        bilateral_pairs.extend(self._extract_cortical_pairs(filtered_data, all_excluded))

        result_df = pd.DataFrame(bilateral_pairs)
        
        if not result_df.empty:
            self.logger.info(f"Found {len(result_df)} bilateral region pairs")
            self.logger.info(f"Excluded {len(all_excluded)} region types from analysis")
        
        return result_df

    def _extract_subcortical_pairs(self, filtered_data, excluded_regions):
        """Extract subcortical bilateral pairs (Left-/Right- pattern)"""
        pairs = []
        
        # Separate left and right subcortical regions
        left_regions = filtered_data[
            filtered_data["StructName"].str.startswith("Left-")
        ].copy()
        right_regions = filtered_data[
            filtered_data["StructName"].str.startswith("Right-")
        ].copy()

        for _, left_row in left_regions.iterrows():
            region_base = left_row["StructName"].replace("Left-", "")
            
            # Skip excluded regions
            if region_base in excluded_regions:
                self.logger.debug(f"Excluding subcortical region: {region_base}")
                continue
                
            right_match = right_regions[
                right_regions["StructName"] == f"Right-{region_base}"
            ]

            if not right_match.empty:
                right_row = right_match.iloc[0]
                pair_data = {
                    "region": region_base,
                    "region_type": "subcortical",
                    "left_volume": left_row["Volume_mm3"],
                    "right_volume": right_row["Volume_mm3"],
                    "left_mean_perfusion": left_row["Mean"],
                    "right_mean_perfusion": right_row["Mean"],
                    "left_std": left_row["StdDev"],
                    "right_std": right_row["StdDev"],
                    "left_nvoxels": left_row["NVoxels"],
                    "right_nvoxels": right_row["NVoxels"],
                    "left_segid": left_row["SegId"],
                    "right_segid": right_row["SegId"],
                }
                pairs.append(pair_data)
                
        return pairs

    def _extract_cortical_pairs(self, filtered_data, excluded_regions):
        """Extract cortical bilateral pairs (ctx-lh-/ctx-rh- pattern)"""
        pairs = []
        
        # Separate left and right cortical regions
        left_regions = filtered_data[
            filtered_data["StructName"].str.startswith("ctx-lh-")
        ].copy()
        right_regions = filtered_data[
            filtered_data["StructName"].str.startswith("ctx-rh-")
        ].copy()

        for _, left_row in left_regions.iterrows():
            region_base = left_row["StructName"].replace("ctx-lh-", "")
            
            # Skip excluded regions
            if region_base in excluded_regions:
                self.logger.debug(f"Excluding cortical region: {region_base}")
                continue
                
            right_match = right_regions[
                right_regions["StructName"] == f"ctx-rh-{region_base}"
            ]

            if not right_match.empty:
                right_row = right_match.iloc[0]
                pair_data = {
                    "region": region_base,
                    "region_type": "cortical",
                    "left_volume": left_row["Volume_mm3"],
                    "right_volume": right_row["Volume_mm3"],
                    "left_mean_perfusion": left_row["Mean"],
                    "right_mean_perfusion": right_row["Mean"],
                    "left_std": left_row["StdDev"],
                    "right_std": right_row["StdDev"],
                    "left_nvoxels": left_row["NVoxels"],
                    "right_nvoxels": right_row["NVoxels"],
                    "left_segid": left_row["SegId"],
                    "right_segid": right_row["SegId"],
                }
                pairs.append(pair_data)
                
        return pairs

    def set_excluded_regions(self, excluded_regions):
        """Set list of regions to exclude from bilateral analysis"""
        self.excluded_regions = excluded_regions or []
        self.logger.info(f"Set {len(self.excluded_regions)} regions to exclude from analysis")

    def get_available_regions(self):
        """Get list of all available bilateral region pairs"""
        if self.measurements.empty:
            raise ValueError("No measurements loaded. Run parse_segstats_file first.")
        
        available_regions = []
        
        # Get subcortical regions
        left_subcortical = self.measurements[
            self.measurements["StructName"].str.startswith("Left-")
        ]["StructName"].str.replace("Left-", "").tolist()
        
        right_subcortical = self.measurements[
            self.measurements["StructName"].str.startswith("Right-")
        ]["StructName"].str.replace("Right-", "").tolist()
        
        subcortical_pairs = set(left_subcortical).intersection(set(right_subcortical))
        available_regions.extend([f"subcortical:{region}" for region in subcortical_pairs])
        
        # Get cortical regions
        left_cortical = self.measurements[
            self.measurements["StructName"].str.startswith("ctx-lh-")
        ]["StructName"].str.replace("ctx-lh-", "").tolist()
        
        right_cortical = self.measurements[
            self.measurements["StructName"].str.startswith("ctx-rh-")
        ]["StructName"].str.replace("ctx-rh-", "").tolist()
        
        cortical_pairs = set(left_cortical).intersection(set(right_cortical))
        available_regions.extend([f"cortical:{region}" for region in cortical_pairs])
        
        return sorted(available_regions)
