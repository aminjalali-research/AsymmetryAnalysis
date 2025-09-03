#!/usr/bin/env python3
"""
Create Laterality Index (LI) NIfTI Map

This script creates a 3D NIfTI file where each voxel's intensity represents
the laterality index (LI) of the brain region it belongs to based on FreeSurfer parcellation.
"""

import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
import sys
import json

sys.path.append("src")


class LateralityMapper:
    """Create spatial LI maps from parcellation and asymmetry results"""

    def __init__(self, data_dir, patient_id="P013"):
        self.data_dir = Path(data_dir)
        self.patient_id = patient_id
        self.output_dir = Path("laterality_maps")
        self.output_dir.mkdir(exist_ok=True)

        # File paths
        self.parcellation_file = self.data_dir / f"{patient_id}" / "aparc+aseg.nii.gz"
        self.t1_file = self.data_dir / f"{patient_id}" / "T1w_acpc_dc_restore.nii.gz"
        self.perfusion_file = (
            self.data_dir
            / f"{patient_id}"
            / f"{patient_id}_perfusion_calib_resampled_to_T1w.nii.gz"
        )
        self.results_file = (
            Path("P013_Analysis_Results") / "P013_comprehensive_asymmetry_analysis.csv"
        )

    def load_asymmetry_results(self):
        """Load previously computed asymmetry results"""

        print("📊 Loading asymmetry results...")

        if not self.results_file.exists():
            raise FileNotFoundError(f"Asymmetry results not found: {self.results_file}")

        results_df = pd.read_csv(self.results_file)
        print(f" Loaded {len(results_df)} bilateral region pairs")

        # Create SegId to LI mapping for both left and right regions
        segid_to_li = {}

        for _, row in results_df.iterrows():
            left_segid = int(row["left_segid"])
            right_segid = int(row["right_segid"])
            li_value = row["laterality_index"]

            # For left regions: positive LI means left > right, so left gets positive value
            # For right regions: negative LI means right > left, so right gets positive value
            segid_to_li[left_segid] = li_value
            segid_to_li[right_segid] = -li_value  # Flip sign for right hemisphere

        print(f" Created SegId-to-LI mapping for {len(segid_to_li)} regions")
        return segid_to_li, results_df

    def load_parcellation_data(self):
        """Load FreeSurfer parcellation data"""

        print(" Loading parcellation data...")

        if not self.parcellation_file.exists():
            raise FileNotFoundError(
                f"Parcellation file not found: {self.parcellation_file}"
            )

        # Load parcellation
        parc_img = nib.load(self.parcellation_file)
        parc_data = parc_img.get_fdata().astype(int)

        print(f" Loaded parcellation: {parc_data.shape}")
        print(f"   Unique regions: {len(np.unique(parc_data))}")
        print(f"   SegId range: {parc_data.min()} to {parc_data.max()}")

        return parc_img, parc_data

    def create_laterality_map(self):
        """Create laterality index map"""

        print("\n Creating laterality index map...")

        # Load data
        segid_to_li, results_df = self.load_asymmetry_results()
        parc_img, parc_data = self.load_parcellation_data()

        # Initialize LI map
        li_map = np.zeros_like(parc_data, dtype=np.float32)

        # Track statistics
        mapped_regions = 0
        unmapped_regions = set()

        # Map each voxel to its region's LI
        unique_segids = np.unique(parc_data)

        for segid in unique_segids:
            if segid == 0:  # Skip background
                continue

            if segid in segid_to_li:
                # Set all voxels of this region to its LI value
                mask = parc_data == segid
                li_map[mask] = segid_to_li[segid]
                mapped_regions += 1
            else:
                unmapped_regions.add(segid)

        print(f" Mapped {mapped_regions} regions to LI values")
        if unmapped_regions:
            print(
                f"⚠️  Unmapped regions: {len(unmapped_regions)} (SegIds: {sorted(list(unmapped_regions))[:10]}{'...' if len(unmapped_regions) > 10 else ''})"
            )

        # Create statistics
        stats = {
            "total_voxels": int(li_map.size),
            "mapped_voxels": int(np.sum(li_map != 0)),
            "li_range": [float(li_map.min()), float(li_map.max())],
            "li_mean": float(np.mean(li_map[li_map != 0])),
            "li_std": float(np.std(li_map[li_map != 0])),
            "mapped_regions": mapped_regions,
            "unmapped_regions": len(unmapped_regions),
            "significant_asymmetry_threshold": 0.1,
        }

        # Count significant asymmetry voxels
        significant_voxels = np.sum(np.abs(li_map) > 0.1)
        stats["significant_asymmetry_voxels"] = int(significant_voxels)
        stats["significant_asymmetry_percentage"] = (
            float(100 * significant_voxels / stats["mapped_voxels"])
            if stats["mapped_voxels"] > 0
            else 0
        )

        print(f"📊 LI Map Statistics:")
        print(f"   Total voxels: {stats['total_voxels']:,}")
        print(
            f"   Mapped voxels: {stats['mapped_voxels']:,} ({100 * stats['mapped_voxels'] / stats['total_voxels']:.1f}%)"
        )
        print(f"   LI range: {stats['li_range'][0]:.4f} to {stats['li_range'][1]:.4f}")
        print(f"   LI mean ± std: {stats['li_mean']:.4f} ± {stats['li_std']:.4f}")
        print(
            f"   Significant asymmetry: {stats['significant_asymmetry_voxels']:,} voxels ({stats['significant_asymmetry_percentage']:.1f}%)"
        )

        return li_map, parc_img, stats, results_df

    def save_laterality_map(self, li_map, template_img, stats, results_df):
        """Save laterality index map as NIfTI"""

        print("\n💾 Saving laterality index map...")

        # Create new NIfTI image with same header as template
        li_img = nib.Nifti1Image(li_map, template_img.affine, template_img.header)

        # Update header description
        li_img.header["descrip"] = b"Laterality Index Map (LI per region)"

        # Save main LI map
        output_file = self.output_dir / f"{self.patient_id}_laterality_index_map.nii.gz"
        nib.save(li_img, output_file)
        print(f" Saved: {output_file}")

        # Create and save binary significant asymmetry mask
        sig_mask = (np.abs(li_map) > 0.1).astype(np.float32)
        sig_img = nib.Nifti1Image(sig_mask, template_img.affine, template_img.header)
        sig_img.header["descrip"] = b"Significant Asymmetry Mask (|LI| > 0.1)"

        sig_output_file = (
            self.output_dir / f"{self.patient_id}_significant_asymmetry_mask.nii.gz"
        )
        nib.save(sig_img, sig_output_file)
        print(f" Saved: {sig_output_file}")

        # Create and save thresholded maps for visualization
        # Positive LI map (Left > Right)
        pos_li_map = np.where(li_map > 0.1, li_map, 0).astype(np.float32)
        pos_img = nib.Nifti1Image(pos_li_map, template_img.affine, template_img.header)
        pos_img.header["descrip"] = b"Left > Right Asymmetry (LI > 0.1)"
        pos_output_file = (
            self.output_dir / f"{self.patient_id}_left_dominant_regions.nii.gz"
        )
        nib.save(pos_img, pos_output_file)
        print(f" Saved: {pos_output_file}")

        # Negative LI map (Right > Left)
        neg_li_map = np.where(li_map < -0.1, np.abs(li_map), 0).astype(np.float32)
        neg_img = nib.Nifti1Image(neg_li_map, template_img.affine, template_img.header)
        neg_img.header["descrip"] = b"Right > Left Asymmetry (LI < -0.1)"
        neg_output_file = (
            self.output_dir / f"{self.patient_id}_right_dominant_regions.nii.gz"
        )
        nib.save(neg_img, neg_output_file)
        print(f" Saved: {neg_output_file}")

        # Save statistics and mapping info
        stats_file = self.output_dir / f"{self.patient_id}_laterality_map_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f" Saved: {stats_file}")

        # Save region mapping table
        mapping_df = results_df[
            [
                "left_segid",
                "right_segid",
                "region",
                "laterality_index",
                "left_mean_perfusion",
                "right_mean_perfusion",
            ]
        ].copy()
        mapping_df["significant_asymmetry"] = mapping_df["laterality_index"].abs() > 0.1

        mapping_file = self.output_dir / f"{self.patient_id}_segid_to_li_mapping.csv"
        mapping_df.to_csv(mapping_file, index=False)
        print(f" Saved: {mapping_file}")

        return output_file, sig_output_file, pos_output_file, neg_output_file

    def create_visualization_info(self, results_df):
        """Create visualization information"""

        print("\n Creating visualization info...")

        significant_regions = results_df[results_df["laterality_index"].abs() > 0.1]

        viz_info = f"""
# {self.patient_id} Laterality Index Visualization Guide

## Files Created:
1. **{self.patient_id}_laterality_index_map.nii.gz** - Full LI map (all regions)
2. **{self.patient_id}_significant_asymmetry_mask.nii.gz** - Binary mask (|LI| > 0.1)
3. **{self.patient_id}_left_dominant_regions.nii.gz** - Left > Right regions (LI > 0.1)
4. **{self.patient_id}_right_dominant_regions.nii.gz** - Right > Left regions (LI < -0.1)

## Visualization Tips:

### In FSLeyes/MRIcroGL:
1. Load T1w image as underlay
2. Load LI map as overlay
3. Set color map to "Red-Yellow" or "Cool-Warm"
4. Set threshold to ±0.1 to show significant asymmetries
5. Use transparency ~70% to see anatomy underneath

### Color Scale Interpretation:
- **Red/Positive values**: Left hemisphere dominance (L > R)
- **Blue/Negative values**: Right hemisphere dominance (R > L)  
- **Zero/Background**: No asymmetry or balanced regions

## Significant Asymmetric Regions ({len(significant_regions)} regions):

"""

        for _, region in significant_regions.sort_values(
            "laterality_index", key=abs, ascending=False
        ).iterrows():
            direction = (
                "Left > Right" if region["laterality_index"] > 0 else "Right > Left"
            )
            viz_info += f"- **{region['region']}** (SegId: {region['left_segid']}/{region['right_segid']}): LI = {region['laterality_index']:.4f} ({direction})\n"

        viz_info += f"""

## Statistical Summary:
- Mean LI: {results_df["laterality_index"].mean():.4f}
- LI Range: {results_df["laterality_index"].min():.4f} to {results_df["laterality_index"].max():.4f}
- Significant asymmetries: {len(significant_regions)}/{len(results_df)} regions ({100 * len(significant_regions) / len(results_df):.1f}%)

## Recommended Visualization Commands:

### FSLeyes:
```bash
fsleyes T1w_acpc_dc_restore.nii.gz {self.patient_id}_laterality_index_map.nii.gz -cm red-yellow -dr -0.4 0.4 -a 70
```

### MRIcroGL:
```bash
mricrogl -o T1w_acpc_dc_restore.nii.gz -l {self.patient_id}_laterality_index_map.nii.gz -h 0.1 -l 0.4
```
        """

        viz_file = self.output_dir / f"{self.patient_id}_visualization_guide.md"
        with open(viz_file, "w") as f:
            f.write(viz_info)
        print(f" Saved: {viz_file}")

        return viz_file


def main():
    """Main execution function"""

    print(" Creating Laterality Index Maps for Brain Visualization")
    print("=" * 70)

    data_dir = "Dataset"
    patient_id = "P013"

    try:
        # Initialize mapper
        mapper = LateralityMapper(data_dir, patient_id)

        # Create laterality map
        li_map, template_img, stats, results_df = mapper.create_laterality_map()

        # Save maps and related files
        output_files = mapper.save_laterality_map(
            li_map, template_img, stats, results_df
        )

        # Create visualization guide
        viz_file = mapper.create_visualization_info(results_df)

        print("\n LATERALITY MAPPING COMPLETED!")
        print("=" * 70)
        print(f"📁 All files saved to: {mapper.output_dir}")

        print("\n Files created:")
        for file in sorted(mapper.output_dir.glob("*")):
            print(f"   ✓ {file.name}")

        print(f"\n📊 SUMMARY:")
        print(f"   • Patient: {patient_id}")
        print(f"   • Regions mapped: {stats['mapped_regions']}")
        print(
            f"   • Significant asymmetry: {stats['significant_asymmetry_percentage']:.1f}% of brain voxels"
        )
        print(
            f"   • LI range: {stats['li_range'][0]:.4f} to {stats['li_range'][1]:.4f}"
        )

        print(f"\n VISUALIZATION:")
        print("   Load the T1w image and laterality map in FSLeyes/MRIcroGL")
        print("   Use Red-Yellow colormap with ±0.1 threshold")
        print(f"   See {viz_file.name} for detailed instructions")

    except Exception as e:
        print(f"❌ Error during mapping: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
