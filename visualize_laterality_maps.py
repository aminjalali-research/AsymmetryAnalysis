#!/usr/bin/env python3
"""
Quick Visualization Script for Laterality Maps

This script provides simple visualization commands and basic analysis of the created laterality maps.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def quick_analysis():
    """Quick analysis of the laterality maps"""

    print("🔍 Quick Analysis of P013 Laterality Maps")
    print("=" * 50)

    maps_dir = Path("laterality_maps")

    # Load the main LI map
    li_file = maps_dir / "P013_laterality_index_map.nii.gz"
    if not li_file.exists():
        print(f"❌ LI map not found: {li_file}")
        return

    li_img = nib.load(li_file)
    li_data = li_img.get_fdata()

    # Basic statistics
    non_zero_mask = li_data != 0
    non_zero_li = li_data[non_zero_mask]

    print(f"📊 Laterality Index Statistics:")
    print(f"   Shape: {li_data.shape}")
    print(f"   Total voxels: {li_data.size:,}")
    print(
        f"   Non-zero voxels: {np.sum(non_zero_mask):,} ({100 * np.sum(non_zero_mask) / li_data.size:.1f}%)"
    )
    print(f"   LI range: {non_zero_li.min():.4f} to {non_zero_li.max():.4f}")
    print(f"   Mean LI: {non_zero_li.mean():.4f}")
    print(f"   Std LI: {non_zero_li.std():.4f}")

    # Count asymmetric regions
    left_dominant = np.sum(li_data > 0.1)
    right_dominant = np.sum(li_data < -0.1)
    balanced = np.sum(np.abs(li_data) <= 0.1) - np.sum(li_data == 0)

    print(f"\n🧠 Asymmetry Distribution:")
    print(f"   Left dominant (LI > 0.1): {left_dominant:,} voxels")
    print(f"   Right dominant (LI < -0.1): {right_dominant:,} voxels")
    print(f"   Balanced (|LI| ≤ 0.1): {balanced:,} voxels")
    print(f"   Background (LI = 0): {np.sum(li_data == 0):,} voxels")

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(non_zero_li, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    plt.axvline(0, color="black", linestyle="--", alpha=0.7, label="No asymmetry")
    plt.axvline(
        0.1, color="red", linestyle="--", alpha=0.7, label="Significance threshold"
    )
    plt.axvline(-0.1, color="red", linestyle="--", alpha=0.7)
    plt.xlabel("Laterality Index (LI)", fontweight="bold")
    plt.ylabel("Number of Voxels", fontweight="bold")
    plt.title("P013 Laterality Index Distribution", fontweight="bold", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)

    # Add statistics text
    stats_text = f"""Statistics:
Mean: {non_zero_li.mean():.3f}
Std: {non_zero_li.std():.3f}
Range: [{non_zero_li.min():.3f}, {non_zero_li.max():.3f}]
Significant: {100 * (left_dominant + right_dominant) / np.sum(non_zero_mask):.1f}%"""

    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
        fontfamily="monospace",
        fontsize=10,
    )

    plt.tight_layout()

    # Save histogram
    hist_file = maps_dir / "P013_LI_histogram.png"
    plt.savefig(hist_file, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Histogram saved: {hist_file}")

    return li_data, li_img


def print_visualization_commands():
    """Print visualization commands for different software"""

    print(f"\n🎨 VISUALIZATION COMMANDS:")
    print("=" * 50)

    print("### FSLeyes (recommended):")
    print("cd /home/amin/AsymmetryAnalysis")
    print("fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \\")
    print("         laterality_maps/P013_laterality_index_map.nii.gz \\")
    print("         -cm red-yellow -dr -0.4 0.4 -a 70 &")

    print("\n### Alternative - Significant regions only:")
    print("fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \\")
    print("         laterality_maps/P013_significant_asymmetry_mask.nii.gz \\")
    print("         -cm red -dr 0 1 -a 80 &")

    print("\n### Separate left/right dominant regions:")
    print("fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \\")
    print("         laterality_maps/P013_left_dominant_regions.nii.gz \\")
    print("         -cm red -dr 0 0.4 -a 70 \\")
    print("         laterality_maps/P013_right_dominant_regions.nii.gz \\")
    print("         -cm blue -dr 0 0.4 -a 70 &")

    print("\n### MRIcroGL:")
    print("mricrogl \\")
    print("  Dataset/P013/T1w_acpc_dc_restore.nii.gz \\")
    print("  -l laterality_maps/P013_laterality_index_map.nii.gz \\")
    print("  -h 0.1 -l 0.4")

    print(f"\n💡 INTERPRETATION:")
    print("   🔴 Red/Warm colors: Left > Right dominance")
    print("   🔵 Blue/Cool colors: Right > Left dominance")
    print("   ⚫ Dark/Zero: Balanced or unmapped regions")
    print("   📏 Threshold ±0.1 shows clinically significant asymmetries")


def main():
    """Main function"""

    try:
        # Quick analysis
        li_data, li_img = quick_analysis()

        # Print visualization commands
        print_visualization_commands()

        print(
            f"\n✅ Analysis complete! Use the commands above to visualize your laterality maps."
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
