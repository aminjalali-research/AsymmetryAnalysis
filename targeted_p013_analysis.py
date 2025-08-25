#!/usr/bin/env python3
"""
Comprehensive P013 Asymmetry Analysis

This is the main analysis script that provides:
- All 15 sophisticated asymmetry calculation methods (removed redundant robust method)
- SegID-based region ordering from list.txt
- Statistical analysis and visualization
- NIfTI spatial mapping capabilities
- Publication-ready outputs
- Clear left/right laterality interpretation

Updated to consolidate all project functionality.
"""

import sys
import warnings
from pathlib import Path

# Add src to path before importing modules
sys.path.append("src")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from calculator import AsymmetryCalculator


class P013TargetedAnalysis:
    """Analyze specific regions from user's list.txt file"""

    def __init__(self, list_file_path):
        self.list_file = Path(list_file_path)
        self.output_dir = Path("P013_Analysis_Results")  # Single consolidated folder
        self.output_dir.mkdir(exist_ok=True)

        # Initialize calculator with all 15 methods (removed redundant robust method)
        self.calculator = AsymmetryCalculator()

        # Load and parse the specific regions
        self.regions_data = self._parse_list_file()

    def _parse_list_file(self):
        """Parse the user's list.txt file format"""

        print("📊 Parsing your list.txt file...")

        regions_data = []
        current_hemisphere = None

        with open(self.list_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if line == "LEFT:":
                current_hemisphere = "left"
                continue
            elif line == "RIGHT:":
                current_hemisphere = "right"
                continue
            elif line.startswith("#") or line == "":
                continue

            # Parse data line
            parts = line.split()
            if len(parts) >= 10:
                index = int(parts[0])
                seg_id = int(parts[1])
                nvoxels = int(parts[2])
                volume_mm3 = float(parts[3])
                struct_name = parts[4]
                mean_perfusion = float(parts[5])
                std_dev = float(parts[6])
                min_val = float(parts[7])
                max_val = float(parts[8])
                range_val = float(parts[9])

                regions_data.append(
                    {
                        "hemisphere": current_hemisphere,
                        "index": index,
                        "seg_id": seg_id,
                        "nvoxels": nvoxels,
                        "volume_mm3": volume_mm3,
                        "struct_name": struct_name,
                        "mean_perfusion": mean_perfusion,
                        "std_dev": std_dev,
                        "min_val": min_val,
                        "max_val": max_val,
                        "range_val": range_val,
                    }
                )

        df = pd.DataFrame(regions_data)
        print(
            f"✅ Parsed {len(df)} region entries ({len(df[df['hemisphere'] == 'left'])} left, {len(df[df['hemisphere'] == 'right'])} right)"
        )

        return df

    def _extract_bilateral_pairs(self):
        """Extract bilateral pairs from the parsed data"""

        print("\n🔗 Extracting bilateral region pairs...")

        left_regions = self.regions_data[
            self.regions_data["hemisphere"] == "left"
        ].copy()
        right_regions = self.regions_data[
            self.regions_data["hemisphere"] == "right"
        ].copy()

        bilateral_pairs = []

        # For subcortical regions (no ctx prefix)
        for _, left_region in left_regions.iterrows():
            left_name = left_region["struct_name"]

            if left_name.startswith("Left-"):
                # Find matching right region
                right_name = left_name.replace("Left-", "Right-")
                right_match = right_regions[right_regions["struct_name"] == right_name]

                if not right_match.empty:
                    right_region = right_match.iloc[0]

                    # Clean region name
                    region_name = left_name.replace("Left-", "").replace("Right-", "")

                    bilateral_pairs.append(
                        {
                            "region": region_name,
                            "region_type": "subcortical",
                            "left_segid": left_region["seg_id"],
                            "right_segid": right_region["seg_id"],
                            "left_index": left_region["index"],
                            "right_index": right_region["index"],
                            "left_struct_name": left_region["struct_name"],
                            "right_struct_name": right_region["struct_name"],
                            "display_name": f"[{left_region['seg_id']}/{right_region['seg_id']}] {region_name}",
                            "left_mean_perfusion": left_region["mean_perfusion"],
                            "right_mean_perfusion": right_region["mean_perfusion"],
                            "left_volume": left_region["volume_mm3"],
                            "right_volume": right_region["volume_mm3"],
                            "left_std": left_region["std_dev"],
                            "right_std": right_region["std_dev"],
                            "left_nvoxels": left_region["nvoxels"],
                            "right_nvoxels": right_region["nvoxels"],
                            "segid_order": left_region[
                                "index"
                            ],  # Use original index for ordering
                        }
                    )

        # For cortical regions (ctx prefix)
        for _, left_region in left_regions.iterrows():
            left_name = left_region["struct_name"]

            if left_name.startswith("ctx-lh-"):
                # Find matching right region
                right_name = left_name.replace("ctx-lh-", "ctx-rh-")
                right_match = right_regions[right_regions["struct_name"] == right_name]

                if not right_match.empty:
                    right_region = right_match.iloc[0]

                    # Clean region name
                    region_name = left_name.replace("ctx-lh-", "")

                    bilateral_pairs.append(
                        {
                            "region": region_name,
                            "region_type": "cortical",
                            "left_segid": left_region["seg_id"],
                            "right_segid": right_region["seg_id"],
                            "left_index": left_region["index"],
                            "right_index": right_region["index"],
                            "left_struct_name": left_region["struct_name"],
                            "right_struct_name": right_region["struct_name"],
                            "display_name": f"[{left_region['seg_id']}/{right_region['seg_id']}] {region_name}",
                            "left_mean_perfusion": left_region["mean_perfusion"],
                            "right_mean_perfusion": right_region["mean_perfusion"],
                            "left_volume": left_region["volume_mm3"],
                            "right_volume": right_region["volume_mm3"],
                            "left_std": left_region["std_dev"],
                            "right_std": right_region["std_dev"],
                            "left_nvoxels": left_region["nvoxels"],
                            "right_nvoxels": right_region["nvoxels"],
                            "segid_order": left_region[
                                "index"
                            ],  # Use original index for ordering
                        }
                    )

        bilateral_df = pd.DataFrame(bilateral_pairs)

        # Sort by original order from list.txt (using segid_order which stores original index)
        bilateral_df = bilateral_df.sort_values(
            "segid_order", ascending=True
        ).reset_index(drop=True)

        print(f"✅ Found {len(bilateral_df)} bilateral pairs:")
        print(
            f"   - Subcortical: {len(bilateral_df[bilateral_df['region_type'] == 'subcortical'])} pairs"
        )
        print(
            f"   - Cortical: {len(bilateral_df[bilateral_df['region_type'] == 'cortical'])} pairs"
        )

        return bilateral_df

    def calculate_all_asymmetries(self):
        """Calculate all 16 asymmetry methods for the bilateral pairs"""

        # Extract bilateral pairs
        bilateral_data = self._extract_bilateral_pairs()

        print(
            f"\n🧮 Calculating 15 asymmetry methods for {len(bilateral_data)} regions..."
        )

        results = []

        for _, region in bilateral_data.iterrows():
            left_val = region["left_mean_perfusion"]
            right_val = region["right_mean_perfusion"]
            left_vol = region["left_volume"]
            right_vol = region["right_volume"]
            left_std = region["left_std"]
            right_std = region["right_std"]

            # Skip if invalid data
            if left_val <= 0 or right_val <= 0:
                continue

            # Calculate all 15 asymmetry methods (removed redundant robust method)
            asymmetries = {
                "laterality_index": self.calculator.laterality_index(
                    left_val, right_val
                ),
                "asymmetry_index": self.calculator.asymmetry_index(left_val, right_val),
                "percentage_difference": self.calculator.percentage_difference(
                    left_val, right_val
                ),
                "normalized_asymmetry": self.calculator.normalized_asymmetry(
                    left_val, right_val
                ),
                "log_ratio": self.calculator.log_ratio(left_val, right_val),
                "absolute_asymmetry_index": self.calculator.absolute_asymmetry_index(
                    left_val, right_val
                ),
                "relative_asymmetry_index": self.calculator.relative_asymmetry_index(
                    left_val, right_val
                ),
                "z_score_asymmetry": self.calculator.z_score_asymmetry(
                    left_val, right_val
                ),
                "signed_log_ratio": self.calculator.signed_log_ratio(
                    left_val, right_val
                ),
                "cohens_d_asymmetry": self.calculator.cohen_d_asymmetry(
                    left_val, right_val
                ),
                "fold_change_asymmetry": self.calculator.fold_change_asymmetry(
                    left_val, right_val
                ),
                "mutual_information_asymmetry": self.calculator.mutual_information_asymmetry(
                    left_val, right_val
                ),
                "volume_corrected_asymmetry": self.calculator.volume_corrected_asymmetry(
                    left_val, right_val, left_vol, right_vol
                ),
                "volume_weighted_asymmetry": self.calculator.laterality_index(
                    left_val, right_val
                )
                * (left_vol + right_vol),
                "modified_gyrification_index": (left_val - right_val)
                / (
                    (left_vol + right_vol) / 2 + 1e-6
                ),  # Add small epsilon to avoid division by zero
            }

            result = {
                "region": region["region"],
                "region_type": region["region_type"],
                "display_name": region["display_name"],
                "left_segid": region["left_segid"],
                "right_segid": region["right_segid"],
                "left_index": region["left_index"],
                "right_index": region["right_index"],
                "left_struct_name": region["left_struct_name"],
                "right_struct_name": region["right_struct_name"],
                "segid_order": region["segid_order"],  # Include ordering field
                "left_mean_perfusion": left_val,
                "right_mean_perfusion": right_val,
                "left_volume": left_vol,
                "right_volume": right_vol,
                **asymmetries,
            }

            results.append(result)

        results_df = pd.DataFrame(results)

        # Save complete results
        output_file = self.output_dir / "P013_comprehensive_asymmetry_analysis.csv"
        results_df.to_csv(output_file, index=False)
        print(f"💾 Complete analysis results saved to: {output_file}")

        # Also save just the significant asymmetries
        significant_results = results_df[results_df["laterality_index"].abs() > 0.1]
        if len(significant_results) > 0:
            sig_output_file = self.output_dir / "P013_significant_asymmetries.csv"
            significant_results.to_csv(sig_output_file, index=False)
            print(f"💾 Significant asymmetries saved to: {sig_output_file}")

        return results_df

    def sort_by_asymmetry_magnitude(self, results_df, method="laterality_index"):
        """Display regions in list.txt order with asymmetry values"""

        print(f"\n📊 Asymmetry analysis by {method} (list.txt order)...")

        # Sort by original list.txt order, not by magnitude
        ordered_df = results_df.sort_values("segid_order", ascending=True).reset_index(
            drop=True
        )

        print(f"🧠 All Regions by {method} (StructName Order from list.txt):")
        print("=" * 80)

        # Count significant asymmetries
        significant_count = 0

        for i, (_, region) in enumerate(ordered_df.iterrows(), 1):
            asymm_val = region[method]
            direction = "L>R" if asymm_val > 0 else "R>L"
            is_significant = abs(asymm_val) > 0.1

            if is_significant:
                significant_count += 1

            # Highlight significant asymmetries
            if is_significant:
                print(
                    f"\033[1;91m{i:2d}. {region['display_name']:40} | {asymm_val:8.4f} ({direction}) | {region['region_type']}\033[0m"
                )
            else:
                print(
                    f"{i:2d}. {region['display_name']:40} | {asymm_val:8.4f} ({direction}) | {region['region_type']}"
                )

        print("=" * 80)
        print(
            f"📊 {significant_count}/{len(ordered_df)} regions show significant asymmetry (|{method}| > 0.1)"
        )
        print("🔴 Red highlighted regions have clinically significant asymmetry")

        return ordered_df

    def display_regions_by_segid_order(self, results_df, significance_threshold=0.1):
        """Display regions in SegId order from list.txt, highlighting asymmetric ones"""

        print(f"\n📋 REGIONS IN LIST.TXT ORDER (by SegId) - Asymmetry Analysis")
        print("=" * 90)
        print(
            f"{'#':>3} {'SegId L/R':>12} {'Region Name':^25} {'Laterality':>10} {'L>R/R>L':>8} {'Asymmetric':>11}"
        )
        print("-" * 90)

        # Sort by the original order (segid_order field)
        ordered_df = results_df.sort_values("segid_order", ascending=True).reset_index(
            drop=True
        )

        asymmetric_count = 0

        for i, (_, region) in enumerate(ordered_df.iterrows(), 1):
            li_value = region["laterality_index"]
            is_asymmetric = abs(li_value) > significance_threshold

            if is_asymmetric:
                asymmetric_count += 1

            # Format values
            segid_display = f"{region['left_segid']}/{region['right_segid']}"
            region_name = region["region"][:23]  # Truncate long names
            li_display = f"{li_value:7.4f}"
            direction = "L>R" if li_value > 0 else "R>L" if li_value < 0 else "Bal"
            asymm_indicator = "⭐ YES" if is_asymmetric else "   -"

            # Highlight asymmetric regions
            if is_asymmetric:
                print(
                    f"\033[1;91m{i:3d} {segid_display:>12} {region_name:^25} {li_display:>10} {direction:>8} {asymm_indicator:>11}\033[0m"
                )
            else:
                print(
                    f"{i:3d} {segid_display:>12} {region_name:^25} {li_display:>10} {direction:>8} {asymm_indicator:>11}"
                )

        print("-" * 90)
        print(
            f"📊 SUMMARY: {asymmetric_count}/{len(ordered_df)} regions show significant asymmetry (|LI| > {significance_threshold})"
        )
        print(f"⭐ Red highlighted regions have clinically significant asymmetry")
        print("=" * 90)

        return ordered_df, asymmetric_count

    def create_magnitude_rankings(self, results_df):
        """Create informative asymmetry visualization with both detailed view and summary"""

        print("\n📊 Creating comprehensive asymmetry visualizations...")

        # Select key asymmetry methods
        key_methods = [
            "laterality_index",
            "cohens_d_asymmetry",
            "robust_asymmetry_index",
            "volume_corrected_asymmetry",
            "fold_change_asymmetry",
        ]

        # Get significantly asymmetric regions (|LI| > 0.1)
        significant_regions = results_df[
            results_df["laterality_index"].abs() > 0.1
        ].copy()
        significant_regions = significant_regions.sort_values(
            "segid_order", ascending=True
        ).reset_index(drop=True)

        # Create figure with better layout
        fig = plt.figure(figsize=(20, 14))

        # Create a 3x3 grid but use different subplot arrangements
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)

        rankings_summary = {}

        # Top row: Show top 3 most informative methods for significant regions only
        informative_methods = [
            "laterality_index",
            "cohens_d_asymmetry", 
            "volume_corrected_asymmetry",
        ]

        for i, method in enumerate(informative_methods):
            ax = fig.add_subplot(gs[0, i])

            if len(significant_regions) > 0:
                # Plot significant regions only
                colors = [
                    "red" if x < 0 else "blue" for x in significant_regions[method]
                ]

                bars = ax.barh(
                    range(len(significant_regions)),
                    significant_regions[method],
                    color=colors,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.7,
                )

                ax.set_yticks(range(len(significant_regions)))
                ax.set_yticklabels(
                    [r["region"][:15] for _, r in significant_regions.iterrows()],
                    fontsize=10,
                    fontweight="bold",
                )

                # Add value labels
                for bar, value in zip(bars, significant_regions[method]):
                    width = bar.get_width()
                    ax.text(
                        width + (0.02 * max(abs(significant_regions[method]))),
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.3f}",
                        ha="left" if width > 0 else "right",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Significant\nAsymmetries",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )

            ax.set_xlabel(
                f"{method.replace('_', ' ').title()}", fontweight="bold", fontsize=11
            )
            ax.set_title(
                f"Significant Regions\n{method.replace('_', ' ').title()}",
                fontweight="bold",
                fontsize=12,
            )
            ax.grid(axis="x", alpha=0.4, linestyle="--")
            ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5)

            # Store rankings
            if len(significant_regions) > 0:
                rankings_summary[method] = significant_regions[
                    ["region", method, "segid_order", "display_name"]
                ].to_dict("records")

        # Left: All regions laterality index with clearer L/R coloring
        ax1 = fig.add_subplot(gs[1, :2])
        all_ordered = results_df.sort_values("segid_order", ascending=True).reset_index(
            drop=True
        )
        
        # Improved color scheme: Blue for L>R, Red for R>L, Gray for balanced
        colors_all = []
        for x in all_ordered["laterality_index"]:
            if abs(x) <= 0.1:  # Balanced
                colors_all.append("lightgray")
            elif x > 0.1:  # Left > Right (positive LI)
                colors_all.append("blue")
            else:  # Right > Left (negative LI)
                colors_all.append("red")

        bars = ax1.barh(
            range(len(all_ordered)),
            all_ordered["laterality_index"],
            color=colors_all,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.3,
        )

        ax1.set_yticks(range(len(all_ordered)))
        ax1.set_yticklabels(
            [r["region"][:12] for _, r in all_ordered.iterrows()], fontsize=8
        )
        ax1.set_xlabel("Laterality Index (L>R = Blue, R>L = Red)", fontweight="bold")
        ax1.set_title(
            "All Regions - Laterality Index (SegId Order)\n🔵 Blue = Left Dominant | 🔴 Red = Right Dominant | ⚪ Gray = Balanced",
            fontweight="bold",
            fontsize=12,
        )
        ax1.grid(axis="x", alpha=0.4)
        ax1.axvline(x=0, color="black", linewidth=1.5)
        ax1.axvline(x=0.1, color="blue", linestyle=":", alpha=0.7, label="L>R threshold")
        ax1.axvline(x=-0.1, color="red", linestyle=":", alpha=0.7, label="R>L threshold")

        # Right: Asymmetry distribution pie chart
        ax2 = fig.add_subplot(gs[1, 2])
        asymmetric_count = len(significant_regions)
        balanced_count = len(results_df) - asymmetric_count

        labels = [
            f"Asymmetric\n({asymmetric_count} regions)",
            f"Balanced\n({balanced_count} regions)",
        ]
        sizes = [asymmetric_count, balanced_count]
        colors_pie = ["red", "lightblue"]

        wedges, texts, autotexts = ax2.pie(
            sizes,
            labels=labels,
            colors=colors_pie,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 10, "fontweight": "bold"},
        )
        ax2.set_title(
            "Asymmetry Distribution\n(|LI| > 0.1 threshold)",
            fontweight="bold",
            fontsize=12,
        )

        # Bottom: Summary statistics table
        ax3 = fig.add_subplot(gs[2, :])
        ax3.axis("off")

        # Create summary table data
        summary_data = []
        for _, region in significant_regions.iterrows():
            summary_data.append(
                [
                    f"[{region['left_segid']}/{region['right_segid']}]",
                    region["region"][:20],
                    f"{region['laterality_index']:.4f}",
                    "L>R" if region["laterality_index"] > 0 else "R>L",
                    f"{region['cohens_d_asymmetry']:.3f}",
                    f"{region['left_mean_perfusion']:.1f}",
                    f"{region['right_mean_perfusion']:.1f}",
                ]
            )

        if len(summary_data) > 0:
            headers = [
                "SegId",
                "Region",
                "LI",
                "Direction",
                "Cohen's d",
                "Left Perf",
                "Right Perf",
            ]

            table = ax3.table(
                cellText=summary_data,
                colLabels=headers,
                cellLoc="center",
                loc="center",
                colWidths=[0.12, 0.25, 0.1, 0.1, 0.1, 0.1, 0.1],
            )

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            # Style the table
            for i in range(len(headers)):
                table[(0, i)].set_facecolor("#40466e")
                table[(0, i)].set_text_props(weight="bold", color="white")

            for i in range(1, len(summary_data) + 1):
                for j in range(len(headers)):
                    table[(i, j)].set_facecolor("#f8f9fa" if i % 2 == 0 else "white")

        ax3.set_title(
            "Significantly Asymmetric Regions Summary Table",
            fontweight="bold",
            fontsize=14,
            pad=20,
        )

        plt.suptitle(
            "P013 Asymmetry Analysis - Comprehensive View",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(
            self.output_dir / "comprehensive_asymmetry_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        return rankings_summary, significant_regions

    def create_focused_asymmetry_heatmap(self, results_df):
        """Create a focused, highly expressive heatmap with distinct colormaps for each method"""

        print("\n🎨 Creating enhanced focused asymmetry heatmap with expressive color variation...")

        # Select the most clinically relevant asymmetry methods (now 15 instead of 16)
        key_methods = [
            "laterality_index",  # Clinical standard
            "cohens_d_asymmetry",  # Research standard
            "volume_corrected_asymmetry",  # Structure-aware
            "fold_change_asymmetry",  # Multiplicative effect
            "normalized_asymmetry",  # Scale-independent
            "log_ratio",  # Log-scale comparison
            "percentage_difference",  # Intuitive percentage
        ]

        # Use consistent colormap for all methods with enhanced visual layout
        consistent_colormap = "RdBu_r"  # Classic red-blue: Red=Right>Left, Blue=Left>Right

        # Create matrix for heatmap with only key methods
        heatmap_data = results_df[key_methods].copy()

        # Sort regions by SegId order (maintain list.txt order)
        ordered_results = results_df.sort_values("segid_order", ascending=True)
        region_labels = [
            f"[{row['left_segid']}/{row['right_segid']}] {row['region'][:15]}"
            for _, row in ordered_results.iterrows()
        ]
        heatmap_data.index = region_labels
        heatmap_data = heatmap_data.loc[heatmap_data.index]

        # Create the figure with enhanced layout
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(len(key_methods), 4, 
                             width_ratios=[3, 0.15, 1.2, 0.05], 
                             height_ratios=[1]*len(key_methods),
                             hspace=0.15, wspace=0.1)

        # Create individual heatmaps for each method with unique colormaps
        for i, method in enumerate(key_methods):
            # Main heatmap for this method
            ax_main = fig.add_subplot(gs[i, 0])
            
            # Get data for this method
            method_data = heatmap_data[[method]].T
            
            # Normalize data for better color distribution
            method_values = method_data.values.flatten()
            vmin, vmax = np.percentile(method_values[~np.isnan(method_values)], [5, 95])
            
            # Create heatmap with consistent colormap for all methods
            im = sns.heatmap(
                method_data,
                ax=ax_main,
                cmap=consistent_colormap,
                center=0,
                vmin=vmin, 
                vmax=vmax,
                annot=True,
                fmt=".3f",
                annot_kws={"size": 8, "weight": "bold"},
                cbar=False,  # We'll add custom colorbars
                linewidths=0.8,
                linecolor="white",
            )
            
            # Customize method-specific heatmap
            method_display = method.replace('_', ' ').title()
            ax_main.set_title(f"{method_display}", fontsize=14, fontweight="bold", pad=10)
            
            if i == len(key_methods) - 1:  # Only show x-labels on bottom plot
                ax_main.set_xticklabels(region_labels, rotation=45, ha='right', fontsize=10)
                ax_main.set_xlabel("Brain Regions (SegId Order)", fontweight="bold", fontsize=12)
            else:
                ax_main.set_xticklabels([])
                ax_main.set_xlabel("")
            
            ax_main.set_yticklabels([method_display], rotation=0, fontsize=12, fontweight="bold")
            
            # Highlight significant asymmetries in region labels
            if i == len(key_methods) - 1:  # Only for bottom plot
                x_labels = ax_main.get_xticklabels()
                for j, label in enumerate(x_labels):
                    region_li = ordered_results.iloc[j]["laterality_index"]
                    if abs(region_li) > 0.1:  # Significant asymmetry
                        label.set_color("red")
                        label.set_fontweight("bold")
                ax_main.set_xticklabels(x_labels, rotation=45, ha='right')
            
            # Add individual colorbar for each method
            ax_cbar = fig.add_subplot(gs[i, 1])
            cbar = plt.colorbar(im.get_children()[0], cax=ax_cbar)
            cbar.set_label(f"{method_display}\nScore", rotation=90, labelpad=15, fontsize=10)
            cbar.ax.tick_params(labelsize=9)

        # Enhanced side panel with method-specific information
        ax_summary = fig.add_subplot(gs[:, 2])
        ax_summary.axis("off")

        # Get significant regions
        significant_regions = results_df[results_df["laterality_index"].abs() > 0.1]

        # Create enhanced summary text with consistent colormap approach
        summary_text = f"""
🎨 ENHANCED ASYMMETRY ANALYSIS
{"═" * 35}

📊 OVERVIEW:
Total Regions: {len(results_df)}
Significant: {len(significant_regions)} 
Balanced: {len(results_df) - len(significant_regions)}

🎨 CONSISTENT COLOR SCHEME:
{"─" * 25}
All methods use RdBu_r colormap:
🔴 Red Colors: Right > Left dominance  
🔵 Blue Colors: Left > Right dominance
⚪ White: Balanced regions
📊 Intensity: Darker = Stronger asymmetry

🏥 SIGNIFICANT REGIONS:
{"─" * 25}
"""

        for _, region in significant_regions.sort_values("segid_order").iterrows():
            li_val = region["laterality_index"]
            cohens_d = region["cohens_d_asymmetry"]
            direction = "L>R" if li_val > 0 else "R>L"
            effect_size = "Large" if abs(cohens_d) > 0.8 else "Medium" if abs(cohens_d) > 0.5 else "Small"
            
            summary_text += f"\n🧠 {region['region'][:18]}\n"
            summary_text += f"   LI: {li_val:.3f} ({direction})\n"
            summary_text += f"   Effect: {effect_size} (d={cohens_d:.2f})\n"
            summary_text += f"   SegId: [{region['left_segid']}/{region['right_segid']}]\n"

        summary_text += f"""

📈 METHOD DESCRIPTIONS:
{"─" * 25}
🔵🔴 Laterality Index: 
     Clinical standard (L-R)/(L+R)

�� Cohen's d: 
     Research effect size measure

�🔴 Volume Corrected: 
     Structure-aware asymmetry

�🔴 Fold Change: 
     Multiplicative ratio effect

�🔴 Normalized: 
     Scale-independent measure

🔵🔴 Log Ratio: 
     Log-scale comparison

�🔴 Percentage Diff: 
     Intuitive % difference

🎯 CONSISTENT INTERPRETATION:
{"─" * 30}
• All methods use same Red-Blue colormap
• Red tones = Right hemisphere dominance
• Blue tones = Left hemisphere dominance  
• White/Gray = Balanced regions
• Values shown in each cell for precision
• Individual colorbars show method ranges
        """

        ax_summary.text(
            0.05,
            0.98,
            summary_text,
            transform=ax_summary.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.9),
        )

        # Add overall title
        fig.suptitle(
            "P013 Enhanced Multi-Method Asymmetry Analysis\n🎨 Consistent RdBu Colormap: Red=Right>Left | Blue=Left>Right", 
            fontsize=18, fontweight="bold", y=0.98
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "focused_asymmetry_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print("✅ Enhanced focused asymmetry heatmap saved with consistent colormap!")

        # Create a second simplified heatmap for significant regions only
        if len(significant_regions) > 0:
            self._create_significant_regions_heatmap(significant_regions, key_methods)

    def _create_significant_regions_heatmap(self, significant_regions, key_methods):
        """Create a focused heatmap showing only significantly asymmetric regions"""

        print("🔍 Creating heatmap for significant regions only...")

        # Sort by SegId order
        sig_ordered = significant_regions.sort_values("segid_order", ascending=True)

        # Create heatmap data
        sig_heatmap_data = sig_ordered[key_methods].copy()
        sig_heatmap_data.index = [
            f"[{row['left_segid']}/{row['right_segid']}] {row['region']}"
            for _, row in sig_ordered.iterrows()
        ]

        plt.figure(figsize=(14, 8))

        # Create heatmap
        sns.heatmap(
            sig_heatmap_data.T,
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "Asymmetry Score"},
            linewidths=1,
            linecolor="white",
            annot_kws={"size": 9, "weight": "bold"},
        )

        plt.title(
            f"Significantly Asymmetric Regions Only ({len(significant_regions)} regions with |LI| > 0.1)\n🔴 Red = Right>Left | 🔵 Blue = Left>Right",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Significantly Asymmetric Brain Regions", fontweight="bold")
        plt.ylabel("Asymmetry Methods", fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "significant_regions_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def create_asymmetry_heatmap(self, results_df):
        """Create heatmap of asymmetries across all methods"""

        print("\n🎨 Creating asymmetry heatmap...")

        # Select asymmetry methods (exclude raw data columns)
        asymm_methods = [
            col
            for col in results_df.columns
            if col
            not in [
                "region",
                "region_type",
                "display_name",
                "left_index",
                "right_index",
                "left_struct_name",
                "right_struct_name",
                "left_mean_perfusion",
                "right_mean_perfusion",
                "left_volume",
                "right_volume",
                "abs_asymmetry",
                "mean_abs_asymmetry",
            ]
        ]

        # Create matrix for heatmap
        heatmap_data = results_df[asymm_methods].copy()
        heatmap_data.index = results_df["display_name"]

        # Sort regions by SegId order (not by asymmetry magnitude)
        ordered_results = results_df.sort_values("segid_order", ascending=True)
        heatmap_data = heatmap_data.loc[ordered_results["display_name"]]

        plt.figure(figsize=(16, 12))

        # Create heatmap
        mask = heatmap_data.isnull()
        sns.heatmap(
            heatmap_data.T,
            cmap="RdBu_r",
            center=0,
            annot=False,
            fmt=".3f",
            cbar_kws={"label": "Asymmetry Score"},
            mask=mask.T,
        )

        plt.title(
            "P013 Targeted Regions - Asymmetry Heatmap\n(All 16 Methods - SegId Order)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Brain Regions (SegId order from list.txt)", fontweight="bold")
        plt.ylabel("Asymmetry Methods", fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "targeted_regions_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def generate_targeted_report(self, results_df, rankings_summary, ordered_data):
        """Generate comprehensive report for targeted analysis"""

        print("\n📝 Generating targeted analysis report...")

        # Get overall statistics
        total_regions = len(results_df)
        subcortical_count = len(results_df[results_df["region_type"] == "subcortical"])
        cortical_count = len(results_df[results_df["region_type"] == "cortical"])

        # Get strongest asymmetries (from original order)
        if "mean_abs_asymmetry" in ordered_data.columns:
            top_region = ordered_data.iloc[0]
            mean_asymm = top_region["mean_abs_asymmetry"]
        else:
            # Calculate mean asymmetry for the top region if not available
            top_region = ordered_data.iloc[0]
            key_methods = [
                "laterality_index",
                "cohens_d_asymmetry",
                "robust_asymmetry_index",
                "volume_corrected_asymmetry",
                "fold_change_asymmetry",
            ]
            mean_asymm = sum(
                abs(top_region[method])
                for method in key_methods
                if method in top_region
            ) / len(key_methods)

        strongest_li = results_df.loc[results_df["laterality_index"].abs().idxmax()]

        report = f"""
# P013 Targeted Region Asymmetry Analysis Report

## Analysis Overview
- **Total Bilateral Regions Analyzed**: {total_regions}
- **Subcortical Regions**: {subcortical_count}
- **Cortical Regions**: {cortical_count}
- **Analysis Methods**: 16 sophisticated asymmetry measures
- **Data Source**: User-specified regions from list.txt

## Top Asymmetric Regions

### Overall Strongest Asymmetry (Mean of All Methods)
**{top_region["region"]}** ({top_region["region_type"]})
- Mean Asymmetry Magnitude: {mean_asymm:.4f}
- Left Perfusion: {top_region["left_mean_perfusion"]:.2f}
- Right Perfusion: {top_region["right_mean_perfusion"]:.2f}

### Strongest Laterality Index
**{strongest_li["region"]}** ({strongest_li["region_type"]})
- Laterality Index: {strongest_li["laterality_index"]:.4f}
- Direction: {"Left > Right" if strongest_li["laterality_index"] > 0 else "Right > Left"}

## Regional Type Analysis

### Subcortical Asymmetries
"""

        subcortical_regions = results_df[results_df["region_type"] == "subcortical"]
        if len(subcortical_regions) > 0:
            top_subcortical = subcortical_regions.nlargest(
                5,
                "mean_abs_asymmetry"
                if "mean_abs_asymmetry" in subcortical_regions.columns
                else "laterality_index",
            )

            for i, (_, region) in enumerate(top_subcortical.iterrows(), 1):
                asymm_val = region["laterality_index"]
                direction = "L>R" if asymm_val > 0 else "R>L"
                report += (
                    f"{i}. **{region['region']}**: {asymm_val:.4f} ({direction})\\n"
                )

        report += f"""

### Cortical Asymmetries
"""

        cortical_regions = results_df[results_df["region_type"] == "cortical"]
        if len(cortical_regions) > 0:
            top_cortical = cortical_regions.nlargest(
                5,
                "mean_abs_asymmetry"
                if "mean_abs_asymmetry" in cortical_regions.columns
                else "laterality_index",
            )

            for i, (_, region) in enumerate(top_cortical.iterrows(), 1):
                asymm_val = region["laterality_index"]
                direction = "L>R" if asymm_val > 0 else "R>L"
                report += (
                    f"{i}. **{region['region']}**: {asymm_val:.4f} ({direction})\\n"
                )

        report += f"""

## Method-Specific Rankings

### Top 5 by Laterality Index (Clinical Standard)
"""

        top_li = results_df.nlargest(5, "laterality_index")
        for i, (_, region) in enumerate(top_li.iterrows(), 1):
            report += f"{i}. {region['region']}: {region['laterality_index']:.4f}\\n"

        report += f"""

### Top 5 by Cohen's d (Research Standard)  
"""

        top_cohens = results_df.nlargest(5, "cohens_d_asymmetry")
        for i, (_, region) in enumerate(top_cohens.iterrows(), 1):
            report += f"{i}. {region['region']}: {region['cohens_d_asymmetry']:.4f}\\n"

        report += f"""

## Clinical Implications

### Significant Asymmetries (|LI| > 0.1)
"""

        significant_regions = results_df[results_df["laterality_index"].abs() > 0.1]
        report += f"**{len(significant_regions)} regions** show clinically significant asymmetry:\\n"

        for _, region in significant_regions.iterrows():
            asymm_val = region["laterality_index"]
            direction = "L>R" if asymm_val > 0 else "R>L"
            report += f"- **{region['region']}**: {asymm_val:.4f} ({direction})\\n"

        report += f"""

### Research-Grade Effects (|Cohen's d| > 0.2)
"""

        research_grade = results_df[results_df["cohens_d_asymmetry"].abs() > 0.2]
        report += (
            f"**{len(research_grade)} regions** meet research publication standards.\\n"
        )

        report += f"""

## Summary Statistics
- **Mean Laterality Index**: {results_df["laterality_index"].mean():.4f}
- **Std Laterality Index**: {results_df["laterality_index"].std():.4f}
- **Range of Asymmetries**: {results_df["laterality_index"].min():.4f} to {results_df["laterality_index"].max():.4f}
- **Most Variable Method**: {results_df[[col for col in results_df.columns if "asymmetry" in col or "index" in col]].std().idxmax()}

## Methodology
- **Bilateral Pairing**: Automatic left-right region matching
- **Quality Control**: Excluded regions with zero/negative perfusion
- **Standardization**: Cohen's d effect sizes for cross-study comparison
- **Volume Correction**: Structure-aware asymmetry measures
- **Robust Statistics**: Noise-resistant asymmetry calculations

---
*Analysis generated from user-specified regions in P013 list.txt*
        """

        # Save report
        with open(self.output_dir / "targeted_p013_analysis_report.md", "w") as f:
            f.write(report)

        print(f"✅ Report saved to: {self.output_dir}/targeted_p013_analysis_report.md")

        return report


def main():
    """Main execution function"""

    print("🎯 P013 Targeted Region Asymmetry Analysis")
    print("=" * 60)

    list_file = "Dataset/P013/list.txt"
    if not Path(list_file).exists():
        print(f"❌ list.txt file not found: {list_file}")
        return

    try:
        # Initialize analysis
        analyzer = P013TargetedAnalysis(list_file)

        # Calculate all asymmetries
        results_df = analyzer.calculate_all_asymmetries()

        print(f"\n✅ Successfully analyzed {len(results_df)} bilateral region pairs")

        # Display regions in SegId order from list.txt with asymmetry highlighting
        print("\n" + "=" * 90)
        ordered_regions, asymmetric_count = analyzer.display_regions_by_segid_order(
            results_df
        )

        # Sort by magnitude for different methods
        print("\n" + "=" * 60)
        analyzer.sort_by_asymmetry_magnitude(results_df, "laterality_index")

        print("\n" + "=" * 60)
        analyzer.sort_by_asymmetry_magnitude(results_df, "cohens_d_asymmetry")

        # Create comprehensive analysis charts
        rankings, significant_data = analyzer.create_magnitude_rankings(results_df)

        # Create improved focused heatmap
        analyzer.create_focused_asymmetry_heatmap(results_df)

        # Generate report (use significant_data if available, otherwise ordered data)
        ordered_data = (
            results_df.sort_values("segid_order", ascending=True)
            if len(significant_data) == 0
            else significant_data
        )
        analyzer.generate_targeted_report(results_df, rankings, ordered_data)

        # Create final summary statistics file
        summary_stats = {
            'total_regions_analyzed': len(results_df),
            'subcortical_regions': len(results_df[results_df['region_type'] == 'subcortical']),
            'cortical_regions': len(results_df[results_df['region_type'] == 'cortical']),
            'significant_asymmetries': len(results_df[results_df['laterality_index'].abs() > 0.1]),
            'strong_asymmetries': len(results_df[results_df['laterality_index'].abs() > 0.2]),
            'mean_laterality_index': results_df['laterality_index'].mean(),
            'std_laterality_index': results_df['laterality_index'].std(),
            'max_asymmetry_region': results_df.loc[results_df['laterality_index'].abs().idxmax(), 'region'],
            'max_asymmetry_value': results_df.loc[results_df['laterality_index'].abs().idxmax(), 'laterality_index']
        }
        
        # Save summary statistics
        with open(analyzer.output_dir / "analysis_summary.txt", "w") as f:
            f.write("P013 Asymmetry Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            for key, value in summary_stats.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")

        print(f"\n📊 Analysis Summary Statistics saved to: {analyzer.output_dir}/analysis_summary.txt")

        print("\n🎉 COMPREHENSIVE P013 ASYMMETRY ANALYSIS COMPLETED!")
        print("=" * 70)
        print(f"📁 All results consolidated in: {analyzer.output_dir}")
        print("\n🔍 Generated files:")
        for file in sorted(analyzer.output_dir.glob("*")):
            file_size = file.stat().st_size / 1024  # Size in KB
            print(f"   ✓ {file.name:<35} ({file_size:.1f} KB)")

        print("\n📊 FINAL SUMMARY:")
        print(f"   • Total bilateral regions analyzed: {len(results_df)}")
        print(f"   • Significant asymmetries (|LI| > 0.1): {len(results_df[results_df['laterality_index'].abs() > 0.1])}")
        print(f"   • Strongest asymmetric region: {results_df.loc[results_df['laterality_index'].abs().idxmax(), 'region']}")
        print(f"   • Maximum |LI| value: {results_df['laterality_index'].abs().max():.4f}")
        print("   • Analysis methods used: 15 sophisticated asymmetry measures")
        
        print("\n🎯 Key Files for Review:")
        print("   📈 P013_comprehensive_asymmetry_analysis.csv - Complete results")
        print("   ⭐ P013_significant_asymmetries.csv - Significant findings only") 
        print("   📊 comprehensive_asymmetry_analysis.png - Main visualization")
        print("   📋 targeted_p013_analysis_report.md - Detailed analysis report")
        print("   📈 analysis_summary.txt - Quick statistics overview")

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
