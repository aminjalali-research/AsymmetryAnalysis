#!/usr/bin/env python3
"""
LI Map Visualization Examples

This script demonstrates different visualization approaches for laterality index maps.
Run this to get hands-on experience with the visualization options.
"""

import subprocess
from pathlib import Path
import time


def run_fsleyes_command(title, command, wait_time=3):
    """Run FSLeyes command with description"""
    print(f"\n🎨 {title}")
    print("=" * 60)
    print("Command:")
    print(f"  {command}")
    print("\nDescription: Opening FSLeyes viewer...")

    try:
        # Run the command
        subprocess.Popen(
            command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"✅ {title} opened successfully")
        print("   Close the FSLeyes window when done viewing")

        # Wait a moment before next visualization
        time.sleep(wait_time)

    except Exception as e:
        print(f"❌ Error opening {title}: {e}")


def check_files():
    """Check if all required files exist"""

    required_files = [
        "Dataset/P013/T1w_acpc_dc_restore.nii.gz",
        "laterality_maps/P013_laterality_index_map.nii.gz",
        "laterality_maps/P013_significant_asymmetry_mask.nii.gz",
        "laterality_maps/P013_left_dominant_regions.nii.gz",
        "laterality_maps/P013_right_dominant_regions.nii.gz",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease run create_laterality_map.py first!")
        return False

    print("✅ All required files found!")
    return True


def demo_basic_visualization():
    """Demonstrate basic LI map visualization"""

    command = """fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \\
         laterality_maps/P013_laterality_index_map.nii.gz \\
         -cm red-yellow -dr -0.4 0.4 -a 70 \\
         -n "Laterality Index Map" &"""

    run_fsleyes_command(
        "Basic LI Map Visualization", command.replace("\\\n         ", " ")
    )


def demo_significance_threshold():
    """Demonstrate significance threshold visualization"""

    command = """fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \\
         laterality_maps/P013_significant_asymmetry_mask.nii.gz \\
         -cm red -dr 0.5 1 -a 80 \\
         -n "Significant Asymmetries |LI| > 0.1" &"""

    run_fsleyes_command(
        "Significant Asymmetries Only", command.replace("\\\n         ", " ")
    )


def demo_hemisphere_comparison():
    """Demonstrate left vs right hemisphere comparison"""

    command = """fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \\
         laterality_maps/P013_left_dominant_regions.nii.gz \\
         -cm red -dr 0.1 0.4 -a 70 -n "Left Dominant" \\
         laterality_maps/P013_right_dominant_regions.nii.gz \\
         -cm blue -dr 0.1 0.4 -a 70 -n "Right Dominant" &"""

    run_fsleyes_command(
        "Left vs Right Hemisphere Dominance", command.replace("\\\n         ", " ")
    )


def demo_multi_threshold():
    """Demonstrate multi-threshold visualization"""

    command = """fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \\
         laterality_maps/P013_laterality_index_map.nii.gz \\
         -cm red-yellow -dr -0.1 0.1 -a 40 -n "Mild |LI| < 0.1" \\
         laterality_maps/P013_laterality_index_map.nii.gz \\
         -cm red -dr -0.4 -0.2 -a 80 -n "Strong Right > Left" \\
         laterality_maps/P013_laterality_index_map.nii.gz \\
         -cm yellow -dr 0.2 0.4 -a 80 -n "Strong Left > Right" &"""

    run_fsleyes_command(
        "Multi-Threshold Analysis", command.replace("\\\n         ", " ")
    )


def demo_quality_control():
    """Demonstrate quality control visualization"""

    command = """fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \\
         Dataset/P013/aparc+aseg.nii.gz \\
         -cm random -dr 1 2035 -a 30 -n "Parcellation" \\
         laterality_maps/P013_laterality_index_map.nii.gz \\
         -cm red-yellow -dr -0.4 0.4 -a 60 -n "LI Map" &"""

    run_fsleyes_command(
        "Quality Control: LI Map + Parcellation", command.replace("\\\n         ", " ")
    )


def show_menu():
    """Show visualization menu"""

    print("\n🎨 LI MAP VISUALIZATION EXAMPLES")
    print("=" * 70)
    print("Choose a visualization to demo:")
    print()
    print("1. Basic LI Map (recommended first view)")
    print("2. Significant Asymmetries Only (|LI| > 0.1)")
    print("3. Left vs Right Hemisphere Comparison")
    print("4. Multi-Threshold Analysis (advanced)")
    print("5. Quality Control View (LI + parcellation)")
    print("6. Run All Examples")
    print("7. Show Quick Commands")
    print("0. Exit")
    print()
    return input("Enter choice (0-7): ").strip()


def show_quick_commands():
    """Display quick reference commands"""

    print("\n📋 QUICK REFERENCE COMMANDS")
    print("=" * 70)
    print()
    print("🔸 Basic viewing:")
    print(
        "fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz laterality_maps/P013_laterality_index_map.nii.gz -cm red-yellow -dr -0.4 0.4 -a 70 &"
    )
    print()
    print("🔸 Significant only:")
    print(
        "fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz laterality_maps/P013_significant_asymmetry_mask.nii.gz -cm red -dr 0 1 -a 80 &"
    )
    print()
    print("🔸 Hemisphere comparison:")
    print(
        "fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz laterality_maps/P013_left_dominant_regions.nii.gz -cm red -a 70 laterality_maps/P013_right_dominant_regions.nii.gz -cm blue -a 70 &"
    )
    print()
    print("🔸 Color interpretation:")
    print("   Red = Right > Left dominance")
    print("   Yellow = Left > Right dominance")
    print("   Dark = Balanced regions")
    print()


def run_all_examples():
    """Run all visualization examples"""

    print("\n🚀 Running all visualization examples...")
    print("Close each FSLeyes window before the next one opens.")
    print("Press Ctrl+C to stop at any time.")

    try:
        demo_basic_visualization()
        input("\nPress Enter when ready for next visualization...")

        demo_significance_threshold()
        input("\nPress Enter when ready for next visualization...")

        demo_hemisphere_comparison()
        input("\nPress Enter when ready for next visualization...")

        demo_multi_threshold()
        input("\nPress Enter when ready for next visualization...")

        demo_quality_control()

        print("\n✅ All examples completed!")

    except KeyboardInterrupt:
        print("\n🛑 Examples stopped by user")


def main():
    """Main menu loop"""

    print("🧠 LI Map Visualization Demo")
    print("=" * 40)

    # Check if files exist
    if not check_files():
        return

    while True:
        choice = show_menu()

        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            demo_basic_visualization()
        elif choice == "2":
            demo_significance_threshold()
        elif choice == "3":
            demo_hemisphere_comparison()
        elif choice == "4":
            demo_multi_threshold()
        elif choice == "5":
            demo_quality_control()
        elif choice == "6":
            run_all_examples()
        elif choice == "7":
            show_quick_commands()
        else:
            print("❌ Invalid choice. Please enter 0-7.")

        if choice != "0" and choice != "6" and choice != "7":
            input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user. Goodbye!")
