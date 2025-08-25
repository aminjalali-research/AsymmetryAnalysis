#!/usr/bin/env python3
"""
Setup script for the Asymmetry Analysis project.

This script helps set up the environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_requirements():
    """Install required Python packages"""
    print("Installing required Python packages...")

    requirements_file = Path(__file__).parent / "requirements.txt"

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        )
        print("✓ Successfully installed all requirements")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("✗ Python 3.7 or higher is required")
        return False

    print(
        f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected"
    )
    return True


def setup_directories():
    """Create necessary directories"""
    print("Setting up directory structure...")

    base_dir = Path(__file__).parent
    directories = [base_dir / "output", base_dir / "data", base_dir / "logs"]

    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")


def run_tests():
    """Run basic tests to verify installation"""
    print("Running basic tests...")

    test_file = Path(__file__).parent / "tests" / "test_asymmetry.py"

    if test_file.exists():
        try:
            subprocess.check_call(
                [sys.executable, str(test_file)], cwd=Path(__file__).parent
            )
            print("✓ All tests passed")
            return True
        except subprocess.CalledProcessError:
            print("⚠ Some tests failed, but installation may still work")
            return False
    else:
        print("⚠ Test file not found, skipping tests")
        return True


def main():
    """Main setup function"""
    print("Asymmetry Analysis Project Setup")
    print("=" * 40)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Setup directories
    setup_directories()

    # Install requirements
    if not install_requirements():
        print("\n⚠ Warning: Failed to install some requirements")
        print("You may need to install them manually:")
        print("pip install pandas numpy matplotlib seaborn scipy plotly statsmodels")

    # Run tests
    print("\nRunning verification tests...")
    run_tests()

    print("\n" + "=" * 40)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Check the examples/ directory for usage examples")
    print("2. Read the docs/ directory for detailed documentation")
    print("3. Try running: python examples/run_analysis.py")
    print("\nFor help, see README.md or the user guide in docs/")


if __name__ == "__main__":
    main()
