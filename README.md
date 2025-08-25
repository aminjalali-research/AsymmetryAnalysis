# FreeSurfer ASL Asymmetry Analysis

A comprehensive toolkit for analyzing brain asymmetry patterns from FreeSurfer ASL (Arterial Spin Labeling) perfusion data. This project provides sophisticated methods for quantifying hemispheric differences in brain perfusion and visualizing laterality patterns.

## Features

### Core Analysis Methods
- **15 asymmetry calculation methods** including:
  - Traditional Laterality Index (LI)
  - Volume-weighted asymmetries
  - Statistical-based indices (Cohen's d, effect sizes)
  - Robust statistical measures (median-based, IQR)
  - Advanced correlation and ratio analyses

### Visualization Capabilities
- **Enhanced heatmap visualizations** with consistent color mapping
- **Spatial brain mapping** with NIfTI file generation for FSLeyes
- **Interactive visualization interface** with multiple viewing modes
- **Quality control overlays** for data validation

### Advanced Analytics
- **Correlation analysis** between asymmetry methods
- **Pattern recognition** and clustering
- **Regional comparison** tools
- **Statistical validation** frameworks

### Prerequisites
- Python 3.7+ with conda environment
- FreeSurfer parcellation data  
- FSLeyes or MRIcroGL for visualization

### Installation & Setup
```bash
# Clone and setup environment
git clone <repository-url>
cd AsymmetryAnalysis
conda activate Asym  # or your preferred environment
pip install -r requirements.txt
```

### Essential Analysis Pipeline
```bash
# 1. Run asymmetry analysis (16 methods)
python targeted_p013_analysis.py

# 2. Create spatial NIfTI maps for brain visualization
python create_laterality_map.py

# 3. Interactive visualization demo
python demo_li_visualization.py
```

## 📊 Project Features

### Asymmetry Methods (16 total)
- **Basic Methods**: Laterality Index, Asymmetry Index, Percentage Difference, Normalized Asymmetry, Log Ratio
- **Statistical Methods**: Cohen's d, Volume-Corrected, Robust, Fold-Change, Z-score based
- **Sophisticated Methods**: Entropy Difference, Mahalanobis Distance, Skewness/Kurtosis Differences, IQR-based

### Key Capabilities
-  **SegID-Based Ordering**: Maintains FreeSurfer parcellation sequence from list.txt
-  **Statistical Significance**: Multiple thresholds (|LI| > 0.1, 0.2, 0.3)
-  **Spatial Visualization**: NIfTI maps for FSLeyes, MRIcroGL, AFNI
-  **Publication Ready**: Statistical reports and visualization guides
-  **Quality Control**: Comprehensive validation and error checking

## 📁 Project Structure

### Core Scripts
```
AsymmetryAnalysis/
├── targeted_p013_analysis.py     # Main analysis with 16 methods
├── create_laterality_map.py      # NIfTI spatial mapping
├── demo_li_visualization.py      # Interactive visualization demo
└── visualize_laterality_maps.py  # Analysis tools for NIfTI maps
```

### Source Code
```
src/
├── calculator.py                 # All 16 asymmetry calculation methods
├── freesurfer_parser.py         # FreeSurfer data parsing
├── analysis.py                  # Main analysis pipeline
└── visualizer.py                # Visualization utilities
```

### Data & Documentation
```
Dataset/P013/                    # Patient data directory
laterality_maps/                 # Generated NIfTI spatial maps
AsymGuide.md                     # Comprehensive reference guide
LATERALITY_VISUALIZATION_GUIDE.md # Detailed visualization instructions
QUICK_LI_REFERENCE.md           # Essential commands reference
```

## 📖 Usage Examples

### Basic Analysis
```python
# Run complete asymmetry analysis
from src.analysis import ComprehensiveAsymmetryAnalysis

analyzer = ComprehensiveAsymmetryAnalysis()
results = analyzer.run_analysis("Dataset/P013/P013_segstats_perfusion.txt")
```

### Spatial Visualization
```bash
# Basic LI map visualization
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
  laterality_maps/P013_laterality_index_map.nii.gz \
  -cm red-yellow -dr -0.4 0.4 -a 70 &

# Significant asymmetries only
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
  laterality_maps/P013_significant_asymmetry_mask.nii.gz \
  -cm red -dr 0 1 -a 80 &
```

## 📈 Expected Results

### P013 Key Findings
- **Total Analyzed Regions**: 74 bilateral pairs
- **Significant Asymmetries**: 9 regions with |LI| > 0.1 (12.2%)
- **Brain Coverage**: 26.4% of voxels show measurable asymmetry
- **Processing Time**: 2-5 minutes for complete analysis

### Significant Asymmetric Regions
1. **Left-Cerebellum-White-Matter** (LI: -0.368) - Strong left dominance
2. **Right-Cerebellum-White-Matter** (LI: 0.368) - Strong right dominance  
3. **ctx-lh-temporalpole** (LI: -0.168) - Moderate left dominance
4. **ctx-rh-temporalpole** (LI: 0.168) - Moderate right dominance
5. **Left-Putamen** (LI: -0.140) - Mild left dominance

## Technical Details

### Statistical Thresholds
- **Mild Asymmetry**: |LI| > 0.05
- **Significant Asymmetry**: |LI| > 0.1 (clinical threshold)
- **Strong Asymmetry**: |LI| > 0.2
- **Extreme Asymmetry**: |LI| > 0.3

### Color Interpretation (NIfTI Maps)
| Color | Meaning | LI Range |
|-------|---------|----------|
| **Red** | Right > Left dominance | LI > 0.1 |
| **Yellow** | Left > Right dominance | LI < -0.1 |
| **Dark** | Balanced regions | \|LI\| ≤ 0.1 |

## 📖 Documentation

### Complete Guides
- **`AsymGuide.md`** - Comprehensive reference (master document)
- **`LATERALITY_VISUALIZATION_GUIDE.md`** - Detailed FSLeyes/MRIcroGL instructions  
- **`QUICK_LI_REFERENCE.md`** - Essential commands and key findings

### Interactive Learning
- **`demo_li_visualization.py`** - Menu-driven visualization examples
- **`targeted_p013_analysis.py`** - Complete analysis with documentation

---
