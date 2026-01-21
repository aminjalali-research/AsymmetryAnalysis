
Check this if there are templates for symmetrical voxel-based bilateral-pairing: https://github.com/CUHK-AIM-Group/NeuroSTORM

# Updates
- The laterality maps are showing values for the entire image volume instead of just the brain regions. Fixed the create_laterality_map.py to properly mask the laterality values to only brain regions (non-zero parcellation areas).
- Updated the save function to handle NaN values properly when creating the NIfTI files. The main LI map should convert NaN values to 0 for proper NIfTI format.

- Before: Laterality values appeared throughout the entire image volume, including background areas outside the brain. 
  After: Laterality values only appear within brain tissue regions defined by the FreeSurfer parcellation. (How: Initialize LI map with NaN for non-brain areas, Create brain mask from parcellation (parc_data > 0), Only assign LI values within brain regions)

- How about it work on this repository: https://github.com/PennLINC/aslprep

# Template Usage and ROI Analysis
Labels: Only 3 territories defined in vascular_territories_atlas_labels.txt:
- LICA: Left Internal Carotid Artery territory
- RICA: Right Internal Carotid Artery territory
- VBA: Vertebrobasilar Artery territory

# Processing Templates
- MNI Standard Space: The pipeline transforms data to MNI space (MNINonLinear processing in stage 10-12)
- T1w Native Space: Primary processing occurs in subject's native T1w space (ASLT1w)
- Surface Templates: Uses HCP surface meshes with MSMAll registration by default

# The templates used are NOT explicitly symmetrical

- Vascular Territories Atlas: Has separate left (LICA) and right (RICA) carotid territories. This suggests anatomical asymmetry is preserved.
- Not symmetrical because it maintains left/right distinctions.
- MNI Template: Standard MNI152 space used for final outputs. While MNI templates can be symmetrical, the pipeline doesn't specify use of a symmetrical version.
- Surface Processing: Uses HCP's MSMAll registration which preserves individual cortical folding patterns. Not symmetrical because it maintains subject-specific anatomy.





-----

# Conferences
- [Neurology Congress 2026](https://magnivelinternational.com/conferences/international-conference-on-neurology-and-neurological-disorders-2026)
- [Intelligent Systems Conference 2026](https://saiconference.com/IntelliSys)
- [Neuroscience, neurology, and brain disorders](https://neuroscience2026.researchconnects.org/)
- NeuroImage 
- MICCAI (Medical Image Computing and Computer-Assisted Intervention)
- IEEE ISBI (International Symposium on Biomedical Imaging)

https://www.ean.org/meet/congresses/future-congresses

# FreeSurfer ASL Asymmetry Analysis

```
https://github.com/stars/aminjalali-research/lists/asl
```
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
