# 🧠 Comprehensive Asymmetry Analysis Guide

*Complete reference for FreeSurfer ASL perfusion data asymmetry analysis*

---

## 📋 Table of Contents

1. [File Management Instructions](#file-management-instructions)
2. [Quick Start](#quick-start)
3. [Project Overview](#project-overview)
4. [Data Requirements](#data-requirements)
5. [Analysis Methods](#analysis-methods)
6. [File Structure](#file-structure)
7. [Step-by-Step Workflow](#step-by-step-workflow)
8. [Visualization Guide](#visualization-guide)
9. [Interpretation Guidelines](#interpretation-guidelines)
10. [Troubleshooting](#troubleshooting)
11. [Reference Commands](#reference-commands)

---

## 📂 File Management Instructions

### Default Behavior
**Always update existing files instead of creating new ones.**

### Code and Document Management
When working with code or documents:

- **Modify and improve existing files** rather than creating new versions
- **Only create a new file** if the changes would be so extensive that they could break existing functionality or make the previous code incompatible
- **If you must create a new file** due to dramatic changes, clearly explain why the existing file cannot be safely updated

### Exception Criteria for New File Creation

✅ **Create new file only when:**
- Complete architectural changes that would render existing code non-functional
- Switching to entirely different frameworks, languages, or paradigms
- Changes that would require extensive refactoring of dependent files

### Best Practice Approach
**Always prefer:** Incremental updates, additions, and improvements to existing files.

---symmetry Analysis Guide

*Complete reference for FreeSurfer ASL perfusion data asymmetry analysis*

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Project Overview](#project-overview)
3. [Data Requirements](#data-requirements)
4. [Analysis Methods](#analysis-methods)
5. [File Structure](#file-structure)
6. [Step-by-Step Workflow](#step-by-step-workflow)
7. [Visualization Guide](#visualization-guide)
8. [Interpretation Guidelines](#interpretation-guidelines)
9. [Troubleshooting](#troubleshooting)
10. [Reference Commands](#reference-commands)

---

## 🚀 Quick Start

### Environment Setup
```bash
# Activate conda environment
conda activate Asym

# Navigate to project directory
cd /home/amin/AsymmetryAnalysis
```

### Basic Analysis Pipeline
```bash
# 1. Run comprehensive asymmetry analysis
python targeted_p013_analysis.py

# 2. Create spatial NIfTI maps
python create_laterality_map.py

# 3. Interactive visualization demo
python demo_li_visualization.py
```

### Essential Visualization Command
```bash
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
  laterality_maps/P013_laterality_index_map.nii.gz \
  -cm red-yellow -dr -0.4 0.4 -a 70 &
```

---

## 📊 Project Overview

### Purpose
Advanced asymmetry analysis for FreeSurfer parcellated ASL perfusion data using sophisticated statistical methods beyond simple left-right ratios.

### Key Features
- **15 Advanced Asymmetry Methods**: Including Cohen's d, volume-corrected, robust statistical measures (refined from 16, removed redundant method)
- **SegID-Based Ordering**: Maintains original FreeSurfer parcellation sequence
- **Enhanced Visualizations**: Multi-method heatmaps with consistent RdBu colormap (Red=Right>Left, Blue=Left>Right)
- **Spatial Visualization**: NIfTI maps for brain visualization in FSLeyes/MRIcroGL
- **Statistical Significance Testing**: Multiple thresholds and quality control
- **Publication-Ready Outputs**: Enhanced figures and statistical tables with clear laterality interpretation
- **Consolidated Results**: Single P013_Analysis_Results folder containing all outputs

### Clinical Applications
- Stroke asymmetry assessment
- Neurodevelopmental studies
- Pre-surgical planning
- Research on hemispheric specialization

### Latest Enhancements (August 2025)
- ✅ Removed redundant robust asymmetry method (identical to laterality index)
- ✅ Enhanced focused heatmap with individual method displays
- ✅ Consistent RdBu colormap across all methods for clear interpretation
- ✅ Improved left/right laterality visualization with color-coded indicators

---

## 📁 Data Requirements

### Required Files
```
Dataset/P013/
├── T1w_acpc_dc_restore.nii.gz    # T1-weighted structural image
├── aparc+aseg.nii.gz             # FreeSurfer parcellation
└── P013_segstats_perfusion.txt   # Perfusion statistics per region
```

### Additional Files
```
list.txt                          # FreeSurfer region definitions with SegIDs
targeted_p013_analysis.py         # Main analysis script
create_laterality_map.py          # NIfTI mapping script  
demo_li_visualization.py          # Visualization demo
```

---

## 🔬 Analysis Methods

### 15 Asymmetry Calculation Methods (Refined Suite)

#### Basic Methods
1. **Asymmetry Index (AI)**: `(L-R)/(L+R)`
2. **Laterality Index (LI)**: `(R-L)/(R+L)` - Clinical standard 
3. **Absolute Difference**: `|L-R|`
4. **Relative Difference**: `(L-R)/mean(L,R)`

#### Advanced Statistical Methods
5. **Cohen's d**: Standardized effect size
6. **Volume-Corrected AI**: Adjusted for region volume
7. **Robust AI**: Using median instead of mean
8. **Fold Change**: `L/R` ratio
9. **Log Fold Change**: `log2(L/R)`
10. **Normalized Difference**: `(L-R)/max(L,R)`

#### Sophisticated Measures
11. **Coefficient of Variation Difference**: Variability-based asymmetry
12. **Interquartile Range Ratio**: Robust spread measure
13. **Skewness Difference**: Distribution shape asymmetry
14. **Kurtosis Difference**: Tail behavior asymmetry
15. **Entropy Difference**: Information-theoretic measure
16. **Mahalanobis Distance**: Multivariate asymmetry

### Statistical Thresholds
- **Mild Asymmetry**: |LI| > 0.05
- **Significant Asymmetry**: |LI| > 0.1
- **Strong Asymmetry**: |LI| > 0.2
- **Extreme Asymmetry**: |LI| > 0.3

---

## 📂 File Structure

### Input Files
```
AsymmetryAnalysis/
├── Dataset/P013/
│   ├── T1w_acpc_dc_restore.nii.gz
│   ├── aparc+aseg.nii.gz
│   └── P013_segstats_perfusion.txt
├── list.txt
└── [analysis scripts]
```

### Generated Outputs

#### Consolidated Analysis Results (NEW!)
```
P013_Analysis_Results/                          # Single consolidated folder
├── P013_comprehensive_asymmetry_analysis.csv  # All 15 methods results
├── P013_significant_asymmetries.csv           # |LI| > 0.1 regions
├── analysis_summary.txt                       # Statistical summary
├── comprehensive_asymmetry_analysis.png       # Main overview visualization
├── focused_asymmetry_heatmap.png              # Enhanced multi-method heatmap
├── significant_regions_heatmap.png            # Significant regions only
└── targeted_p013_analysis_report.md           # Detailed analysis report
```

#### Legacy Spatial Maps (if generated separately)
```
laterality_maps/
├── P013_laterality_index_map.nii.gz           # Full LI map
├── P013_significant_asymmetry_mask.nii.gz     # |LI| > 0.1 mask
├── P013_left_dominant_regions.nii.gz          # L > R regions
├── P013_right_dominant_regions.nii.gz         # R > L regions
├── P013_asymmetry_statistics.txt              # Map statistics
├── P013_region_mapping.csv                    # SegID mappings
└── P013_laterality_histogram.png              # LI distribution
```

#### Documentation
```
├── LATERALITY_VISUALIZATION_GUIDE.md          # Complete visualization guide
├── QUICK_LI_REFERENCE.md                      # Quick reference commands
├── AsymGuide.md                               # This comprehensive guide (updated)
└── test_colormap_preview.py                   # Colormap demonstration script
```

---

## 🔄 Step-by-Step Workflow

### Step 1: Data Preparation
1. Ensure FreeSurfer parcellation is complete
2. Extract perfusion statistics using FreeSurfer's `mri_segstats`
3. Verify all required files are present

### Step 2: Asymmetry Analysis
```bash
# Run comprehensive analysis with SegID ordering
python targeted_p013_analysis.py
```

**What this does:**
- Loads perfusion data and region definitions
- Calculates 16 different asymmetry measures
- Orders results by original SegID sequence
- Highlights significantly asymmetric regions
- Generates statistical summaries

### Step 3: Spatial Mapping
```bash
# Create NIfTI maps for visualization
python create_laterality_map.py
```

**What this does:**
- Maps region-based LI values to brain voxels
- Creates multiple visualization files
- Generates quality control statistics
- Produces distribution histograms

### Step 4: Visualization
```bash
# Interactive demo with multiple options
python demo_li_visualization.py
```

**Available visualizations:**
1. Basic LI map overlay
2. Significant asymmetries only
3. Hemisphere comparison (left vs right)
4. Multi-threshold analysis
5. Quality control view

### Step 5: Interpretation
- Review significant asymmetries (|LI| > 0.1)
- Check clinical relevance of affected regions
- Validate results with anatomical knowledge
- Generate publication figures if needed

---

## 🎨 Visualization Guide

### Enhanced Asymmetry Visualizations

#### 📊 Python-Generated Visualizations

**1. Enhanced Focused Asymmetry Heatmap** (New!)
- File: `P013_Analysis_Results/focused_asymmetry_heatmap.png`
- Features: 7 key methods displayed separately with consistent RdBu colormap
- Individual colorbars for each method showing specific ranges
- Consistent interpretation: 🔴 Red = Right>Left | 🔵 Blue = Left>Right
- Numerical values displayed in each cell for precision

**2. Comprehensive Analysis Overview**
- File: `P013_Analysis_Results/comprehensive_asymmetry_analysis.png`
- Features: Multi-panel visualization with rankings, distributions, and summaries

**3. Significant Regions Heatmap**
- File: `P013_Analysis_Results/significant_regions_heatmap.png`
- Features: Focused view of only significantly asymmetric regions (|LI| > 0.1)

### FSLeyes Commands

#### Basic LI Map
```bash
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
  laterality_maps/P013_laterality_index_map.nii.gz \
  -cm red-yellow -dr -0.4 0.4 -a 70 -n "Laterality Index" &
```

#### Significant Asymmetries Only
```bash
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
  laterality_maps/P013_significant_asymmetry_mask.nii.gz \
  -cm red -dr 0 1 -a 80 -n "Significant |LI| > 0.1" &
```

#### Hemisphere Comparison
```bash
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
  laterality_maps/P013_left_dominant_regions.nii.gz \
  -cm red -dr 0.1 0.4 -a 70 -n "Left Dominant" \
  laterality_maps/P013_right_dominant_regions.nii.gz \
  -cm blue -dr 0.1 0.4 -a 70 -n "Right Dominant" &
```

### Enhanced Color Interpretation (Updated)
| Context | Color | Meaning | LI Value |
|---------|-------|---------|----------|
| **Python Heatmaps** | **🔴 Red** | Right > Left dominance | LI > 0 |
| **Python Heatmaps** | **🔵 Blue** | Left > Right dominance | LI < 0 |
| **Python Heatmaps** | **⚪ White/Gray** | Balanced regions | LI ≈ 0 |
| **FSLeyes Maps** | **Red** | Right > Left dominance | LI > 0.1 |
| **FSLeyes Maps** | **Yellow** | Left > Right dominance | LI < -0.1 |
| **FSLeyes Maps** | **Dark/Black** | Balanced regions | \|LI\| ≤ 0.1 |

### Alternative Software

#### MRIcroGL
```bash
# Open with overlay
MRIcroGL -o Dataset/P013/T1w_acpc_dc_restore.nii.gz \
  -l laterality_maps/P013_laterality_index_map.nii.gz
```

#### AFNI
```bash
# Load in AFNI
afni -dset Dataset/P013/T1w_acpc_dc_restore.nii.gz \
  -func laterality_maps/P013_laterality_index_map.nii.gz &
```

---

## 📖 Interpretation Guidelines

### Clinical Significance

#### P013 Key Findings
- **Total Regions Analyzed**: 74 bilateral pairs
- **Significant Asymmetries**: 9 regions (12.2%)
- **Brain Coverage**: 26.4% of voxels show asymmetry
- **Dominant Pattern**: Mixed left/right with regional specificity

#### Significantly Asymmetric Regions (|LI| > 0.1)
1. **Left-Cerebellum-White-Matter** (LI: -0.368) - Strong left dominance
2. **Right-Cerebellum-White-Matter** (LI: 0.368) - Strong right dominance
3. **Left-Cerebellum-Cortex** (LI: -0.219) - Moderate left dominance
4. **ctx-lh-temporalpole** (LI: -0.168) - Moderate left dominance
5. **ctx-rh-temporalpole** (LI: 0.168) - Moderate right dominance
6. **Left-Putamen** (LI: -0.140) - Mild left dominance
7. **Right-Putamen** (LI: 0.140) - Mild right dominance
8. **ctx-lh-superiorfrontal** (LI: -0.116) - Mild left dominance
9. **ctx-rh-superiorfrontal** (LI: 0.116) - Mild right dominance

### Statistical Interpretation

#### Effect Sizes (Cohen's d)
- **Small effect**: d = 0.2
- **Medium effect**: d = 0.5  
- **Large effect**: d = 0.8

#### Clinical Thresholds
- **|LI| < 0.05**: No meaningful asymmetry
- **0.05 ≤ |LI| < 0.1**: Mild asymmetry
- **0.1 ≤ |LI| < 0.2**: Significant asymmetry
- **|LI| ≥ 0.2**: Strong asymmetry

### Quality Control Checks
1. **Anatomical Consistency**: Check if asymmetries make anatomical sense
2. **Bilateral Pairing**: Verify left-right pairs show opposite LI signs
3. **Regional Coherence**: Similar regions should show similar patterns
4. **Statistical Significance**: Use multiple methods for validation

---

## 🔧 Troubleshooting

### Common Issues

#### File Not Found Errors
```bash
# Check if all required files exist
ls -la Dataset/P013/
ls -la laterality_maps/

# Regenerate missing files
python create_laterality_map.py
```

#### FSLeyes Display Issues
```bash
# Check FSLeyes installation
which fsleyes

# Alternative visualization
python demo_li_visualization.py
# Choose option 7 for quick commands
```

#### Memory Issues with Large Datasets
```python
# In create_laterality_map.py, reduce precision
laterality_data = laterality_data.astype(np.float32)  # Instead of float64
```

#### Parcellation Mismatches
```bash
# Verify parcellation integrity
mri_info Dataset/P013/aparc+aseg.nii.gz
fslstats Dataset/P013/aparc+aseg.nii.gz -R
```

### Error Messages and Solutions

#### "No bilateral pairs found"
- **Cause**: Region naming doesn't follow Left-/Right- or ctx-lh-/ctx-rh- convention
- **Solution**: Check list.txt format and region naming consistency

#### "NIfTI file creation failed"
- **Cause**: Missing parcellation file or incorrect dimensions
- **Solution**: Verify T1 and parcellation files have matching dimensions

#### "Visualization command not found"
- **Cause**: FSLeyes not installed or not in PATH
- **Solution**: Install FSL/FSLeyes or use alternative visualization methods

---

## 📚 Reference Commands

### Essential Analysis Commands
```bash
# Complete analysis pipeline
python targeted_p013_analysis.py          # Main analysis
python create_laterality_map.py           # Create NIfTI maps
python visualize_laterality_maps.py       # Quick analysis
python demo_li_visualization.py           # Interactive demo
```

### File Management
```bash
# Check analysis results
ls -la results/
head -20 results/P013_comprehensive_analysis.csv

# Check spatial maps
ls -la laterality_maps/
fslstats laterality_maps/P013_laterality_index_map.nii.gz -R
```

### Quick Statistics
```bash
# Count significant asymmetries
grep -v "SegID" results/P013_significant_asymmetries.csv | wc -l

# Show LI distribution
cat laterality_maps/P013_asymmetry_statistics.txt
```

### Batch Processing (Future Extension)
```bash
# Process multiple subjects (template)
for subject in P001 P002 P003; do
    python targeted_analysis.py --subject $subject
    python create_laterality_map.py --subject $subject
done
```

---

## 🔬 Advanced Features

### Custom Thresholds
Modify thresholds in analysis scripts:
```python
# In targeted_p013_analysis.py
SIGNIFICANCE_THRESHOLD = 0.1    # Adjust as needed
STRONG_ASYMMETRY_THRESHOLD = 0.2
```

### Additional Asymmetry Methods
Add new methods to the calculator:
```python
def custom_asymmetry_method(left_values, right_values):
    """Your custom asymmetry calculation"""
    return custom_result
```

### Batch Visualization
Create multiple views automatically:
```bash
# Run all visualization examples
python demo_li_visualization.py
# Choose option 6: "Run All Examples"
```

### Publication Figures
Generate high-quality figures:
```bash
# High-resolution screenshot commands included in visualization guide
# See LATERALITY_VISUALIZATION_GUIDE.md for details
```

---

## 📊 Expected Results Summary

### Typical Analysis Output
- **Processing Time**: 2-5 minutes for full analysis
- **Significant Regions**: 5-15% of total regions typically show |LI| > 0.1
- **File Sizes**: ~50-100 MB for complete output set
- **Quality Metrics**: >95% successful region mapping expected

### Validation Checklist
- [ ] All bilateral pairs identified correctly
- [ ] LI values between -1 and +1
- [ ] Significant asymmetries anatomically plausible
- [ ] NIfTI maps load correctly in FSLeyes
- [ ] Statistical summaries make sense

---

## 🆘 Support and Resources

### Documentation Files
- `LATERALITY_VISUALIZATION_GUIDE.md` - Detailed visualization instructions
- `QUICK_LI_REFERENCE.md` - Essential commands and findings
- `AsymGuide.md` - This comprehensive guide (you are here!)

### Key Scripts
- `targeted_p013_analysis.py` - Main analysis with 16 methods
- `create_laterality_map.py` - Spatial NIfTI mapping
- `demo_li_visualization.py` - Interactive visualization demo

### External Resources
- [FreeSurfer Documentation](https://surfer.nmr.mgh.harvard.edu/fswiki)
- [FSLeyes User Guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes)
- [NIfTI Format Specification](https://nifti.nimh.nih.gov/)

---

## 📝 Version History

- **v1.0** (Current): Complete asymmetry analysis pipeline with 16 methods
- **v0.5**: Basic 5-method calculator with simple visualization
- **v0.1**: Initial concept and data structure

---

## 📧 Quick Reference Card

### Most Important Commands
```bash
# 1. Run analysis
python targeted_p013_analysis.py

# 2. Create maps  
python create_laterality_map.py

# 3. Visualize
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
  laterality_maps/P013_laterality_index_map.nii.gz \
  -cm red-yellow -dr -0.4 0.4 -a 70 &
```

### Key Findings (P013 - Updated Analysis)
- **9/37 bilateral regions** significantly asymmetric (|LI| > 0.1)
- **15 sophisticated asymmetry methods** refined from 16 (removed redundant method)
- **Strongest asymmetries**: entorhinal (-0.398), frontalpole (-0.332), parsorbitalis (-0.215)
- **Predominantly right-dominant** asymmetries in this subject
- **Mixed patterns**: Some left-dominant regions (amygdala, temporalpole, bankssts)
- **Enhanced visualization** with consistent RdBu colormap for clear interpretation

### Analysis Evolution (August 2025)
- ✅ **Method Refinement**: From 16 to 15 methods (removed redundant robust asymmetry)
- ✅ **Visual Enhancement**: New focused heatmap with individual method displays
- ✅ **Consistent Colors**: RdBu colormap across all methods (Red=R>L, Blue=L>R)
- ✅ **Consolidated Output**: Single P013_Analysis_Results folder
- ✅ **Clear Interpretation**: Enhanced laterality indicators and documentation

---

*This guide provides comprehensive coverage of the asymmetry analysis pipeline. Refer to individual documentation files for detailed technical instructions.*
