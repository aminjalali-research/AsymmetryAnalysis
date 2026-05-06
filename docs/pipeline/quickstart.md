# Quickstart — Reproduce the Canonical Results

> Run these commands in order to reproduce the manuscript's Pipeline A and Pipeline B results from the released codebase. Assumes you have followed [`install.md`](install.md).

## Prerequisites

1. Cloned this repository
2. Activated a Python ≥ 3.9 virtual environment with the requirements installed (see `install.md`)
3. `Dataset/` and `DatasetControls/` populated (FreeSurfer-preprocessed; the released codebase does not run FreeSurfer itself)
4. `clinical_spreadsheet.xlsx` present at repo root
5. (Optional, only for `interactive_viewer.py`) FSLeyes on `PATH`

## Reproduce Pipeline B (voxel-wise zAI)

```bash
# 1. Build the control normative database (one-time, ~5 min)
python 01_build_control_normative.py --group F_20_39

# 2. Compute control-referenced zAI maps for every patient (~1 min/patient)
python 02_compute_zai.py

# 3. Apply gray-matter masking + clinical thresholds
python 03_clinical_maps.py

# 4. Render publication figures 10-16
python 04_publication_figures.py
```

Outputs of interest after these four commands:

- `results_zscore/groups/F_20_39/{mean,sd}_perfusion.nii.gz` — control normative
- `results_zscore/asymmetry/patients/<pid>/<pid>_asymmetry_zscore.nii.gz` — per-patient zAI
- `results_zscore/clinical/<pid>/<pid>_clinical_*.nii.gz` — clinical-grade maps
- `visual_analysis/{10..16}_zai_*.png` — figures 10–16

## Reproduce Pipeline A (ROI MDT discrimination)

Pipeline A is independent of Pipeline B and can be run on its own.

```bash
# 5. ROI-based discrimination using MDT-reviewed labels
python 05_roi_discrimination.py
```

Outputs:

- `ez_analysis_mdt/analysis_report.txt` — top-line summary
- `ez_analysis_mdt/method_comparison.png` — AUC across the 15 indices
- `ez_analysis_mdt/method_performance_*.csv` — full performance tables

## Optional methodology + QC

```bash
# zAI threshold-justification distributions (referenced in supplementary)
python analyze_zai_distributions.py

# Interactive 3D inspection (requires FSLeyes)
python interactive_viewer.py
# pick a patient, then option 19 for the cleanest presurgical view
```

## Expected runtime end-to-end

| Stage | Time |
|-------|------|
| Step 1 (build normative, one-time) | ~5 min |
| Step 2 (zAI, all patients) | ~15 min |
| Step 3 (clinical maps, all patients) | ~5 min |
| Step 4 (figures) | ~3 min |
| Step 5 (ROI discrimination) | <1 min |
| **Pipeline A + B total** | **~30 min** |

## What to check after each step

- After step 1: `ls results_zscore/groups/F_20_39/` should list `mean_perfusion.nii.gz`, `sd_perfusion.nii.gz`, `consensus_parcellation.nii.gz`, `brain_mask.nii.gz`.
- After step 2: `ls results_zscore/asymmetry/patients/` lists 15 patient subdirectories.
- After step 3: `ls results_zscore/clinical/P013/` shows `*_clinical_lateralized_*.nii.gz`.
- After step 4: `ls visual_analysis/*_zai_*.png` shows seven new PNGs (numbered 10–16).
- After step 5: `cat ez_analysis_mdt/analysis_report.txt` shows the AUC summary.

## Read next

- [`01-overview.md`](01-overview.md) for the full pipeline diagram
- Per-step detail: [`02-build-control-normative.md`](02-build-control-normative.md) … [`06-roi-discrimination.md`](06-roi-discrimination.md)
