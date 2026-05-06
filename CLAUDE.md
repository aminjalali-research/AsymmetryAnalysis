# AsymmetryAnalysis — Project Guide for Claude Code

## What This Project Does

This is a **clinical neuroimaging research project** for **epilepsy presurgical planning**. It analyzes ASL (Arterial Spin Labeling) perfusion MRI scans to find brain regions with abnormal blood flow that may indicate epileptogenic zones.

**The core question:** "Where in this patient's brain is blood flow abnormal compared to healthy people, and is the abnormality lateralized (left vs right)?"

## Key Directories

| Directory | Contents |
|-----------|----------|
| `Dataset/` | **Patient data** — 15 epilepsy patients (P013-P027). Each has: `aparc+aseg.nii.gz`, `T1w_acpc_dc_restore.nii.gz`, `{ID}_perfusion_calib_resampled_to_T1w.nii.gz` (actually MNI space despite filename) |
| `DatasetControls/` | **Healthy control data** — organized by age/sex (e.g., `20_29F/`, `30_39F/`). Each subject has: `*_MNISpace_perfusion_calib_upsampled.nii.gz`, `aparc+aseg.nii.gz` |
| `src/` | Core modules — `calculator.py` (11 asymmetry methods) |
| `results_zscore/` | Z-score pipeline outputs: `groups/` (mean/SD/parcellation), `patients/` (z-score maps), `clinical/` (gray-matter-only maps), `asymmetry/` (asymmetry of z-scores) |
| `results_voxel/` | Within-subject asymmetry results (no control comparison) |
| `results/` | ROI-based asymmetry results |
| `visual_analysis/` | Publication-quality PNG figures |

## The Full Pipeline

```
1. CONTROLS → GROUP STATS
   01_build_control_normative.py
   Loads 30 control scans → Welford's algorithm → mean + SD perfusion maps
   Output: results_zscore/groups/F_20_39/{mean,sd}_perfusion.nii.gz

2. PATIENT → Z-SCORE MAP
   01_build_control_normative.py
   z = (patient_perfusion - mean_controls) / sd_controls  per voxel
   Output: results_zscore/patients/{pid}/{pid}_vs_F_20_39_zscore.nii.gz

3. ASYMMETRY → CONTROL-REFERENCED zAI (Pipeline B canonical)
   02_compute_zai.py
   AI = 100 * (L - R) / ((L+R)/2) per voxel; then z = (AI - mean_AI) / sd_AI
   Output: results_zscore/asymmetry/patients/{pid}/{pid}_asymmetry_zscore.nii.gz

4. CLINICAL MAPS (gray matter only, strict threshold)
   03_clinical_maps.py
   Mask zAI to GM, |zAI|>=3.0, cluster>=50 voxels, lateralized maps
   Output: results_zscore/clinical/{pid}/{pid}_clinical_*.nii.gz

5. PUBLICATION FIGURES
   04_publication_figures.py
   Figures 10-16 (zAI panels, distributions, cluster overlays)
   Output: visual_analysis/

6. ROI MDT DISCRIMINATION (Pipeline A canonical)
   05_roi_discrimination.py + ez_ground_truth.py
   ROI-based, 15 indices x 4 aggregation strategies, ROC + LOOCV
   Output: ez_analysis_mdt/

7. CLINICAL INTERPRETATION (surgeon-facing report)  -- NEW 2026-05-05
   06_clinical_interpretation.py
   Surgical thresholds (|zAI|>=4, >=100 vox) + 37-ROI scope + lobe aggregation
   + lateralization confidence (Bartolomei 2008 framework) + MDT concordance
   Output: results_zscore/clinical/<PID>/<PID>_surgeon_report.{txt,json}
           results_zscore/clinical/cohort_concordance.csv (cohort summary)
```

## Surgical Threshold Tiers (2026-05-05 policy)

| Use case | zAI threshold | Cluster size | Source |
|----------|---------------|--------------|--------|
| Research / sensitivity | \|zAI\|>=3 | >=50 voxels | Shang 2021 |
| Standard clinical view | \|zAI\|>=4 | >=100 voxels | Boscolo Galazzo 2016 |
| Surgical confidence | \|zAI\|>=5 | >=200 voxels | Bartolomei 2008 |
| Concordance-required | \|zAI\|>=4 + within 2cm of MRI lesion | + EEG/neuropsych | Gennari 2025 |

`06_clinical_interpretation.py` defaults to **standard clinical view**; pass `--peak-threshold` and `--size-threshold` to override.

## Critical Technical Facts

- **All data is in MNI152 space** at 227x272x227 voxels, 0.8mm isotropic
- Patient files named `*_resampled_to_T1w.nii.gz` are actually in MNI space (confirmed by affine matrix)
- Control files: `*_MNISpace_perfusion_calib_upsampled.nii.gz`
- **No registration needed** — all volumes share identical geometry
- Group stats use **Welford's online algorithm** (numerically stable, memory-efficient)
- Consensus parcellation: majority-vote across 30 control aparc+aseg files
- Patient demographics: `clinical_spreadsheet.xlsx` (cols: ID, AGE, SEX)

## 9 Control Groups (3 age ranges x 3 sex groups)

| Group | Subdirectories | Status |
|-------|---------------|--------|
| F_20_39 | 20_29F/, 30_39F/ | Available (30 subjects) |
| F_40_59 | 40_49F/, 50_59F/ | In processing |
| F_60_79 | 60_69F/, 70_79F/ | In processing |
| M_20_39 | 20_29M/, 30_39M/ | In processing |
| M_40_59 | 40_49M/, 50_59M/ | In processing |
| M_60_79 | 60_69M/, 70_79M/ | In processing |
| FM_20_39 | All 20-39 | In processing |
| FM_40_59 | All 40-59 | In processing |
| FM_60_79 | All 60-79 | In processing |

## 15 Asymmetry Indices (src/calculator.py — literature-cited, 2026-05-04 formula-fix)

**Ratio-based:** laterality_index (LI), absolute_asymmetry_index (AAI), log_ratio, simple_difference, ratio
**Effect sizes:** cohen_d_asymmetry (Cohen 1988), hedges_g_asymmetry (Hedges 1981, small-sample-corrected), glass_delta_asymmetry (Glass 1976)
**Distribution-based:** zscore_asymmetry, robust_zscore_asymmetry (Huber 1981 / Leys 2013, MAD-based)
**Advanced:** percent_difference, cv_ratio (Galaburda 1978), hyperperfusion_ratio (max/min form, Van Bogaert 2000), normalized_difference (Toga & Thompson 2003), coefficient_of_asymmetry

All formulas literature-cited. Audit doc: `docs/superpowers/audits/2026-05-04-formula-fix.md`.

## 13 Thresholding/Clustering Methods (src/thresholding.py)

M1: Fixed Z + cluster, M2: Quality + percentile, M3: FDR (BH), M4: Bonferroni, M5: TFCE, M6: GRF, M7: Permutation, M8: GMM, M9: Otsu, M10: Random baseline, M11: Quality+TFCE (recommended), M12: Quality+GMM, M13: Quality+Otsu

Module: `src/thresholding.py` (extracted from deleted `control_zscore_asymmetry_clustering.py` on 2026-05-04, 18/18 smoke tests passing).

## Clinical Scope — 37 ROI Pairs (POLICY, 2026-05-05)

**Per user policy, the canonical clinical scope for surgeon-facing Pipeline B output is the 37 ROI pairs:**

- **3 subcortical pairs** (FreeSurfer labels): Thalamus (10/49), Hippocampus (17/53), Amygdala (18/54)
- **34 cortical Desikan-Killiany regions**: Left labels 1001-1035, Right labels 2001-2035

Total: 76 FreeSurfer labels = 37 paired regions. This matches Pipeline A's ROI scope and the manuscript's reported analysis space.

**Implication for Pipeline B (voxel zAI):**

- Voxel-level zAI maps still cover all gray matter (no information loss in raw output).
- **All clinical/surgeon-facing visualizations and cluster reports MUST be masked to the 37 ROIs by default** (e.g., `interactive_viewer.py` option 11, `03_clinical_maps.py` clinical output).
- Voxels outside the 37 ROIs (e.g., paracentral lobule, isthmus cingulate, occipital pole, basal ganglia other than thalamus) are excluded from the "where is the EZ" decision.

**Excluded basal ganglia structures** (intentionally not in the 37): Caudate (11/50), Putamen (12/51), Pallidum (13/52). These are not part of the surgeon-facing EZ region set per the manuscript.

**Code reference:** see `interactive_viewer.py::_clinical_roi_labels()` and the policy follow-up F-18 in `.claude/NEXT.md` (refactor `03_clinical_maps.py` and `04_publication_figures.py` to default to this mask).

## Code Conventions

- Standalone Python scripts at project root
- Class-based analyzers (e.g., `VoxelBasedAsymmetryAnalyzer`, `ControlGroupBuilder`)
- NIfTI I/O via `nibabel`, analysis via `numpy`/`scipy`/`pandas`
- Visualizations: `matplotlib` with `seaborn-v0_8-darkgrid` style, dpi=300
- Output figures numbered sequentially in `visual_analysis/` (01_, 02_, etc.)
- FSLeyes commands printed for interactive 3D viewing
- Interactive viewer: `interactive_viewer.py` (menu-driven, options 1-20)

## How to Run

```bash
# Step 1: Build control normative DB (mean/SD perfusion + per-patient z-score)
python 01_build_control_normative.py --group F_20_39

# Step 2: Compute control-referenced zAI per patient
python 02_compute_zai.py

# Step 3: Clinical-grade gray-matter zAI maps
python 03_clinical_maps.py

# Step 4: Publication figures
python 04_publication_figures.py

# Step 5: ROI-based MDT discrimination (Pipeline A)
python 05_roi_discrimination.py

# Step 6: Clinical interpretation report (surgeon-facing) — NEW 2026-05-05
python 06_clinical_interpretation.py --patient P015     # one patient
python 06_clinical_interpretation.py --all              # cohort summary

# Diagnostic / methodology aids
python analyze_zai_distributions.py     # AI distribution histograms

# Interactive 3D viewing (option 15: Clinical Report; option 10: Top-N clusters; option 11: 37-ROI mask)
python interactive_viewer.py
```



## Helper Scripts

`scripts/` directory holds non-Python preprocessing helpers:
- `scripts/resample_parcellation_to_mni.sh` — bash helper to resample a parcellation NIfTI to MNI space (uses FSL `flirt`)

## Known Issues & Solutions

1. **Patient coverage mismatch**: Patients have fewer non-zero voxels than controls. Solution: only compute z-scores where patient has data (`patient_data > 0`).
2. **Extreme z-scores**: Very small SD at some voxels. Solution: clip z-scores to [-20, 20].
3. **White matter noise**: 70% of "significant" voxels were in white matter. Solution: gray-matter-only clinical maps (`03_clinical_maps.py`).
4. **Everything highlighted at z=1.96**: Too liberal for 2.6M voxels. Solution: use stricter threshold (|z|>=3.0) and larger clusters (50+ voxels), or use TFCE/FDR methods.
