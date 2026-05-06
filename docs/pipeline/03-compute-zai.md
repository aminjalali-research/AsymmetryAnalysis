# Step 2: Compute Control-Referenced zAI Maps

> Part of the canonical voxel-based zAI pipeline (Pipeline B). Step 2 of 4. Implements the literature-standard control-referenced asymmetry z-score (zAI).

## Purpose

For every patient, compute a per-voxel z-score of *asymmetry*, not raw perfusion. First the asymmetry index is computed at every voxel from the patient's left and right hemispheres,

```
AI(voxel) = 100 × (Left − Right) / ((Left + Right) / 2)
```

then it is z-scored against the control distribution of asymmetry at that same voxel,

```
zAI(voxel) = (AI_patient − mean_AI_controls) / SD_AI_controls.
```

This is the literature-standard zAI of Shang et al. (2021), Boscolo Galazzo et al. (2016), and Gennari et al. (2025). The audit at `docs/superpowers/audits/2026-05-04-control_asymmetry_zscore-audit.md` confirms this implementation correctly reproduces the published methodology.

**Why this and not raw z-scores.** Raw perfusion z-scores (step 1's by-product) answer "is this spot's blood flow abnormal?" — they yield scattered, bilateral, often diffuse results. zAI answers "is this spot's *left-vs-right* difference abnormal?" — yielding lateralized clusters that align with the clinical question.

## Script

`02_compute_zai.py` (canonical, at repo root)

## Inputs

- `results_zscore/groups/<group>/mean_perfusion.nii.gz` (from step 1)
- `results_zscore/groups/<group>/sd_perfusion.nii.gz` (from step 1; also used to compute the asymmetry control distribution)
- `results_zscore/groups/<group>/consensus_parcellation.nii.gz` (from step 1)
- `Dataset/<pid>/<pid>_perfusion_calib_resampled_to_T1w.nii.gz` (patient ASL, MNI space)

## Outputs

- `results_zscore/asymmetry/patients/<pid>/<pid>_asymmetry_zscore.nii.gz` (signed zAI map)
- `results_zscore/asymmetry/patients/<pid>/<pid>_asymmetry_cluster_report.csv`
- `results_zscore/asymmetry/groups/<group>/brain_mask.nii.gz`

## Usage

```bash
python 02_compute_zai.py                                  # all patients, F_20_39
python 02_compute_zai.py --patient P013                   # single patient
python 02_compute_zai.py --threshold 3.0 --min-cluster 50 # custom thresholds
```

Typical runtime: ~1 minute per patient.

## What this step produces (relative to the manuscript)

- The voxel-wise zAI maps that back Figures 10–14 (`visual_analysis/10_zai_*.png` through `14_zai_*.png`)
- Cluster-level reports used in §3 results tables for Pipeline B
- The input substrate for clinical (gray-matter only) maps in step 3

## Sign convention

- `zAI > 0`: abnormally **left-dominant** perfusion (left higher than right vs. controls)
- `zAI < 0`: abnormally **right-dominant** perfusion (right higher than left vs. controls)

This sign convention applies to *asymmetry of perfusion*, not the EZ side. The EZ-classification convention (used in Pipeline A) inverts this because the EZ shows interictal hypoperfusion — see [`06-roi-discrimination.md`](06-roi-discrimination.md).

## Dependencies

- Python: `numpy`, `scipy`, `pandas`, `nibabel`, `matplotlib`
- Data: `Dataset/`
- Other scripts: `01_build_control_normative.py` must be run first

## Cross-references

- Manuscript section: §2.10 "Pipeline B: Voxel-Wise zAI Analysis"; §3 Pipeline B results
- Audit confirming methodology: `docs/superpowers/audits/2026-05-04-control_asymmetry_zscore-audit.md`
- Related canonical scripts: `01_build_control_normative.py` (input), `03_clinical_maps.py` (consumer), `04_publication_figures.py` (figures)

## Citations

- Shang, S., et al. (2021). *Quantitative Voxel-Based Z-Score Mapping Analysis of Arterial Spin Labeling MR Imaging for Localization of the Epileptogenic Zone in Patients with Focal Cortical Dysplasia.*
- Boscolo Galazzo, I., et al. (2016). *Cerebral metabolism and perfusion in MR-negative individuals with refractory focal epilepsy assessed by simultaneous acquisition of 18F-FDG PET and arterial spin labeling.* NeuroImage: Clinical.
- Gennari, A. G., et al. (2025). *Arterial spin labeling perfusion z-score mapping for epileptogenic zone localization.* (Recent zAI methodology.)
