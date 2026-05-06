# Step 1: Build Control Normative Database

> Part of the canonical voxel-based zAI pipeline (Pipeline B). Required prerequisite for steps 2–4.

## Purpose

Build the per-voxel statistical reference used by every downstream patient analysis: mean and standard-deviation perfusion maps across an age/sex-matched control cohort, plus a majority-vote *consensus parcellation* derived from the controls' FreeSurfer `aparc+aseg` files. The script also emits per-patient raw perfusion z-score maps `(patient − mean)/SD` and a brain mask. All outputs share the MNI152 0.8 mm isotropic geometry (227×272×227).

## Script

`01_build_control_normative.py` (canonical, at repo root)

## Inputs

- `DatasetControls/<group>/<subdir>/` — controls organized by age/sex (e.g. `20_29F/`, `30_39F/`)
- Each control has `*_MNISpace_perfusion_calib_upsampled.nii.gz` and `aparc+aseg.nii.gz`
- `Dataset/<patient_id>/` — patient ASL files (named `*_resampled_to_T1w.nii.gz`, but actually MNI space)
- `clinical_spreadsheet.xlsx` — patient demographics for age/sex matching

## Outputs

- `results_zscore/groups/<group>/mean_perfusion.nii.gz`
- `results_zscore/groups/<group>/sd_perfusion.nii.gz`
- `results_zscore/groups/<group>/consensus_parcellation.nii.gz`
- `results_zscore/groups/<group>/brain_mask.nii.gz`
- `results_zscore/patients/<pid>/<pid>_vs_<group>_zscore.nii.gz`
- `results_zscore/patients/<pid>/<pid>_cluster_report.csv`

## Usage

```bash
python 01_build_control_normative.py                  # all matchable patients, F_20_39
python 01_build_control_normative.py --patient P013   # single patient
python 01_build_control_normative.py --group F_20_39  # specific control group
python 01_build_control_normative.py --rebuild-group  # force rebuild group stats
```

Typical runtime: ~5 minutes for the group statistics build (one-time, cached), ~1 minute per patient.

## What this step produces (relative to the manuscript)

- The control normative model used by Pipeline B (referenced in §2.10 "Pipeline B: Voxel-Wise zAI Analysis")
- Backs the per-voxel SD reported in supplementary methodological detail
- Provides the consensus parcellation used for region-level reporting throughout

## Dependencies

- Python: `numpy`, `scipy`, `pandas`, `nibabel`
- Data: `DatasetControls/`, `Dataset/`, `clinical_spreadsheet.xlsx`
- Other scripts: none (this is step 1)

## Implementation notes

- **Welford's online algorithm** for mean/SD — numerically stable and memory-efficient (avoids loading 30 control volumes simultaneously).
- **Consensus parcellation** — majority vote at each voxel across the per-control `aparc+aseg.nii.gz` after they are resampled to a common space. Falls back to "no label" when no label receives a plurality.
- **z-score clipping** — patient z-scores are clipped to [−20, 20] to mitigate extreme outliers at voxels with very small SD.
- **Coverage rule** — z-scores are only computed where the patient has nonzero data (`patient_data > 0`). This prevents spurious z-scores in patient-side dropouts.

## Cross-references

- Manuscript section: §2.10 "Pipeline B: Voxel-Wise zAI Analysis" (control normative model)
- Audit: `docs/superpowers/audits/2026-05-04-control_asymmetry_zscore-audit.md`
- Related canonical scripts: `02_compute_zai.py` (consumes the outputs)

## Citations

- Welford, B. P. (1962). *Note on a method for calculating corrected sums of squares and products.* Technometrics, 4(3), 419–420.
- Desikan, R. S., et al. (2006). *An automated labeling system for subdividing the human cerebral cortex on MRI scans.* NeuroImage, 31(3), 968–980. (Desikan–Killiany atlas used by FreeSurfer `aparc+aseg`.)
