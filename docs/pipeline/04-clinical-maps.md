# Step 3: Clinical-Grade zAI Maps

> Part of the canonical voxel-based zAI pipeline (Pipeline B). Step 3 of 4. Produces clean, presentation-ready maps suitable for presurgical review.

## Purpose

Convert the raw whole-brain zAI maps from step 2 into clinically interpretable maps for presurgical planning. Three things change:

1. **Gray-matter masking.** The consensus parcellation is used to restrict zAI to the 34 paired cortical regions (Desikan–Killiany, FreeSurfer labels 1001–1035 / 2001–2035) plus the 6 paired subcortical structures (thalamus, hippocampus, amygdala, caudate, putamen, pallidum). White matter, ventricles, and unlabeled voxels are zeroed. This addresses the known issue that ~70% of unmasked "significant" voxels lie in white matter (`CLAUDE.md` known-issues §3).
2. **Strict thresholding.** `|zAI| ≥ 3.0` (p < 0.003 two-tailed at the voxel level). Much more specific than the conventional `|z| ≥ 1.96` for the ~2.6 M voxels in our brain mask.
3. **Cluster-size filtering.** Minimum 50 voxels (~25.6 mm³ at 0.8 mm isotropic) using 18-connectivity. Removes scattered single-voxel noise; retains focal abnormalities.

Outputs include both a full gray-matter zAI map and a *lateralized* version showing only the dominant (more abnormal) side at each location — the cleanest view for unambiguous left-vs-right reading.

## Script

`03_clinical_maps.py` (canonical, at repo root)

## Inputs

- `results_zscore/asymmetry/patients/<pid>/<pid>_asymmetry_zscore.nii.gz` (from step 2)
- `results_zscore/groups/<group>/consensus_parcellation.nii.gz` (from step 1)
- `results_zscore/groups/<group>/brain_mask.nii.gz` (from step 1)

## Outputs

In `results_zscore/clinical/<pid>/`:

- `<pid>_clinical_gm_zscore.nii.gz` — gray-matter-masked zAI
- `<pid>_clinical_significant_clusters.nii.gz` — surviving clusters
- `<pid>_clinical_hyper.nii.gz` / `<pid>_clinical_hypo.nii.gz` — separated by sign
- `<pid>_clinical_lateralized_significant.nii.gz` — dominant-side-only
- `<pid>_clinical_lateralized_hyper.nii.gz` / `<pid>_clinical_lateralized_hypo.nii.gz`
- `<pid>_clinical_region_report.csv`

The "hyper / hypo" filenames are retained for backwards compatibility with earlier outputs. Semantically for zAI inputs they mean "left-dominant" / "right-dominant" — see `02_compute_zai.py` sign convention.

## Usage

```bash
python 03_clinical_maps.py                                  # all patients
python 03_clinical_maps.py --patient P013                   # single patient
python 03_clinical_maps.py --threshold 3.0 --min-cluster 50 # custom thresholds
```

Typical runtime: ~30 seconds per patient.

## What this step produces (relative to the manuscript)

- Backs the per-patient lateralized panels in Figure 13/14
- Provides the cluster reports cited in §3 Pipeline B results
- Outputs are what `interactive_viewer.py` options 18–20 display

## Dependencies

- Python: `numpy`, `scipy.ndimage` (for connected-component labeling), `pandas`, `nibabel`, `matplotlib`
- Data: `results_zscore/asymmetry/` and `results_zscore/groups/`
- Other scripts: `01_build_control_normative.py`, `02_compute_zai.py` must have been run

## Cross-references

- Manuscript section: §2.11 "Clinical-grade zAI maps"; §3 Pipeline B per-patient results
- Related canonical scripts: `02_compute_zai.py` (input), `04_publication_figures.py` (figures), `interactive_viewer.py` (display)
- Known issue addressed: `CLAUDE.md` known-issues §3 (white-matter noise)
