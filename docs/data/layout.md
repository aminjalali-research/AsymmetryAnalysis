# Data Layout

This document describes every on-disk dataset used by the canonical pipeline. It serves two audiences:

- **Open-source visitor** — to understand what data each script consumes and what it produces.
- **Reviewer reproducing the paper** — to map every figure and number in the manuscript back to a concrete file on disk.

All paths are relative to the repository root unless noted otherwise.

---

## `Dataset/` — patient cohort

15 patients with drug-resistant focal epilepsy (`P013` through `P027`), each in their own subdirectory.

### Per-patient files

| File | Description |
|---|---|
| `aparc+aseg.nii.gz` | FreeSurfer 7.3 parcellation (Desikan–Killiany cortex + subcortical aseg). MNI152 space, 227 × 272 × 227 voxels at 0.8 mm isotropic. |
| `T1w_acpc_dc_restore.nii.gz` | Anterior-Commissure–Posterior-Commissure-aligned, distortion-corrected, intensity-restored T1-weighted volume. MNI space, 0.8 mm iso. |
| `{ID}_perfusion_calib_resampled_to_T1w.nii.gz` | Calibrated CBF map derived from pCASL (BASIL pipeline). **The filename mentions `T1w` but the affine confirms MNI152 space**; this is a historical naming artifact. mL/100 g/min units, 0.8 mm iso. |

### Used by

- `01_build_control_normative.py` — patient demographic match against control groups.
- `02_compute_zai.py` — patient AI map and zAI map computation (Pipeline B).
- `03_clinical_maps.py` — gray-matter-masked clinical zAI maps for surgical review.
- `05_roi_discrimination.py` — ROI-mean perfusion extraction and asymmetry indices (Pipeline A).

---

## `DatasetControls/` — healthy control cohort

Reference cohort organized by demographic subgroup directories (e.g., `20_29F/`, `30_39F/`). Each subgroup directory contains one subdirectory per subject.

### Per-subject files

| File | Description |
|---|---|
| `*_MNISpace_perfusion_calib_upsampled.nii.gz` | Calibrated CBF map, pre-registered to MNI152 at 0.8 mm iso (227 × 272 × 227). The filename's `MNISpace` and `upsampled` markers reflect that these have already been resampled from native pCASL resolution. |
| `aparc+aseg.nii.gz` | FreeSurfer 7.3 parcellation in the same MNI geometry. Used to build the consensus parcellation (majority vote across controls). |

### Available demographic groups

| Group | Subdirectories | Status |
|---|---|---|
| `F_20_39` | `20_29F/`, `30_39F/` | **Available — n = 30** |
| `F_40_59` | `40_49F/`, `50_59F/` | Pending data acquisition |
| `F_60_79` | `60_69F/`, `70_79F/` | Pending data acquisition |
| `M_20_39` | `20_29M/`, `30_39M/` | Pending data acquisition |
| `M_40_59` | `40_49M/`, `50_59M/` | Pending data acquisition |
| `M_60_79` | `60_69M/`, `70_79M/` | Pending data acquisition |
| `FM_20_39` | union of all 20–39 | Pending |
| `FM_40_59` | union of all 40–59 | Pending |
| `FM_60_79` | union of all 60–79 | Pending |

The Pipeline B results in the present manuscript draft therefore cover only the five patients (P013, P014, P015, P020, P026) whose age × sex matches the available `F_20_39` group. Once additional control groups are processed, the remaining 10 patients can be analyzed without code changes.

### Used by

- `01_build_control_normative.py` — Welford-streamed per-voxel mean and SD of perfusion (and of AI) across control subjects per demographic group.
- `02_compute_zai.py` — uses `mean_asymmetry.nii.gz` and `sd_asymmetry.nii.gz` from the matching control group to z-score patient AI maps.
- `analyze_zai_distributions.py` — distributional QC of zAI versus the control reference.

---

## `Dataset MNI/` — companion patient dataset (different preprocessing target)

> Note the **space in the directory name** — `Dataset MNI/`, not `DatasetMNI/`. Quote the path in shell commands.

This is the same 15 patients (`P013`–`P027`) as `Dataset/`, but with files at a different stage of preprocessing. Investigated 2026-05-04 (see `.claude/ISSUES.md` I-2 — RESOLVED, KEEP). It is not a duplicate.

### Per-patient files

| File | Description |
|---|---|
| `T1w_restore.nii.gz` | Restored T1, **not** ACPC-aligned (different preprocessing target than `Dataset/`'s `T1w_acpc_dc_restore.nii.gz`). |
| `perfusion_calib.nii.gz` | CBF at native pCASL resolution (72 × 87 × 72), prior to MNI upsampling. |
| `aparc+aseg.nii.gz` | A second parcellation with a different MD5 from `Dataset/`'s — 3.26 M non-zero voxels here vs 1.96 M in `Dataset/`. Different segmentation pass. |
| `aparc+aseg_mni_resampled.nii.gz` | The above parcellation downsampled to perfusion-native resolution. |

### Used by

- `analyze_zai_distributions.py` — uses `perfusion_calib.nii.gz` to construct the bilateral quality mask at the perfusion-native scale.
- `interactive_viewer.py` — uses `T1w_restore.nii.gz` as the FSLeyes anatomical underlay (the non-ACPC orientation matches the orientation expected by some downstream visualization tools).

### Decision

Keep. Both canonical scripts above will silently fail without it. See `.claude/ISSUES.md` I-2 for the full investigation.

---

## `clinical_spreadsheet.xlsx`

Single-sheet Excel file with patient-level demographic and clinical metadata. One row per patient.

### Columns

| Column | Used for |
|---|---|
| `ID` | Patient identifier (e.g., `sub-P013`). Used to map rows to subdirectories under `Dataset/`. |
| `AGE (SCAN)` | Age at scan, integer years. Used for control demographic matching. |
| `SEX` | `Male` / `Female`. Used for control demographic matching. |
| `EPILEPSY TYPE` | Free text (e.g., "Focal Epilepsy"). |
| `AGE (ONSET)` | Age at first seizure. |
| `MEDICATIONS` | Current AED regimen. |
| `EEG/EMU FINDINGS` | Ictal/interictal monitoring summary — input to MDT lateralization. |
| `CLINICAL MRI` | Radiology read of the structural MRI. |
| `NEUROPSYCHOLOGY` | Memory and language lateralization assessment. |
| `RESEARCH MRI` | Volumetry, MELDgraph, AIDHS, language fMRI, etc. |
| `SURGICAL PLAN AFTER MDT` | MDT-decided surgical plan. |
| `SURGERY DATE` | If operated. |
| `HISTOLOGY` | Resected tissue pathology. |
| `SCANNER` | Field strength / model. |
| `POST-SURGICAL OUTCOME` | Engel-class equivalent (free text). |
| `HANDEDNESS` | `R`/`L`/`A`. |

The MDT-reviewed lateralization label used by the manuscript lives in `ez_ground_truth.py` rather than in this spreadsheet — see [docs/data/ground-truth.md](ground-truth.md).

### Used by

- `01_build_control_normative.py` — reads `AGE (SCAN)` and `SEX` for demographic matching of patients to control groups.

---

## Coordinate system and geometry

All NIfTI files used by the canonical pipeline share one of two geometries:

| Use | Geometry | Voxel size | Source files |
|---|---|---|---|
| Canonical pipeline (Pipelines A and B) | MNI152, 227 × 272 × 227 | 0.8 mm isotropic | `Dataset/`, `DatasetControls/` |
| Native pCASL (auxiliary visualization, QC) | 72 × 87 × 72 | pCASL native (≈ 3 × 3 × 5 mm) | `Dataset MNI/perfusion_calib.nii.gz`, `aparc+aseg_mni_resampled.nii.gz` |

Within each row, all files share identical affines, so **no registration is required at runtime**. The canonical pipeline never mixes the two geometries: pipeline scripts read only the 0.8 mm files; `interactive_viewer.py` and `analyze_zai_distributions.py` read only the perfusion-native files.

If you add a new dataset, ensure its perfusion file passes `nibabel.load(...).affine` equality with the existing controls before placing it in `DatasetControls/`.

---

## FreeSurfer file conventions

### From FREESURFER_FILES_GUIDE.md (consolidated 2026-05-05)

Per-patient `.gz` inventory and recommended FSLeyes overlay combinations for the legacy ROI laterality maps. Most of the FSLeyes commands are also available through `interactive_viewer.py`, but the table and reasoning are useful when constructing custom overlays.

#### Core FreeSurfer files in `Dataset/{patient}/`

| File | Approx. size (P013) | Purpose | Visualization tip |
|---|---|---|---|
| `T1w_acpc_dc_restore.nii.gz` | 38.1 MB | High-resolution T1-weighted anatomical (ACPC-aligned, distortion-corrected) | Use as base/underlay for all overlays |
| `aparc+aseg.nii.gz` | 0.4 MB | FreeSurfer parcellation (Desikan–Killiany cortex + aseg subcortex; 68 cortical + ~30 subcortical regions) | Overlay with `random` colormap to see region boundaries |
| `perfusion_calib.nii.gz` | 0.2 MB | Original ASL perfusion data at native pCASL resolution | Lower resolution; use `hot` colormap |
| `{ID}_perfusion_calib_resampled_to_T1w.nii.gz` | 9.3 MB | Calibrated CBF resampled to T1w/MNI grid | Preferred for analysis — matches T1w resolution |

#### Legacy laterality maps in `laterality_maps/{patient}_*.nii.gz`

| File | Approx. size (P013) | Purpose | Visualization tip |
|---|---|---|---|
| `{ID}_laterality_index_map.nii.gz` | 0.8 MB | Full laterality index map for all regions | `red-yellow` colormap, range −0.4 to 0.4 |
| `{ID}_significant_asymmetry_mask.nii.gz` | 0.4 MB | Binary mask of significant asymmetries (\|LI\| > 0.1) | `green` colormap to highlight survivors |
| `{ID}_left_dominant_regions.nii.gz` | 0.4 MB | Regions where Left > Right (LI > 0.1) | `blue` colormap, range 0.1–0.4 |
| `{ID}_right_dominant_regions.nii.gz` | 0.4 MB | Regions where Right > Left (LI < −0.1) | `red` colormap, range 0.1–0.4 |

#### Recommended overlay combinations (FSLeyes)

```bash
# Basic laterality analysis
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
        laterality_maps/P013_laterality_index_map.nii.gz \
        -cm red-yellow -dr -0.4 0.4 -a 70

# Hemisphere dominance comparison
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
        laterality_maps/P013_left_dominant_regions.nii.gz \
        -cm blue -dr 0.1 0.4 -a 80 -n "Left>Right" \
        laterality_maps/P013_right_dominant_regions.nii.gz \
        -cm red -dr 0.1 0.4 -a 80 -n "Right>Left"

# Perfusion + laterality
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
        Dataset/P013/P013_perfusion_calib_resampled_to_T1w.nii.gz \
        -cm hot -dr 20 80 -a 50 -n "Perfusion" \
        laterality_maps/P013_laterality_index_map.nii.gz \
        -cm red-yellow -dr -0.4 0.4 -a 60 -n "LI Map"

# Quality control (parcellation + significance overlay)
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
        Dataset/P013/aparc+aseg.nii.gz \
        -cm random -dr 1 2035 -a 30 -n "Parcellation" \
        laterality_maps/P013_significant_asymmetry_mask.nii.gz \
        -cm green -dr 0.5 1 -a 70 -n "Significant"

# Comprehensive multi-layer view
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
        Dataset/P013/aparc+aseg.nii.gz \
        -cm random -dr 1 2035 -a 25 -n "Parcellation" \
        Dataset/P013/P013_perfusion_calib_resampled_to_T1w.nii.gz \
        -cm hot -dr 30 80 -a 40 -n "Perfusion" \
        laterality_maps/P013_laterality_index_map.nii.gz \
        -cm red-yellow -dr -0.4 0.4 -a 50 -n "LI Map"
```

#### Color interpretation

- Laterality maps (`red-yellow`): red/warm = right hemisphere dominance (R > L); yellow = left dominance (L > R); dark/zero = balanced.
- Perfusion maps (`hot`): hot colours = higher perfusion; cool/dark = lower perfusion. Typical range 20–80 mL/100 g/min.
- Parcellation (`random`): each colour = a distinct FreeSurfer region.
- Binary masks (`green` etc.): bright = significant findings (\|LI\| > 0.1); dark = non-significant.

#### Dimensions sanity check

- T1w resolution: 256 × 256 × 256 voxels (1 mm³ in the legacy native ACPC volume; the canonical Pipeline-B ACPC-MNI volume is 227 × 272 × 227 at 0.8 mm iso — see "Coordinate system and geometry" above).
- Perfusion native resolution: 64 × 64 × 30 → resampled to T1w space.
- Laterality maps share the same resolution as T1w for precise overlay.

---

## MNI transform notes

### From TRANSFORM_PARCELLATION_TO_MNI.md (consolidated 2026-05-05)

The voxel-ROI extraction step (`voxel_roi_extraction.py`, exposed as Option 9 in the legacy interactive viewer) requires the FreeSurfer parcellation in MNI space. The canonical Pipeline-B scripts already use MNI-aligned parcellations (see `Dataset/`'s `aparc+aseg.nii.gz` and `Dataset MNI/`'s `aparc+aseg_mni_resampled.nii.gz`); the recipe below is preserved for users who need to regenerate `Dataset MNI/{patient}/aparc+aseg_mni.nii.gz` from a fresh FreeSurfer run.

#### Required files per patient

| Role | Path |
|---|---|
| Input — native parcellation | `Dataset/{patient}/aparc+aseg.nii.gz` |
| Input — native T1w | `Dataset/{patient}/T1w_acpc_dc_restore.nii.gz` |
| Reference — MNI T1w | `Dataset MNI/{patient}/T1w_restore.nii.gz` |
| Output | `Dataset MNI/{patient}/aparc+aseg_mni.nii.gz` |

#### Method 1 — ANTs (recommended)

If you already have a transform from native T1w → MNI:

```bash
PATIENT="P022"

antsApplyTransforms -d 3 \
  -i "Dataset/${PATIENT}/aparc+aseg.nii.gz" \
  -r "Dataset MNI/${PATIENT}/T1w_restore.nii.gz" \
  -o "Dataset MNI/${PATIENT}/aparc+aseg_mni.nii.gz" \
  -n NearestNeighbor \
  -t path/to/your/transform_matrix.mat
```

`-n NearestNeighbor` is essential — preserves integer FreeSurfer label values. Do NOT use linear interpolation.

If you need to create the transform first:

```bash
PATIENT="P022"

# Register native T1w to MNI T1w
antsRegistrationSyNQuick.sh -d 3 \
  -f "Dataset MNI/${PATIENT}/T1w_restore.nii.gz" \
  -m "Dataset/${PATIENT}/T1w_acpc_dc_restore.nii.gz" \
  -o "transforms/${PATIENT}_"

# Apply to parcellation
antsApplyTransforms -d 3 \
  -i "Dataset/${PATIENT}/aparc+aseg.nii.gz" \
  -r "Dataset MNI/${PATIENT}/T1w_restore.nii.gz" \
  -o "Dataset MNI/${PATIENT}/aparc+aseg_mni.nii.gz" \
  -n NearestNeighbor \
  -t "transforms/${PATIENT}_1Warp.nii.gz" \
  -t "transforms/${PATIENT}_0GenericAffine.mat"
```

#### Method 2 — FSL FLIRT/FNIRT

```bash
PATIENT="P022"

applywarp \
  --ref="Dataset MNI/${PATIENT}/T1w_restore.nii.gz" \
  --in="Dataset/${PATIENT}/aparc+aseg.nii.gz" \
  --out="Dataset MNI/${PATIENT}/aparc+aseg_mni.nii.gz" \
  --warp=path/to/your/warp.nii.gz \
  --premat=path/to/your/affine.mat \
  --interp=nn
```

`--interp=nn` is essential — preserves integer label values.

#### Batch script for all 15 patients

```bash
#!/bin/bash
PATIENTS=(P013 P014 P015 P016 P017 P018 P019 P020 P021 P022 P023 P024 P025 P026 P027)

for PATIENT in "${PATIENTS[@]}"; do
    echo "Processing ${PATIENT}..."

    if [ ! -f "Dataset/${PATIENT}/aparc+aseg.nii.gz" ]; then
        echo "  WARNING: native parcellation not found, skipping..."
        continue
    fi
    if [ ! -f "Dataset MNI/${PATIENT}/T1w_restore.nii.gz" ]; then
        echo "  WARNING: MNI T1w not found, skipping..."
        continue
    fi

    antsApplyTransforms -d 3 \
      -i "Dataset/${PATIENT}/aparc+aseg.nii.gz" \
      -r "Dataset MNI/${PATIENT}/T1w_restore.nii.gz" \
      -o "Dataset MNI/${PATIENT}/aparc+aseg_mni.nii.gz" \
      -n NearestNeighbor \
      -t "transforms/${PATIENT}_1Warp.nii.gz" \
      -t "transforms/${PATIENT}_0GenericAffine.mat"

    if [ $? -eq 0 ]; then echo "  OK: ${PATIENT}"; else echo "  FAILED: ${PATIENT}"; fi
done
```

#### Verification

```bash
# Existence + dimensions
ls -lh "Dataset MNI/P022/aparc+aseg_mni.nii.gz"
fslinfo "Dataset MNI/P022/aparc+aseg_mni.nii.gz"
fslinfo "Dataset MNI/P022/T1w_restore.nii.gz"   # must match

# Visual overlay check
fsleyes "Dataset MNI/P022/T1w_restore.nii.gz" \
        "Dataset MNI/P022/aparc+aseg_mni.nii.gz" \
        -cm random -dr 1 2035 -a 50 &
```

Expected dimensions match the project MNI grid (227 × 272 × 227 at 0.8 mm) or your downstream-target template (commonly 91 × 109 × 91 or 72 × 87 × 72 at the perfusion-native scale).

#### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Labels are blurred / non-integer | Linear interpolation used | Re-run with `-n NearestNeighbor` (ANTs) or `--interp=nn` (FSL) |
| Parcellation doesn't align with MNI T1w | Wrong transform matrix | Verify the matrix maps native → MNI in the correct direction |
| File size huge (~30 MB+) | Likely interpolation issue or wrong reference | Output should be similar size to MNI T1w (~0.5–2 MB) |
| Missing transform matrices | Never ran the registration | Run `antsRegistrationSyNQuick.sh` (see "create the transform" above) |

#### After transformation — running the dependent step

```bash
python voxel_roi_extraction.py
python interactive_viewer.py     # then choose option 9 for the patient
# Outputs land in:
#   results_voxel_roi/{PID}/{PID}_voxel_roi_statistics.csv
#   results_voxel_roi/all_patients_voxel_roi_statistics.csv
```

The transformation preserves FreeSurfer label IDs (e.g. `1001` = `ctx-lh-bankssts`). All 37 bilateral region pairs are extracted; statistics include mean, median, std, z-score, and significance percentage; bilateral statistics are computed across both left and right ROIs.
