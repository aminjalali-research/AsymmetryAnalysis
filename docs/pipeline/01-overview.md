# Pipeline Overview

> The canonical reproducible pipeline for the AsymmetryAnalysis project. Two parallel pipelines on the same patient cohort answer one clinical question: *"Where is the epileptogenic zone, and is it lateralized?"*

## What this project answers

For 15 epilepsy patients (P013–P027) with arterial spin labeling (ASL) perfusion MRI, find:

1. **Pipeline A — ROI-based MDT discrimination** (`05_roi_discrimination.py`): Does a patient's left-vs-right perfusion laterality, summarized as a single per-patient score, agree with the MDT-reviewed seizure-onset side?
2. **Pipeline B — Voxel-wise zAI** (`02_compute_zai.py` etc.): At which *voxels* is this patient's left-vs-right asymmetry statistically abnormal compared to age/sex-matched healthy controls?

The two pipelines are independent. Pipeline A is the discrimination claim; Pipeline B is the spatial localization claim. Together they form the manuscript's two core results.

## End-to-end diagram

```
                       Dataset/         DatasetControls/
                          |                    |
                          |                    v
                          |       01_build_control_normative.py
                          |          (per-voxel mean / SD / consensus parc)
                          |                    |
                          v                    v
        +------------ 02_compute_zai.py ------+
        |                    |
        |                    v
        |         03_clinical_maps.py        --> visual_analysis/
        |                    |                  (via 04_publication_figures.py)
        |                    v
        |   results_zscore/clinical/<pid>/...
        |
        v
  05_roi_discrimination.py + ez_ground_truth.py
        |
        v
  ez_analysis_mdt/   (ROC, LOOCV, method comparison)
```

## The 8 canonical scripts

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `01_build_control_normative.py` | Per-voxel control normative DB (mean, SD, consensus parcellation) |
| 2 | `02_compute_zai.py` | Compute control-referenced asymmetry z-scores (zAI) per patient |
| 3 | `03_clinical_maps.py` | Gray-matter masking, strict thresholds, lateralized maps |
| 4 | `04_publication_figures.py` | Publication-quality figures 10–16 |
| 5 | `05_roi_discrimination.py` | ROI-based MDT discrimination, 15 indices × 4 strategies |
| – | `ez_ground_truth.py` | MDT-reviewed clinical labels (imported by step 5) |
| – | `analyze_zai_distributions.py` | Methodology validation (threshold sensitivity) |
| – | `interactive_viewer.py` | FSLeyes-based interactive QC (options 1–20) |

## Claims supported

- **Pipeline A:** AUC and LOOCV accuracy for left-vs-right EZ classification using a single per-patient laterality score (`ez_analysis_mdt/method_performance_*.csv`).
- **Pipeline B:** Per-patient zAI maps with cluster-level reports linking abnormal voxels to FreeSurfer Desikan–Killiany regions (`results_zscore/asymmetry/patients/<pid>/`, `results_zscore/clinical/<pid>/`).

## Read next

- [`02-build-control-normative.md`](02-build-control-normative.md) — control normative DB
- [`03-compute-zai.md`](03-compute-zai.md) — voxel-wise zAI
- [`06-roi-discrimination.md`](06-roi-discrimination.md) — ROI MDT discrimination
- [`quickstart.md`](quickstart.md) — minimal commands to reproduce
- [`install.md`](install.md) — environment setup

---

## From ZSCORE_ANALYSIS_PLAN.md (consolidated 2026-05-05)

Original methodological narrative for the control-referenced z-score asymmetry pipeline (i.e. what is now Pipeline B). Preserved for historical context — most operational details have moved to the per-step pages, but the issues / fixes log and the consensus-parcellation rationale below are unique to this document.

> Note: this document predates the 2026-05-04 pivot. The historical "asymmetry-of-z-score" ordering described below was inverted relative to the literature standard (AI-then-z), and the canonical Pipeline B now does AI-then-z. See [`../manuscript/notes.md` §1](../manuscript/notes.md#1-methods-pivot-history-2026-05-04). The text below is preserved verbatim for the consensus-parcellation, Welford, and issues sections, which still apply.

### Two clinical questions answered by the pipeline

1. **Localization** — Where in this patient's brain is blood flow abnormal compared to healthy controls?
2. **Lateralization** — Where is the patient's left-right asymmetry abnormal compared to healthy controls?

### The 9 control groups (recap)

180 healthy controls (90F, 90M) across 6 age decades are divided into 9 groups (3 sex × 3 age bands) — see [`../data/layout.md`](../data/layout.md) for the canonical table. Each group produces 1 mean scan + 1 SD scan = 18 normative maps total (9 groups × 2). A 25-year-old female is compared to both `F_20_39` (sex-matched) and `FM_20_39` (sex-pooled).

### Welford's online algorithm (rationale)

Used in `01_build_control_normative.py` because:

- Loading all 30 control volumes simultaneously would use ~1.7 GB.
- Welford processes one volume at a time (peak ~340 MB).
- Numerically stable — no catastrophic cancellation from the sum / sum-of-squares approach.

### Consensus parcellation (rationale)

The brain mask and consensus parcellation are built from the controls, not from any single patient:

- Brain mask: voxels where ≥ 75% of subjects have non-zero perfusion.
- Consensus parcellation: majority-vote across all control subjects' `aparc+aseg.nii.gz` files. 110 unique labels from the Desikan–Killiany atlas. Provides unbiased region boundaries that are not distorted by patient pathology.

### Original z-score formulation (historical — superseded by AI-then-z)

The original prototype computed:

```
z(voxel) = (patient_perfusion(voxel) − mean_controls(voxel)) / sd_controls(voxel)
```

with z-scores clipped to [−20, 20] and computed only where the brain mask is valid AND SD > 1e−6 AND the patient also has non-zero perfusion. The asymmetry was then computed on z, using the 11 calculator methods. The current Pipeline B inverts this — AI then z — for the reasons documented in the manuscript pivot notes.

### Issues discovered & solutions (from original prototyping)

| # | Issue | Cause | Fix |
|---|---|---|---|
| 1 | Extreme z-scores (z = 1002) | Patients have fewer non-zero voxels than controls. At ~1.3 M voxels, patient had zero perfusion but controls had signal → z = (0 − mean) / sd → very large false negative. | Only compute z-scores where the patient also has non-zero perfusion. Clip to [−20, 20]. |
| 2 | Everything highlighted (35% of brain at z = 1.96) | White matter (35% of significant voxels) + unlabeled regions (34%) + too-liberal threshold for 2.6 M simultaneous tests. | Gray-matter-only clinical maps with stricter threshold (\|z\| ≥ 3.0, 50+ voxel clusters). Coverage drops from 35% to 5–12%. Implemented in `03_clinical_maps.py`. |
| 3 | Bilateral redundancy | If both L-hippocampus and R-hippocampus show hypoperfusion, both light up. For presurgical planning, only the more abnormal side matters. | Lateralized maps that keep only the dominant side by comparing each voxel to its contralateral mirror. |

### Why an asymmetry-based approach (not raw z-score thresholding) for lateralization

Direct thresholding of absolute z-scores answers "where is the abnormality?" but not "is the abnormality lateralized?". Computing left-vs-right z-asymmetry directly answers the lateralization question, and the 13 thresholding methods (especially TFCE / Quality+TFCE) produce clean, focal results suitable for clinical review. The current canonical pipeline implements this on AI-then-z (Pipeline B) rather than z-then-AI.

### TFCE recommendation (M11)

Threshold-free cluster enhancement integrates cluster size and peak height. Score 0.778 (best of 13 methods on the original benchmarks). Coverage ~3% of brain — clean, focal results. Combined with the quality mask (Pipeline B M11) it remains the recommended thresholding method for clinical reporting.

### Consensus map across thresholding methods

For each voxel, count how many of the 12 non-random methods (excluding M10 random baseline) flag it as significant. Higher values = more confidence. Use `consensus ≥ 6` as a high-confidence threshold for clinical reporting.

**Step 7 (clinical interpretation):** `06_clinical_interpretation.py` produces a surgeon-facing per-patient report with surgical-grade thresholds (|zAI|>=4, >=100 vox), 37-ROI clinical scope, lobe-level aggregation, lateralization confidence (Bartolomei 2008 framework), and explicit MDT concordance. Cohort-level concordance summary at `results_zscore/clinical/cohort_concordance.csv`. See [`09-clinical-interpretation.md`](./09-clinical-interpretation.md) for details.
